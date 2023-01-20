#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        /**
         * @brief Create duplicates of all of the nodes downstream of the provided
         *        start nodes.
         *        Add the duplicates to the provided graph.
         *        Return the location of the new start nodes.
         *
         * @param graph
         * @param startNodes
         * @return std::vector<int>
         */
        std::vector<int> duplicateControlNodes(KernelGraph& graph, std::set<int> const& startNodes)
        {
            std::vector<int> newStartNodes;
            GraphReindexer   reindexer;

            // Create duplicates of all of the nodes downstream of the startNodes
            for(auto const& node :
                graph.control.depthFirstVisit(startNodes, Graph::Direction::Downstream))
            {
                // Only do this step if element is a node
                if(graph.control.getElementType(node) == Graph::ElementType::Node)
                {
                    auto op = graph.control.addElement(graph.control.getElement(node));
                    reindexer.control[node] = op;
                }
            }

            for(auto const& reindex : reindexer.control)
            {
                // Create all edges within new sub-graph
                auto location = graph.control.getLocation(reindex.first);
                for(auto const& output : location.outgoing)
                {
                    int child = *graph.control.getNeighbours<Graph::Direction::Downstream>(output)
                                     .begin();
                    graph.control.addElement(graph.control.getElement(output),
                                             {reindex.second},
                                             {reindexer.control[child]});
                }

                // Use the same coordinate graph mappings
                for(auto const& c : graph.mapper.getConnections(reindex.first))
                {
                    auto coord = c.coordinate;
                    // If one of the mappings represents storage, duplicate it in the CoordinateGraph.
                    // This will allow the RegisterTagManager to recognize that the nodes are pointing
                    // to different data and to use different registers.
                    // Note: no edges are being added, to there will be loose Dimensions within the
                    // Coordinate graph.
                    if(c.tindex == typeid(MacroTile))
                    {
                        if(reindexer.coordinates.count(c.coordinate) == 0)
                        {
                            auto dim = graph.coordinates.addElement(
                                graph.coordinates.getElement(c.coordinate));
                            reindexer.coordinates[c.coordinate] = dim;
                        }
                        coord = reindexer.coordinates[c.coordinate];
                    }
                    graph.mapper.connect(reindex.second, coord, c.tindex, c.subDimension);
                }
            }

            // Change coordinate values in Expressions
            for(auto const& reindex : reindexer.control)
            {
                auto elem = graph.control.getElement(reindex.first);
                if(isOperation<Assign>(elem))
                {
                    auto new_assign = graph.control.getNode<Assign>(reindex.second);
                    ReindexExpressionVisitor visitor(reindexer);
                    new_assign.expression = visitor.call(new_assign.expression);
                    graph.control.setElement(reindex.second, new_assign);
                }
            }

            // Return the new start nodes
            for(auto const& startNode : startNodes)
            {
                newStartNodes.push_back(reindexer.control[startNode]);
            }

            return newStartNodes;
        }

        /**
         * @brief Determine whether to unroll a specified loop
         *
         */
        bool performUnroll(KernelGraph& graph, int loopTag)
        {
            auto dimTag = graph.mapper.get<Dimension>(loopTag);
            auto dim    = std::get<Dimension>(graph.coordinates.getElement(dimTag));
            if(identical(getSize(dim), Expression::literal(1u)))
                return false;

            // TODO: Better loop dependency checker
            // Do not unroll loops that have a dependency between iterations.
            // At the moment, we are saying that if the first node in a loop's
            // body is a Multiply, then we know it has a dependency. This should
            // be improved in the future.
            auto directBodyNodes = graph.control.getOutputNodeIndices<Body>(loopTag);

            for(auto const& directBodyNode : directBodyNodes)
            {
                auto x = std::get<Operation>(graph.control.getElement(directBodyNode));
                if(std::holds_alternative<Multiply>(x))
                    return false;
            }

            return true;
        }

        struct UnrollLoopsVisitor : public BaseGraphVisitor
        {
            UnrollLoopsVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context, Graph::Direction::Upstream, false)
            {
            }

            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        ForLoopOp const&   op) override
            {
                copyOperation(graph, original, reindexer, tag);
                auto newTag = reindexer.control.at(tag);
                auto bodies = graph.control.getOutputNodeIndices<Body>(newTag).to<std::set>();

                if(!performUnroll(graph, newTag))
                    return;

                // ---------------------------------
                // Add Unroll dimension to the coordinates graph

                // Use the same coordinate graph mappings
                auto loopIterator = original.mapper.getConnections(tag)[0].coordinate;

                // The loop iterator should have a dataflow link to the ForLoop dimension
                auto forLoopDimension
                    = graph.coordinates.getOutputNodeIndices<DataFlowEdge>(loopIterator)
                          .to<std::vector>()[0];

                // Find all incoming PassThrough edges to the ForLoop dimension and replace them with
                // a Split edge with an Unroll dimension.
                auto forLoopLocation = graph.coordinates.getLocation(forLoopDimension);
                auto unrollDimension = graph.coordinates.addElement(Unroll(UNROLL_AMOUNT));
                for(auto const& input : forLoopLocation.incoming)
                {
                    if(isEdge<PassThrough>(std::get<Edge>(graph.coordinates.getElement(input))))
                    {
                        int parent
                            = *graph.coordinates.getNeighbours<Graph::Direction::Upstream>(input)
                                   .begin();
                        graph.coordinates.addElement(
                            Split(), {parent}, {forLoopDimension, unrollDimension});
                        graph.coordinates.deleteElement(input);
                    }
                }

                // Find all outgoing PassThrough edges from the ForLoop dimension and replace them with
                // a Join edge with an Unroll dimension.
                for(auto const& output : forLoopLocation.outgoing)
                {
                    if(isEdge<PassThrough>(std::get<Edge>(graph.coordinates.getElement(output))))
                    {
                        int child
                            = *graph.coordinates.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        graph.coordinates.addElement(
                            Join(), {forLoopDimension, unrollDimension}, {child});
                        graph.coordinates.deleteElement(output);
                    }
                }

                // ------------------------------
                // Change the loop increment calculation
                // Multiply the increment amount by the unroll amount

                // Find the ForLoopIcrement calculation
                // TODO: Handle multiple ForLoopIncrement edges that might be in a different
                // format, such as ones coming from ComputeIndex.
                auto loopIncrement = graph.control.getOutputNodeIndices<ForLoopIncrement>(newTag)
                                         .to<std::vector>();
                AssertFatal(loopIncrement.size() == 1, "Should only have 1 loop increment edge");

                auto loopIncrementOp = graph.control.getNode<Assign>(loopIncrement[0]);

                auto [lhs, rhs] = getForLoopIncrement(graph, newTag);

                auto newAddExpr            = lhs + (rhs * Expression::literal(UNROLL_AMOUNT));
                loopIncrementOp.expression = newAddExpr;

                graph.control.setElement(loopIncrement[0], loopIncrementOp);

                // ------------------------------
                // Add a setCoordinate node in between the original ForLoopOp and the loop bodies

                // Delete edges between original ForLoopOp and original loop body
                for(auto const& child :
                    graph.control.getNeighbours<Graph::Direction::Downstream>(newTag))
                {
                    if(isEdge<Body>(graph.control.getElement(child)))
                    {
                        graph.control.deleteElement(child);
                    }
                }

                // Function for adding a SetCoordinate node inbetween the ForLoop
                // and a list of nodes.
                auto connectWithSetCoord = [&](auto const& toConnect, unsigned int coordValue) {
                    auto setCoord
                        = graph.control.addElement(SetCoordinate(Expression::literal(coordValue)));
                    graph.mapper.connect<Unroll>(setCoord, unrollDimension);
                    graph.control.addElement(Body(), {newTag}, {setCoord});
                    for(auto const& body : toConnect)
                    {
                        graph.control.addElement(Body(), {setCoord}, {body});
                    }
                };

                // Add setCoordinate nodes to original body
                connectWithSetCoord(bodies, 0u);

                // ------------------------------
                // Create duplicates of the loop body and add a setCoordinate node in between
                // the ForLoopOp and the new bodies
                for(unsigned int i = 1; i < UNROLL_AMOUNT; i++)
                {
                    auto newBodies = duplicateControlNodes(graph, bodies);

                    connectWithSetCoord(newBodies, i);
                }
            }

        private:
            const unsigned int UNROLL_AMOUNT = 2;
        };

        KernelGraph unrollLoops(KernelGraph const& k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::unrollLoops");
            auto visitor = UnrollLoopsVisitor(context);
            return rewrite(k, visitor);
        }
    }
}
