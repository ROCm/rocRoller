
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/UnrollLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        using GD = Graph::Direction;

        using namespace ControlGraph;
        using namespace CoordinateGraph;

        enum RAW
        {
            INVALID,
            NAUGHT,
            WRITE,
            READ,
        };

        /**
         * @brief Gets the name of the current for loop.
         *
         * @param graph
         * @param start
         * @return std::string
         */
        std::string getForLoopName(KernelGraph& graph, int start)
        {
            // Find the number of forLoops downstream from start
            auto forLoop = graph.control.get<ForLoopOp>(start);
            return forLoop->name;
        }

        /**
         * @brief Determine how many times to unroll the loop.
         *
         * A value of 1 means do not unroll it.
         * Use getForLoopName to determine which forLoop we are attempting to unroll
         * Checks unrollX(Y) value, 0 is default unroll it all if we can.
         */
        unsigned int
            getUnrollAmount(KernelGraph& graph, int loopTag, KernelOptions const& kernelOptions)
        {
            auto dimTag        = graph.mapper.get<Dimension>(loopTag);
            auto forLoopLength = getSize(std::get<Dimension>(graph.coordinates.getElement(dimTag)));

            auto unrollX = kernelOptions.unrollX;
            auto unrollY = kernelOptions.unrollY;
            auto unrollK = kernelOptions.unrollK;
            // Find the number of forLoops following this for loop.
            auto name = getForLoopName(graph, loopTag);
            if(name == rocRoller::XLOOP && unrollX > 0)
                return unrollX;
            else if(name == rocRoller::YLOOP && unrollY > 0)
                return unrollY;
            else if(name == rocRoller::KLOOP && unrollK > 0)
                return unrollK;

            // Use default behavior if the above isn't true
            // If loop length is a constant, unroll the loop by that amount
            if(Expression::evaluationTimes(forLoopLength)[Expression::EvaluationTime::Translate])
            {
                auto length = Expression::evaluate(forLoopLength);
                if(isInteger(length))
                    return getUnsignedInt(length);
            }

            return 1u;
        }

        std::optional<int> getForLoopCoordinate(KernelGraph const& kgraph, int forLoop)
        {
            auto iterator = kgraph.mapper.get<Dimension>(forLoop);
            return only(kgraph.coordinates.getOutputNodeIndices<DataFlowEdge>(iterator));
        }

        /**
         * @brief Container to describe a loop-carried dependency.
         */
        struct LoopCarriedDependency
        {
            int coordinate; //< Coordinate that has a loop-carried dependency
            int control; //< Operation that needs to be made sequential within the unroll to satisfy loop-carried dependencies.
        };

        /**
         * @brief Find loop-carried dependencies.
         *
         * If a coordinate:
         *
         * 1. is written to only once
         * 2. is read-after-write
         * 3. has the loop coordinate in it's coordinate transform
         *    for the write
         *
         * then we take a giant leap of faith and say that it
         * doesn't have loop-carried-dependencies.
         *
         * Returns a map with a coordinate node as a key and a list of
         * every control node that uses that coordinate node as its value.
         */
        std::map<int, std::vector<int>> findLoopCarriedDependencies(KernelGraph const& kgraph,
                                                                    int                forLoop)
        {
            using RW = ControlFlowRWTracer::ReadWrite;
            std::map<int, std::vector<int>> result;

            rocRoller::Log::getLogger()->debug("KernelGraph::findLoopDependencies({})", forLoop);

            auto maybeTopForLoopCoord = getForLoopCoordinate(kgraph, forLoop);
            if(!maybeTopForLoopCoord)
                return result;
            auto topForLoopCoord = *maybeTopForLoopCoord;

            ControlFlowRWTracer tracer(kgraph);
            tracer.trace(forLoop);

            auto readwrite = tracer.coordinatesReadWrite();

            // Coordinates that are read/write in loop body
            std::unordered_set<int> coordinates;
            for(auto m : readwrite)
                coordinates.insert(m.coordinate);

            // Coordinates with loop-carried-dependencies; true by
            // default.
            std::unordered_set<int> loopCarriedDependencies;
            for(auto m : readwrite)
                loopCarriedDependencies.insert(m.coordinate);

            // Now we want to determine which coordinates don't have
            // loop carried dependencies...
            //
            // If a coordinate:
            //
            // 1. is written to only once
            // 2. is read-after-write
            // 3. has the loop coordinate in it's coordinate transform
            //    for the write
            //
            // then we take a giant leap of faith and say that it
            // doesn't have loop-carried-dependencies.
            for(auto coordinate : coordinates)
            {
                std::optional<int> writeOperation;

                // Is coordinate RAW?
                //
                // RAW finite state machine moves through:
                //   NAUGHT -> WRITE   upon write
                //   WRITE  -> READ    upon read
                //   READ   -> READ    upon read
                //          -> INVALID otherwise
                //
                // If it ends in READ, we have a RAW
                RAW raw = RAW::NAUGHT;
                for(auto x : readwrite)
                {
                    if(x.coordinate != coordinate)
                        continue;
                    // Note: "or raw == RAW::WRITE" is absent on purpose;
                    // only handle single write for now
                    if((raw == RAW::NAUGHT) && (x.rw == RW::WRITE || x.rw == RW::READWRITE))
                    {
                        raw            = RAW::WRITE;
                        writeOperation = x.control;
                        continue;
                    }
                    if((raw == RAW::WRITE || raw == RAW::READ) && (x.rw == RW::READ))
                    {
                        raw = RAW::READ;
                        continue;
                    }
                    raw = RAW::INVALID;
                }

                if(raw != RAW::READ)
                    continue;

                AssertFatal(writeOperation);

                // Does the coordinate transform for the write
                // operation contain the loop iteration?

                auto [target, direction] = getOperationTarget(*writeOperation, kgraph);
                auto [required, path]    = findRequiredCoordinates(target, direction, kgraph);
                if(required.size() == 0)
                {
                    // Local variable
                    loopCarriedDependencies.erase(coordinate);
                    continue;
                }
                auto forLoopCoords = filterCoordinates<ForLoop>(required, kgraph);
                for(auto forLoopCoord : forLoopCoords)
                    if(forLoopCoord == topForLoopCoord)
                        loopCarriedDependencies.erase(coordinate);
            }

            // For the loop-carried-depencies, find the write operation.
            for(auto coordinate : loopCarriedDependencies)
            {
                for(auto x : readwrite)
                {
                    if(x.coordinate != coordinate)
                        continue;
                    if(x.rw == RW::WRITE || x.rw == RW::READWRITE)
                        result[coordinate].push_back(x.control);
                }
            }

            return result;
        }

        /**
         * @brief Make operations sequential.
         *
         * Make operations in `sequentialOperations` execute
         * sequentially.  This changes concurrent patterns similar to
         *
         *     Kernel/Loop/Scope
         *       |           |
         *      ...         ...
         *       |           |
         *     OperA       OperB
         *       |           |
         *      ...         ...
         *
         * into a sequential pattern
         *
         *     Kernel/Loop/Scope
         *       |           |
         *      ...         ...
         *       |           |
         *   SetCoord --->SetCoord ---> remaining
         *       |           |
         *     OperA       OperB
         *
         */
        void makeSequential(KernelGraph&                         graph,
                            const std::vector<std::vector<int>>& sequentialOperations)
        {
            for(int i = 0; i < sequentialOperations.size() - 1; i++)
            {
                graph.control.addElement(Sequence(),
                                         {sequentialOperations[i].back()},
                                         {sequentialOperations[i + 1].front()});
            }
        }

        struct UnrollLoopsVisitor : public BaseGraphVisitor
        {
            UnrollLoopsVisitor(ContextPtr context)
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
                auto bodies = graph.control.getOutputNodeIndices<Body>(newTag).to<std::vector>();

                auto unrollAmount = getUnrollAmount(graph, newTag, m_context->kernelOptions());
                if(unrollAmount == 1)
                    return;

                auto loopCarriedDependencies = findLoopCarriedDependencies(original, tag);
                if(loopCarriedDependencies.empty())
                    return;

                // TODO: Iron out storage node duplication
                //
                // If we aren't doing the KLoop: some assumptions made
                // during subsequent lowering stages (LDS) are no
                // longer satisifed.
                //
                // For now, to avoid breaking those assumptions, only
                // follow through with loop-carried analysis when we
                // are doing the K loop.
                //
                // To disable loop-carried analysis, just set the
                // dependencies to an empty list...
                //
                // The only test which tries to unroll the K-loop is
                // the BasicGEMMUnrollK test.
                if(getForLoopName(graph, newTag) != rocRoller::KLOOP || unrollAmount <= 1)
                {
                    loopCarriedDependencies = {};
                }

                std::unordered_set<int> dontDuplicate;
                for(auto const& [coord, controls] : loopCarriedDependencies)
                {
                    dontDuplicate.insert(coord);
                }

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
                auto unrollDimension = graph.coordinates.addElement(Unroll(unrollAmount));
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

                auto newAddExpr            = lhs + (rhs * Expression::literal(unrollAmount));
                loopIncrementOp.expression = newAddExpr;

                graph.control.setElement(loopIncrement[0], loopIncrementOp);

                // ------------------------------
                // Add a setCoordinate node in between the original ForLoopOp and the loop bodies

                // Delete edges between original ForLoopOp and original loop body
                for(auto const& child :
                    graph.control.getNeighbours<Graph::Direction::Downstream>(newTag)
                        .to<std::vector>())
                {
                    if(isEdge<Body>(graph.control.getElement(child)))
                    {
                        graph.control.deleteElement(child);
                    }
                }

                // Function for adding a SetCoordinate nodes around all load and
                // store operations. Only adds a SetCoordinate node if the coordinate is needed
                // by the load/store operation.
                auto connectWithSetCoord = [&](const auto& toConnect, unsigned int coordValue) {
                    for(auto const& body : toConnect)
                    {
                        graph.control.addElement(Body(), {newTag}, {body});
                        for(auto const& op : findComputeIndexCandidates(graph, body))
                        {
                            auto [target, direction] = getOperationTarget(op, graph);
                            auto [required, path]
                                = findRequiredCoordinates(target, direction, graph);

                            if(path.count(unrollDimension) > 0)
                            {
                                auto setCoord = replaceWith(graph,
                                                            op,
                                                            graph.control.addElement(SetCoordinate(
                                                                Expression::literal(coordValue))),
                                                            false);
                                graph.mapper.connect<Unroll>(setCoord, unrollDimension);

                                graph.control.addElement(Body(), {setCoord}, {op});
                            }
                        }
                    }
                };

                std::vector<std::vector<int>> duplicatedBodies;

                // ------------------------------
                // Create duplicates of the loop body and populate the sequentialOperations
                // data structure. sequentialOperations is a map that uses a coordinate with
                // a loop carried dependency as a key and contains a vector of vectors
                // with every control node that uses that coordinate in each new body of the loop.
                std::map<int, std::vector<std::vector<int>>> sequentialOperations;
                for(unsigned int i = 0; i < unrollAmount; i++)
                {
                    GraphReindexer unrollReindexer;

                    auto dontDuplicatePedicate = [&](int x) { return dontDuplicate.contains(x); };

                    if(i == 0)
                        duplicatedBodies.push_back(bodies);
                    else
                        duplicatedBodies.push_back(duplicateControlNodes(
                            graph, unrollReindexer, bodies, dontDuplicatePedicate));

                    for(auto const& [coord, controls] : loopCarriedDependencies)
                    {
                        std::vector<int> dupControls;
                        for(auto const& control : controls)
                        {
                            auto newOp = reindexer.control.at(control);
                            if(i == 0)
                            {
                                dupControls.push_back(newOp);
                            }
                            else
                            {
                                auto dupOp = unrollReindexer.control.at(newOp);
                                dupControls.push_back(dupOp);
                            }
                        }
                        sequentialOperations[coord].emplace_back(std::move(dupControls));
                    }
                }

                // Connect the duplicated bodies to the loop, adding SetCoordinate nodes
                // as needed.
                for(int i = 0; i < unrollAmount; i++)
                {
                    connectWithSetCoord(duplicatedBodies[i], i);
                }

                // If there are any loop carried dependencies, add Sequence nodes
                // between the control nodes with dependencies.
                for(auto [coord, allControls] : sequentialOperations)
                {
                    makeSequential(graph, allControls);
                }
            }
        };

        KernelGraph UnrollLoops::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::unrollLoops");
            auto visitor = UnrollLoopsVisitor(m_context);
            return rewrite(original, visitor);
        }
    }
}
