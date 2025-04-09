/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        /**
         * @brief Fuse Loops Transformation
         *
         * This transformation looks for the following pattern:
         *
         *     ForLoop
         * |            |
         * Set Coord    Set Coord
         * |            |
         * For Loop     For Loop
         *
         * If it finds it, it will fuse the lower loops into a single
         * loop, as long as they are the same size.
         */
        namespace FuseLoopsNS
        {
            using GD = rocRoller::Graph::Direction;

            std::vector<std::unordered_set<int>> gatherForLoops(KernelGraph& graph, int start)
            {
                auto bodies      = graph.control.getOutputNodeIndices<Body>(start).to<std::set>();
                auto isForLoopOp = graph.control.isElemType<ForLoopOp>();
                auto isSetCoordinate = graph.control.isElemType<SetCoordinate>();
                auto isSequence      = graph.control.isElemType<Sequence>();
                auto isBody          = graph.control.isElemType<Body>();

                auto maybeForLoops
                    = graph.control.depthFirstVisit(bodies, isSequence, GD::Downstream)
                          .filter([&](int tag) -> bool {
                              if(isForLoopOp(tag))
                              {
                                  return true;
                              }
                              if(isSetCoordinate(tag))
                              {
                                  return graph.mapper.get<CoordinateGraph::ForLoop>(tag) > 0;
                              }
                              return false;
                          })
                          .to<std::unordered_set>();

                std::vector<int> forLoops;
                for(auto const& maybeForLoop : maybeForLoops)
                {
                    if(isForLoopOp(maybeForLoop))
                    {
                        forLoops.insert(forLoops.begin(), maybeForLoop);
                    }
                    else
                    {
                        // The filter means that this is now a SetCoordinate connected to an Unroll dimension
                        // Tail loops are always downstream of these
                        std::optional<int> tag = maybeForLoop;
                        while(tag && isSetCoordinate(*tag))
                        {
                            tag = only(graph.control.getOutputNodeIndices<Body>(*tag));
                        }
                        if(tag && isForLoopOp(*tag))
                        {
                            forLoops.push_back(*tag);
                        }
                    }
                }

                std::vector<std::unordered_set<int>> loopGroupsToFuse;
                while(!forLoops.empty())
                {
                    std::unordered_set<int>   loopGroup;
                    Expression::ExpressionPtr loopIncrement;
                    Expression::ExpressionPtr loopLength;
                    std::map<int, int>        baseLoopContents;
                    for(auto const& loop : forLoops)
                    {
                        if(loopGroup.count(loop) != 0)
                            continue;

                        auto loopDim = getSize(std::get<CT::Dimension>(graph.coordinates.getElement(
                            graph.mapper.get(loop, NaryArgument::DEST))));

                        // Loops to be fused must have the same length
                        if(loopLength)
                        {
                            if(!identical(loopDim, loopLength))
                                continue;
                        }
                        else
                        {
                            loopLength = loopDim;
                        }

                        // Loops to be fused must have the same increment value
                        auto [dataTag, increment] = getForLoopIncrement(graph, loop);
                        if(loopIncrement)
                        {
                            if(!identical(loopIncrement, increment))
                                continue;
                        }
                        else
                        {
                            loopIncrement = increment;
                        }

                        // Loop similarity heuristic
                        // We only want to fuse loops which are "the same" or similar.
                        // Currently this is just counting the number of each type of node in the body.
                        // We may want to replace this with a better heuristic in the future.
                        auto               loopContentsGenerator = graph.control.nodesInBody(loop);
                        std::map<int, int> loopContents;
                        std::for_each(loopContentsGenerator.begin(),
                                      loopContentsGenerator.end(),
                                      [&](int tag) -> void {
                                          auto type = graph.control.get<Operation>(tag)->index();
                                          if(loopContents.contains(type))
                                          {
                                              loopContents[type]++;
                                          }
                                          else
                                          {
                                              loopContents[type] = 1;
                                          }
                                      });

                        if(!baseLoopContents.empty())
                        {
                            if(baseLoopContents != loopContents)
                                continue;
                        }
                        else
                        {
                            baseLoopContents = loopContents;
                        }

                        loopGroup.insert(loop);
                    }

                    for(auto loop : loopGroup)
                    {
                        std::erase(forLoops, loop);
                    }

                    if(loopGroup.size() > 1)
                    {
                        loopGroupsToFuse.push_back(loopGroup);
                    }
                }

                return loopGroupsToFuse;
            }

            void fuseNode(KernelGraph& graph, int fusedNodeTag, int nodeTag)
            {
                for(auto const& child :
                    graph.control.getOutputNodeIndices<Sequence>(nodeTag).to<std::vector>())
                {
                    if(fusedNodeTag != child)
                    {
                        graph.control.addElement(Sequence(), {fusedNodeTag}, {child});
                    }
                    graph.control.deleteElement<Sequence>(std::vector<int>{nodeTag},
                                                          std::vector<int>{child});
                    std::unordered_set<int> toDelete;
                    for(auto descSeqOfChild :
                        filter(graph.control.isElemType<Sequence>(),
                               graph.control.depthFirstVisit(child, GD::Downstream)))
                    {
                        if(graph.control.getNeighbours<GD::Downstream>(descSeqOfChild)
                               .to<std::unordered_set>()
                               .contains(fusedNodeTag))
                        {
                            toDelete.insert(descSeqOfChild);
                        }
                    }
                    for(auto edge : toDelete)
                    {
                        graph.control.deleteElement(edge);
                    }
                }

                for(auto const& child :
                    graph.control.getOutputNodeIndices<Body>(nodeTag).to<std::vector>())
                {
                    graph.control.addElement(Body(), {fusedNodeTag}, {child});
                    graph.control.deleteElement<Body>(std::vector<int>{nodeTag},
                                                      std::vector<int>{child});
                }

                for(auto const& parent :
                    graph.control.getInputNodeIndices<Sequence>(nodeTag).to<std::vector>())
                {
                    auto descOfFusedLoop
                        = graph.control
                              .depthFirstVisit(fusedNodeTag,
                                               graph.control.isElemType<Sequence>(),
                                               GD::Downstream)
                              .to<std::unordered_set>();

                    if(!descOfFusedLoop.contains(parent))
                    {
                        graph.control.addElement(Sequence(), {parent}, {fusedNodeTag});
                    }
                    graph.control.deleteElement<Sequence>(std::vector<int>{parent},
                                                          std::vector<int>{nodeTag});
                }

                for(auto const& parent :
                    graph.control.getInputNodeIndices<Body>(nodeTag).to<std::vector>())
                {
                    graph.control.addElement(Body(), {parent}, {fusedNodeTag});
                    graph.control.deleteElement<Body>(std::vector<int>{parent},
                                                      std::vector<int>{nodeTag});
                }
            }

            struct IsSameOperationVisitor
            {
                template <CConcreteOperation OpA, CConcreteOperation OpB>
                bool operator()(int, OpA const&, int, OpB const&)
                {
                    return false;
                }

                bool operator()(int tagA, SetCoordinate const& A, int tagB, SetCoordinate const& B)
                {
                    auto connA = graph.mapper.getConnections(tagA);
                    auto connB = graph.mapper.getConnections(tagB);

                    if(connA.size() != connB.size())
                    {
                        return false;
                    }
                    for(auto iterA = connA.begin(), iterB = connB.begin(); iterA != connA.end();
                        iterA++, iterB++)
                    {
                        if(iterA->coordinate != iterB->coordinate)
                            return false;
                        if(iterA->connection != iterB->connection)
                            return false;
                    }
                    return identical(A.value, B.value);
                }

                bool call(int tagA, int tagB)
                {
                    return std::visit(*this,
                                      std::variant<int>(tagA),
                                      graph.control.getNode(tagA),
                                      std::variant<int>(tagB),
                                      graph.control.getNode(tagB));
                }

                KernelGraph const& graph;
            };

            void fuseScopes(KernelGraph& graph, int tag)
            {
                auto parentsWithEdges
                    = graph.control.getInputNodeIndices<Body>(tag).template to<std::vector>();
                std::set<std::set<int>> nodeSetsToMerge;
                IsSameOperationVisitor  visitor{graph};

                for(auto const& A : parentsWithEdges)
                {
                    std::set<int> sameAsThis;
                    for(auto const& B : parentsWithEdges)
                    {
                        if(A == B)
                            continue;
                        if(visitor.call(A, B))
                        {
                            sameAsThis.insert(B);
                        }
                    }
                    if(!sameAsThis.empty())
                    {
                        sameAsThis.insert(A);
                        nodeSetsToMerge.insert(sameAsThis);
                    }
                }

                for(auto mergeSet : nodeSetsToMerge)
                {
                    auto mergedNodeTag = *mergeSet.begin();
                    for(auto const& nodeTag : mergeSet)
                    {
                        if(nodeTag == mergedNodeTag)
                            continue;
                        fuseNode(graph, mergedNodeTag, nodeTag);

                        graph.control.deleteElement(nodeTag);
                        graph.mapper.purge(nodeTag);
                    }
                    fuseScopes(graph, mergedNodeTag);
                }
            }

            void fuseLoops(KernelGraph& graph, int tag)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::fuseLoops({})", tag);

                auto loopGroupsToFuse = gatherForLoops(graph, tag);
                for(auto forLoopsToFuse : loopGroupsToFuse)
                {
                    if(forLoopsToFuse.size() <= 1)
                        return;

                    auto dontWalkPastForLoop = [&](int tag) -> bool {
                        for(auto neighbour : graph.control.getNeighbours(tag, GD::Downstream))
                        {
                            if(graph.control.get<ForLoopOp>(neighbour))
                            {
                                return false;
                            }
                        }
                        return true;
                    };
                    auto fusedLoopTag = *forLoopsToFuse.begin();

                    for(auto const& forLoopTag : forLoopsToFuse)
                    {
                        if(forLoopTag == fusedLoopTag)
                            continue;

                        fuseNode(graph, fusedLoopTag, forLoopTag);
                        purgeFor(graph, forLoopTag);
                    }

                    fuseScopes(graph, fusedLoopTag);

                    auto children
                        = graph.control.getOutputNodeIndices<Body>(fusedLoopTag).to<std::vector>();

                    auto loads = filter(graph.control.isElemType<LoadTiled>(),
                                        graph.control.depthFirstVisit(
                                            children, dontWalkPastForLoop, GD::Downstream))
                                     .to<std::vector>();

                    auto ldsLoads = filter(graph.control.isElemType<LoadLDSTile>(),
                                           graph.control.depthFirstVisit(
                                               children, dontWalkPastForLoop, GD::Downstream))
                                        .to<std::vector>();

                    auto stores = filter(graph.control.isElemType<StoreTiled>(),
                                         graph.control.depthFirstVisit(
                                             children, dontWalkPastForLoop, GD::Downstream))
                                      .to<std::vector>();

                    auto ldsStores = filter(graph.control.isElemType<StoreLDSTile>(),
                                            graph.control.depthFirstVisit(
                                                children, dontWalkPastForLoop, GD::Downstream))
                                         .to<std::vector>();

                    orderMemoryNodes(graph, loads, true);
                    orderMemoryNodes(graph, ldsLoads, true);
                    orderMemoryNodes(graph, stores, true);
                    orderMemoryNodes(graph, ldsStores, true);
                }
            }
        }

        KernelGraph FuseLoops::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::fuseLoops");

            auto newGraph = k;

            for(const auto node :
                newGraph.control.depthFirstVisit(*newGraph.control.roots().begin()))
            {
                if(isOperation<ForLoopOp>(newGraph.control.getElement(node)))
                {
                    FuseLoopsNS::fuseLoops(newGraph, node);
                }
            }

            return newGraph;
        }

        ConstraintStatus BodyOfOnlyOneNode(const KernelGraph& graph)
        {
            ConstraintStatus retval;
            for(auto tag : graph.control.leaves().to<std::vector>())
            {
                auto containing = graph.control.nodesContaining(tag).to<std::unordered_set>();
                for(auto contA : containing)
                {
                    for(auto contB : containing)
                    {
                        if(contA == contB)
                        {
                            continue;
                        }
                        auto order = graph.control.compareNodes(contA, contB);
                        if(!(order == NodeOrdering::LeftInBodyOfRight
                             || order == NodeOrdering::RightInBodyOfLeft))
                        {
                            retval.combine(
                                false,
                                concatenate(
                                    "Nodes ", contA, " and ", contB, "have intersecting bodies."));
                        }
                    }
                }
            }
            return retval;
        }

        std::vector<GraphConstraint> FuseLoops::postConstraints() const
        {
            return {BodyOfOnlyOneNode};
        }
    }
}
