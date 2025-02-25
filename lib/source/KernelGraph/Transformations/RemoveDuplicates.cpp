
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveDuplicates.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace LoadStoreSpecification
        {
            /**
             * @brief Load/store specification.
             *
             * This uniquely identifies an load/store operation.
             */
            struct Spec
            {
                /// Target coordinate (source for loads, destination
                /// for stores).  This is a storage node: User, LDS
                /// etc
                int targetCoord;

                /// Parent operation.  If the load is within a loop,
                /// this is the ForLoopOp.  If not, this is -1.
                int parentOp;

                /// Operation variant.  Differentiates between the
                /// operation types.
                int opType;

                /// Unroll colour: unroll coordinate to unroll value
                /// mapping.
                std::map<int, int> unrollColour;

                auto operator<=>(Spec const&) const = default;
            };

            std::string toString(Spec const& x)
            {
                std::stringstream os;
                os << "{";
                os << " target: " << x.targetCoord;
                os << ", parent: " << x.parentOp;
                os << ", type: " << x.opType;
                os << ", colour: [ ";
                for(auto [coord, value] : x.unrollColour)
                {
                    os << "(" << coord << ", " << value << "), ";
                }
                os << "] }";
                return os.str();
            }

            /**
             * @brief Return load/store specifier for the load/store operation.
             */
            Spec getLoadStoreSpecifier(KernelGraph const&     k,
                                       UnrollColouring const& colouring,
                                       int                    opTag)
            {
                auto [target, direction] = getOperationTarget(opTag, k);

                auto maybeForLoop = findContainingOperation<ForLoopOp>(opTag, k);

                auto parentOp = -1;
                if(maybeForLoop)
                    parentOp = *maybeForLoop;

                auto unrollColour = std::map<int, int>{};
                if(maybeForLoop && colouring.operationColour.contains(opTag))
                {
                    unrollColour = colouring.operationColour.at(opTag);
                }

                int opType = k.control.getNode<Operation>(opTag).index();

                return {target, parentOp, opType, unrollColour};
            }
        }

        namespace RD
        {
            bool loadForLDS(int loadTag, KernelGraph const& graph)
            {
                namespace CT = rocRoller::KernelGraph::CoordinateGraph;

                auto dst = graph.mapper.get<MacroTile>(loadTag);
                dst      = only(graph.coordinates.getOutputNodeIndices(dst, CT::isEdge<Duplicate>))
                          .value_or(dst);

                bool hasLDS = false;
                for(auto output : graph.coordinates.getInputNodeIndices(dst, CT::isEdge<DataFlow>))
                {
                    if(graph.coordinates.get<LDS>(output).has_value())
                        return true;
                }
                return false;
            }

            /**
             * @brief Return set of workgroup operations to consider
             * for deduplication.
             *
             * This looks for global loads and LDS stores that operate
             * on workgroup tiles.
             */
            std::unordered_set<int> getWGCandidates(KernelGraph const& graph)
            {
                std::unordered_set<int> candidates;
                for(auto op : graph.control.getNodes())
                {
                    auto isLoadTiled    = graph.control.get<LoadTiled>(op).has_value();
                    auto isStoreLDSTile = graph.control.get<StoreLDSTile>(op).has_value();
                    if(!(isLoadTiled || isStoreLDSTile))
                        continue;

                    // Only consider within the KLoop
                    auto maybeForLoop = findContainingOperation<ForLoopOp>(op, graph);
                    if(!maybeForLoop)
                        continue;
                    auto forLoop = *graph.control.get<ForLoopOp>(*maybeForLoop);
                    if(forLoop.loopName != rocRoller::KLOOP)
                        continue;

                    // If LoadTiled, don't consider when loading directly
                    if(isLoadTiled && !loadForLDS(op, graph))
                        continue;

                    // If we get this far, the operation is a
                    // "workgroup" operation.  This implies that
                    // jammed colours should be squashed.
                    candidates.insert(op);
                }
                return candidates;
            }

            /**
             * @brief Return set of wave operations to consider for
             * deduplication.
             *
             * This looks for LDS loads that operate on wave tiles.
             */
            std::unordered_set<int> getWaveCandidates(KernelGraph const& graph)
            {
                std::unordered_set<int> candidates;
                for(auto op : graph.control.getNodes())
                {
                    auto isLoadTiled   = graph.control.get<LoadTiled>(op).has_value();
                    auto isLoadLDSTile = graph.control.get<LoadLDSTile>(op).has_value();

                    if(!(isLoadTiled || isLoadLDSTile))
                        continue;

                    // Only consider within the KLoop
                    auto maybeForLoop = findContainingOperation<ForLoopOp>(op, graph);
                    if(!maybeForLoop)
                        continue;
                    auto forLoop = *graph.control.get<ForLoopOp>(*maybeForLoop);
                    if(forLoop.loopName != rocRoller::KLOOP)
                        continue;

                    // If LoadTiled, only consider when loading directly
                    if(isLoadTiled && loadForLDS(op, graph))
                        continue;

                    candidates.insert(op);
                }
                return candidates;
            }

            /**
             * @brief Return set of Unroll coordinates that are
             * neighbours of ForLoops.
             *
             * These are "jammed" coordinates.
             */
            std::unordered_set<int> jammedColours(KernelGraph const& graph)
            {
                std::unordered_set<int> rv;
                for(auto forLoopOpTag : graph.control.getNodes<ForLoopOp>())
                {
                    auto forLoopOp = *graph.control.get<ForLoopOp>(forLoopOpTag);

                    if(forLoopOp.loopName != rocRoller::KLOOP)
                    {
                        auto [forLoopCoord, _ignore] = getForLoopCoords(forLoopOpTag, graph);
                        auto maybeUnroll             = findUnrollNeighbour(graph, forLoopCoord);
                        if(maybeUnroll)
                            rv.insert(*maybeUnroll);
                    }
                }
                return rv;
            }

            /**
             * @brief Return mapping of "existing" operation to set of
             * "duplicate" operations (subset of `candidates`) that
             * should be removed from the control graph.
             *
             * Note that the "existing" operation is
             * NoderOrdering::LeftFirst of all members of the
             * "duplicate" set.
             *
             * An operation is a duplicate of another if:
             * - they are the same operations (LoadTiled, StoreLDSTile, LoadLDSTile)
             * - they have the same body-parent (typically ForLoopOp)
             * - they have the same target (User, LDS)
             * - they have the same unroll colouring
             *
             * If `squashJammedColours` is enabled: any colouring
             * associated with a JammedWaveTileNumber is squashed (set
             * to zero) when comparing unroll colouring.
             */
            std::map<int, std::unordered_set<int>> findDuplicates(KernelGraph const&     graph,
                                                                  UnrollColouring const& colouring,
                                                                  auto const&            candidates,
                                                                  bool squashJammedColours)
            {
                using namespace LoadStoreSpecification;

                std::map<int, std::unordered_set<int>> duplicates;
                std::map<Spec, int>                    opSpecs;

                // First pass, create specification to earliest-operation mapping
                for(auto op : candidates)
                {
                    auto spec = getLoadStoreSpecifier(graph, colouring, op);
                    if(opSpecs.contains(spec))
                    {
                        auto originalOp = opSpecs[spec];
                        auto ordering   = graph.control.compareNodes(op, originalOp);
                        if(ordering == NodeOrdering::LeftFirst)
                            opSpecs[spec] = op;
                    }
                    else
                    {
                        opSpecs[spec] = op;
                    }
                }

                std::unordered_set<int> squashColours;
                if(squashJammedColours)
                    squashColours = jammedColours(graph);

                // Second pass, mark duplicates
                for(auto op : candidates)
                {
                    auto spec = getLoadStoreSpecifier(graph, colouring, op);

                    auto originalSpec = spec;
                    if(!squashColours.empty())
                    {
                        for(auto kv : originalSpec.unrollColour)
                        {
                            if(squashColours.contains(kv.first))
                                originalSpec.unrollColour[kv.first] = 0;
                        }
                    }

                    if(opSpecs.contains(originalSpec))
                    {
                        auto originalOp = opSpecs[originalSpec];
                        if(op != originalOp)
                        {
                            duplicates[originalOp].insert(op);
                            Log::debug("Marking as duplicate: {} (orignal {}) {}",
                                       op,
                                       originalOp,
                                       toString(spec));
                        }
                    }
                }

                return duplicates;
            }

            /**
             * @brief Remove duplicates from the graph.
             *
             * The operation (either the original operation, or it's
             * top-most SetCoordinate) is replaced by a NOP.
             *
             * References to duplicate tiles are updated to the
             * non-duplicate tiles.
             */
            KernelGraph removeDuplicates(KernelGraph const& original, auto const& duplicates)
            {
                auto graph = original;

                GraphReindexer expressionReindexer;

                for(auto [existingOp, duplicateOps] : duplicates)
                {
                    auto topExistingOp = getTopSetCoordinate(graph, existingOp);
                    for(auto duplicateOp : duplicateOps)
                    {
                        auto topDuplicateOp = getTopSetCoordinate(graph, duplicateOp);

                        // Replace top-duplicate with a NOP, then purge the top-duplicate.
                        //
                        // Recall that (see findDuplicates) any
                        // operation we replace with a NOP happens
                        // *after* the existing operation that it
                        // duplicates.
                        auto nop = graph.control.addElement(NOP());
                        replaceWith(graph, topDuplicateOp, nop, false);
                        purgeNodeAndChildren(graph, topDuplicateOp);

                        Log::debug("RemoveDuplicates: NOPing {} (top of {}) because it duplicates "
                                   "{} (top of {}).  NOP is {}.",
                                   topDuplicateOp,
                                   duplicateOp,
                                   topExistingOp,
                                   existingOp,
                                   nop);

                        // Mark for update: references to the
                        // duplicate tile need to be updated to the
                        // existing tile.
                        auto tileTag = original.mapper.get<MacroTile>(duplicateOp);
                        if(tileTag != -1)
                        {
                            auto existingTileTag = original.mapper.get<MacroTile>(existingOp);
                            expressionReindexer.coordinates.emplace(tileTag, existingTileTag);
                        }
                    }
                }

                // Update tile references
                auto kernel = *graph.control.roots().begin();
                reindexExpressions(graph, kernel, expressionReindexer);

                // ZZZ Remove old tiles

                // Update tile connections
                for(auto [oldTag, newTag] : expressionReindexer.coordinates)
                {
                    auto connections = graph.mapper.getCoordinateConnections(oldTag);
                    graph.mapper.purgeMappingsTo(oldTag);
                    for(auto conn : connections)
                    {
                        conn.coordinate = newTag;
                        graph.mapper.connect(conn);
                    }
                }
                return graph;
            }
        }

        KernelGraph RemoveDuplicates::apply(KernelGraph const& original)
        {
            auto colouring = colourByUnrollValue(original);
            auto graph     = removeRedundantSequenceEdges(original);

            // Two passes:
            //   1. Workgroup (LoadTiled and StoreLDSTile) and
            //   2. Wave (LoadLDSTile)
            //
            // We assume that these passes are independent of each
            // other.  In particular, the first pass doesn't change
            // the colouring etc that is used for the second pass.

            // LoadTile and StoreLDSTile operate on Workgroup tiles
            Log::debug("RemoveDuplicates Workgroup pass");
            {
                auto candidates = RD::getWGCandidates(original);
                auto duplicates = RD::findDuplicates(original, colouring, candidates, true);
                graph           = RD::removeDuplicates(graph, duplicates);
            }

            // LoadLDSTile operation on Wave tiles
            Log::debug("RemoveDuplicates Wave pass");
            {
                auto candidates = RD::getWaveCandidates(original);
                auto duplicates = RD::findDuplicates(original, colouring, candidates, false);
                graph           = RD::removeDuplicates(graph, duplicates);
            }

            return removeRedundantNOPs(graph);
        }
    }
}
