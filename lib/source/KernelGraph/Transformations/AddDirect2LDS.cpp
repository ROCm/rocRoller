#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDirect2LDS.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
namespace rocRoller
{
    namespace KernelGraph
    {
        bool isChain(KernelGraph const& kgraph, int op1, int op2)
        {
            for(auto tag : kgraph.control.depthFirstVisit(op1, Graph::Direction::Downstream))
            {
                if(tag == op2)
                {
                    return true;
                }
            }
            return false;
        }

        bool isLeaf(KernelGraph const& kgraph, int tag)
        {
            auto children
                = kgraph.control.getNeighbours<Graph::Direction::Downstream>(tag).to<std::vector>();
            return children.empty();
        }

        std::vector<std::pair<int, int>> searchCandidates(KernelGraph const&   kgraph,
                                                          CommandParametersPtr params,
                                                          ContextPtr           context)
        {
            auto loadTiledNodes    = kgraph.control.getNodes<LoadTiled>().to<std::vector>();
            auto storeLDSTileNodes = kgraph.control.getNodes<StoreLDSTile>().to<std::vector>();
            std::vector<std::pair<int, int>> result;
            for(auto loadGlobal : loadTiledNodes)
            {
                auto internalMacroTile = kgraph.mapper.get<MacroTile>(loadGlobal);
                auto macTile           = kgraph.coordinates.getNode<MacroTile>(internalMacroTile);

                for(auto storeLDS : storeLDSTileNodes)
                {

                    bool sameMacroTile
                        = (kgraph.mapper.get<MacroTile>(storeLDS) == internalMacroTile);
                    auto useSwappedAccess = params->transposeMemoryAccess[macTile.layoutType];
                    auto ldsWriteStride   = useSwappedAccess
                                                ? (macTile.sizes[1] / macTile.subTileSizes[1])
                                                : (macTile.sizes[0] / macTile.subTileSizes[0]);

                    auto LDSTileTag = kgraph.mapper.get<LDS>(storeLDS);
                    auto LDSTile    = kgraph.coordinates.getNode<LDS>(LDSTileTag);
                    if(!LDSTile.isDirect2LDS)
                    {
                        Log::debug("  LDSTile {} is not Direct2LDS.", LDSTileTag);
                        continue;
                    }

                    if(sameMacroTile)
                    {
                        auto const lanesPerWavefront = context->targetArchitecture().GetCapability(
                            GPUCapability::DefaultWavefrontSize);
                        AssertFatal(ldsWriteStride % lanesPerWavefront == 0);
                        result.push_back({loadGlobal, storeLDS});
                    }
                }
            }
            return result;
        }

        void mergeOperations(KernelGraph& kgraph, int globalOp, int ldsOp)
        {
            auto variableType = getVariableType(kgraph, globalOp);

            if(isChain(kgraph, globalOp, ldsOp))
            {
                Log::debug("  Merge LoadTiled {} and StoreLDSTile {} chain.", globalOp, ldsOp);

                // create LoadTileDirect2LDS operation
                auto direct2lds = kgraph.control.addElement(LoadTileDirect2LDS(variableType));
                moveConnections(kgraph, globalOp, direct2lds);
                moveConnections(kgraph, ldsOp, direct2lds);

                // replace operations
                replaceWith(kgraph, globalOp, kgraph.control.addElement(NOP()), false);
                replaceWith(kgraph, ldsOp, direct2lds, false);

                purgeNodes(kgraph, {globalOp});
                purgeNodes(kgraph, {ldsOp});
            }
            else
            {
                Log::debug("  Merge LoadTiled {} and StoreLDSTile {} leaf.", globalOp, ldsOp);

                // the LoadTiled and StoreLDSTile have to be the leaf node
                AssertFatal(isLeaf(kgraph, globalOp) && isLeaf(kgraph, ldsOp));

                // create LoadTileDirect2LDS operation
                auto direct2lds = kgraph.control.addElement(LoadTileDirect2LDS(variableType));
                moveConnections(kgraph, globalOp, direct2lds);
                moveConnections(kgraph, ldsOp, direct2lds);

                // find the barrier before StoreLDSTile operation
                int ldsBarrier      = -1;
                int computeIndexTag = -1;
                for(auto parent : kgraph.control.depthFirstVisit(ldsOp, Graph::Direction::Upstream))
                {
                    bool containing = ldsBarrier != -1
                                      && (kgraph.control.get<Body>(ldsBarrier)
                                          || kgraph.control.get<Sequence>(ldsBarrier));
                    ldsBarrier = parent;
                    if(kgraph.control.get<ComputeIndex>(parent))
                    {
                        computeIndexTag = parent;
                    }

                    auto maybeBarrier = kgraph.control.get<Barrier>(parent);
                    if(maybeBarrier && containing)
                    {
                        break;
                    }
                }
                AssertFatal(computeIndexTag != -1 && ldsBarrier != -1);

                // add LoadTileDirect2LDS operation to the graph
                auto barrier = kgraph.control.addElement(Barrier());
                reconnect<Graph::Direction::Upstream>(kgraph, -1, computeIndexTag);
                kgraph.control.addElement(Sequence(), {globalOp}, {computeIndexTag});
                replaceWith(kgraph, globalOp, kgraph.control.addElement(NOP()), false);
                replaceWith(kgraph, ldsBarrier, kgraph.control.addElement(NOP()), false);
                replaceWith(kgraph, ldsOp, barrier, false);
                kgraph.control.addElement(Sequence(), {barrier}, {direct2lds});
                purgeNodes(kgraph, {globalOp});
                purgeNodes(kgraph, {ldsOp});
                purgeNodes(kgraph, {ldsBarrier});
            }
        }
        /** This transformation does:
         *
         *    1. Search the pairs of LoadTiled and StoreLDSTile operations that connects to the same internal MacroTile
         *
         *    2. Merge each pair
         */
        KernelGraph AddDirect2LDS::apply(KernelGraph const& original)
        {
            auto kgraph = original;

            Log::debug("  AddDirect2LDS control graph transform.");
            auto candidates = searchCandidates(kgraph, m_params, m_context);

            if(candidates.size() > 0)
            {
                AssertFatal(
                    m_context->targetArchitecture().HasCapability(GPUCapability::HasDirectToLds),
                    "Not have DirectToLds capability");

                // TODO: remove it after fixing the wait count observer for Direct-to-LDS
                m_context->setWaitZeroBeforeBarrier(true);
            }

            for(auto loadAndStore : candidates)
            {
                mergeOperations(kgraph, loadAndStore.first, loadAndStore.second);
            }

            return kgraph;
        }
    }
}
