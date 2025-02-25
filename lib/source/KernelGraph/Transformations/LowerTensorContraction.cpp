
#include <algorithm>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTensorContraction.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        void duplicateMacroTile(KernelGraph& graph, int load)
        {
            auto original = graph.mapper.get<MacroTile>(load);
            auto newMacroTile
                = graph.coordinates.addElement(graph.coordinates.getElement(original));
            graph.coordinates.addElement(Duplicate(), {newMacroTile}, {original});
            graph.mapper.disconnect<MacroTile>(load, original);
            graph.mapper.connect<MacroTile>(load, newMacroTile);
        }

        void addConnectionsMultiply(KernelGraph& graph,
                                    int          waveMult,
                                    int          loadTag,
                                    NaryArgument argType)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTensorContraction::addConnectionsMultiply(): Multiply({})",
                waveMult);

            auto userTag   = graph.mapper.get<User>(loadTag);
            auto macroTile = graph.mapper.get<MacroTile>(loadTag);

            auto [waveTag, wave] = graph.getDimension<WaveTile>(loadTag);

            graph.mapper.connect(
                waveMult, macroTile, Connections::typeArgument<MacroTile>(argType));
            graph.mapper.connect(waveMult, waveTag, Connections::typeArgument<WaveTile>(argType));
        }

        int getUserFromDataFlow(KernelGraph const& graph, int start, Graph::Direction direction)
        {
            auto predicate
                = [&](int x) -> bool { return graph.coordinates.get<DataFlow>(x).has_value(); };
            for(auto elem : graph.coordinates.depthFirstVisit(start, predicate, direction))
            {
                auto maybeUser = graph.coordinates.get<User>(elem);
                if(maybeUser)
                    return elem;
            }
            return -1;
        }

        struct LoadStoreInfo
        {
            int global   = -1; //< Global memory operation: LoadTiled or StoreTiled.
            int storeLDS = -1; //< Store to LDS operation: StoreLDSTile
            int loadLDS  = -1; //< Load from LDS operation: LoadLDSTile

            /// If true, this is a
            ///     LoadTile+StoreLDSTile+LoadLDSTile
            /// chain.  If false, this is a
            ///     StoreLDSTile+LoadLDSTile+StoreTiled
            /// chain.
            bool isLoad;

            /// For load operations, this is the operation and fills
            /// the destination tile.  For non-LDS loads this is the
            /// global operation.  For LDS loads this is the loadLDS
            /// operation.
            int load() const
            {
                return loadLDS == -1 ? global : loadLDS;
            }
        };

        Graph::Direction getCoordinateAddDirection(KernelGraph& graph, int opTag)
        {
            auto isLoadTiled   = graph.control.get<LoadTiled>(opTag).has_value();
            auto isLoadLDSTile = graph.control.get<LoadLDSTile>(opTag).has_value();
            if(isLoadTiled || isLoadLDSTile)
                return Graph::Direction::Downstream;
            auto isStoreTiled   = graph.control.get<StoreTiled>(opTag).has_value();
            auto isStoreLDSTile = graph.control.get<StoreLDSTile>(opTag).has_value();
            if(isStoreTiled || isStoreLDSTile)
                return Graph::Direction::Upstream;
            Throw<FatalError>("Cannot determine direction: invalid operation.");
        }

        /**
         * @brief
         */
        void connectJammedOperation(KernelGraph& graph, int opTag, int waveTilesX, int waveTilesY)
        {
            if(opTag == -1)
                return;

            if(getCoordinateAddDirection(graph, opTag) == Graph::Direction::Downstream)
            {
                auto jammedX = graph.mapper.get<JammedWaveTileNumber>(opTag, 0);
                if(jammedX != -1)
                    graph.coordinates.addElement(PassThrough(), {jammedX}, {waveTilesX});
                auto jammedY = graph.mapper.get<JammedWaveTileNumber>(opTag, 1);
                if(jammedY != -1)
                    graph.coordinates.addElement(PassThrough(), {jammedY}, {waveTilesY});
            }
            else
            {
                auto jammedX = graph.mapper.get<JammedWaveTileNumber>(opTag, 0);
                if(jammedX != -1)
                    graph.coordinates.addElement(PassThrough(), {waveTilesX}, {jammedX});
                auto jammedY = graph.mapper.get<JammedWaveTileNumber>(opTag, 1);
                if(jammedY != -1)
                    graph.coordinates.addElement(PassThrough(), {waveTilesY}, {jammedY});
            }
        }

        void connectJammedOperation(KernelGraph&         graph,
                                    LoadStoreInfo const& info,
                                    int                  waveTilesX,
                                    int                  waveTilesY)
        {
            if(info.global == -1)
                return;
            connectJammedOperation(graph, info.global, waveTilesX, waveTilesY);
            connectJammedOperation(graph, info.storeLDS, waveTilesX, waveTilesY);
            connectJammedOperation(graph, info.loadLDS, waveTilesX, waveTilesY);
        }

        void connectJammedOperation(KernelGraph&            graph,
                                    std::vector<int> const& opTags,
                                    int                     waveTilesX,
                                    int                     waveTilesY)
        {
            for(auto opTag : opTags)
                connectJammedOperation(graph, opTag, waveTilesX, waveTilesY);
        }

        void connectJammedOperation(KernelGraph&                 graph,
                                    std::optional<LoadStoreInfo> maybeInfo,
                                    int                          waveTilesX,
                                    int                          waveTilesY)
        {
            if(maybeInfo)
                connectJammedOperation(graph, *maybeInfo, waveTilesX, waveTilesY);
        }

        void connectGlobalLoadOperations(KernelGraph& graph, int forLoop, LoadStoreInfo const& info)
        {
            if(info.loadLDS == -1)
                return;

            auto edge = *only(graph.control.getNeighbours(info.global, Graph::Direction::Upstream));
            graph.control.deleteElement(edge);
            graph.control.addElement(Body(), {forLoop}, {info.global});

            // XXX: Attach MacroTileNumber of LDS buffers to K ForLoop coordinate
            // graph.coordinates.addElement(
            //     PassThrough(), {K}, {graph.mapper.get<MacroTileNumber>(chain->storeLDSTile)});
            // graph.coordinates.addElement(
            //     PassThrough(), {graph.mapper.get<MacroTileNumber>(chain->loadLDSTile)}, {K});
        }

        /**
         * Returns global-load-and-store-to-lds chain ({top, bottom} pair) above LoadLDSTile.
         */
        std::optional<LoadStoreInfo> getLoadStoreInfo(std::optional<int> maybeOp,
                                                      KernelGraph const& graph)
        {
            if(!maybeOp)
                return {};

            auto op = maybeOp.value();

            auto load = graph.control.get<LoadTiled>(op);
            if(load)
            {
                return LoadStoreInfo{op, -1, -1, true};
            }

            auto loadLDS = graph.control.get<LoadLDSTile>(op);
            if(loadLDS)
            {
                auto storeLDSTag = *only(graph.control.getInputNodeIndices<Sequence>(op));
                auto storeLDS    = graph.control.get<StoreLDSTile>(storeLDSTag);
                if(!storeLDS)
                    return {};

                auto loadTileTag = *only(graph.control.getInputNodeIndices<Sequence>(storeLDSTag));
                auto loadTile    = graph.control.get<LoadTiled>(loadTileTag);
                if(!loadTile)
                    return {};

                return LoadStoreInfo{loadTileTag, storeLDSTag, op, true};
            }

            auto store = graph.control.get<StoreTiled>(op);
            if(store)
            {
                auto loadLDSTag = *only(graph.control.getInputNodeIndices<Sequence>(op));
                auto loadLDS    = graph.control.get<LoadLDSTile>(loadLDSTag);
                if(!loadLDS)
                    return LoadStoreInfo{op, -1, -1, false};

                auto storeLDSTag = *only(graph.control.getInputNodeIndices<Sequence>(loadLDSTag));
                auto storeLDS    = graph.control.get<StoreLDSTile>(storeLDSTag);
                if(!storeLDS)
                    return {};

                return LoadStoreInfo{op, storeLDSTag, loadLDSTag, false};
            }

            return {};
        }

        struct MatrixMultiplyInfo
        {
            int                          kernel; //< Kernel operation
            LoadStoreInfo                loadA; //< Load operation that loads the A (LHS) operand
            LoadStoreInfo                loadB; //< Load operation that loads the B (RHS) operand
            std::optional<LoadStoreInfo> loadAScale; //< Load operation that loads the A (LHS) scale
            std::optional<LoadStoreInfo> loadBScale; //< Load operation that loads the B (RHS) scale
            std::optional<LoadStoreInfo> storeD; //< Store operation that stores the result (D)
            int                          userA; //< Tag of global A Tensor
            int                          userB; //< Tag of global B Tensor
            int                          userAScale; //< Tag of global A Scale Tensor
            int                          userBScale; //< Tag of global B Scale Tensor

            std::vector<int> dependentAssigns; //< Assign operations that use the result (D)
            std::vector<int> siblingLoads; //< Load operations that flow into dependentAssigns
            std::vector<int> siblingOps; //< Other operations that flow into dependentAssigns
        };

        MatrixMultiplyInfo getMatrixMultiplyInfo(KernelGraph const& graph, int tensorContractionTag)
        {
            MatrixMultiplyInfo info;

            auto parents = graph.control.parentNodes(tensorContractionTag).to<std::vector>();
            AssertFatal(parents.size() == 2 || parents.size() == 4, ShowValue(parents.size()));

            // Get tensor contraction operands

            std::optional<int> operandA, operandAScale;
            std::optional<int> operandB, operandBScale;

            std::map<int, int> parentTags;
            for(auto p : parents)
            {
                auto mapped        = graph.mapper.get<MacroTile>(p);
                parentTags[mapped] = p;
            }

            {
                auto [aTag, aTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::LHS);
                operandA = parentTags.at(aTag);
                parentTags.erase(aTag);
                info.userA = getUserFromDataFlow(graph, aTag, Graph::Direction::Upstream);
            }

            {
                auto [bTag, bTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::RHS);
                operandB = parentTags.at(bTag);
                parentTags.erase(bTag);
                info.userB = getUserFromDataFlow(graph, bTag, Graph::Direction::Upstream);
            }

            if(parents.size() == 4)
            {
                // TODO: We will need to implement versions of getDimension that return an
                // optional<>.  Once that is done, we can support one scaled input and one
                // non-scaled input.

                auto [aScaledTag, aScaledTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::LHS_SCALE);
                operandAScale = parentTags.at(aScaledTag);
                parentTags.erase(aScaledTag);
                info.userAScale
                    = getUserFromDataFlow(graph, aScaledTag, Graph::Direction::Upstream);

                auto [bScaledTag, bScaledTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::RHS_SCALE);
                operandBScale = parentTags.at(bScaledTag);
                parentTags.erase(bScaledTag);
                info.userBScale
                    = getUserFromDataFlow(graph, bScaledTag, Graph::Direction::Upstream);
            }

            AssertFatal(parentTags.empty());

            info.loadA      = getLoadStoreInfo(operandA, graph).value();
            info.loadB      = getLoadStoreInfo(operandB, graph).value();
            info.loadAScale = getLoadStoreInfo(operandAScale, graph);
            info.loadBScale = getLoadStoreInfo(operandBScale, graph);

            // Find loads, stores, assigns etc
            auto reachableFromTC
                = graph.control.depthFirstVisit(tensorContractionTag).to<std::unordered_set>();

            std::vector<int> stores;
            for(auto const index : graph.control.getNodes())
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{[&](auto op) {},
                                            [&](StoreTiled const& store) {
                                                if(reachableFromTC.contains(index))
                                                    stores.push_back(index);
                                            },
                                            [&](Assign const& op) {
                                                if(reachableFromTC.contains(index))
                                                    info.dependentAssigns.push_back(index);
                                            }},
                      std::get<Operation>(elem));
            }

            AssertFatal(info.loadA.global != -1);
            AssertFatal(info.loadB.global != -1);

            AssertFatal(stores.size() <= 1);
            if(!stores.empty())
                info.storeD = getLoadStoreInfo(stores[0], graph);

            auto root = only(graph.control.roots());
            AssertFatal(root, "More than one Kernel node not supported");
            info.kernel = *root;

            // Find sibling loads and ops
            auto upstreamOfTC
                = graph.control.depthFirstVisit(tensorContractionTag, Graph::Direction::Upstream)
                      .to<std::unordered_set>();

            auto filterOutUpstreamOfTC = [&](int x) { return !upstreamOfTC.contains(x); };
            auto kernelOutputs
                = filter(filterOutUpstreamOfTC, graph.control.childNodes(info.kernel))
                      .to<std::vector>();
            for(auto const index : kernelOutputs)
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) { info.siblingOps.push_back(index); },
                          [&](LoadTiled const& load) {
                              auto reachableFromLoad
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              for(auto const& assign : info.dependentAssigns)
                              {
                                  if(reachableFromLoad.contains(assign))
                                  {
                                      info.siblingLoads.push_back(index);
                                      break;
                                  }
                              }
                          },
                      },
                      std::get<Operation>(elem));
            }

            if(!info.loadAScale.has_value() && info.loadBScale.has_value())
            {
                AssertFatal(info.siblingLoads.size() <= 1, ShowValue(info.siblingLoads.size()));
            }

            return info;
        }

        ExpressionPtr getAccumulationLoopSize(KernelGraph const& graph, int tileTag, int userTag)
        {
            auto sdims = graph.coordinates
                             .getOutputNodeIndices(
                                 userTag, rocRoller::KernelGraph::CoordinateGraph::isEdge<Split>)
                             .to<std::vector>();

            auto userA = graph.coordinates.getNode<User>(userTag);
            auto tileA = graph.coordinates.getNode<MacroTile>(tileTag);
            auto matK  = graph.coordinates.getNode<SubDimension>(sdims[1]).size;
            auto macK  = literal(static_cast<uint>(tileA.sizes[1])); // M x K

            auto toUInt32 = [](ExpressionPtr expr) -> ExpressionPtr {
                return std::make_shared<Expression::Expression>(
                    Expression::Convert{{.arg{expr}}, DataType::UInt32});
            };

            return toUInt32(matK / macK);
        }

        int getMacroTileNumber(KernelGraph const& graph, int userTag, int sdim)
        {
            auto [required, path]
                = findRequiredCoordinates(userTag, Graph::Direction::Downstream, graph);
            auto macroTileNumbers = filterCoordinates<MacroTileNumber>(required, graph);
            for(auto mtnTag : macroTileNumbers)
            {
                for(auto input : graph.coordinates.getInputNodeIndices(
                        mtnTag, rocRoller::KernelGraph::CoordinateGraph::isEdge<Tile>))
                {
                    auto maybeSubDimension = graph.coordinates.get<SubDimension>(input);
                    if(!maybeSubDimension)
                        continue;
                    if(maybeSubDimension->dim == sdim)
                        return mtnTag;
                }
            }
            return -1;
        }

        /**
         * Lower rank-2 TensorContraction into a matrix multiply.
         */
        void lowerMatrixMultiply(KernelGraph&             graph,
                                 int                      tag,
                                 int                      a,
                                 int                      b,
                                 int                      d,
                                 CommandParametersPtr     params,
                                 std::shared_ptr<Context> context)
        {
            rocRoller::Log::debug("KernelGraph::lowerMatrixMultiply({})", tag);

            auto info = getMatrixMultiplyInfo(graph, tag);

            AssertFatal(info.loadAScale.has_value() == info.loadBScale.has_value(),
                        "A and B must both be scaled or neither.",
                        ShowValue(info.loadAScale.has_value()),
                        ShowValue(info.loadBScale.has_value()));
            bool scaled    = info.loadAScale.has_value();
            auto scaleMode = scaled ? Operations::ScaleMode::Separate : Operations::ScaleMode::None;

            std::optional<int> scaleSize, scaledK;

            auto accumulationCoordSize = getAccumulationLoopSize(graph, a, info.userA);

            auto [K, forK] = rangeFor(graph, accumulationCoordSize, rocRoller::KLOOP);

            if(scaled)
            {
                scaledK = K;
            }

            // A row block is x-workgroup, column block is for loop index
            // B row block is for loop index, column block is y-workgroup
            //
            // TODO: For macTileNumYA: Look for Number siblings of
            // the first bound Index nodes of the WORKGROUP Tensor
            // above `b`.  Similarly for A.
            auto macTileNumYA = getMacroTileNumber(graph, info.userA, 1);
            auto macTileNumXB = getMacroTileNumber(graph, info.userB, 0);

            rocRoller::Log::debug("  Load A {} MTN {}; Load B {} MTN {}",
                                  info.loadA.load(),
                                  macTileNumYA,
                                  info.loadB.load(),
                                  macTileNumXB);

            graph.coordinates.addElement(PassThrough(), {macTileNumYA}, {K});
            graph.coordinates.addElement(PassThrough(), {macTileNumXB}, {K});

            if(scaled)
            {
                auto macTileNumYAScale = getMacroTileNumber(graph, info.userAScale, 1);
                auto macTileNumXBScale = getMacroTileNumber(graph, info.userBScale, 0);

                graph.coordinates.addElement(PassThrough(), {macTileNumYAScale}, {*scaledK});
                graph.coordinates.addElement(PassThrough(), {macTileNumXBScale}, {*scaledK});
            }

            auto [waveATag, waveA] = graph.getDimension<WaveTile>(info.loadA.load());
            auto [waveBTag, waveB] = graph.getDimension<WaveTile>(info.loadB.load());
            uint num_elements      = waveA.sizes[0] * waveB.sizes[1];
            uint wfs               = context->kernel()->wavefront_size();
            uint numAGPRs          = num_elements / wfs; // number of output registers per thread

            auto initD = graph.control.addElement(
                Assign{Register::Type::Accumulator, literal(0.f), numAGPRs});

            graph.mapper.connect(initD, d, NaryArgument::DEST);

            auto waveTileNumYA = graph.mapper.get<WaveTileNumber>(info.loadA.load(), 1);
            auto waveTileNumXB = graph.mapper.get<WaveTileNumber>(info.loadB.load(), 0);

            rocRoller::Log::debug("  Load A {} WTN {}; Load B {} WTN {}",
                                  info.loadA.load(),
                                  waveTileNumYA,
                                  info.loadB.load(),
                                  waveTileNumXB);

            std::optional<int>      waveAScaleTag, waveBScaleTag;
            std::optional<WaveTile> waveAScale, waveBScale;
            std::optional<int>      waveTileNumYAScale;
            std::optional<int>      waveTileNumXBScale;

            if(scaled)
            {
                std::tie(waveAScaleTag, waveAScale)
                    = graph.getDimension<WaveTile>(info.loadAScale->load());
                std::tie(waveBScaleTag, waveBScale)
                    = graph.getDimension<WaveTile>(info.loadBScale->load());
                waveTileNumYAScale = graph.mapper.get<WaveTileNumber>(info.loadAScale->load(), 1);
                waveTileNumXBScale = graph.mapper.get<WaveTileNumber>(info.loadBScale->load(), 0);
            }

            // Add an unroll dimension that connects to both A's WaveTileNumber[1] and B's
            // WaveTileNumber[0]. This is because we are unrolling the "small k" loop.
            auto tileA = graph.coordinates.getNode<MacroTile>(a);

            uint const numWaveTiles = tileA.sizes[1] / waveA.sizes[1];
            auto       smallKUnroll = graph.coordinates.addElement(Unroll(numWaveTiles));
            graph.coordinates.addElement(PassThrough(), {waveTileNumYA}, {smallKUnroll});
            graph.coordinates.addElement(PassThrough(), {waveTileNumXB}, {smallKUnroll});

            if(scaled)
            {
                graph.coordinates.addElement(PassThrough(), {*waveTileNumYAScale}, {smallKUnroll});
                graph.coordinates.addElement(PassThrough(), {*waveTileNumXBScale}, {smallKUnroll});
            }

            int lastWaveMult   = -1;
            int lastSetCoordA  = -1;
            int lastSetCoordB  = -1;
            int firstSetCoordA = -1;
            int firstSetCoordB = -1;

            std::optional<int> lastSetCoordAScale;
            std::optional<int> lastSetCoordBScale;

            std::vector<int> nodesToOrder;

            for(uint k = 0; k < numWaveTiles; k++)
            {
                auto createUnrollLoad = [&, forK = forK](int load) -> std::tuple<int, int> {
                    auto setCoord = graph.control.addElement(SetCoordinate(literal(k)));
                    graph.mapper.connect<Unroll>(setCoord, smallKUnroll);
                    graph.control.addElement(Body(), {forK}, {setCoord});

                    auto newLoad = duplicateControlNode(graph, load);
                    if(k != 0)
                        duplicateMacroTile(graph, newLoad);
                    graph.control.addElement(Body(), {setCoord}, {newLoad});

                    return {setCoord, newLoad};
                };

                auto [setCoordA, newLoadA] = createUnrollLoad(info.loadA.load());
                auto [setCoordB, newLoadB] = createUnrollLoad(info.loadB.load());

                if(firstSetCoordA == -1)
                    firstSetCoordA = setCoordA;
                if(firstSetCoordB == -1)
                    firstSetCoordB = setCoordB;

                std::optional<int> setCoordAScale, newLoadAScale;
                std::optional<int> setCoordBScale, newLoadBScale;

                if(scaled)
                {
                    std::tie(setCoordAScale, newLoadAScale)
                        = createUnrollLoad(info.loadAScale->load());
                    std::tie(setCoordBScale, newLoadBScale)
                        = createUnrollLoad(info.loadBScale->load());
                }

                auto waveMult = graph.control.addElement(Multiply(scaleMode, scaleMode));
                graph.mapper.connect(
                    waveMult, d, Connections::typeArgument<MacroTile>(NaryArgument::DEST));

                addConnectionsMultiply(graph, waveMult, newLoadA, NaryArgument::LHS);
                addConnectionsMultiply(graph, waveMult, newLoadB, NaryArgument::RHS);

                if(scaled)
                {
                    addConnectionsMultiply(
                        graph, waveMult, *newLoadAScale, NaryArgument::LHS_SCALE);
                    addConnectionsMultiply(
                        graph, waveMult, *newLoadBScale, NaryArgument::RHS_SCALE);
                }

                nodesToOrder.insert(nodesToOrder.end(), {setCoordA, setCoordB});
                if(scaled)
                    nodesToOrder.insert(nodesToOrder.end(), {*setCoordAScale, *setCoordBScale});
                nodesToOrder.push_back(waveMult);
            }

            auto prev = nodesToOrder.begin();
            for(auto cur = prev + 1; prev != nodesToOrder.end() && cur != nodesToOrder.end();
                prev++, cur++)
            {
                graph.control.addElement(Sequence(), {*prev}, {*cur});
            }

            // Add loops to iterate over wavetiles within a wavefront
            auto wavetilesPerWavefront = params->getWaveTilesPerWavefront();
            AssertFatal(wavetilesPerWavefront.size() > 1);

            auto [WaveTilesX, forWaveTilesX]
                = rangeFor(graph, literal(wavetilesPerWavefront[0]), rocRoller::XLOOP);
            auto [WaveTilesY, forWaveTilesY]
                = rangeFor(graph, literal(wavetilesPerWavefront[1]), rocRoller::YLOOP);

            auto forWaveTilesEpilogueX = *only(
                duplicateControlNodes(graph, nullptr, {forWaveTilesX}, [](int x) { return true; }));

            auto forWaveTilesEpilogueY = *only(
                duplicateControlNodes(graph, nullptr, {forWaveTilesY}, [](int x) { return true; }));

            auto forWaveTilesEpilogueYNOP = graph.control.addElement(NOP());

            graph.control.addElement(Body(), {info.kernel}, {forWaveTilesX});
            graph.control.addElement(Body(), {forWaveTilesX}, {forWaveTilesY});
            graph.control.addElement(Body(), {forWaveTilesY}, {initD});
            graph.control.addElement(Sequence(), {initD}, {forK});
            graph.control.addElement(Sequence(), {forWaveTilesX}, {forWaveTilesEpilogueX});
            graph.control.addElement(Body(), {forWaveTilesEpilogueX}, {forWaveTilesEpilogueY});
            graph.control.addElement(Body(), {forWaveTilesEpilogueY}, {forWaveTilesEpilogueYNOP});

            // Connect ops after contraction to forK, remove contraction and its incoming edges
            auto tcOutgoingEdges
                = graph.control.getNeighbours<Graph::Direction::Downstream>(tag).to<std::vector>();
            for(auto const e : tcOutgoingEdges)
            {
                auto elem = graph.control.getElement(e);
                auto dst  = graph.control.getNeighbours<Graph::Direction::Downstream>(e)
                               .to<std::vector>();
                graph.control.deleteElement(e);
                graph.control.addElement(
                    Sequence(), std::vector<int>{forWaveTilesEpilogueYNOP}, dst);
            }
            auto tcIncomingEdges
                = graph.control.getNeighbours<Graph::Direction::Upstream>(tag).to<std::vector>();
            for(auto const e : tcIncomingEdges)
                graph.control.deleteElement(e);
            graph.control.deleteElement(tag);
            graph.mapper.purge(tag);

            // Add siblings...
            for(auto const index : info.siblingLoads)
            {
                for(auto e : graph.control.getNeighbours<Graph::Direction::Upstream>(index)
                                 .to<std::vector>())
                {
                    graph.control.deleteElement(e);
                }
                // TODO: This explicitly puts the + beta * C portion of a GEMM after the
                //       forK loop. We might want to remove this after the dynamic
                //       scheduling has been implemented.
                graph.control.addElement(Sequence(), {forWaveTilesEpilogueYNOP}, {index});
            }

            for(auto const index : info.siblingOps)
            {
                auto e = only(graph.control.getNeighbours<Graph::Direction::Downstream>(index))
                             .value();
                auto elem = graph.control.getElement(e);
                graph.control.deleteElement(e);
                graph.control.addElement(
                    e, elem, std::vector<int>{index}, std::vector<int>{forWaveTilesX});
            }

            // Add PassThrough edges from all JammedWaveTileNumbers to
            // their matching jammed ForLoop coordinate
            connectJammedOperation(graph, info.loadA, WaveTilesX, WaveTilesY);
            connectJammedOperation(graph, info.loadB, WaveTilesX, WaveTilesY);
            connectJammedOperation(graph, info.siblingLoads, WaveTilesX, WaveTilesY);
            connectJammedOperation(graph, info.loadAScale, WaveTilesX, WaveTilesY);
            connectJammedOperation(graph, info.loadBScale, WaveTilesX, WaveTilesY);
            connectJammedOperation(graph, info.storeD, WaveTilesX, WaveTilesY);

            // Delete original loadA and loadB.
            purgeNodes(graph, {info.loadA.load(), info.loadB.load()});
            if(scaled)
                purgeNodes(graph, {info.loadAScale->load(), info.loadBScale->load()});

            // If the original loads were through LDS, attach their
            // LoadTiled+StoreLDSTile operations to the ForLoop.
            // Barriers and/or prefetching will be added during the
            // AddPrefetch transform.
            connectGlobalLoadOperations(graph, forK, info.loadA);
            connectGlobalLoadOperations(graph, forK, info.loadB);

            if(scaled)
            {
                connectGlobalLoadOperations(graph, forK, *info.loadAScale);
                connectGlobalLoadOperations(graph, forK, *info.loadBScale);
            }

            // Memory ordering:
            {
                auto loadAChain = info.loadA.storeLDS != -1;
                auto loadBChain = info.loadB.storeLDS != -1;
                auto loadAScaleChain
                    = info.loadAScale.has_value() && info.loadAScale->storeLDS != -1;
                auto loadBScaleChain
                    = info.loadBScale.has_value() && info.loadBScale->storeLDS != -1;

                if(loadAChain)
                    graph.control.addElement(Sequence(), {info.loadA.storeLDS}, {firstSetCoordA});
                if(loadAScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadAScale->storeLDS}, {firstSetCoordA});

                if(loadBChain)
                    graph.control.addElement(Sequence(), {info.loadB.storeLDS}, {firstSetCoordB});
                if(loadBScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadBScale->storeLDS}, {firstSetCoordB});

                if(loadAChain && loadAScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadA.storeLDS}, {info.loadAScale->storeLDS});
                if(loadBChain && loadBScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadB.storeLDS}, {info.loadBScale->storeLDS});

                if(loadAChain && !loadBChain)
                    graph.control.addElement(Sequence(), {info.loadA.storeLDS}, {firstSetCoordB});
                if(loadAScaleChain && !loadBScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadAScale->storeLDS}, {firstSetCoordB});

                if(!loadAChain && loadBChain)
                    graph.control.addElement(Sequence(), {firstSetCoordA}, {info.loadB.global});
                if(!loadAScaleChain && loadBScaleChain)
                    graph.control.addElement(
                        Sequence(), {firstSetCoordA}, {info.loadBScale->global});

                if(loadAChain && loadBChain)
                    graph.control.addElement(Sequence(), {info.loadA.global}, {info.loadB.global});
                if(loadAScaleChain && loadBScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadAScale->global}, {info.loadBScale->global});

                if(loadAChain && loadAScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadA.global}, {info.loadAScale->global});
                if(loadBChain && loadBScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadB.global}, {info.loadBScale->global});
                if(loadBChain && loadAScaleChain)
                    graph.control.addElement(
                        Sequence(), {info.loadB.global}, {info.loadAScale->global});
            }

            // Order StoreLDSTile operations
            auto toOrder = filter(graph.control.isElemType<StoreLDSTile>(),
                                  graph.control.depthFirstVisit(forK, Graph::Direction::Downstream))
                               .to<std::vector>();
            orderMemoryNodes(graph, toOrder, false);
        }

        KernelGraph LowerTensorContraction::apply(KernelGraph const& graph)
        {
            TIMER(t, "KernelGraph::lowerTensorContraction");

            auto contractions = graph.control.getNodes<TensorContraction>().to<std::vector>();
            AssertFatal(contractions.size() <= 1,
                        "More than one TensorContraction not supported yet.");

            if(contractions.size() < 1)
                return graph;

            auto kgraph       = graph;
            auto tag          = contractions[0];
            auto op           = kgraph.control.getNode<TensorContraction>(tag);
            auto [aTag, aMac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::LHS);
            auto [bTag, bMac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::RHS);
            auto [dTag, dMac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::DEST);
            if(aMac.rank == 2 && bMac.rank == 2 && op.aDims == std::vector<int>{1}
               && op.bDims == std::vector<int>{0})
            {
                lowerMatrixMultiply(kgraph, tag, aTag, bTag, dTag, m_params, m_context);
            }
            else
            {
                Throw<FatalError>("General contraction not implemented yet.");
            }

            // XXX REMOVE THIS
            {
                std::ofstream dfile;
                dfile.open("tmp.dot", std::ofstream::out | std::ofstream::trunc);
                dfile << kgraph.toDOT();
                dfile.close();
            }

            return kgraph;
        }

        ConstraintStatus NoDanglingJammedNumbers(const KernelGraph& graph)
        {
            using GD = rocRoller::Graph::Direction;

            ConstraintStatus retval;
            for(auto tag : graph.coordinates.getNodes<JammedWaveTileNumber>())
            {
                auto noIncoming = empty(graph.coordinates.getNeighbours<GD::Upstream>(tag));
                auto noOutgoing = empty(graph.coordinates.getNeighbours<GD::Downstream>(tag));
                if(noIncoming || noOutgoing)
                {
                    retval.combine(false, concatenate("Dangling JammedWaveTileNumber: ", tag));
                }
            }
            return retval;
        }

        std::vector<GraphConstraint> LowerTensorContraction::postConstraints() const
        {
            return {NoDanglingJammedNumbers};
        }
    }
}
