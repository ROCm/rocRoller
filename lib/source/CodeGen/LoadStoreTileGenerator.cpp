
#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        LoadStoreTileGenerator::LoadStoreTileGenerator(std::shared_ptr<KernelGraph> graph,
                                                       ContextPtr                   context,
                                                       unsigned int workgroupSizeTotal)
            : m_graph(graph)
            , m_context(context)
            , m_fastArith(context)
            , m_workgroupSizeTotal(workgroupSizeTotal)
        {
        }

        inline Generator<Instruction> LoadStoreTileGenerator::generate(auto&         dest,
                                                                       ExpressionPtr expr) const
        {
            co_yield Expression::generate(dest, m_fastArith(expr), m_context);
        }

        inline ExpressionPtr L(auto const& x)
        {
            return Expression::literal(x);
        }

        inline std::shared_ptr<BufferDescriptor> LoadStoreTileGenerator::getBufferDesc(int tag)
        {
            auto bufferTag = m_graph->mapper.get<Buffer>(tag);
            auto bufferSrd = m_context->registerTagManager()->getRegister(bufferTag);
            return std::make_shared<BufferDescriptor>(bufferSrd, m_context);
        }

        /**
             * @brief Build unrolled offset expression.
             *
             * Offsets inside unrolled loops look like:
             *
             *    offset = offset + unroll-iteration * stride
             *
             * where the additional piece is a local/independent
             * expression.
             *
             * When requesting an Offset register, this routines looks
             * nearby for Stride expressions connected to Unroll
             * coordinates, and returns the
             *
             *     + unroll-iteration * stride
             *
             * part of the offset above.
             */
        ExpressionPtr LoadStoreTileGenerator::getOffsetExpr(int                offsetTag,
                                                            Transformer const& coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::getOffsetExpr(offsetTag: {})", offsetTag);

            // Find storage node connected to Offset edge.
            auto maybeTargetTag = findStorageNeighbour(offsetTag, *m_graph);
            if(!maybeTargetTag)
                return nullptr;
            auto [targetTag, direction] = *maybeTargetTag;

            // Find all required coordinates for the storage node,
            // and filter out Unroll coordinates.
            auto [required, path] = findRequiredCoordinates(targetTag, direction, *m_graph);
            auto unrolls          = filterCoordinates<Unroll>(required, *m_graph);

            if(unrolls.size() == 0)
                return nullptr;

            ExpressionPtr result = Expression::literal(0u);

            for(auto const& unroll : unrolls)
            {
                // Find the neighbour of the Unroll that:
                // 1. is in the load/store coordinate transform path
                // 2. has a Stride edge connected to it
                std::optional<int> maybeStrideTag;
                std::vector<int>   neighbourNodes;
                if(direction == Graph::Direction::Downstream)
                    neighbourNodes = m_graph->coordinates.parentNodes(unroll).to<std::vector>();
                else
                    neighbourNodes = m_graph->coordinates.childNodes(unroll).to<std::vector>();
                for(auto neighbourNode : neighbourNodes)
                {
                    if(path.contains(neighbourNode))
                    {
                        auto neighbourEdges = m_graph->coordinates.getNeighbours(
                            neighbourNode, Graph::opposite(direction));
                        for(auto neighbourEdge : neighbourEdges)
                        {
                            auto maybeStride = m_graph->coordinates.get<Stride>(neighbourEdge);
                            if(maybeStride
                               && m_context->registerTagManager()->hasExpression(neighbourEdge))
                            {
                                maybeStrideTag = neighbourEdge;
                            }
                        }
                    }
                }

                if(!maybeStrideTag)
                    continue;

                auto [strideExpr, _dtype]
                    = m_context->registerTagManager()->getExpression(*maybeStrideTag);

                rocRoller::Log::getLogger()->debug(
                    "  unroll coord {} value: {}", unroll, toString(coords.getCoordinate(unroll)));

                result = result + coords.getCoordinate(unroll) * strideExpr;
            }

            return result;
        }

        Generator<Instruction> LoadStoreTileGenerator::getOffset(LoadStoreTileInfo& info,
                                                                 Transformer        coords,
                                                                 int                tag,
                                                                 bool               preserveOffset)
        {
            auto offsetTag = m_graph->mapper.get<Offset>(tag, 0);
            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::getOffset(tag:"
                                               " {}, offsetTag: {})",
                                               tag,
                                               offsetTag);

            AssertFatal(offsetTag >= 0, "No Offset found");

            ExpressionPtr rowOffsetExpr;

            if(m_context->registerTagManager()->hasRegister(offsetTag))
            {
                info.rowOffsetReg = m_context->registerTagManager()->getRegister(offsetTag);
                rowOffsetExpr     = getOffsetExpr(offsetTag, coords);
            }
            else if(m_baseOffsets.count(offsetTag) > 0)
            {
                auto baseTag      = m_baseOffsets[offsetTag];
                info.rowOffsetReg = m_context->registerTagManager()->getRegister(baseTag);

                info.rowOffsetReg->setName(concatenate("offset", offsetTag));
                m_context->getScopeManager()->addRegister(offsetTag);
                m_context->registerTagManager()->addRegister(offsetTag, info.rowOffsetReg);
                rowOffsetExpr = getOffsetExpr(offsetTag, coords);
            }
            else
            {
                Throw<FatalError>("Base offset not found");
            }

            if(rowOffsetExpr)
                rocRoller::Log::getLogger()->debug("  rowOffsetExpr: {}", toString(rowOffsetExpr));

            if(rowOffsetExpr
               && Expression::evaluationTimes(rowOffsetExpr)[EvaluationTime::Translate]
               && info.offset->regType() == Register::Type::Literal)
            {
                info.offset
                    = Register::Value::Literal(getUnsignedInt(evaluate(rowOffsetExpr))
                                               + getUnsignedInt(info.offset->getLiteralValue()));
                rowOffsetExpr.reset();
            }

            if(rowOffsetExpr)
            {
                auto unrolledRowOffsetExpr
                    = m_fastArith(info.rowOffsetReg->expression() + rowOffsetExpr);
                auto tmp = info.rowOffsetReg->placeholder();
                co_yield generate(tmp, unrolledRowOffsetExpr);
                info.rowOffsetReg = tmp;
            }
            else if(preserveOffset)
            {
                auto tmp = info.rowOffsetReg->placeholder();
                co_yield m_context->copier()->copy(tmp, info.rowOffsetReg);
                info.rowOffsetReg = tmp;
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::generateStride(Register::ValuePtr& stride,
                                                                      int                 tag,
                                                                      int                 dimension)
        {
            auto strideTag = m_graph->mapper.get<Stride>(tag, dimension);
            if(strideTag >= 0)
            {
                auto [strideExpr, strideDataType]
                    = m_context->registerTagManager()->getExpression(strideTag);

                stride = nullptr;
                co_yield generate(stride, strideExpr);
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::genComputeIndex(int                 tag,
                                                                       ComputeIndex const& ci,
                                                                       Transformer         coords)
        {
            auto tagger = m_context->registerTagManager();

            auto base = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BASE});
            auto offset = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::OFFSET});
            auto stride = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::STRIDE});
            auto target = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::TARGET});
            auto increment = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::INCREMENT});

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::ComputeIndex({}): "
                "target {} increment {} base {} offset {} stride {}",
                tag,
                target,
                increment,
                base,
                offset,
                stride);

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = m_graph->coordinates.get<LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of computing indexes,
                // use the parent LDS as the target instead.
                namespace CT = rocRoller::KernelGraph::CoordinateGraph;

                auto maybeParentLDS = only(
                    m_graph->coordinates.getInputNodeIndices(target, CT::isEdge<PassThrough>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            auto scope    = m_context->getScopeManager();
            uint numBytes = DataTypeInfo::Get(ci.valueType).elementSize;

            coords.setCoordinate(increment, L(0u));
            for(int idx = 0;; ++idx)
            {
                auto zeroTag = m_graph->mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::ZERO, idx});
                if(zeroTag < 0)
                    break;
                coords.setCoordinate(zeroTag, L(0u));
            }

            if(base < 0)
            {
                // no base coordinate to copy offset from, so need
                // to explicity compute our own offset

                auto offsetReg
                    = tagger->getRegister(offset, Register::Type::Vector, ci.offsetType, 1);
                offsetReg->setName(concatenate("offset", tag));
                scope->addRegister(offset);

                auto indexExpr
                    = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

                rocRoller::Log::getLogger()->debug("  Offset({}): {}", offset, toString(indexExpr));

                co_yield generate(offsetReg, indexExpr * L(numBytes));
            }
            else
            {
                m_baseOffsets.insert_or_assign(offset, base);
            }

            if(stride > 0)
            {
                auto indexExpr = ci.forward ? coords.forwardStride(increment, L(1), {target})[0]
                                            : coords.reverseStride(increment, L(1), {target})[0];
                rocRoller::Log::getLogger()->debug("  Stride({}): {}", stride, toString(indexExpr));

                // We have to manually invoke m_fastArith here since it can't traverse into the
                // RegisterTagManager.
                // TODO: Revisit storing expressions in the RegisterTagManager.
                tagger->addExpression(stride, m_fastArith(indexExpr * L(numBytes)), ci.strideType);
                scope->addRegister(stride);
            }

            auto buffer = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});
            if(buffer > 0)
            {
                auto user = m_graph->coordinates.get<User>(target);
                if(user && !tagger->hasRegister(buffer))
                {
                    auto bufferReg = tagger->getRegister(
                        buffer, Register::Type::Scalar, {DataType::None, PointerType::Buffer}, 1);
                    bufferReg->setName(concatenate("buffer", tag));
                    if(bufferReg->allocationState() == Register::AllocationState::Unallocated)
                    {
                        co_yield Register::AllocateIfNeeded(bufferReg);
                        Register::ValuePtr basePointer;
                        auto               bufDesc = BufferDescriptor(bufferReg, m_context);
                        co_yield m_context->argLoader()->getValue(user->argumentName(),
                                                                  basePointer);
                        co_yield bufDesc.setBasePointer(basePointer);
                        co_yield bufDesc.setDefaultOpts();
                    }
                    scope->addRegister(buffer);
                }
            }
        }

        /**
         * @brief Load a tile from memory where all of the strides are literals and the datatype
         *        is packed.
         *
         * @param info
         * @return Generator<Instruction>
         */
        Generator<Instruction>
            LoadStoreTileGenerator::loadTileLiteralStridesPack(LoadStoreTileInfo& info)
        {
            auto proc      = Settings::getInstance()->get(Settings::Scheduler);
            auto cost      = Settings::getInstance()->get(Settings::SchedulerCost);
            auto scheduler = Component::GetNew<Scheduling::Scheduler>(proc, cost, m_context);
            std::vector<Generator<Instruction>> generators;

            // If all of the strides are literals, we can load everything using offsets
            // without using a runtime counter
            auto offsetValue = getUnsignedInt(info.offset->getLiteralValue());
            auto rowStride   = getUnsignedInt(info.rowStrideReg->getLiteralValue());
            auto colStride   = getUnsignedInt(info.colStrideReg->getLiteralValue());
            for(uint64_t i = 0; i < info.m; ++i)
            {
                for(uint64_t j = 0; j < info.n; j += 2)
                {
                    uint a = i * info.n + j;

                    generators.push_back(m_context->mem()->loadAndPack(
                        info.kind,
                        info.data->element({static_cast<int>(a / 2)}),
                        info.rowOffsetReg,
                        Register::Value::Literal(offsetValue + j * colStride),
                        info.rowOffsetReg,
                        Register::Value::Literal(offsetValue + (j + 1) * colStride),
                        "",
                        info.bufDesc));
                }
                offsetValue += rowStride;
            }
            co_yield (*scheduler)(generators);
        }

        /**
         * @brief Load or Store a tile where all of the strides are literal values.
         *
         * @tparam Dir
         * @param info
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction>
            LoadStoreTileGenerator::moveTileLiteralStrides(LoadStoreTileInfo& info)
        {
            // If all of the strides are literals, we can load everything using offsets
            // without using a runtime counter
            auto offsetValue = getUnsignedInt(info.offset->getLiteralValue());
            auto rowStride   = getUnsignedInt(info.rowStrideReg->getLiteralValue());
            auto colStride   = getUnsignedInt(info.colStrideReg->getLiteralValue());

            if(colStride == info.elementSize)
            {
                for(uint64_t i = 0; i < info.m; ++i)
                {
                    auto start = (i * info.n) / info.packedAmount;
                    auto stop  = (i * info.n + info.n) / info.packedAmount;
                    co_yield m_context->mem()->moveData<Dir>(
                        info.kind,
                        info.rowOffsetReg,
                        info.data->element(Generated(iota(start, stop))),
                        Register::Value::Literal(offsetValue),
                        info.elementSize * info.n,
                        "",
                        false,
                        info.bufDesc);
                    offsetValue += rowStride;
                }
            }
            else
            {
                if(Dir == MemoryInstructions::MemoryDirection::Load && info.packedAmount > 1)
                {
                    co_yield loadTileLiteralStridesPack(info);
                    co_return;
                }

                for(uint64_t i = 0; i < info.m; ++i)
                {
                    for(uint64_t j = 0; j < info.n; ++j)
                    {
                        co_yield m_context->mem()->moveData<Dir>(
                            info.kind,
                            info.rowOffsetReg,
                            info.data->element(
                                {static_cast<int>((i * info.n + j) / info.packedAmount)}),
                            Register::Value::Literal(offsetValue + j * colStride),
                            info.elementSize,
                            "",
                            j % info.packedAmount == 1,
                            info.bufDesc);
                    }
                    offsetValue += rowStride;
                }
            }
        }

        /**
         * @brief Load or store a tile where the column stride is known to be a single element, but
         *        the row stride is only known at runtime.
         *
         * @tparam Dir
         * @param info
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction> LoadStoreTileGenerator::moveTileColStrideOne(LoadStoreTileInfo& info)
        {
            for(uint64_t i = 0; i < info.m; ++i)
            {
                auto start = (i * info.n) / info.packedAmount;
                auto stop  = (i * info.n + info.n) / info.packedAmount;
                co_yield m_context->mem()->moveData<Dir>(
                    info.kind,
                    info.rowOffsetReg->subset({0}),
                    info.data->element(Generated(iota(start, stop))),
                    info.offset,
                    info.elementSize * info.n,
                    "",
                    false,
                    info.bufDesc);

                if(i < info.m - 1)
                {
                    co_yield generate(info.rowOffsetReg,
                                      info.rowOffsetReg->expression()
                                          + info.rowStrideReg->expression());
                }
            }
        }

        /**
         * @brief Load a tile where the strides are only known at runtime, and the datatype
         *        is packed.
         *
         * @param info
         * @return Generator<Instruction>
         */
        Generator<Instruction>
            LoadStoreTileGenerator::loadTileRuntimeStridesPack(LoadStoreTileInfo& info)
        {
            auto proc      = Settings::getInstance()->get(Settings::Scheduler);
            auto cost      = Settings::getInstance()->get(Settings::SchedulerCost);
            auto scheduler = Component::GetNew<Scheduling::Scheduler>(proc, cost, m_context);
            std::vector<Generator<Instruction>> generators;

            auto gen = [this, &info](uint64_t i, uint64_t j) -> Generator<Instruction> {
                Register::ValuePtr offset1;
                Register::ValuePtr offset2;

                co_yield generate(offset1,
                                  info.rowOffsetReg->expression()
                                      + info.colStrideReg->expression() * Expression::literal(j));
                co_yield generate(offset2, offset1->expression() + info.colStrideReg->expression());

                co_yield m_context->mem()->loadAndPack(
                    info.kind,
                    info.data->element({static_cast<int>((i * info.n + j) / 2)}),
                    offset1,
                    info.offset,
                    offset2,
                    info.offset,
                    "",
                    info.bufDesc);
            };

            // Uses a generator so that scheduler can pick between packing and loading instructions
            for(uint64_t i = 0; i < info.m; ++i)
            {
                for(uint64_t j = 0; j < info.n; j += 2)
                {
                    generators.push_back(gen(i, j));
                }
                co_yield (*scheduler)(generators);
                generators.clear();
                if(i < info.m - 1)
                {
                    co_yield generate(info.rowOffsetReg,
                                      info.rowOffsetReg->expression()
                                          + info.rowStrideReg->expression());
                }
            }
        }

        /**
         * @brief Load or store a tile where the strides are only known at runtime.
         *
         * @tparam Dir
         * @param info
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction>
            LoadStoreTileGenerator::moveTileRuntimeStrides(LoadStoreTileInfo& info)
        {
            if(Dir == MemoryInstructions::MemoryDirection::Load && info.packedAmount > 1)
            {
                co_yield loadTileRuntimeStridesPack(info);
                co_return;
            }

            auto colOffsetReg = info.rowOffsetReg->placeholder();

            for(uint64_t i = 0; i < info.m; ++i)
            {
                co_yield m_context->copier()->copy(colOffsetReg, info.rowOffsetReg);

                for(uint64_t j = 0; j < info.n; ++j)
                {
                    co_yield m_context->mem()->moveData<Dir>(
                        info.kind,
                        colOffsetReg->subset({0}),
                        info.data->element(
                            {static_cast<int>((i * info.n + j) / info.packedAmount)}),
                        info.offset,
                        info.elementSize,
                        "",
                        j % info.packedAmount == 1,
                        info.bufDesc);

                    if(j < info.n - 1)
                    {
                        co_yield generate(colOffsetReg,
                                          colOffsetReg->expression()
                                              + info.colStrideReg->expression());
                    }
                }

                if(i < info.m - 1)
                {
                    co_yield generate(info.rowOffsetReg,
                                      info.rowOffsetReg->expression()
                                          + info.rowStrideReg->expression());
                }
            }
        }

        /**
             * @brief Load or store a tile
             *
             * @param kind The kind of memory instruction to use
             * @param m Number of rows in the tile
             * @param n Number of columns in the tile
             * @param dataType The type of the data being loaded
             * @param tag The tag of the control graph node generating the load or store
             * @param vgpr The registers to store the data in (null is loading)
             * @param offset Offset from the starting index
             * @param coords Transformer object
             * @return Generator<Instruction>
             */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction> LoadStoreTileGenerator::moveTile(MemoryInstructions::MemoryKind kind,
                                                                uint64_t                       m,
                                                                uint64_t                       n,
                                                                VariableType       dataType,
                                                                int                tag,
                                                                Register::ValuePtr vgpr,
                                                                Register::ValuePtr offset,
                                                                Transformer&       coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::moveTile<{}>({})", ToString(Dir), tag);

            LoadStoreTileInfo info;
            info.kind   = kind;
            info.m      = m;
            info.n      = n;
            info.offset = offset;

            if(Dir == MemoryInstructions::MemoryDirection::Load)
            {
                auto macTileTag = m_graph->mapper.get<MacroTile>(tag);

                Register::ValuePtr tmpl;
                if(dataType == DataType::Half && n > 1)
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, m * n / 2);
                }
                else
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, dataType, m * n);
                }

                info.data = m_context->registerTagManager()->getRegister(macTileTag, tmpl);
                co_yield Register::AllocateIfNeeded(info.data);
            }
            else
            {
                if(!m_context->targetArchitecture().HasCapability(
                       GPUCapability::ArchAccUnifiedRegs))
                {
                    co_yield m_context->copier()->ensureType(vgpr, vgpr, Register::Type::Vector);
                }

                // Convert the data to the expected datatype
                if(DataTypeInfo::Get(vgpr->variableType()).segmentVariableType != dataType)
                {
                    co_yield m_context->copier()->ensureType(vgpr, vgpr, Register::Type::Vector);
                    info.data = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, dataType, vgpr->valueCount());
                    co_yield info.data->allocate();
                    for(int i = 0; i < vgpr->valueCount(); ++i)
                    {
                        Register::ValuePtr tmp = info.data->element({i});
                        co_yield generate(
                            tmp,
                            convert(dataType.dataType,
                                    std::make_shared<Expression::Expression>(vgpr->element({i}))));
                    }
                }
                else
                {
                    info.data = vgpr;
                }
            }

            if(!info.offset)
            {
                info.offset = Register::Value::Literal(0u);
            }

            if(kind == MemoryInstructions::MemoryKind::Buffer)
            {
                info.bufDesc = getBufferDesc(tag);
            }

            if(m > 1)
                co_yield generateStride(info.rowStrideReg, tag, 0);
            else
                info.rowStrideReg = Register::Value::Literal(0u);
            co_yield generateStride(info.colStrideReg, tag, 1);

            AssertFatal(info.rowStrideReg, "Invalid row stride register.");
            AssertFatal(info.colStrideReg, "Invalid col stride register.");

            info.elementSize  = (uint)DataTypeInfo::Get(dataType).elementSize;
            info.packedAmount = DataTypeInfo::Get(info.data->variableType()).packing;

            bool colStrideIsLiteral = (info.colStrideReg->regType() == Register::Type::Literal);
            bool allStridesAreLiteral
                = (info.rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && info.offset->regType() == Register::Type::Literal);
            bool colStrideIsOne
                = colStrideIsLiteral
                  && (getUnsignedInt(info.colStrideReg->getLiteralValue()) == info.elementSize);

            // Get the values from the associated ComputeIndex node
            co_yield getOffset(info, coords, tag, !allStridesAreLiteral && info.m > 1);
            AssertFatal(info.rowOffsetReg, "Invalid row offset register.");

            if(allStridesAreLiteral)
            {
                co_yield moveTileLiteralStrides<Dir>(info);
            }
            else if(colStrideIsOne)
            {
                co_yield moveTileColStrideOne<Dir>(info);
            }
            else
            {
                co_yield moveTileRuntimeStrides<Dir>(info);
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileVGPR(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileVGPR()");
            co_yield Instruction::Comment("GEN: loadMacroTileVGPRCI");

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto const n           = getUnsignedInt(evaluate(elemY.size));

            AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                m,
                n,
                load.vtype,
                tag,
                nullptr,
                nullptr,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileLDS(int                tag,
                                                                        LoadLDSTile const& load,
                                                                        Transformer        coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileLDS()");
            co_yield_(Instruction::Comment("GEN: loadMacroTileLDS"));

            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto const m = tile.subTileSizes[0];
            auto const n = tile.subTileSizes[1];

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Local,
                m,
                n,
                load.vtype,
                tag,
                nullptr,
                ldsOffset,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVELDS(int                tag,
                                                                            LoadLDSTile const& load,
                                                                            Transformer coords)
        {
            co_yield_(Instruction::Comment("GEN: loadMacroTileWAVELDSCI"));

            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVELDS({}, {})",
                ldsTag,
                waveTileTag);

            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;
            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Local,
                1,
                numVgpr,
                load.vtype,
                tag,
                nullptr,
                ldsOffset,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVE(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVE({})", tag);
            co_yield Instruction::Comment("GEN: loadMacroTileWAVECI");

            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                1,
                numVgpr,
                load.vtype,
                tag,
                nullptr,
                nullptr,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVECIACCUM(
            int tag, LoadTiled const& load, Transformer coords)

        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVECIACCUM({})", tag);
            co_yield Instruction::Comment("GEN: loadMacroTileWAVECIACCUM");

            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                numVgpr / 4,
                4,
                load.vtype,
                tag,
                nullptr,
                nullptr,
                coords);
        }

        Generator<Instruction>
            LoadStoreTileGenerator::genLoadTile(int tag, LoadTiled const& load, Transformer coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
                co_yield loadMacroTileVGPR(tag, load, coords);
                break;
            case MemoryType::WAVE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                case LayoutType::MATRIX_B:
                    co_yield loadMacroTileWAVE(tag, load, coords);
                    break;
                case LayoutType::MATRIX_ACCUMULATOR:
                    co_yield loadMacroTileWAVECIACCUM(tag, load, coords);
                    break;
                default:
                    Throw<FatalError>("Layout type not supported yet for LoadTiled.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for LoadTiled.");
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::genLoadLDSTile(int                tag,
                                                                      LoadLDSTile const& load,
                                                                      Transformer        coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
                co_yield loadMacroTileLDS(tag, load, coords);
                break;
            case MemoryType::WAVE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                case LayoutType::MATRIX_B:
                    co_yield loadMacroTileWAVELDS(tag, load, coords);
                    break;
                default:
                    Throw<FatalError>("Layout type not supported yet for LoadLDSTile.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for LoadLDSTile.");
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileLDS(int                 tag,
                                                                         StoreLDSTile const& store,
                                                                         Transformer         coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileLDS()");
            co_yield Instruction::Comment("GEN: storeMacroTileLDS");

            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            // Temporary register(s) that is used to copy the data from global memory to
            // local memory.
            auto vgpr  = m_context->registerTagManager()->getRegister(tileTag);
            auto vtype = store.dataType;

            auto numElements = product(tile.subTileSizes) * m_workgroupSizeTotal;
            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(m_context, vtype, numElements);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto const n           = getUnsignedInt(evaluate(elemY.size));

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Local, m, n, vtype, tag, vgpr, ldsOffset, coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileVGPR(int               tag,
                                                                          StoreTiled const& store,
                                                                          Transformer       coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileVGPR({})", tag);
            co_yield Instruction::Comment("GEN: storeMacroTileVGPR");

            rocRoller::Log::getLogger()->debug(" tile {}", macTileTag);

            auto vgpr = m_context->registerTagManager()->getRegister(macTileTag);

            auto const m = macTile.subTileSizes[0];
            auto const n = macTile.subTileSizes[1];

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Buffer,
                m,
                n,
                store.dataType,
                tag,
                vgpr,
                nullptr,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVELDS(
            int tag, StoreLDSTile const& store, Transformer coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileWAVELDS()");
            co_yield Instruction::Comment("GEN: storeMacroTileWAVELDS");

            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto macrotileNumElements    = product(macTile.sizes);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);
            uint waveTileNumElements     = waveTile.sizes[0] * waveTile.sizes[1];
            auto vtype                   = store.dataType;

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation
                    = Register::Value::AllocateLDS(m_context, vtype, macrotileNumElements);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            uint wfs     = m_context->kernel()->wavefront_size();
            uint numVgpr = waveTileNumElements / wfs;
            auto agpr    = m_context->registerTagManager()->getRegister(macTileTag);
            AssertFatal(agpr->registerCount() == numVgpr);

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Local,
                numVgpr / 4,
                4,
                vtype,
                tag,
                agpr,
                ldsOffset,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVE(int               tag,
                                                                          StoreTiled const& store,
                                                                          Transformer       coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileWAVE()");
            co_yield Instruction::Comment("GEN: storeMacroTileWAVE");

            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            auto agpr = m_context->registerTagManager()->getRegister(macTileTag);

            AssertFatal(agpr->registerCount() == numVgpr);

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Buffer,
                numVgpr / 4,
                4,
                store.dataType,
                tag,
                agpr,
                nullptr,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::genStoreTile(int               tag,
                                                                    StoreTiled const& store,
                                                                    Transformer       coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
                co_yield storeMacroTileVGPR(tag, store, coords);
                break;
            case MemoryType::WAVE:
                co_yield storeMacroTileWAVE(tag, store, coords);
                break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for StoreTiled.");
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::genStoreLDSTile(int                 tag,
                                                                       StoreLDSTile const& store,
                                                                       Transformer         coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
                co_yield storeMacroTileLDS(tag, store, coords);
                break;
            case MemoryType::WAVE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_ACCUMULATOR:
                    co_yield storeMacroTileWAVELDS(tag, store, coords);
                    break;
                default:
                    Throw<FatalError>("Layout type not supported yet for StoreLDSTile.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for StoreLDSTile.");
            }
        }
    }
}
