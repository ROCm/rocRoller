
#include <memory>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
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
        namespace CT         = rocRoller::KernelGraph::CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        LoadStoreTileGenerator::LoadStoreTileGenerator(KernelGraphPtr graph,
                                                       ContextPtr     context,
                                                       unsigned int   workgroupSizeTotal)
            : m_graph(graph)
            , m_context(context)
            , m_fastArith(FastArithmetic(context))
            , m_workgroupSizeTotal(workgroupSizeTotal)
        {
        }

        inline Generator<Instruction> LoadStoreTileGenerator::generate(auto&         dest,
                                                                       ExpressionPtr expr) const
        {
            co_yield Expression::generate(dest, expr, m_context);
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
            {
                return nullptr;
            }

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

                auto [strideExpr, strideAttrs]
                    = m_context->registerTagManager()->getExpression(*maybeStrideTag);

                Log::debug(
                    "  unroll coord {} value: {}", unroll, toString(coords.getCoordinate(unroll)));

                result = result + coords.getCoordinate(unroll) * strideExpr;
            }

            return result;
        }

        Generator<Instruction> LoadStoreTileGenerator::getOffset(LoadStoreTileInfo& info,
                                                                 Transformer        coords,
                                                                 int                tag,
                                                                 bool               preserveOffset,
                                                                 bool               direct2LDS)
        {
            auto offsetTag = m_graph->mapper.get<Offset>(tag, direct2LDS ? 1 : 0);
            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::getOffset(tag:"
                                               " {}, offsetTag: {})",
                                               tag,
                                               offsetTag);

            AssertFatal(offsetTag >= 0, "No Offset found");

            ExpressionPtr rowOffsetExpr;

            if(m_context->registerTagManager()->hasRegister(offsetTag))
            {
                if(direct2LDS)
                {
                    auto tmp = m_context->registerTagManager()->getRegister(offsetTag);
                    co_yield generate(info.data, info.data->expression() + tmp->expression());
                }
                else
                {
                    info.rowOffsetReg = m_context->registerTagManager()->getRegister(offsetTag);
                }

                rowOffsetExpr = getOffsetExpr(offsetTag, coords);
            }
            else if(m_baseOffsets.count(offsetTag) > 0)
            {
                auto baseTag = m_baseOffsets[offsetTag];
                if(direct2LDS)
                {
                    auto tmp = m_context->registerTagManager()->getRegister(baseTag);
                    co_yield generate(info.data, info.data->expression() + tmp->expression());
                    m_context->getScopeManager()->addRegister(offsetTag);
                    m_context->registerTagManager()->addRegister(offsetTag, info.data);
                }
                else
                {
                    info.rowOffsetReg = m_context->registerTagManager()->getRegister(baseTag);
                    info.rowOffsetReg->setName(concatenate("offset", offsetTag));
                    m_context->getScopeManager()->addRegister(offsetTag);
                    m_context->registerTagManager()->addRegister(offsetTag, info.rowOffsetReg);
                }

                rowOffsetExpr = getOffsetExpr(offsetTag, coords);
            }
            else
            {
                Throw<FatalError>("Base offset not found");
            }

            if(direct2LDS)
            {
                co_return;
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
                auto unrolledRowOffsetExpr = info.rowOffsetReg->expression() + rowOffsetExpr;
                auto tmp = info.rowOffsetReg->placeholder(Register::Type::Vector, {});
                co_yield generate(tmp, unrolledRowOffsetExpr);
                info.rowOffsetReg = tmp;
            }
            else if(preserveOffset)
            {
                auto tmp = info.rowOffsetReg->placeholder(Register::Type::Vector, {});
                co_yield m_context->copier()->copy(tmp, info.rowOffsetReg);
                info.rowOffsetReg = tmp;
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::generateStride(
            Register::ValuePtr& stride, RegisterExpressionAttributes& attrs, int tag, int dimension)
        {
            auto strideTag = m_graph->mapper.get<Stride>(tag, dimension);
            if(strideTag >= 0)
            {
                auto [strideExpr, strideAttributes]
                    = m_context->registerTagManager()->getExpression(strideTag);

                attrs = strideAttributes;

                if(!Expression::evaluationTimes(strideExpr)[EvaluationTime::Translate])
                {
                    stride = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, strideAttributes.dataType, 1);
                }
                else
                {
                    stride = nullptr;
                }

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
            auto buffer = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});

            Log::debug("KernelGraph::LoadStoreTileGenerator::ComputeIndex({}): "
                       "target {} increment {} base {} offset {} stride {} buffer {}",
                       tag,
                       target,
                       increment,
                       base,
                       offset,
                       stride,
                       buffer);

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = m_graph->coordinates.get<LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of computing indexes,
                // use the parent LDS as the target instead.
                namespace CT = rocRoller::KernelGraph::CoordinateGraph;

                auto maybeParentLDS = only(
                    m_graph->coordinates.getOutputNodeIndices(target, CT::isEdge<Duplicate>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            bool needsPadding
                = std::visit(rocRoller::overloaded{[&](User user) { return user.needsPadding; },
                                                   [&](auto coord) { return false; }},
                             std::get<Dimension>(m_graph->coordinates.getElement(target)));
            bool ldsHoldsTransposedTile
                = std::visit(rocRoller::overloaded{[&](LDS lds) { return lds.holdsTransposedTile; },
                                                   [&](auto coord) { return false; }},
                             std::get<Dimension>(m_graph->coordinates.getElement(target)));

            auto scope = m_context->getScopeManager();

            auto toBytes = [&](ExpressionPtr expr) -> ExpressionPtr {
                uint numBits = DataTypeInfo::Get(ci.valueType).elementBits;

                // TODO: This would be a good place to add a GPU
                // assert.  If numBits is not a multiple of 8, assert
                // that (expr * numBits) is a multiple of 8.
                Log::debug("  toBytes: {}: numBits {}", toString(ci.valueType), numBits);

                if(numBits % 8u == 0)
                    return expr * L(numBits / 8u);
                return (expr * L(numBits)) / L(8u);
            };

            // Set the zero-coordinates to zero
            auto fullStop  = [&](int tag) { return tag == increment; };
            auto direction = ci.forward ? Graph::Direction::Upstream : Graph::Direction::Downstream;
            auto [required, path] = findRequiredCoordinates(target, direction, fullStop, *m_graph);

            for(auto tag : required)
                if((tag != increment) && (!coords.hasCoordinate(tag)))
                    coords.setCoordinate(tag, L(0u));

            // Set the increment coordinate to zero if it doesn't
            // already have a value
            bool initializeIncrement = !coords.hasPath({target}, ci.forward);
            if(initializeIncrement)
            {
                coords.setCoordinate(increment, L(0u));
            }

            // Compute an offset address if we don't have an
            // associated base address to inherit from
            if(base < 0)
            {
                auto offsetReg
                    = tagger->getRegister(offset, Register::Type::Vector, ci.offsetType, 1);
                offsetReg->setName(concatenate("Offset", tag));
                scope->addRegister(offset);

                auto indexExpr
                    = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

                uint          numBits = DataTypeInfo::Get(ci.valueType).elementBits;
                ExpressionPtr paddingBytes{L(0u)};
                if(numBits == 6 && ldsHoldsTransposedTile)
                {
                    uint elementsPerTrLoad = bitsPerTransposeLoad(numBits) / numBits;
                    auto extraLdsBytes     = extraLDSBytesPerElementBlock(numBits);
                    paddingBytes           = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
                }

                Log::debug("  Offset({}): indexExpr: {}", offset, toString(indexExpr));
                Log::debug("  Offset({}): paddingBytes: {}", offset, toString(paddingBytes));

                co_yield generate(offsetReg, toBytes(indexExpr) + paddingBytes);
            }
            else
            {
                m_baseOffsets.insert_or_assign(offset, base);
            }

            // Compute a stride
            if(stride > 0)
            {
                auto indexExpr = ci.forward ? coords.forwardStride(increment, L(1), {target})[0]
                                            : coords.reverseStride(increment, L(1), {target})[0];

                // We have to manually invoke m_fastArith here since it can't traverse into the
                // RegisterTagManager.
                // TODO: Revisit storing expressions in the RegisterTagManager.
                bool unitStride = false;
                if(Expression::evaluationTimes(indexExpr)[EvaluationTime::Translate])
                {
                    if(getUnsignedInt(evaluate(indexExpr)) == 1u)
                        unitStride = true;
                }

                uint          elementBlockSize = 0;
                ExpressionPtr elementBlockStride;
                ExpressionPtr trLoadPairStride;
                ExpressionPtr elementBlockStridePaddingBytes{L(0u)};
                ExpressionPtr trLoadPairStridePaddingBytes{L(0u)};
                ExpressionPtr indexExprPaddingBytes{L(0u)};

                // For F16, F8, and F4 data types and either 16x16x(32|128) or
                // 32x32x(16|64) MFMA instructions, each lane loads (MN * K)/64
                // matrix elements via 2 VGPR blocks. These two sets of elements
                // are not contiguous; and hence we need to break the loads into
                // two, with a stride between them.
                //
                // Setting the elementBlockSize to 16 for other 8bit
                // MFMA configurations is harmless; as the generated
                // stride between sets will be contiguous (and the
                // loads are dwordx4, so the same code will be
                // generated).
                uint numBits = DataTypeInfo::Get(ci.valueType).elementBits;
                if(numBits == 16 || numBits == 8 || numBits == 6 || numBits == 4)
                {
                    auto bitsPerTrLoad = bitsPerTransposeLoad(numBits);
                    elementBlockSize   = /*number of loads*/ 2 * (bitsPerTrLoad / numBits);

                    // Padding is added after every 16 elements, thus for F6 datatypes that will
                    // be transpose loaded from LDS elementBlockSize is set to 16 instead of 32.
                    if(numBits == 6 && (needsPadding || ldsHoldsTransposedTile))
                        elementBlockSize = 16;

                    elementBlockStride
                        = ci.forward
                              ? coords.forwardStride(increment, L(elementBlockSize), {target})[0]
                              : coords.reverseStride(increment, L(elementBlockSize), {target})[0];

                    uint elementsPerTrLoad = bitsPerTrLoad / numBits;
                    trLoadPairStride
                        = ci.forward
                              ? coords.forwardStride(increment, L(elementsPerTrLoad), {target})[0]
                              : coords.reverseStride(increment, L(elementsPerTrLoad), {target})[0];

                    if(numBits == 6 && ldsHoldsTransposedTile)
                    {
                        uint elementsPerTrLoad = bitsPerTransposeLoad(numBits) / numBits;
                        auto extraLdsBytes     = extraLDSBytesPerElementBlock(numBits);
                        elementBlockStridePaddingBytes
                            = elementBlockStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                        trLoadPairStridePaddingBytes
                            = trLoadPairStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                        indexExprPaddingBytes = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
                    }
                }

                Log::debug("  Stride({}): indexExpr: {}", stride, toString(indexExpr));
                Log::debug("  Stride({}): indexExprPaddingBytes: {}",
                           stride,
                           toString(indexExprPaddingBytes));
                Log::debug("  Stride({}): unitStride: {} vgprBlockSize: {}",
                           stride,
                           unitStride,
                           elementBlockSize);
                Log::debug(
                    "  Stride({}): elementBlockStride: {} elementBlockStridePaddingBytes: {}",
                    stride,
                    toString(elementBlockStride),
                    toString(elementBlockStridePaddingBytes));
                Log::debug("  Stride({}): trLoadPairStride:  {} "
                           "trLoadPairStridePaddingBytes: {}",
                           stride,
                           toString(trLoadPairStride),
                           toString(trLoadPairStridePaddingBytes));

                tagger->addExpression(stride,
                                      m_fastArith(toBytes(indexExpr) + indexExprPaddingBytes),
                                      {ci.strideType,
                                       unitStride,
                                       elementBlockSize,
                                       toBytes(elementBlockStride) + elementBlockStridePaddingBytes,
                                       toBytes(trLoadPairStride) + trLoadPairStridePaddingBytes});
                scope->addRegister(stride);
            }

            // Create a buffer descriptor
            if(buffer > 0)
            {
                auto user = m_graph->coordinates.get<User>(target);
                if(user && !tagger->hasRegister(buffer))
                {
                    AssertFatal(
                        user->size, "Invalid User dimension: missing size.", ShowValue(target));

                    auto bufferReg = tagger->getRegister(
                        buffer, Register::Type::Scalar, {DataType::None, PointerType::Buffer}, 1);
                    bufferReg->setName(concatenate("Buffer", buffer));
                    if(bufferReg->allocationState() == Register::AllocationState::Unallocated)
                    {
                        Register::ValuePtr basePointer;
                        auto               bufDesc = BufferDescriptor(bufferReg, m_context);
                        co_yield m_context->argLoader()->getValue(user->argumentName, basePointer);

                        if(user->offset && !Expression::identical(user->offset, literal(0u)))
                        {
                            Register::ValuePtr tmpRegister;
                            co_yield generate(tmpRegister,
                                              simplify(basePointer->expression() + user->offset));
                            co_yield bufDesc.setBasePointer(tmpRegister);
                        }
                        else
                        {
                            co_yield bufDesc.setBasePointer(basePointer);
                        }

                        co_yield bufDesc.setDefaultOpts();
                        Register::ValuePtr limitValue;
                        co_yield generate(limitValue, toBytes(user->size));
                        // TODO: Handle sizes larger than 32 bits
                        auto limit = (limitValue->regType() == Register::Type::Literal)
                                         ? limitValue
                                         : limitValue->subset({0});
                        co_yield bufDesc.setSize(limit);
                    }
                    scope->addRegister(buffer);
                }
            }
        }

        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction>
            LoadStoreTileGenerator::moveTileDirect2LDS(LoadStoreTileInfo& info,
                                                       int                numBytes,
                                                       bool               setM0,
                                                       Register::ValuePtr readOffset,
                                                       Register::ValuePtr readAddr)
        {
            AssertFatal(numBytes == 1 || numBytes == 2 || numBytes == 4, ShowValue(numBytes));
            auto m0 = m_context->getM0();
            if(setM0)
            {
                auto tmp = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::UInt32, 1);
                co_yield generate(tmp, info.data->expression());
                co_yield generate(m0, tmp->expression());
            }
            else
            {
                co_yield generate(m0, m0->expression() + Expression::literal(info.ldsWriteStride));
            }

            co_yield m_context->mem()->moveData<Dir>(info.kind,
                                                     readAddr,
                                                     nullptr,
                                                     readOffset,
                                                     numBytes,
                                                     "",
                                                     false,
                                                     info.bufDesc,
                                                     info.bufOpts);
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
            Log::debug("KernelGraph::LoadStoreTileGenerator::moveTileLiteralStrides<{}>",
                       toString(Dir));

            auto unsegmentedDataType = DataTypeInfo::Get(info.data->variableType().dataType);

            // If all of the strides are literals, we can load everything using offsets
            // without using a runtime counter
            auto offsetValue = getUnsignedInt(info.offset->getLiteralValue());
            auto rowStride   = getUnsignedInt(info.rowStrideReg->getLiteralValue());
            auto colStride   = getUnsignedInt(info.colStrideReg->getLiteralValue());

            if(!info.isTransposedTile && info.colStrideAttributes.unitStride)
            {
                uint numVGPRBlocks = 1;
                if(info.colStrideAttributes.elementBlockSize > 0
                   && info.n > info.colStrideAttributes.elementBlockSize)
                {
                    AssertFatal(info.n % info.colStrideAttributes.elementBlockSize == 0);
                    numVGPRBlocks = info.n / info.colStrideAttributes.elementBlockSize;
                }

                // Segmented
                auto bitsPerMove  = info.n * info.elementBits / numVGPRBlocks;
                auto bytesPerMove = bitsPerMove / 8u;

                // Unsegmented
                auto elementsPerMove = bitsPerMove / unsegmentedDataType.elementBits;

                auto elementBlockStride
                    = (numVGPRBlocks > 1)
                          ? getUnsignedInt(evaluate(info.colStrideAttributes.elementBlockStride))
                          : 0;

                Log::debug("  M {} N {} elementsPerMove {} bytesPerMove {} rowStride {} colStride "
                           "{} vgprBlockSize {} numVGPRBlocks {}",
                           info.m,
                           info.n,
                           elementsPerMove,
                           bytesPerMove,
                           rowStride,
                           colStride,
                           info.colStrideAttributes.elementBlockSize,
                           numVGPRBlocks);

                for(uint64_t i = 0; i < info.m; ++i)
                {
                    for(uint r = 0; r < numVGPRBlocks; ++r)
                    {
                        auto start = (i * numVGPRBlocks + r) * elementsPerMove;
                        auto stop  = (i * numVGPRBlocks + r + 1) * elementsPerMove;
                        if(info.bufOpts.lds)
                        {
                            co_yield moveTileDirect2LDS<Dir>(
                                info,
                                bytesPerMove,
                                (i == 0 && r == 0),
                                Register::Value::Literal(offsetValue + r * elementBlockStride),
                                info.rowOffsetReg);
                        }
                        else
                        {
                            co_yield m_context->mem()->moveData<Dir>(
                                info.kind,
                                info.rowOffsetReg,
                                info.data->element(Generated(iota(start, stop))),
                                Register::Value::Literal(offsetValue + r * elementBlockStride),
                                bytesPerMove,
                                "",
                                false,
                                info.bufDesc,
                                info.bufOpts);
                        }
                    }
                    offsetValue += rowStride;
                }
            }
            else if(info.isTransposedTile)
            {
                const auto bitsPerTrLoad     = bitsPerTransposeLoad(info.elementBits);
                const auto extraLDSBytes     = extraLDSBytesPerElementBlock(info.elementBits);
                const auto bytesPerTrLoad    = bitsPerTrLoad / 8;
                const auto elementsPerTrLoad = bitsPerTrLoad / info.elementBits;
                const auto numVGPRsPerLoad   = bitsPerTrLoad / Register::bitsPerRegister;
                const auto numVGPRBlocks     = numVGPRsPerLoad / unsegmentedDataType.registerCount;
                const auto numTrLoads        = info.n / elementsPerTrLoad;
                const auto elementBlockStride
                    = getUnsignedInt(evaluate(info.colStrideAttributes.elementBlockStride));
                const auto trLoadPairStride
                    = getUnsignedInt(evaluate(info.colStrideAttributes.trLoadPairStride));

                AssertFatal(info.n % elementsPerTrLoad == 0,
                            "WaveTileN must be multiple of the number of elements loaded by each "
                            "transpose load!");
                AssertFatal(numTrLoads % 2 == 0, "Transpose loads must be executed in pairs!");

                Log::debug("  M {} N {} elementsPerTrLoad {} bytesPerTrLoad {} extraLDSBytes {} "
                           "elementBits {} "
                           "rowStride {} colStride {} vgprPerLoad {} numTrLoads {}",
                           info.m,
                           info.n,
                           elementsPerTrLoad,
                           bytesPerTrLoad,
                           extraLDSBytes,
                           info.elementBits,
                           rowStride,
                           colStride,
                           numVGPRsPerLoad,
                           numTrLoads);

                for(uint64_t i = 0; i < info.m; ++i)
                {
                    for(uint64_t j = 0; j < numTrLoads; ++j)
                    {
                        auto start = (i * numTrLoads + (j + 0)) * numVGPRBlocks;
                        auto stop  = (i * numTrLoads + (j + 1)) * numVGPRBlocks;
                        co_yield m_context->mem()->transposeLoadLocal(
                            info.data->element(Generated(iota(start, stop))),
                            info.rowOffsetReg,
                            offsetValue + (j % 2) * trLoadPairStride + (j / 2) * elementBlockStride,
                            bytesPerTrLoad + extraLDSBytes,
                            info.elementBits);
                    }
                    offsetValue += rowStride;
                }
            }
            else
            {
                for(uint64_t i = 0; i < info.m; ++i)
                {
                    for(uint64_t j = 0; j < info.n; ++j)
                    {
                        if(info.bufOpts.lds)
                        {
                            co_yield moveTileDirect2LDS<Dir>(
                                info,
                                CeilDivide(info.elementBits, 8u),
                                (i == 0 && j == 0),
                                Register::Value::Literal(offsetValue + j * colStride),
                                info.rowOffsetReg);
                        }
                        else
                        {
                            co_yield m_context->mem()->moveData<Dir>(
                                info.kind,
                                info.rowOffsetReg,
                                info.data->element(
                                    {static_cast<int>((i * info.n + j) / info.packedAmount)}),
                                Register::Value::Literal(offsetValue + j * colStride),
                                CeilDivide(info.elementBits, 8u),
                                "",
                                j % info.packedAmount == 1,
                                info.bufDesc);
                        }
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
            Log::debug("KernelGraph::LoadStoreTileGenerator::moveTileColStrideOne<{}>",
                       toString(Dir));

            auto unsegmentedDataType = DataTypeInfo::Get(info.data->variableType().dataType);

            auto offsetValue = getUnsignedInt(info.offset->getLiteralValue());

            uint numVGPRBlocks = 1;
            if(info.colStrideAttributes.elementBlockSize > 0
               && info.n > info.colStrideAttributes.elementBlockSize)
            {
                AssertFatal(info.n % info.colStrideAttributes.elementBlockSize == 0);
                numVGPRBlocks = info.n / info.colStrideAttributes.elementBlockSize;
            }

            // Segmented
            auto bitsPerMove  = info.n * info.elementBits / numVGPRBlocks;
            auto bytesPerMove = bitsPerMove / 8u;

            // Unsegmented
            auto elementsPerMove = bitsPerMove / unsegmentedDataType.elementBits;

            Log::debug("  M {} N {} elementsPerMove {} bytesPerMove {} rowStride {} colStride "
                       "{} vgprBlockSize {} numVGPRBlocks {}",
                       info.m,
                       info.n,
                       elementsPerMove,
                       bytesPerMove,
                       toString(info.rowStrideReg->expression()),
                       toString(info.colStrideReg->expression()),
                       info.colStrideAttributes.elementBlockSize,
                       numVGPRBlocks);

            for(uint64_t i = 0; i < info.m; ++i)
            {
                for(uint r = 0; r < numVGPRBlocks; ++r)
                {
                    auto start = (i * numVGPRBlocks + r) * elementsPerMove;
                    auto stop  = (i * numVGPRBlocks + r + 1) * elementsPerMove;
                    if(info.bufOpts.lds)
                    {
                        co_yield moveTileDirect2LDS<Dir>(
                            info,
                            bytesPerMove,
                            (i == 0 && r == 0),
                            Register::Value::Literal(offsetValue + r * bytesPerMove),
                            info.rowOffsetReg->subset({0}));
                    }
                    else
                    {
                        co_yield m_context->mem()->moveData<Dir>(
                            info.kind,
                            info.rowOffsetReg->subset({0}),
                            info.data->element(Generated(iota(start, stop))),
                            Register::Value::Literal(offsetValue + r * bytesPerMove),
                            bytesPerMove,
                            "",
                            false,
                            info.bufDesc,
                            info.bufOpts);
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
            Log::debug("KernelGraph::LoadStoreTileGenerator::moveTileRuntimeStrides<{}>",
                       toString(Dir));

            auto colOffsetReg = info.rowOffsetReg->placeholder();

            for(uint64_t i = 0; i < info.m; ++i)
            {
                co_yield m_context->copier()->copy(colOffsetReg, info.rowOffsetReg);

                for(uint64_t j = 0; j < info.n; ++j)
                {
                    if(info.bufOpts.lds)
                    {
                        co_yield moveTileDirect2LDS<Dir>(info,
                                                         CeilDivide(info.elementBits, 8u),
                                                         i == 0 && j == 0,
                                                         info.offset,
                                                         colOffsetReg->subset({0}));
                    }
                    else
                    {
                        co_yield m_context->mem()->moveData<Dir>(
                            info.kind,
                            colOffsetReg->subset({0}),
                            info.data->element(
                                {static_cast<int>((i * info.n + j) / info.packedAmount)}),
                            info.offset,
                            CeilDivide(info.elementBits, 8u),
                            "",
                            j % info.packedAmount == 1,
                            info.bufDesc,
                            info.bufOpts);
                    }

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
         * @param isTransposedTile if tile needs to be transposed
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
                                                                VariableType             dataType,
                                                                int                      tag,
                                                                Register::ValuePtr       vgpr,
                                                                Register::ValuePtr       offset,
                                                                Transformer&             coords,
                                                                BufferInstructionOptions bufOpts,
                                                                bool isTransposedTile,
                                                                bool isPadded)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::moveTile<{}>({})", toString(Dir), tag);

            LoadStoreTileInfo info;
            info.kind             = kind;
            info.m                = m;
            info.n                = n;
            info.offset           = offset;
            info.bufOpts          = bufOpts;
            info.isTransposedTile = isTransposedTile;

            Register::ValuePtr finalVGPR;

            if(!info.offset)
            {
                info.offset = Register::Value::Literal(0u);
            }

            if(kind == MemoryInstructions::MemoryKind::Buffer
               || kind == MemoryInstructions::MemoryKind::Buffer2LDS)
            {
                info.bufDesc = getBufferDesc(tag);
            }

            info.elementBits = (uint)DataTypeInfo::Get(dataType).elementBits;

            if(m > 1)
                co_yield generateStride(info.rowStrideReg, info.rowStrideAttributes, tag, 0);
            else
                info.rowStrideReg = Register::Value::Literal(0u);
            co_yield generateStride(info.colStrideReg, info.colStrideAttributes, tag, 1);

            AssertFatal(info.rowStrideReg, "Invalid row stride register.");
            AssertFatal(info.colStrideReg, "Invalid col stride register.");

            bool colStrideIsLiteral = (info.colStrideReg->regType() == Register::Type::Literal);

            bool allStridesAreLiteral
                = (info.rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && info.offset->regType() == Register::Type::Literal);
            bool colStrideIsOne = colStrideIsLiteral && info.colStrideAttributes.unitStride;

            if(Dir == MemoryInstructions::MemoryDirection::Load)
            {
                auto macTileTag = m_graph->mapper.get<MacroTile>(tag);

                macTileTag
                    = only(m_graph->coordinates.getInputNodeIndices(macTileTag, CT::isEdge<View>))
                          .value_or(macTileTag);

                auto unsegmentedVariableType
                    = DataTypeInfo::Get(dataType).unsegmentedVariableType();

                Register::ValuePtr tmpl;
                if(unsegmentedVariableType
                   && ((n * info.elementBits) % Register::bitsPerRegister == 0)
                   && (colStrideIsOne || (allStridesAreLiteral && info.isTransposedTile)))
                {
                    auto allocOptions = Register::AllocationOptions::FullyContiguous();
                    auto elementBits  = DataTypeInfo::Get(dataType).elementBits;
                    if(elementBits == 6 && isPadded && !info.isTransposedTile)
                    {
                        auto registerCount
                            = DataTypeInfo::Get(*unsegmentedVariableType).registerCount;
                        allocOptions = {.contiguousChunkWidth = int(registerCount), .alignment = 2};
                    }
                    tmpl = Register::Value::Placeholder(
                        m_context,
                        Register::Type::Vector,
                        *unsegmentedVariableType,
                        m * n * info.elementBits
                            / DataTypeInfo::Get(*unsegmentedVariableType).elementBits,
                        allocOptions);
                }
                else
                {
                    tmpl = Register::Value::Placeholder(
                        m_context,
                        Register::Type::Vector,
                        dataType,
                        m * n,
                        Register::AllocationOptions::FullyContiguous());
                }

                if(kind == MemoryInstructions::MemoryKind::Buffer2LDS)
                {
                    info.data        = vgpr;
                    info.bufOpts.lds = 1;

                    // get lds write stride
                    Register::ValuePtr           ldsWriteStrideRegister;
                    RegisterExpressionAttributes _ignore;
                    co_yield generateStride(ldsWriteStrideRegister, _ignore, tag, 2);
                    auto ldsStride      = getUnsignedInt(ldsWriteStrideRegister->getLiteralValue());
                    info.ldsWriteStride = ldsStride;
                }
                else if(m_context->registerTagManager()->hasRegister(macTileTag))
                {
                    auto reg = m_context->registerTagManager()->getRegister(macTileTag);

                    if(!m_context->targetArchitecture().HasCapability(
                           GPUCapability::ArchAccUnifiedRegs)
                       && reg->regType() != Register::Type::Vector)
                    {
                        // If no unified acc/vgpr registers, create a temporary vgpr register.
                        // The result of the load will be copied into finalVGPR after the load
                        // has been performed.
                        info.data = tmpl;
                        finalVGPR = reg;
                    }
                    else
                    {
                        info.data = reg;
                    }
                }
                else
                {
                    info.data = m_context->registerTagManager()->getRegister(macTileTag, tmpl);

                    auto viewTileTags
                        = m_graph->coordinates.getOutputNodeIndices(macTileTag, CT::isEdge<View>);
                    for(auto viewTileTag : viewTileTags)
                    {
                        m_context->registerTagManager()->addRegister(viewTileTag, info.data);
                    }
                }

                Log::debug("  tag {} tile coord {} registers {}",
                           tag,
                           macTileTag,
                           info.data->description());
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
                    for(int i = 0; i < vgpr->valueCount(); ++i)
                    {
                        Register::ValuePtr dst = info.data->element({i});
                        co_yield generate(
                            dst,
                            convert(dataType.dataType,
                                    std::make_shared<Expression::Expression>(vgpr->element({i}))));
                    }
                }
                else
                {
                    info.data = vgpr;
                }
            }

            info.packedAmount = DataTypeInfo::Get(info.data->variableType()).packing;

            // Get the values from the associated ComputeIndex node
            co_yield getOffset(info, coords, tag, !allStridesAreLiteral && info.m > 1);
            AssertFatal(info.rowOffsetReg, "Invalid row offset register.");

            if(kind == MemoryInstructions::MemoryKind::Buffer2LDS)
            {
                co_yield m_context->copier()->ensureType(
                    info.data, info.data, Register::Type::Vector);
                co_yield getOffset(
                    info, coords, tag, false /* preserveOffset */, true /* direct2LDS */);
            }

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

            if(finalVGPR)
            {
                co_yield m_context->copier()->copy(finalVGPR, info.data);
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileVGPR(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

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
                load.varType,
                tag,
                nullptr,
                nullptr,
                coords,
                {},
                /*isTransposedTile=*/false,
                /*isPadded=*/tile.paddingBytes() > 0);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileLDS(int                tag,
                                                                        LoadLDSTile const& load,
                                                                        Transformer        coords)
        {
            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileLDS: OP {} LDS {} MacroTile {}",
                tag,
                ldsTag,
                tileTag);
            co_yield_(Instruction::Comment(concatenate(
                "GEN: loadMacroTileLDS OP ", tag, " LDS ", ldsTag, "MacroTile ", tileTag)));

            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto const n           = getUnsignedInt(evaluate(elemY.size));

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Local,
                m,
                n,
                load.varType,
                tag,
                nullptr,
                ldsOffset,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileDirect2LDS(
            int tag, LoadTileDirect2LDS const& load, Transformer coords)
        {

            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);
            auto dataType        = load.varType;

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "loadMacroTileDirect2LDS: OP {} LDS {} MacroTile {}",
                                               tag,
                                               ldsTag,
                                               tileTag);
            co_yield Instruction::Comment(concatenate(
                "GEN: loadMacroTileDirect2LDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag));

            auto numElements = product(tile.subTileSizes) * m_workgroupSizeTotal;

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(m_context, dataType, numElements);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            auto offsetValue = ldsAllocation->getLDSAllocation()->offset();

            auto ldsOffset = Register::Value::Literal(offsetValue);

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto const n           = getUnsignedInt(evaluate(elemY.size));

            co_yield Instruction::Lock(Scheduling::Dependency::M0, "Lock M0");
            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer2LDS,
                m,
                n,
                dataType,
                tag,
                ldsOffset,
                nullptr,
                coords,
                {});
            co_yield Instruction::Unlock("Unlock M0");
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVELDS(int                tag,
                                                                            LoadLDSTile const& load,
                                                                            Transformer coords)
        {
            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "loadMacroTileWAVELDS: OP {} LDS {} WaveTile {}",
                                               tag,
                                               ldsTag,
                                               waveTileTag);
            co_yield_(Instruction::Comment(concatenate(
                "GEN: loadMacroTileWAVELDS OP ", tag, " LDS ", ldsTag, " WaveTile ", waveTileTag)));

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
                load.varType,
                tag,
                nullptr,
                ldsOffset,
                coords,
                {},
                load.isTransposedTile);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVE(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVE: OP {} WaveTile {}",
                tag,
                waveTileTag);
            co_yield Instruction::Comment(
                concatenate("GEN: loadMacroTileWAVE OP", tag, " WaveTile ", waveTileTag));

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                1,
                numVgpr,
                load.varType,
                tag,
                nullptr,
                nullptr,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVECIACCUM(
            int tag, LoadTiled const& load, Transformer coords)

        {
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVECIACCUM: OP {} WaveTile {}",
                tag,
                waveTileTag);
            co_yield Instruction::Comment(
                concatenate("GEN: loadMacroTileWAVECIACCUM OP ", tag, " WaveTile ", waveTileTag));

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                numVgpr / 4,
                4,
                load.varType,
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
            case MemoryType::WAVE_SWIZZLE:
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
            case MemoryType::WAVE_SPLIT:
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

        Generator<Instruction> LoadStoreTileGenerator::genLoadTileDirect2LDS(
            int tag, LoadTileDirect2LDS const& load, Transformer coords)
        {
            co_yield loadMacroTileDirect2LDS(tag, load, coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileLDS(int                 tag,
                                                                         StoreLDSTile const& store,
                                                                         Transformer         coords)
        {
            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileLDS: OP {} LDS {} MacroTile {} "
                "layoutType {} memoryType {} paddingBytes {}",
                tag,
                ldsTag,
                tileTag,
                toString(tile.layoutType),
                toString(tile.memoryType),
                tile.paddingBytes());
            co_yield Instruction::Comment(concatenate(
                "GEN: storeMacroTileLDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag));

            // Temporary register(s) that is used to copy the data from global memory to
            // local memory.
            auto vgpr     = m_context->registerTagManager()->getRegister(tileTag);
            auto dataType = store.dataType;

            auto numElements  = product(tile.subTileSizes) * m_workgroupSizeTotal;
            auto paddingBytes = tile.paddingBytes();
            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(
                    m_context, dataType, numElements, /*alignment*/ 4, paddingBytes);
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
                MemoryInstructions::MemoryKind::Local,
                m,
                n,
                dataType,
                tag,
                vgpr,
                ldsOffset,
                coords,
                {},
                /*isTransposedTile*/ false,
                /*isPadded*/ paddingBytes > 0);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileVGPR(int               tag,
                                                                          StoreTiled const& store,
                                                                          Transformer       coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileVGPR: OP {} MacroTile {}",
                tag,
                macTileTag);
            co_yield Instruction::Comment(
                concatenate("GEN: storeMacroTileVGPR OP ", tag, " MacroTile ", macTileTag));

            macTileTag
                = only(m_graph->coordinates.getOutputNodeIndices(macTileTag, CT::isEdge<View>))
                      .value_or(macTileTag);

            auto vgpr = m_context->registerTagManager()->getRegister(macTileTag);

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto const n           = getUnsignedInt(evaluate(elemY.size));

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Buffer,
                m,
                n,
                store.dataType,
                tag,
                vgpr,
                nullptr,
                coords,
                store.bufOpts);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVELDS(
            int tag, StoreLDSTile const& store, Transformer coords)
        {
            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto macrotileNumElements    = product(macTile.sizes);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);
            uint waveTileNumElements     = waveTile.sizes[0] * waveTile.sizes[1];
            auto dataType                = store.dataType;

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileWAVELDS: OP {} LDS {} "
                "MacroTile {} WaveTile {}",
                tag,
                ldsTag,
                macTileTag,
                waveTileTag);
            co_yield Instruction::Comment(concatenate("GEN: storeMacroTileWAVELDS OP ",
                                                      tag,
                                                      " LDS ",
                                                      ldsTag,
                                                      " MacroTile ",
                                                      macTileTag,
                                                      " WaveTile ",
                                                      waveTileTag));

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation
                    = Register::Value::AllocateLDS(m_context, dataType, macrotileNumElements);
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
                dataType,
                tag,
                agpr,
                ldsOffset,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVE(int               tag,
                                                                          StoreTiled const& store,
                                                                          Transformer       coords)
        {
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "storeMacroTileWAVE: OP {} MacroTile {} WaveTile {}",
                                               tag,
                                               macTileTag,
                                               waveTileTag);
            co_yield Instruction::Comment(concatenate("GEN: storeMacroTileWAVE OP ",
                                                      tag,
                                                      " MacroTile ",
                                                      macTileTag,
                                                      " WaveTile ",
                                                      waveTileTag));

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
                coords,
                store.bufOpts);
        }

        Generator<Instruction> LoadStoreTileGenerator::genStoreTile(int               tag,
                                                                    StoreTiled const& store,
                                                                    Transformer       coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::WAVE_SPLIT:
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
