
#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
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

        /*
         * Code generation
         */
        struct CodeGeneratorVisitor
        {
            CodeGeneratorVisitor(KernelGraph graph, std::shared_ptr<AssemblyKernel> kernel)
                : m_graph(graph)
                , m_kernel(kernel)
                , m_context(kernel->context())
                , m_fastArith{kernel->context()}
            {
            }

            Generator<Instruction> generate()
            {
                auto coords = Transformer(
                    std::make_shared<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph>(
                        m_graph.coordinates),
                    m_context,
                    m_fastArith);

                co_yield Instruction::Comment("CodeGeneratorVisitor::generate() begin");
                co_yield setup();
                auto candidates = m_graph.control.roots().to<std::set>();
                co_yield generate(candidates, coords);
                co_yield Instruction::Comment("CodeGeneratorVisitor::generate() end");
            }

            inline Register::ValuePtr MkVGPR(VariableType const& type, int count = 1) const
            {
                return Register::Value::Placeholder(m_context, Register::Type::Vector, type, count);
            }

            inline Register::ValuePtr MkSGPR(VariableType const& type, int count = 1) const
            {
                return Register::Value::Placeholder(m_context, Register::Type::Scalar, type, count);
            }

            inline Register::ValuePtr getBufferSrd(int tag)
            {
                auto offsetTag = m_graph.mapper.get<Buffer>(tag);
                return m_context->registerTagManager()->getRegister(offsetTag);
            }

            std::pair<Register::ValuePtr, Register::ValuePtr> getOffsetAndStride(int tag,
                                                                                 int dimension)
            {
                Register::ValuePtr offset, stride;

                auto offsetTag = m_graph.mapper.get<Offset>(tag, dimension);
                if(offsetTag >= 0)
                    offset = m_context->registerTagManager()->getRegister(offsetTag);
                auto strideTag = m_graph.mapper.get<Stride>(tag, dimension);
                if(strideTag >= 0)
                    stride = m_context->registerTagManager()->getRegister(strideTag);

                return {offset, stride};
            }

            inline ExpressionPtr L(auto const& x)
            {
                return Expression::literal(x);
            }

            inline Generator<Instruction> generate(auto& dest, ExpressionPtr expr) const
            {
                co_yield Expression::generate(dest, expr, m_context);
            }

            inline Generator<Instruction> copy(auto& dest, auto const& src) const
            {
                co_yield m_context->copier()->copy(dest, src);
            }

            Generator<Instruction> setup()
            {
                for(auto x : m_kernel->workgroupSize())
                    m_workgroupSize.push_back(x);
                for(auto x : m_kernel->workgroupIndex())
                    if(x)
                        m_workgroup.push_back(x->expression());
                for(auto x : m_kernel->workitemIndex())
                    if(x)
                        m_workitem.push_back(x->expression());

                co_return;
            }

            /**
             * Generate an index from `expr` and store in `dst`
             * register.  Destination register should be an Int64.
             */
            Generator<Instruction> generateOffset(Register::ValuePtr&       dst,
                                                  Expression::ExpressionPtr expr,
                                                  DataType                  dtype)
            {
                auto const& info     = DataTypeInfo::Get(dtype);
                auto        numBytes = Expression::literal(static_cast<uint>(info.elementSize));

                // TODO: Consider moving numBytes into input of this function.
                co_yield Expression::generate(dst, expr * numBytes, m_context);
            }

            bool hasGeneratedInputs(int const& tag)
            {
                auto inputTags = m_graph.control.getInputNodeIndices<Sequence>(tag);
                for(auto const& itag : inputTags)
                {
                    if(m_completedControlNodes.find(itag) == m_completedControlNodes.end())
                        return false;
                }
                return true;
            }

            /**
             * Generate code for the specified nodes and their standard (Sequence) dependencies.
             */
            Generator<Instruction> generate(std::set<int> candidates, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    concatenate("KernelGraph::CodeGenerator::generate: ", candidates));

                co_yield Instruction::Comment(concatenate("generate(", candidates, ")"));

                while(!candidates.empty())
                {
                    std::set<int> nodes;

                    // Find all candidate nodes whose inputs have been satisfied
                    for(auto const& tag : candidates)
                        if(hasGeneratedInputs(tag))
                            nodes.insert(tag);

                    // If there are none, we have a problem.
                    AssertFatal(!nodes.empty(),
                                "Invalid control graph!",
                                ShowValue(m_graph.control),
                                ShowValue(candidates));

                    // Generate code for all the nodes we found.

                    for(auto const& tag : nodes)
                    {
                        auto op = std::get<Operation>(m_graph.control.getElement(tag));
                        co_yield call(tag, op, coords);
                    }

                    // Add output nodes to candidates.

                    for(auto const& tag : nodes)
                    {
                        auto outTags = m_graph.control.getOutputNodeIndices<Sequence>(tag);
                        candidates.insert(outTags.begin(), outTags.end());
                    }

                    // Delete generated nodes from candidates.

                    for(auto const& node : nodes)
                        candidates.erase(node);
                }
            }

            Generator<Instruction>
                call(int tag, Operation const& operation, Transformer const& coords)
            {
                auto opName = toString(operation);
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::{}({})", opName, tag);
                co_yield Instruction::Comment(opName + " BEGIN");

                AssertFatal(m_completedControlNodes.find(tag) == m_completedControlNodes.end(),
                            ShowValue(operation));

                co_yield std::visit(
                    *this, std::variant<int>(tag), operation, std::variant<Transformer>(coords));

                co_yield Instruction::Comment(opName + " END");

                m_completedControlNodes.insert(tag);
            }

            Generator<Instruction> operator()(int tag, Kernel const& edge, Transformer coords)
            {
                auto scope = std::make_shared<ScopeManager>(m_context);
                m_context->setScopeManager(scope);

                scope->pushNewScope();
                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
                scope->popAndReleaseScope();

                m_context->setScopeManager(nullptr);
            }

            Generator<Instruction> operator()(int tag, Scope const&, Transformer coords)
            {
                auto scope = m_context->getScopeManager();
                scope->pushNewScope();
                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
                scope->popAndReleaseScope();
            }

            Generator<Instruction> operator()(int tag, ForLoopOp const& op, Transformer coords)
            {
                auto topLabel = m_context->labelAllocator()->label("ForLoopTop");
                auto botLabel = m_context->labelAllocator()->label("ForLoopBottom");

                co_yield Instruction::Comment("Initialize For Loop");
                auto init = m_graph.control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
                co_yield generate(init, coords);

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1);
                auto loop_incr_tag = connections[0].coordinate;
                auto iterReg       = m_context->registerTagManager()->getRegister(loop_incr_tag);
                {
                    auto loopDims
                        = m_graph.coordinates.getOutputNodeIndices<DataFlowEdge>(loop_incr_tag);
                    for(auto const& dim : loopDims)
                    {
                        coords.setCoordinate(dim, iterReg->expression());
                    }
                }

                co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock For Loop");
                auto [conditionRegisterType, conditionVariableType]
                    = Expression::resultType(op.condition);
                auto conditionResult = conditionRegisterType == Register::Type::Special
                                               && conditionVariableType == DataType::Bool
                                           ? m_context->getSCC()
                                           : m_context->getVCC();
                co_yield Expression::generate(
                    conditionResult, coords.getTransducer()(op.condition), m_context);
                co_yield m_context->brancher()->branchIfZero(
                    botLabel,
                    conditionResult,
                    concatenate("Condition: Top (jump to " + botLabel->toString() + " if false)"));

                co_yield Instruction::Label(topLabel);

                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);

                co_yield Instruction::Comment("For Loop Increment");
                auto incr
                    = m_graph.control.getOutputNodeIndices<ForLoopIncrement>(tag).to<std::set>();
                co_yield generate(incr, coords);
                co_yield Instruction::Comment("Condition: Bottom (jump to " + topLabel->toString()
                                              + " if true)");

                co_yield Expression::generate(
                    conditionResult, coords.getTransducer()(op.condition), m_context);
                co_yield m_context->brancher()->branchIfNonZero(
                    topLabel,
                    conditionResult,
                    concatenate("Condition: Bottom (jump to " + topLabel->toString()
                                + " if true)"));

                co_yield Instruction::Label(botLabel);
                co_yield Instruction::Unlock("Unlock For Loop");
            }

            Generator<Instruction> operator()(int tag, UnrollOp const& unroll, Transformer coords)
            {
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(int tag, Assign const& assign, Transformer coords)
            {
                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Invalid Assign operation; coordinate missing.");
                auto dim_tag = connections[0].coordinate;

                rocRoller::Log::getLogger()->debug("  assigning dimension: {}", dim_tag);

                auto scope = m_context->getScopeManager();
                scope->addRegister(dim_tag);

                auto deferred = resultVariableType(assign.expression).dataType == DataType::None
                                && !m_context->registerTagManager()->hasRegister(dim_tag);

                Register::ValuePtr dest;
                if(!deferred)
                {
                    rocRoller::Log::getLogger()->debug("  immediate: count {}", assign.valueCount);
                    dest = m_context->registerTagManager()->getRegister(
                        dim_tag,
                        assign.regType,
                        resultVariableType(assign.expression),
                        assign.valueCount);
                }
                co_yield Expression::generate(dest, assign.expression, m_context);

                if(deferred)
                {
                    m_context->registerTagManager()->addRegister(dim_tag, dest);
                }
            }

            Generator<Instruction>
                operator()(int tag, Deallocate const& deallocate, Transformer coords)
            {
                auto dim_tag = m_graph.mapper.get<Dimension>(tag);
                rocRoller::Log::getLogger()->debug("  deallocate dimension: {}", dim_tag);
                m_context->registerTagManager()->deleteRegister(dim_tag);
                co_return;
            }

            Generator<Instruction> operator()(int, Barrier const&, Transformer)
            {
                co_yield m_context->mem()->barrier();
            }

            Generator<Instruction> operator()(int tag, ComputeIndex const& ci, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::ComputeIndex({}): {}/{}",
                    tag,
                    ci.offset,
                    ci.stride);

                auto scope    = m_context->getScopeManager();
                uint numBytes = DataTypeInfo::Get(ci.valueType).elementSize;

                coords.setCoordinate(ci.increment, L(0u));
                for(auto const tag_ : ci.zero)
                    coords.setCoordinate(tag_, L(0u));

                auto offsetReg = m_context->registerTagManager()->getRegister(
                    ci.offset, Register::Type::Vector, ci.offsetType, 1);
                offsetReg->setName(concatenate("offset", tag));
                co_yield Register::AllocateIfNeeded(offsetReg);
                scope->addRegister(ci.offset);

                if(ci.base < 0)
                {
                    auto indexExpr = ci.forward ? coords.forward({ci.target})[0]
                                                : coords.reverse({ci.target})[0];
                    rocRoller::Log::getLogger()->debug(
                        "  Offset({}): {}", ci.offset, toString(indexExpr));
                    co_yield generate(offsetReg, indexExpr * L(numBytes));
                }

                auto strideReg = m_context->registerTagManager()->getRegister(
                    ci.stride, Register::Type::Scalar, ci.strideType, 1);
                strideReg->setName(concatenate("stride", tag));
                co_yield Register::AllocateIfNeeded(strideReg);
                scope->addRegister(ci.stride);

                if(ci.stride > 0)
                {
                    auto indexExpr = ci.forward
                                         ? coords.forwardStride(ci.increment, L(1), {ci.target})[0]
                                         : coords.reverseStride(ci.increment, L(1), {ci.target})[0];
                    rocRoller::Log::getLogger()->debug(
                        "  Stride({}): {}", ci.stride, toString(indexExpr));
                    co_yield generate(strideReg, indexExpr * L(numBytes));
                }

                auto user = m_graph.coordinates.get<User>(ci.target);
                if(user)
                {
                    VariableType bufferPointer{DataType::None, PointerType::Buffer};
                    auto         bufferReg = m_context->registerTagManager()->getRegister(
                        ci.buffer, Register::Type::Scalar, bufferPointer, 1);
                    bufferReg->setName(concatenate("buffer", tag));
                    co_yield Register::AllocateIfNeeded(bufferReg);
                    auto basePointer = MkSGPR(DataType::Int64);
                    auto bufDesc     = BufferDescriptor(bufferReg, m_context);
                    co_yield m_context->argLoader()->getValue(user->argumentName(), basePointer);
                    co_yield bufDesc.setBasePointer(basePointer);
                    co_yield bufDesc.setDefaultOpts();
                    scope->addRegister(ci.buffer);
                }
            }

            Generator<Instruction>
                operator()(int tag, SetCoordinate const& setCoordinate, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::SetCoordinate({}): {}",
                    tag,
                    Expression::toString(setCoordinate.value));

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Invalid SetCoordinate operation; coordinate missing.");
                coords.setCoordinate(connections[0].coordinate, setCoordinate.value);

                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
            }

            Generator<Instruction> operator()(int tag, LoadLinear const& edge, Transformer coords)
            {
                Throw<FatalError>("LoadLinear present in kernel graph.");
            }

            Generator<Instruction>
                loadMacroTileVGPRCI(int tag, LoadTiled const& load, Transformer coords, int sdim)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileVGPRCI()");
                co_yield Instruction::Comment("GEN: loadMacroTileVGPRCI");

                auto [user_tag, user]         = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto bufOpt = BufferInstructionOptions();

                auto [mac_offset_reg, mac_stride_reg] = getOffsetAndStride(tag, -1);
                auto [row_offset_reg, row_stride_reg] = getOffsetAndStride(tag, 0);
                auto [col_offset_reg, col_stride_reg] = getOffsetAndStride(tag, 1);

                auto bufferSrd = getBufferSrd(tag);
                auto bufDesc   = BufferDescriptor(bufferSrd, m_context);

                std::shared_ptr<Register::Value> tmpl;
                if(load.vtype == DataType::Half)
                    tmpl = MkVGPR(DataType::Halfx2, product(mac_tile.subTileSizes));
                else
                    tmpl = MkVGPR(load.vtype, product(mac_tile.subTileSizes));

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                auto numBytes = DataTypeInfo::Get(load.vtype).elementSize;

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

                co_yield copy(row_offset_reg, mac_offset_reg);

                // TODO: multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->loadBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            col_offset_reg->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                        }
                    }

                    if(i < m - 1)
                    {
                        co_yield generate(row_offset_reg,
                                          row_offset_reg->expression()
                                              + row_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction>
                loadMacroTileVGPR(int tag, LoadTiled const& load, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileVGPR({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileVGPR");

                auto [user_tag, user]         = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto bufOpt = BufferInstructionOptions();

                auto [row_offset_reg, row_stride_reg] = getOffsetAndStride(tag, 0);
                auto [col_offset_reg, col_stride_reg] = getOffsetAndStride(tag, 1);
                auto bufferSrd                        = getBufferSrd(tag);
                auto bufDesc                          = BufferDescriptor(bufferSrd, m_context);

                auto tmpl = MkVGPR(load.vtype, product(mac_tile.subTileSizes));
                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

                rocRoller::Log::getLogger()->debug(
                    "  macro tile: {}; sub tile size: {}x{}", mac_tile_tag, m, n);

                // TODO: multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->loadBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            col_offset_reg->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                        }
                    }

                    if(i < m - 1)
                    {
                        co_yield generate(row_offset_reg,
                                          row_offset_reg->expression()
                                              + row_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction>
                loadMacroTileLDS(int tag, LoadLDSTile const& load, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileLDS()");
                co_yield_(Instruction::Comment("GEN: loadMacroTileLDS"));

                auto [lds_tag, lds]   = m_graph.getDimension<LDS>(tag);
                auto [tile_tag, tile] = m_graph.getDimension<MacroTile>(tag);

                auto [row_offset_reg, row_stride_reg] = getOffsetAndStride(tag, 0);
                auto [col_offset_reg, col_stride_reg] = getOffsetAndStride(tag, 1);

                // Find the LDS allocation that contains the tile and store
                // the offset of the beginning of the allocation into lds_offset.
                auto ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                auto lds_offset_expr
                    = Expression::literal(ldsAllocation->getLDSAllocation()->offset());
                co_yield generate(lds_offset, lds_offset_expr);

                auto vtype    = ldsAllocation->variableType();
                auto numBytes = DataTypeInfo::Get(vtype).elementSize;

                auto vgpr = m_context->registerTagManager()->getRegister(tile_tag);

                auto const m = tile.subTileSizes[0];
                auto const n = tile.subTileSizes[1];

                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->load(
                            MemoryInstructions::MemoryKind::Local,
                            lds_offset,
                            vgpr->element({static_cast<int>(i * n + j)}),
                            col_offset_reg->subset({0}),
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                        }
                    }

                    if(i < m - 1)
                    {
                        co_yield generate(row_offset_reg,
                                          row_offset_reg->expression()
                                              + row_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction> loadMacroTileWAVELDSCI(int                tag,
                                                          LoadLDSTile const& load,
                                                          Transformer        coords,
                                                          int                sdim)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVELDSCI()");
                co_yield_(Instruction::Comment("GEN: loadMacroTileWAVELDSCI"));

                auto [lds_tag, lds]             = m_graph.getDimension<LDS>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<MacroTile>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);

                // Find the LDS allocation that contains the tile and store
                // the offset of the beginning of the allocation into lds_offset.
                auto ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                auto lds_offset_expr
                    = Expression::literal(ldsAllocation->getLDSAllocation()->offset());
                co_yield generate(lds_offset, lds_offset_expr);

                auto vtype    = ldsAllocation->variableType();
                auto numBytes = DataTypeInfo::Get(vtype).elementSize;

                auto n_wave_tag = m_graph.mapper.get<WaveTileNumber>(tag, sdim);

                auto [wave_offset_reg, wave_stride_reg] = getOffsetAndStride(tag, 0);
                auto [vgpr_offset_reg, vgpr_stride_reg] = getOffsetAndStride(tag, 1);

                AssertFatal(wave_offset_reg, "Invalid WAVE offset register.");
                AssertFatal(vgpr_offset_reg, "Invalid VGPR offset register.");
                AssertFatal(vgpr_stride_reg, "Invalid VGPR stride register.");

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                if(load.vtype == DataType::Half)
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto offset1 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int32, 1);
                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int32, 1);

                    co_yield copy(vgpr_offset_reg, wave_offset_reg);

                    for(uint a = 0; a < num_vgpr; a += 2)
                    {
                        co_yield copy(offset1, vgpr_offset_reg);
                        co_yield generate(vgpr_offset_reg,
                                          vgpr_offset_reg->expression()
                                              + vgpr_stride_reg->expression());
                        co_yield copy(offset2, vgpr_offset_reg);
                        co_yield generate(vgpr_offset_reg,
                                          vgpr_offset_reg->expression()
                                              + vgpr_stride_reg->expression());

                        co_yield m_context->mem()->loadAndPack(
                            MemoryInstructions::MemoryKind::Local,
                            vgpr->element({static_cast<int>(a / 2)}),
                            lds_offset,
                            offset1,
                            lds_offset,
                            offset2);
                    }
                }
                else
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    co_yield copy(vgpr_offset_reg, wave_offset_reg);

                    for(uint a = 0; a < num_vgpr; ++a)
                    {
                        co_yield m_context->mem()->load(MemoryInstructions::MemoryKind::Local,
                                                        vgpr->element({static_cast<int>(a)}),
                                                        lds_offset,
                                                        vgpr_offset_reg->subset({0}),
                                                        numBytes);

                        if(a < num_vgpr - 1)
                            co_yield generate(vgpr_offset_reg,
                                              vgpr_offset_reg->expression()
                                                  + vgpr_stride_reg->expression());
                    }
                }
            }

            // CI : compute index
            Generator<Instruction>
                loadMacroTileWAVECI(int tag, LoadTiled const& load, Transformer coords, int sdim)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVECI({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileWAVECI");

                auto [user_tag, user]           = m_graph.getDimension<User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<MacroTile>(tag);

                Register::ValuePtr basePointer;
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto [wave_offset_reg, wave_stride_reg] = getOffsetAndStride(tag, 0);
                auto [vgpr_offset_reg, vgpr_stride_reg] = getOffsetAndStride(tag, 1);
                auto bufferSrd                          = getBufferSrd(tag);

                auto bufDesc = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt  = BufferInstructionOptions();

                AssertFatal(wave_offset_reg, "Invalid WAVE offset register.");
                AssertFatal(vgpr_offset_reg, "Invalid VGPR offset register.");
                AssertFatal(vgpr_stride_reg, "Invalid VGPR stride register.");

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                if(load.vtype == DataType::Half)
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto offset1 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);
                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);

                    co_yield copy(vgpr_offset_reg, wave_offset_reg);

                    for(uint a = 0; a < num_vgpr; a += 2)
                    {
                        co_yield copy(offset1, vgpr_offset_reg);
                        co_yield generate(vgpr_offset_reg,
                                          vgpr_offset_reg->expression()
                                              + vgpr_stride_reg->expression());
                        co_yield copy(offset2, vgpr_offset_reg);
                        co_yield generate(vgpr_offset_reg,
                                          vgpr_offset_reg->expression()
                                              + vgpr_stride_reg->expression());

                        co_yield m_context->mem()->loadAndPackBuffer(
                            vgpr->element({static_cast<int>(a / 2)}),
                            offset1,
                            offset2,
                            bufDesc,
                            bufOpt);
                    }
                }
                else
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto numBytes = (uint)DataTypeInfo::Get(vgpr->variableType()).elementSize;

                    co_yield copy(vgpr_offset_reg, wave_offset_reg);

                    for(uint a = 0; a < num_vgpr; ++a)
                    {
                        co_yield m_context->mem()->loadBuffer(vgpr->element({static_cast<int>(a)}),
                                                              vgpr_offset_reg->subset({0}),
                                                              0,
                                                              bufDesc,
                                                              bufOpt,
                                                              numBytes);

                        if(a < num_vgpr - 1)
                            co_yield generate(vgpr_offset_reg,
                                              vgpr_offset_reg->expression()
                                                  + vgpr_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction>
                loadMacroTileWAVECIACCUM(int tag, LoadTiled const& load, Transformer coords)

            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVECIACCUM({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileWAVECIACCUM");

                auto [user_tag, user]           = m_graph.getDimension<User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);
                auto mac_tile_tag               = m_graph.mapper.get<MacroTile>(tag);

                // Move the argument pointer into v_ptr
                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto [vgpr_block_offset_reg, vgpr_block_stride_reg] = getOffsetAndStride(tag, 0);
                auto [vgpr_index_offset_reg, vgpr_index_stride_reg] = getOffsetAndStride(tag, 1);
                auto bufferSrd                                      = getBufferSrd(tag);

                auto bufDesc = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt  = BufferInstructionOptions();

                AssertFatal(vgpr_block_offset_reg, "Invalid VGPR BLOCK offset register.");
                AssertFatal(vgpr_block_stride_reg, "Invalid VGPR BLOCK stride register.");
                AssertFatal(vgpr_index_stride_reg, "Invalid VGPR INDEX stride register.");

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                if(load.vtype == DataType::Half)
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto offset1 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);
                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);

                    for(uint ablk = 0; ablk < num_vgpr / 4; ++ablk)
                    {
                        co_yield copy(vgpr_index_offset_reg, vgpr_block_offset_reg);
                        for(uint aidx = 0; aidx < 4; aidx += 2)
                        {
                            uint a = ablk * 4 + aidx;

                            co_yield copy(offset1, vgpr_index_offset_reg);
                            co_yield generate(vgpr_index_offset_reg,
                                              vgpr_index_offset_reg->expression()
                                                  + vgpr_index_stride_reg->expression());
                            co_yield copy(offset2, vgpr_index_offset_reg);
                            co_yield generate(vgpr_index_offset_reg,
                                              vgpr_index_offset_reg->expression()
                                                  + vgpr_index_stride_reg->expression());

                            co_yield m_context->mem()->loadAndPackBuffer(
                                vgpr->element({static_cast<int>(a / 2)}),
                                offset1,
                                offset2,
                                bufDesc,
                                bufOpt);
                        }
                        if(ablk < num_vgpr / 4 - 1)
                            co_yield generate(vgpr_block_offset_reg,
                                              vgpr_block_offset_reg->expression()
                                                  + vgpr_block_stride_reg->expression());
                    }
                }
                else
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto numBytes = (uint)DataTypeInfo::Get(vgpr->variableType()).elementSize;

                    for(uint ablk = 0; ablk < num_vgpr / 4; ++ablk)
                    {
                        co_yield copy(vgpr_index_offset_reg, vgpr_block_offset_reg);
                        for(uint aidx = 0; aidx < 4; ++aidx)
                        {
                            uint a = ablk * 4 + aidx;

                            co_yield m_context->mem()->loadBuffer(
                                vgpr->element({static_cast<int>(a)}),
                                vgpr_index_offset_reg->subset({0}),
                                0,
                                bufDesc,
                                bufOpt,
                                numBytes);

                            if(aidx < 3)
                                co_yield generate(vgpr_index_offset_reg,
                                                  vgpr_index_offset_reg->expression()
                                                      + vgpr_index_stride_reg->expression());
                        }
                        if(ablk < num_vgpr / 4 - 1)
                            co_yield generate(vgpr_block_offset_reg,
                                              vgpr_block_offset_reg->expression()
                                                  + vgpr_block_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction> operator()(int tag, LoadTiled const& load, Transformer coords)
            {
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                {
                    switch(mac_tile.layoutType)
                    {
                    case LayoutType::MATRIX_A:
                        co_yield loadMacroTileVGPRCI(tag, load, coords, 1);
                        break;
                    case LayoutType::MATRIX_B:
                        co_yield loadMacroTileVGPRCI(tag, load, coords, 0);
                        break;
                    default:
                        co_yield loadMacroTileVGPR(tag, load, coords);
                        break;
                    }
                }
                break;
                case MemoryType::WAVE:
                {
                    switch(mac_tile.layoutType)
                    {
                    case LayoutType::MATRIX_A:
                        co_yield loadMacroTileWAVECI(tag, load, coords, 1);
                        break;
                    case LayoutType::MATRIX_B:
                        co_yield loadMacroTileWAVECI(tag, load, coords, 0);
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
                    Throw<FatalError>("Tile affinity type not supported yet.");
                }
            }

            Generator<Instruction> operator()(int tag, LoadLDSTile const& load, Transformer coords)
            {
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::LDS:
                    co_yield loadMacroTileLDS(tag, load, coords);
                    break;
                case MemoryType::WAVE:
                {
                    switch(mac_tile.layoutType)
                    {
                    case LayoutType::MATRIX_A:
                        co_yield loadMacroTileWAVELDSCI(tag, load, coords, 1);
                        break;
                    case LayoutType::MATRIX_B:
                        co_yield loadMacroTileWAVELDSCI(tag, load, coords, 0);
                        break;
                    default:
                        Throw<FatalError>("Layout type not supported yet for LoadLDSTile.");
                    }
                }
                break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet.");
                }
            }

            Generator<Instruction> operator()(int tag, LoadVGPR const& load, Transformer coords)
            {
                auto [userTag, user] = m_graph.getDimension<User>(tag);
                auto [vgprTag, vgpr] = m_graph.getDimension<VGPR>(tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR({}): User({}), VGPR({})",
                    tag,
                    userTag,
                    vgprTag);

                auto dst = m_context->registerTagManager()->getRegister(
                    vgprTag, Register::Type::Vector, load.vtype.dataType);
                co_yield Register::AllocateIfNeeded(dst);

                if(load.scalar)
                {
                    if(load.vtype.isPointer())
                        co_yield loadVGPRFromScalarPointer(user, dst, coords);
                    else
                        co_yield loadVGPRFromScalarValue(user, dst, coords);
                }
                else
                {
                    co_yield loadVGPRFromGlobalArray(userTag, user, dst, coords);
                }
            }

            Generator<Instruction> loadVGPRFromScalarValue(User                             user,
                                                           std::shared_ptr<Register::Value> vgpr,
                                                           Transformer                      coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): scalar value");
                co_yield Instruction::Comment("GEN: LoadVGPR; scalar value");

                Register::ValuePtr s_value;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_value);
                co_yield m_context->copier()->copy(vgpr, s_value, "Move value");
            }

            Generator<Instruction> loadVGPRFromScalarPointer(User                             user,
                                                             std::shared_ptr<Register::Value> vgpr,
                                                             Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): scalar pointer");
                co_yield Instruction::Comment("GEN: LoadVGPR; scalar pointer");

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, vgpr, v_ptr, nullptr, numBytes);
            }

            Generator<Instruction> loadVGPRFromGlobalArray(int                              userTag,
                                                           User                             user,
                                                           std::shared_ptr<Register::Value> vgpr,
                                                           Transformer                      coords)
            {
                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                co_yield Instruction::Comment("GEN: LoadVGPR; user index");

                auto indexes = coords.reverse({userTag});
                co_yield generateOffset(offset, indexes[0], vgpr->variableType().dataType);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, vgpr, v_ptr, offset, numBytes);
            }

            Generator<Instruction> operator()(int tag, Multiply const& mult, Transformer coords)
            {
                auto loads = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::vector>();
                AssertFatal(loads.size() == 2, "Multiply op needs two operands");

                auto loadA = m_graph.control.getElement(loads[0]);
                auto loadB = m_graph.control.getElement(loads[1]);

                int sourceA_tag = -1;
                if(isOperation<LoadTiled>(loadA))
                    sourceA_tag = m_graph.mapper.get<User>(tag, 0);
                else if(isOperation<LoadLDSTile>(loadA))
                    sourceA_tag = m_graph.mapper.get<LDS>(tag, 0);

                int sourceB_tag = -1;
                if(isOperation<LoadTiled>(loadB))
                    sourceB_tag = m_graph.mapper.get<User>(tag, 1);
                else if(isOperation<LoadLDSTile>(loadB))
                    sourceB_tag = m_graph.mapper.get<LDS>(tag, 1);

                AssertFatal(sourceA_tag > 0 && sourceB_tag > 0, "User or LDS dimensions not found");

                auto [waveA_tag, waveA] = m_graph.getDimension<WaveTile>(tag, 0);
                auto [waveB_tag, waveB] = m_graph.getDimension<WaveTile>(tag, 1);

                auto [macA_tag, macA] = m_graph.getDimension<MacroTile>(tag, 0);
                auto [macB_tag, macB] = m_graph.getDimension<MacroTile>(tag, 1);

                auto n_waveA_y_tags
                    = m_graph.coordinates
                          .findNodes(sourceA_tag,
                                     [&](int index) -> bool {
                                         auto node = m_graph.coordinates.get<WaveTileNumber>(index);
                                         if(node)
                                             return node->dim == 1;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_waveA_y_tags.size() == 1);

                auto n_waveB_x_tags
                    = m_graph.coordinates
                          .findNodes(sourceB_tag,
                                     [&](int index) -> bool {
                                         auto node = m_graph.coordinates.get<WaveTileNumber>(index);
                                         if(node)
                                             return node->dim == 0;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_waveB_x_tags.size() == 1);

                auto loadAB = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();

                auto [mac_offset_x_reg, mac_stride_x_reg]   = getOffsetAndStride(loads[0], -1);
                auto [wave_offset_x_reg, wave_stride_x_reg] = getOffsetAndStride(loads[0], 0);

                auto [mac_offset_y_reg, mac_stride_y_reg]   = getOffsetAndStride(loads[1], -1);
                auto [wave_offset_y_reg, wave_stride_y_reg] = getOffsetAndStride(loads[1], 0);

                AssertFatal(macA.sizes[1] == macB.sizes[0], "MacroTile size mismatch.");

                uint num_elements = waveA.sizes[0] * waveB.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_agpr     = num_elements / wfs;

                auto [D_tag, _D] = m_graph.getDimension<MacroTile>(tag, 2);

                auto D = m_context->registerTagManager()->getRegister(
                    D_tag, Register::Type::Accumulator, DataType::Float, num_agpr);

                auto completed = m_completedControlNodes;

                // D is not initialized here

                if(mac_offset_x_reg)
                    co_yield copy(wave_offset_x_reg, mac_offset_x_reg);
                if(mac_offset_y_reg)
                    co_yield copy(wave_offset_y_reg, mac_offset_y_reg);

                // saving the offsets to be restored for each macrotile in LDS
                // TODO : Need more design thought (how to seed an offset register)
                auto reset_offset_x = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                if(isOperation<LoadLDSTile>(loadA))
                    co_yield copy(reset_offset_x, wave_offset_x_reg);

                auto reset_offset_y = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                if(isOperation<LoadLDSTile>(loadB))
                    co_yield copy(reset_offset_y, wave_offset_y_reg);

                uint const num_wave_tiles = macA.sizes[1] / waveA.sizes[1];
                for(uint k = 0; k < num_wave_tiles; k++)
                {
                    m_completedControlNodes = completed; // TODO: remove this?

                    // A WaveTile number; tall-skinny column block
                    coords.setCoordinate(n_waveA_y_tags.front(), literal(k));
                    // B WaveTile number; short-fat row block
                    coords.setCoordinate(n_waveB_x_tags.front(), literal(k));

                    co_yield generate(loadAB, coords);

                    waveA.vgpr = m_context->registerTagManager()->getRegister(macA_tag);
                    waveB.vgpr = m_context->registerTagManager()->getRegister(macB_tag);

                    Expression::ExpressionPtr A = std::make_shared<Expression::Expression>(
                        std::make_shared<WaveTile>(waveA));
                    Expression::ExpressionPtr B = std::make_shared<Expression::Expression>(
                        std::make_shared<WaveTile>(waveB));

                    co_yield generate(D,
                                      std::make_shared<Expression::Expression>(
                                          Expression::MatrixMultiply(A, B, D->expression())));

                    co_yield generate(wave_offset_x_reg,
                                      wave_offset_x_reg->expression()
                                          + wave_stride_x_reg->expression());

                    co_yield generate(wave_offset_y_reg,
                                      wave_offset_y_reg->expression()
                                          + wave_stride_y_reg->expression());
                }

                if(isOperation<LoadLDSTile>(loadA))
                    co_yield copy(wave_offset_x_reg, reset_offset_x);
                if(isOperation<LoadLDSTile>(loadB))
                    co_yield copy(wave_offset_y_reg, reset_offset_y);
            }

            Generator<Instruction>
                operator()(int tag, TensorContraction const& mul, Transformer coords)
            {
                Throw<FatalError>("TensorContraction present in kernel graph.");
            }

            Generator<Instruction> operator()(int tag, StoreLinear const& edge, Transformer coords)
            {
                Throw<FatalError>("StoreLinear present in kernel graph.");
            }

            Generator<Instruction>
                storeMacroTileVGPR(int tag, StoreTiled const& store, Transformer coords)
            {
                auto [user_tag, user]         = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileVGPR({})", tag);
                co_yield Instruction::Comment("GEN: storeMacroTileVGPR");

                rocRoller::Log::getLogger()->debug("  user {}; tile {}", user_tag, mac_tile_tag);

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                auto [row_offset_reg, row_stride_reg] = getOffsetAndStride(tag, 0);
                auto [col_offset_reg, col_stride_reg] = getOffsetAndStride(tag, 1);

                auto bufferSrd = getBufferSrd(tag);
                auto bufDesc   = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt    = BufferInstructionOptions();

                // TODO multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->storeBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            col_offset_reg->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                        }
                    }
                    if(i < m - 1)
                    {
                        co_yield generate(row_offset_reg,
                                          row_offset_reg->expression()
                                              + row_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction>
                storeMacroTileWAVECI(int tag, StoreTiled const& store, Transformer coords)

            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileWAVE()");
                co_yield Instruction::Comment("GEN: storeMacroTileWAVE");

                auto [user_tag, user]           = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<MacroTile>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto agpr = m_context->registerTagManager()->getRegister(mac_tile_tag);

                AssertFatal(agpr->registerCount() == num_vgpr);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto [vgpr_block_offset_reg, vgpr_block_stride_reg] = getOffsetAndStride(tag, 0);
                auto [vgpr_index_offset_reg, vgpr_index_stride_reg] = getOffsetAndStride(tag, 1);
                auto bufferSrd                                      = getBufferSrd(tag);

                auto bufDesc = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt  = BufferInstructionOptions();

                AssertFatal(vgpr_block_offset_reg, "Invalid VGPR BLOCK offset register.");
                AssertFatal(vgpr_block_stride_reg, "Invalid VGPR BLOCK stride register.");
                AssertFatal(vgpr_index_stride_reg, "Invalid VGPR INDEX stride register.");

                auto numBytes  = DataTypeInfo::Get(store.dataType).elementSize;
                auto value     = MkVGPR(agpr->variableType());
                auto converted = MkVGPR(store.dataType);

                for(uint ablk = 0; ablk < num_vgpr / 4; ++ablk)
                {
                    co_yield copy(vgpr_index_offset_reg, vgpr_block_offset_reg);
                    for(uint aidx = 0; aidx < 4; ++aidx)
                    {
                        uint a = ablk * 4 + aidx;
                        if(value->variableType() != store.dataType)
                        {
                            co_yield m_context->copier()->copy(
                                value, agpr->element({static_cast<int>(a)}));
                            co_yield Expression::generate(
                                converted,
                                convert(store.dataType,
                                        std::make_shared<Expression::Expression>(value)),
                                m_context);
                        }
                        else
                        {
                            co_yield m_context->copier()->copy(
                                converted, agpr->element({static_cast<int>(a)}));
                        }

                        co_yield m_context->mem()->storeBuffer(converted,
                                                               vgpr_index_offset_reg->subset({0}),
                                                               0,
                                                               bufDesc,
                                                               bufOpt,
                                                               numBytes);

                        if(aidx < 3)
                            co_yield generate(vgpr_index_offset_reg,
                                              vgpr_index_offset_reg->expression()
                                                  + vgpr_index_stride_reg->expression());
                    }
                    if(ablk < num_vgpr / 4 - 1)
                        co_yield generate(vgpr_block_offset_reg,
                                          vgpr_block_offset_reg->expression()
                                              + vgpr_block_stride_reg->expression());
                }
            }

            Generator<Instruction> operator()(int tag, StoreTiled const& store, Transformer coords)
            {
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                    co_yield storeMacroTileVGPR(tag, store, coords);
                    break;
                case MemoryType::WAVE:
                    co_yield storeMacroTileWAVECI(tag, store, coords);
                    break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet.");
                }
            }

            Generator<Instruction>
                operator()(int tag, StoreLDSTile const& store, Transformer coords)
            {
                auto [lds_tag, lds]   = m_graph.getDimension<LDS>(tag);
                auto [tile_tag, tile] = m_graph.getDimension<MacroTile>(tag);

                // Temporary register(s) that is used to copy the data from global memory to
                // local memory.
                auto vgpr     = m_context->registerTagManager()->getRegister(tile_tag);
                auto vtype    = store.dataType;
                auto numBytes = DataTypeInfo::Get(vtype).elementSize;

                auto [row_offset_reg, row_stride_reg] = getOffsetAndStride(tag, 0);
                auto [col_offset_reg, col_stride_reg] = getOffsetAndStride(tag, 1);

                auto numElements = product(tile.subTileSizes) * product(m_workgroupSize);
                // Allocate LDS memory, and store the offset of the beginning of the allocation
                // into lds_offset.
                Register::ValuePtr ldsAllocation;
                if(!m_context->registerTagManager()->hasRegister(lds_tag))
                {
                    ldsAllocation = Register::Value::AllocateLDS(m_context, vtype, numElements);
                    m_context->registerTagManager()->addRegister(lds_tag, ldsAllocation);
                }
                else
                {
                    ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);
                }

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                auto lds_offset_expr
                    = Expression::literal(ldsAllocation->getLDSAllocation()->offset());
                co_yield generate(lds_offset, lds_offset_expr);

                auto const m = tile.subTileSizes[0];
                auto const n = tile.subTileSizes[1];

                // saving the offsets to be restored for each macrotile in LDS
                // TODO : Need more design thought (how to seed an offset register)
                auto reset_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield copy(reset_offset, row_offset_reg);

                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->store(
                            MemoryInstructions::MemoryKind::Local,
                            lds_offset,
                            vgpr->element({static_cast<int>(i * n + j)}),
                            col_offset_reg->subset({0}),
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                        }
                    }

                    if(i < m - 1)
                    {
                        co_yield generate(row_offset_reg,
                                          row_offset_reg->expression()
                                              + row_stride_reg->expression());
                    }
                }
                co_yield copy(row_offset_reg, reset_offset);
            }

            Generator<Instruction> operator()(int tag, StoreVGPR const& store, Transformer coords)
            {
                co_yield Instruction::Comment("GEN: StoreVGPR");

                auto [vgprTag, vgpr] = m_graph.getDimension<VGPR>(tag);
                auto [userTag, user] = m_graph.getDimension<User>(tag);

                auto src = m_context->registerTagManager()->getRegister(vgprTag);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                auto indexes = coords.forward({userTag});

                co_yield Instruction::Comment("GEN: StoreVGPR; user index");
                co_yield offset->allocate();
                co_yield generateOffset(offset, indexes[0], src->variableType().dataType);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, src->variableType().getPointer(), 1);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(src->variableType()).elementSize;
                co_yield m_context->mem()->store(
                    MemoryInstructions::MemoryKind::Flat, v_ptr, src, offset, numBytes);
            }

        private:
            KernelGraph                     m_graph;
            std::shared_ptr<Context>        m_context;
            std::shared_ptr<AssemblyKernel> m_kernel;

            std::set<int> m_completedControlNodes;

            std::vector<ExpressionPtr> m_workgroup;
            std::vector<ExpressionPtr> m_workitem;
            std::vector<unsigned int>  m_workgroupSize;
            FastArithmetic             m_fastArith;
        };

        Generator<Instruction> generate(KernelGraph graph, std::shared_ptr<AssemblyKernel> kernel)
        {
            TIMER(t, "KernelGraph::generate");
            rocRoller::Log::getLogger()->debug("KernelGraph::generate(); DOT\n{}",
                                               graph.toDOT(true));

            auto visitor = CodeGeneratorVisitor(graph, kernel);

            co_yield visitor.generate();
        }
    }
}
