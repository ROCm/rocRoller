/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include "CustomMatchers.hpp"
#include "TestContext.hpp"
#include <common/CommonGraphs.hpp>
#include <common/GEMMProblem.hpp>
#include <common/TestValues.hpp>
#include <common/Utilities.hpp>
#include <common/WidenTo64bit.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace rocRoller;

namespace AddressCalculationTest
{
    namespace KernelG     = rocRoller::KernelGraph;
    namespace ControlG    = rocRoller::KernelGraph::ControlGraph;
    namespace CoordG      = rocRoller::KernelGraph::CoordinateGraph;
    using KernelGraphType = typename rocRoller::KernelGraph::KernelGraph;

    class AddressTrace
    {
    public:
        AddressTrace(KernelGraphType const& graph, ContextPtr ctx)
            : m_kGraph(graph)
            , m_context(ctx){};
        std::vector<Expression::ExpressionPtr> traceComputeIndexWithBuffer();

    private:
        KernelGraphType m_kGraph;
        ContextPtr      m_context;
    };

    std::vector<Expression::ExpressionPtr> AddressTrace::traceComputeIndexWithBuffer()
    {
        auto isComputeIndex = [&](int tag) {
            return isOperation<ControlG::ComputeIndex>(this->m_kGraph.control.getElement(tag));
        };

        std::vector<Expression::ExpressionPtr> rv;
        auto                                   root  = m_kGraph.control.roots().only();
        int                                    count = 0;
        for(auto ciTag : filter(isComputeIndex, m_kGraph.control.depthFirstVisit(root.value())))
        {

            auto maybeCi = m_kGraph.control.get<ControlG::ComputeIndex>(ciTag);
            AssertFatal(maybeCi.has_value());
            auto ci = maybeCi.value();

            auto buffer
                = m_kGraph.mapper.get(ciTag,
                                      KernelG::Connections::ComputeIndex{
                                          KernelG::Connections::ComputeIndexArgument::BUFFER});
            if(buffer == -1)
                continue;

            auto base = m_kGraph.mapper.get(ciTag,
                                            KernelG::Connections::ComputeIndex{
                                                KernelG::Connections::ComputeIndexArgument::BASE});

            // Currently, only base < 0 case is being covered.
            if(base >= 0)
                continue;

            {
                // Debugging log
                if(m_context->kernel())
                    Log::debug("kernel is non-null \n");

                auto const& kernelWorkgroupIndices = m_context->kernel()->workgroupIndex();
                Log::debug("Size of kernelWorkGroupIndices: {}", kernelWorkgroupIndices.size());
            }

            auto offset
                = m_kGraph.mapper.get(ciTag,
                                      KernelG::Connections::ComputeIndex{
                                          KernelG::Connections::ComputeIndexArgument::OFFSET});
            auto stride
                = m_kGraph.mapper.get(ciTag,
                                      KernelG::Connections::ComputeIndex{
                                          KernelG::Connections::ComputeIndexArgument::STRIDE});
            auto target
                = m_kGraph.mapper.get(ciTag,
                                      KernelG::Connections::ComputeIndex{
                                          KernelG::Connections::ComputeIndexArgument::TARGET});
            auto increment
                = m_kGraph.mapper.get(ciTag,
                                      KernelG::Connections::ComputeIndex{
                                          KernelG::Connections::ComputeIndexArgument::INCREMENT});

            // Note that identity_transduer is intentionally being used here in place of FastArithmetic.
            // It is alright to use FastArithmetic here but
            // fastArithmetic is being used eventually when the expressions are generated.
            auto identity_transducer = [&](auto expr) { return expr; };
            auto coords              = CoordG::Transformer(
                std::make_shared<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph>(
                    m_kGraph.coordinates),
                m_context,
                identity_transducer);

            auto fullStop  = [&](int tag) { return tag == increment; };
            auto direction = ci.forward ? Graph::Direction::Upstream : Graph::Direction::Downstream;
            auto [required, path] = findRequiredCoordinates(target, direction, fullStop, m_kGraph);

            for(auto tag : required)
                if((tag != increment) && (!coords.hasCoordinate(tag)))
                    coords.setCoordinate(tag, Expression::literal(0u));

            // Set the increment coordinate to zero if it doesn't
            // already have a value
            bool initializeIncrement = !coords.hasPath({target}, ci.forward);
            if(initializeIncrement)
            {
                coords.setCoordinate(increment, Expression::literal(0u));
            }

            // Compute an offset address if we don't have an
            // associated base address to inherit from
            {
                // base < 0 by the time control reacheds here.
                auto indexExpr
                    = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

                rv.push_back(indexExpr);

                // Rests are for logging for debugging.
                Log::debug("ci.forward for tag {} dir {}, stride {}, buffer {}",
                           ciTag,
                           ci.forward,
                           stride,
                           buffer);
                Log::debug("IndexExpr base < 0 for tag {} in the original graph {}",
                           ciTag,
                           toString(indexExpr));

                // Buffer Descriptor's base
                auto user = m_kGraph.coordinates.get<CoordG::User>(target);
                if(user)
                {
                    Log::debug("User for tag {} {}, offset {}",
                               ciTag,
                               ShowValue(target),
                               (user->offset ? "yes" : "no"));
                    Log::debug("argument name {}, real name {}",
                               user->argumentName,
                               m_context->kernel()->findArgument(user->argumentName).name);

                    // 1. user->argumentName + user->offset if any --> set to base of BufferDesc
                    //   (Expression)
                    // 2. user->size --> set to size field (32-bit) of BuferDesc
                }
            }
            // book-keeping for debugging purpose
            count++;
        }

        Log::debug("Count of computeIndex investigated: {}", count);

        return rv;
    }

    struct AddressCalculationTest
    {
        /**
         * gemmGraph should be a graph initialized by prob
         */
        AddressCalculationTest(rocRoller::ContextPtr       context,
                               GEMMProblem const&          prob,
                               rocRollerTest::Graphs::GEMM gemmGraph)
            : m_context(context)
            , m_problem(prob)
            , m_gemmGraph(gemmGraph)
        {
        }

        bool check_uint32_overflow(uint a, uint b)
        {
            uint64_t prod = static_cast<uint64_t>(a) * b;
            return prod > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
        }

        std::pair<std::array<Expression::ExpressionPtr, 3>, uint>
            getWorkItemCount(CommandParametersPtr params, GEMMProblem const& problem)
        {
            int M = problem.m;
            int N = problem.n;
            int K = problem.k;

            AssertFatal(M > 0 && N > 0 && K > 0);

            auto workGroupSizes = params->getManualWorkgroupSize();

            AssertFatal(workGroupSizes.has_value());
            auto workgroupSizeX = workGroupSizes.value()[0];
            auto workgroupSizeY = workGroupSizes.value()[1];

            // compute NumWorkGroups
            uint numWorkgroupX;
            uint numWorkgroupY;

            if(problem.loopOverTiles > 0)
            {
                // multiple output macro tiles per workgroup
                numWorkgroupX = M * N / problem.macM / problem.macN / 2;
                numWorkgroupY = 1;
            }
            else if(problem.streamK)
            {
                numWorkgroupX = problem.numWGs;
                numWorkgroupY = 1;
            }
            else
            {
                // one output macro tile per workgroup
                numWorkgroupX = M / problem.macM;
                numWorkgroupY = N / problem.macN;
            }

            AssertFatal(!check_uint32_overflow(numWorkgroupX, workgroupSizeX));
            AssertFatal(!check_uint32_overflow(numWorkgroupY, workgroupSizeY));

            auto NX_literal = numWorkgroupX * workgroupSizeX;
            auto NY_literal = numWorkgroupY * workgroupSizeY;

            auto NX = std::make_shared<Expression::Expression>(NX_literal);
            auto NY = std::make_shared<Expression::Expression>(NY_literal);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            auto totalWorkitemCounts = NX_literal * NY_literal;
            {
                Log::debug("Calculated workitemcount[0]: {}", toString(NX));
                Log::debug("Calculated workitemcount[1]: {}", toString(NY));
                Log::debug("Calculated workitemcount[2]: {}", toString(NZ));
                Log::debug("totalWorkitemCounts: {}", totalWorkitemCounts);
            }

            return {{NX, NY, NZ}, totalWorkitemCounts};
        }

        static Expression::ExpressionPtr
            get64BitVectorOffset(ContextPtr                                      context,
                                 std::array<Expression::ExpressionPtr, 3> const& workitemCount,
                                 std::array<unsigned int, 3> const&              workgroupSize)
        {
            std::array<Expression::ExpressionPtr, 3> thread_index;
            for(int i = 0; i < 3; i++)
                thread_index[i] = std::make_shared<Expression::Expression>(
                    context->kernel()->workitemIndex()[i]);

            std::array<Expression::ExpressionPtr, 3> workgroup_index;
            for(int i = 0; i < 3; i++)
                workgroup_index[i] = std::make_shared<Expression::Expression>(
                    context->kernel()->workgroupIndex()[i]);

            std::array<Expression::ExpressionPtr, 3> workgroup_size;
            for(int i = 0; i < 3; i++)
                workgroup_size[i] = std::make_shared<Expression::Expression>(
                    Register::Value::Literal<unsigned int>(workgroupSize[i]));

            auto idx_x = thread_index[0] + workgroup_index[0] * workgroup_size[0];
            auto idx_y = thread_index[1] + workgroup_index[1] * workgroup_size[1];

            auto compare_res_pointer = idx_x + idx_y * workitemCount[0];
            auto elementSize         = std::make_shared<Expression::Expression>(
                Register::Value::Literal<int32_t>(sizeof(uint64_t)));
            compare_res_pointer = compare_res_pointer * elementSize;

            Log::debug("Offset in kb: {}", toString(compare_res_pointer));

            return compare_res_pointer;
        }

        // This is for printing out workgroup index and thread index.
        auto kb_sanity_indices(ContextPtr                                      context,
                               std::array<Expression::ExpressionPtr, 3> const& workitemCount,
                               std::array<unsigned int, 3> const&              workgroupSize)
        {
            return [context, workitemCount, workgroupSize]() -> Generator<Instruction> {
                // store base addr
                Register::ValuePtr s_ptr;
                co_yield context->argLoader()->getValue("rv_ptr", s_ptr);

                Register::ValuePtr s_ptr2;
                co_yield context->argLoader()->getValue("rv_ptr2", s_ptr2);

                auto compare_res_pointer
                    = get64BitVectorOffset(context, workitemCount, workgroupSize);
                Log::debug("Offset in kb: {}", toString(compare_res_pointer));

                Register::ValuePtr v_offset_1 = nullptr;
                co_yield Expression::generate(
                    v_offset_1, compare_res_pointer + s_ptr->expression(), context);

                Register::ValuePtr v_offset_2 = nullptr;
                co_yield Expression::generate(
                    v_offset_2, compare_res_pointer + s_ptr2->expression(), context);

                // workgroupIndex x
                auto v_wg_x = Register::Value::Placeholder(
                    context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield context->copier()->copy(
                    v_wg_x, context->kernel()->workgroupIndex()[0], "copy wgi.x to v");
                co_yield context->mem()->storeGlobal(v_offset_1, v_wg_x, 0, 4);

                co_yield context->mem()->storeGlobal(
                    v_offset_2, (context->kernel()->workitemIndex())[0], 0, 4);
            };
        }

        // Just print out passed expressions to device memory.
        // If passed expressions are workitemCounts, the expectation is that
        // the generated expressions' values are the same with the host-side computed values.
        auto kb_implicit_workitemcount(ContextPtr                         context,
                                       Expression::ExpressionPtr const&   workitemcount_X,
                                       Expression::ExpressionPtr const&   workitemcount_Y,
                                       std::array<unsigned int, 3> const& workgroupSize)
        {
            return [context,
                    workitemcount_X,
                    workitemcount_Y,
                    workgroupSize]() -> Generator<Instruction> {
                // store base addrs
                Register::ValuePtr s_ptr;
                co_yield context->argLoader()->getValue("rv_ptr", s_ptr);
                Register::ValuePtr s_ptr2;
                co_yield context->argLoader()->getValue("rv_ptr2", s_ptr2);
                auto compare_res_pointer = get64BitVectorOffset(
                    context, {workitemcount_X, workitemcount_Y}, workgroupSize);
                Log::debug("Offset in kb: {}", toString(compare_res_pointer));

                Register::ValuePtr v_offset_1 = nullptr;
                co_yield Expression::generate(
                    v_offset_1, compare_res_pointer + s_ptr->expression(), context);

                Register::ValuePtr v_offset_2 = nullptr;
                co_yield Expression::generate(
                    v_offset_2, compare_res_pointer + s_ptr2->expression(), context);

                // Compute the value 1
                Register::ValuePtr s_value_1 = nullptr;
                co_yield Expression::generate(s_value_1, workitemcount_X, context);
                auto v_value_11 = Register::Value::Placeholder(
                    context, Register::Type::Vector, DataType::Int64, 1);
                co_yield context->copier()->copy(v_value_11, s_value_1, "copy to v1");

                // Compute the value 2
                Register::ValuePtr s_value_2 = nullptr;
                co_yield Expression::generate(s_value_2, workitemcount_Y, context);
                auto v_value_22 = Register::Value::Placeholder(
                    context, Register::Type::Vector, DataType::Int64, 1);
                co_yield context->copier()->copy(v_value_22, s_value_2, "copy to v2");

                co_yield context->mem()->storeGlobal(v_offset_1, v_value_11, 0, 8);
                co_yield context->mem()->storeGlobal(v_offset_2, v_value_22, 0, 8);
            };
        }

        // For now, mainly for debugging, directly copy the two results values.
        // workitemCount is passed as argument. Since it is 64-bit
        // different global_store_dwordx2 format should be used.
        auto kb_equal_one(ContextPtr                                      context,
                          Expression::ExpressionPtr const&                input,
                          Expression::ExpressionPtr const&                widenedInput,
                          std::array<Expression::ExpressionPtr, 3> const& workitemCount,
                          std::array<unsigned int, 3> const&              workgroupSize)
        {
            return [context, input, widenedInput, workitemCount, workgroupSize]()
                       -> Generator<Instruction> {
                // store base addr
                Register::ValuePtr s_ptr;
                co_yield context->argLoader()->getValue("rv_ptr", s_ptr);

                Register::ValuePtr s_ptr2;
                co_yield context->argLoader()->getValue("rv_ptr2", s_ptr2);

                auto compare_res_pointer
                    = get64BitVectorOffset(context, workitemCount, workgroupSize);
                Log::debug("Offset in kb: {}", toString(compare_res_pointer));

                Register::ValuePtr v_offset_1 = nullptr;
                co_yield Expression::generate(
                    v_offset_1, compare_res_pointer + s_ptr->expression(), context);

                Register::ValuePtr v_offset_2 = nullptr;
                co_yield Expression::generate(
                    v_offset_2, compare_res_pointer + s_ptr2->expression(), context);

                // Compute the value 1
                Register::ValuePtr v_value_1 = nullptr;
                co_yield Expression::generate(v_value_1, input, context);

                // Compute the value 2
                Register::ValuePtr v_value_2 = nullptr;
                co_yield Expression::generate(v_value_2, widenedInput, context);

                co_yield context->mem()->storeGlobal(v_offset_1, v_value_1, 0, 8);
                co_yield context->mem()->storeGlobal(v_offset_2, v_value_2, 0, 8);
            };
        }

        void setTensorArguments(CommandArguments&                  commandArgs,
                                GEMMProblem const&                 problem,
                                rocRollerTest::Graphs::GEMM const& gemm,
                                CommandKernelPtr const&            commandKernelPtr) const
        {
            // calling setArgument - needed
            TensorDescriptor descA(
                gemm.mTa, {size_t(problem.m), size_t(problem.k)}, problem.transA);
            TensorDescriptor descB(
                gemm.mTb, {size_t(problem.k), size_t(problem.n)}, problem.transB);
            TensorDescriptor descC(gemm.mTd, {size_t(problem.m), size_t(problem.n)}, "N");
            TensorDescriptor descD(gemm.mTd, {size_t(problem.m), size_t(problem.n)}, "N");

            // Note that actually large matrix hipMalloc is not needed.
            // But at the same time, larger malloc may incurr larger base address, which can lead to overflow.
            // ?? how to get float from m_dt? gemm.m_ta?
            auto deviceA = make_shared_device<float /* came from DataType::Float for now.*/>();
            auto deviceB = make_shared_device<float /* came from DataType::Float for now.*/>();
            auto deviceC = make_shared_device<float /* came from DataType::Float for now.*/>();
            auto deviceD = make_shared_device<float /* came from DataType::Float for now.*/>();

            // Note that gemm built the CommandGraph, and store OperationTags. We are reusing
            // that original Command, but getting away building KernelGraph and its lowering.
            setCommandTensorArg(commandArgs, gemm.mTagTensorA, descA, deviceA.get());
            setCommandTensorArg(commandArgs, gemm.mTagTensorB, descB, deviceB.get());
            setCommandTensorArg(commandArgs, gemm.mTagTensorC, descC, deviceC.get());
            setCommandTensorArg(commandArgs, gemm.mTagTensorD, descD, deviceD.get());

            commandArgs.setArgument(gemm.mTagScalarAlpha, ArgumentType::Value, problem.alpha);
            commandArgs.setArgument(gemm.mTagScalarBeta, ArgumentType::Value, problem.beta);
            // seed doesn't seem to be relevant currently.
            //if(seed.has_value())
            //    commandArgs.setArgument(gemm.m_tagScalarSeed, ArgumentType::Value, seed.value());

            // Create scratch space
            if(problem.streamK)
            {
                commandArgs.setArgument(gemm.mTagNumWGs, ArgumentType::Value, problem.numWGs);
            }

            // ?? Is it correct to use original commandKernel or addrTestCommandKernel
            auto scratchSpaceRequired
                = commandKernelPtr->scratchSpaceRequired(commandArgs.runtimeArguments());
            auto deviceScratch = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
            commandArgs.setArgument(gemm.mTagScratch, ArgumentType::Value, deviceScratch.get());
        }

        void generateOrigKernelByProlog()
        {
            m_commandKernel = std::make_shared<CommandKernel>(m_gemmGraph.getCommand(), "");
            m_commandKernel->setContext(m_context);
            m_commandKernel->setCommandParameters(m_gemmGraph.getCommandParameters());

            Log::debug("lazyAddArguments: {}", m_context->kernelOptions().lazyAddArguments);

            // Add extra kargs for testing
            m_rvTag = m_commandKernel->getCommand()->allocateTag();
            m_commandKernel->getCommand()->allocateArgument(
                VariableType(DataType::UInt64, PointerType::PointerGlobal),
                m_rvTag,
                ArgumentType::Value,
                DataDirection::WriteOnly,
                "rv_ptr");

            m_rvTag2 = m_commandKernel->getCommand()->allocateTag();
            m_commandKernel->getCommand()->allocateArgument(
                VariableType(DataType::UInt64, PointerType::PointerGlobal),
                m_rvTag2,
                ArgumentType::Value,
                DataDirection::WriteOnly,
                "rv_ptr2");

            m_commandKernel->generateKernelGraphOnlyAfterTransforms();

            auto k = m_context->kernel();
            m_context->schedule(k->preamble());
            {
                { // Original Command Arguments
                    auto commandArguments = m_commandKernel->getCommand()->getArguments();
                    for(auto arg : commandArguments)
                        Log::debug("Original Arg: {}", arg->toString());
                }

                { // workitemcount
                    for(auto wit : k->workitemCount())
                        Log::debug("Original Workitem expr: {}", toString(wit));
                }
            }

            m_commandKernel->lowerToKernelArguments();
            auto commandParameters = m_commandKernel->getCommandParameters();
            auto workgroupSize     = commandParameters->getManualWorkgroupSize();
            CHECK(workgroupSize.has_value());

            { // workitemcount
                for(auto wit : k->workitemCount())
                    Log::debug("== Original Workitem expr: {}", toString(wit));
            }

            m_context->schedule(k->prolog());
        }

        void launchKernelAndCopyBackToHost()
        {
            // Setting workitem counts before launching
            // Notice that addrTestCommandKernel's generateKernel() won't be called.
            auto launch = std::make_shared<CommandLaunchParameters>();

            // See the comments down there. Computation of workitemCount is redundant.
            // "totalWorkitemCounts" is for debugging or logging.
            std::tie(m_workitemCount, m_totalWorkitemCount)
                = getWorkItemCount(m_commandKernel->getCommandParameters(), m_problem);

            // Notice that without following line of "setManualWorkitemCount".
            //    launch->setManualWorkitemCount(workitemCount);
            // - workitemcounts were computed correctly
            // within the workitemcount_kernelbody's execution. That, I think, is because
            // The original graph already had workitemcount expressions as a function of
            // input tensor sizes, one of the commandArguments. (e.g. Tensor_0_size_0_8)
            // Thus, only the command arguments are set, workitemcounts can be computed.
            // In ohter words, this manual computation of workitemcounts and setting by
            // setManualWorkitemCount is not needed. Still, "getWorkItemCount" is kept
            // for debugging purposes. In order to allocate device and host memory for storing
            // results of kernel's execution, we still need a concrete number of allocation sizes.

            m_commandKernel->setLaunchParameters(launch);

            CommandArguments commandArgs = m_commandKernel->getCommand()->createArguments();
            setTensorArguments(commandArgs, m_problem, m_gemmGraph, m_commandKernel);

            auto outputSize = m_totalWorkitemCount;
            auto rvPointer  = make_shared_device<uint64_t>(outputSize, 0);
            commandArgs.setArgument(m_rvTag, ArgumentType::Value, rvPointer.get());

            auto rvPointer2 = make_shared_device<uint64_t>(outputSize, 0);
            commandArgs.setArgument(m_rvTag2, ArgumentType::Value, rvPointer2.get());

            m_commandKernel->launchKernel(commandArgs.runtimeArguments());

            m_hostBuffer.resize(outputSize, 10);
            CHECK_THAT(hipMemcpy(m_hostBuffer.data(),
                                 rvPointer.get(),
                                 sizeof(uint64_t) * outputSize,
                                 hipMemcpyDefault),
                       HasHipSuccess(0));

            m_hostBuffer2.resize(outputSize, 20);
            CHECK_THAT(hipMemcpy(m_hostBuffer2.data(),
                                 rvPointer2.get(),
                                 sizeof(uint64_t) * outputSize,
                                 hipMemcpyDefault),
                       HasHipSuccess(0));

            // check hostBuffer's value
            Log::debug("outputSize: {}", outputSize);
        }

        void test_implicit_workitemcount()
        {
            generateOrigKernelByProlog();

            auto k = m_context->kernel();
            m_context->schedule(kb_implicit_workitemcount(
                m_context,
                k->workitemCount()[0],
                k->workitemCount()[1],
                (m_commandKernel->getCommandParameters()->getManualWorkgroupSize()).value())());

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            launchKernelAndCopyBackToHost();

            // Remove ":" and subsequent parts to extract only leading literals.
            auto const& host_x_string = toString(m_workitemCount[0]);
            size_t      del1          = host_x_string.find_first_of(":");
            auto const& host_x        = host_x_string.substr(0, del1);

            auto const& host_y_string = toString(m_workitemCount[1]);
            del1                      = host_y_string.find_first_of(":");
            auto const& host_y        = host_y_string.substr(0, del1);

            for(int i = 0, size = m_hostBuffer.size(); i < m_totalWorkitemCount; i++)
            {
                // For 128 by 128 output matrix, workgroupCount computed is {512, 2, 1}
                if(toString(m_hostBuffer[i]) != host_x || toString(m_hostBuffer2[i]) != host_y)
                {
                    std::cout
                        << "workitemCount.x and workitemCount.y in kernel for global workitem " << i
                        << ": " << m_hostBuffer[i] << ", " << m_hostBuffer2[i] << "\n";
                    std::cout << "workitemCount.x and workitemCount.y in host for global workitem "
                              << i << ": " << host_x << ", " << host_y << "\n";
                }
                CHECK(toString(m_hostBuffer[i]) == host_x);
                CHECK(toString(m_hostBuffer2[i]) == host_y);
            }
        }

        void test_sanity_indices()
        {
            generateOrigKernelByProlog();

            auto k = m_context->kernel();

            m_context->schedule(kb_sanity_indices(
                m_context,
                k->workitemCount(),
                (m_commandKernel->getCommandParameters()->getManualWorkgroupSize()).value())());

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            launchKernelAndCopyBackToHost();

            for(int i = 0; i < m_totalWorkitemCount; i++)
                Log::debug("wgx {} thx {}", m_hostBuffer[i], m_hostBuffer2[i]);

            // Remove ":" and subsequent parts to extract only leading literals.
            auto const& host_x_string   = toString(m_workitemCount[0]);
            size_t      del1            = host_x_string.find_first_of(":");
            auto const& host_x          = host_x_string.substr(0, del1);
            auto        workitemcount_x = std::stoi(host_x);

            auto const& host_y_string   = toString(m_workitemCount[1]);
            del1                        = host_y_string.find_first_of(":");
            auto const& host_y          = host_y_string.substr(0, del1);
            auto        workitemcount_y = std::stoi(host_y);

            Log::debug("workitemcount_x {} workitemcount_y {}", workitemcount_x, workitemcount_y);

            auto workgroupsize_x
                = ((m_commandKernel->getCommandParameters()->getManualWorkgroupSize()).value())[0];
            for(int rows = 0, numRows = workitemcount_y; rows < numRows; rows++)
            {
                auto blockIdx_y = rows % workgroupsize_x;
                auto base       = rows * workitemcount_x;
                for(int cols = 0, numCols = workitemcount_x; cols < numCols; cols++)
                {
                    auto blockIdx_x  = cols / workgroupsize_x;
                    auto threadIdx_x = cols % workgroupsize_x;

                    auto linearIdx = cols + base;

                    CHECK(m_hostBuffer[linearIdx] == blockIdx_x);
                    CHECK(m_hostBuffer2[linearIdx] == threadIdx_x);
                }
            }
        }

        void test_equal_one()
        {
            generateOrigKernelByProlog();

            // Get the expression to compute
            std::vector<Expression::ExpressionPtr> indexExprPtrs;
            indexExprPtrs = AddressTrace(m_commandKernel->getKernelGraph(), m_context)
                                .traceComputeIndexWithBuffer();
            std::vector<Expression::ExpressionPtr> widenedExprPtrs;
            for(int i = 0, size = indexExprPtrs.size(); i < size; i++)
            {
                auto eptr = indexExprPtrs[i];
                Log::debug("== Expr : {} ", toString(eptr));

                widenedExprPtrs.push_back(rocRollerTest::widenTo64bit(eptr));
                Log::debug("++ Widen : {} ", toString(widenedExprPtrs.back()));
            }

            auto k = m_context->kernel();
            m_context->schedule(kb_equal_one(
                m_context,
                indexExprPtrs[0],
                widenedExprPtrs[0],
                k->workitemCount(),
                (m_commandKernel->getCommandParameters()->getManualWorkgroupSize()).value())());

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            launchKernelAndCopyBackToHost();

            // for test_kernelbody
            for(int i = 0, size = m_hostBuffer.size(); i < size; i++)
            {
                if(m_hostBuffer[i] != m_hostBuffer2[i])
                {
                    Log::debug("diff at {}: {} {}", i, m_hostBuffer[i], m_hostBuffer2[i]);
                }
                CHECK(m_hostBuffer[i] == m_hostBuffer2[i]);
            }
        }

        void test_equal()
        {
            generateOrigKernelByProlog();

            // Get the expression to compute
            std::vector<Expression::ExpressionPtr> indexExprPtrs;
            indexExprPtrs = AddressTrace(m_commandKernel->getKernelGraph(), m_context)
                                .traceComputeIndexWithBuffer();
            std::vector<Expression::ExpressionPtr> widenedExprPtrs;
            for(int i = 0, size = indexExprPtrs.size(); i < size; i++)
            {
                auto eptr = indexExprPtrs[i];
                Log::debug("== Expr : {} ", toString(eptr));

                widenedExprPtrs.push_back(rocRollerTest::widenTo64bit(eptr));
                Log::debug("++ Widen : {} ", toString(widenedExprPtrs.back()));
            }

            auto allone_uint64
                = std::make_shared<Expression::Expression>(static_cast<uint64_t>(0xFFFFFFFF));

            auto kb = [&]()
                // return [context, input, widenedInput, workitemCount, workgroupSize, allone_uint64]()
                -> Generator<Instruction> {
                    // store base addr
                    Register::ValuePtr s_ptr;

                    co_yield m_context->argLoader()->getValue("rv_ptr", s_ptr);

                    Register::ValuePtr s_ptr2;

                    co_yield m_context->argLoader()->getValue("rv_ptr2", s_ptr2);

                    // 2-D
                    auto compare_res_pointer = get64BitVectorOffset(
                        m_context,
                        m_context->kernel()->workitemCount(),
                        (m_commandKernel->getCommandParameters()->getManualWorkgroupSize())
                            .value());
                    Log::debug("Offset in kb: {}", toString(compare_res_pointer));

                    Register::ValuePtr v_offset = nullptr;
                    co_yield Expression::generate(
                        v_offset, compare_res_pointer + s_ptr->expression(), m_context);

                    // may not be needed.
                    //co_yield_(Instruction::Wait(WaitCount::LGKMCnt(0, "extra waitcnt for debug")));

                    // boolean_diff was allocated to s[0:1]
                    // diff should be computed per lane,
                    // but is_zero_diff is actually one-bit
                    auto boolean_true = Register::Value::WavefrontPlaceholder(m_context);
                    Register::ValuePtr v_allone;
                    co_yield Expression::generate(v_allone, allone_uint64, m_context);

                    co_yield m_context->copier()->copy(
                        boolean_true, v_allone, "set to true for all lanes");

                    Register::ValuePtr temp_res;
                    for(int i = 0, size = indexExprPtrs.size(); i < size; i++)
                    {
                        // Compute the value 1
                        Register::ValuePtr v_value_1 = nullptr;
                        co_yield Expression::generate(v_value_1, indexExprPtrs[i], m_context);

                        // Compute the value 2
                        Register::ValuePtr v_value_2 = nullptr;
                        co_yield Expression::generate(v_value_2, widenedExprPtrs[i], m_context);

                        // Compute diff (value_1 - value_2)
                        auto is_zero_diff = v_value_1->expression() == v_value_2->expression();
                        {
                            auto boolType = resultVariableType(is_zero_diff).dataType;
                            AssertFatal(boolType == DataType::Bool64,
                                        "is_zero type {}",
                                        toString(boolType));
                        }

                        // boolean_true = boolean_true | is_zero_diff
                        auto accumRes
                            = std::make_shared<Expression::Expression>(Expression::BitwiseAnd{
                                boolean_true->expression(), is_zero_diff, "accum"});
                        {
                            auto accumType = resultVariableType(accumRes).dataType;
                            AssertFatal(accumType == DataType::Bool64,
                                        "accum type {}",
                                        toString(accumType));
                        }
                        co_yield Expression::generate(boolean_true, accumRes, m_context);
                    }

                    auto v_value = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::UInt64, 1);
                    co_yield m_context->copier()->copy(v_value, boolean_true, "Move value");
                    co_yield m_context->mem()->storeGlobal(v_offset, v_value, 0, 8);
                };

            m_context->schedule(kb());

            auto k = m_context->kernel();
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            launchKernelAndCopyBackToHost();

            for(int i = 0, size = m_hostBuffer.size(); i < size; i++)
            {
                if(m_hostBuffer[i] != 0xFFFFFFFF)
                {
                    std::cout << "The addresses are not same at " << i << " " << m_hostBuffer[i]
                              << "\n";
                }
                CHECK(m_hostBuffer[i] == 0xFFFFFFFF);
            }

            // hipFree will be taken care of by make_shared_device
        }

    private:
        ContextPtr                  m_context;
        CommandKernelPtr            m_commandKernel;
        GEMMProblem const&          m_problem;
        rocRollerTest::Graphs::GEMM m_gemmGraph;

        rocRoller::Operations::OperationTag m_rvTag;
        rocRoller::Operations::OperationTag m_rvTag2;

        // For checking results on host-side
        std::array<Expression::ExpressionPtr, 3> m_workitemCount;
        uint                                     m_totalWorkitemCount;
        std::vector<uint64_t>                    m_hostBuffer;
        std::vector<uint64_t>                    m_hostBuffer2;
    };

    TEST_CASE("address calculation test generate and run", "[expression][gpu]")
    {
        // Single here means applied to all three A, B, C matrices.
        // TODO: Add more dataTypes
        std::vector<DataType> singleDataTypes = {DataType::Float};

        for(auto dataType : singleDataTypes)
        {
            for(auto [m, n, macM, macN] : TestValues::gemmProblemSizes)
            {
                // Come up with a string from problem_size and data type, to be given to ForTestDevice();
                auto suffixForKernelName = std::to_string(m) + "x" + std::to_string(n) + "_"
                                           + std::to_string(macM) + "_" + std::to_string(macN);
                auto context = TestContext::ForTestDevice({}, suffixForKernelName);

                GEMMProblem                 problem{.m = m, .n = n, .macM = macM, .macN = macN};
                rocRollerTest::Graphs::GEMM gemm(dataType);
                gemm.setProblem(problem);
                CAPTURE(dataType, m, n, macM, macN);

                AddressCalculationTest kernel(context.get(), problem, gemm);
                // Generate a kernel for testing address calculation and run.
                // Verification of the result is done.
                kernel.test_equal();
            }
        }
    }

    TEST_CASE("address calculation test generate and run one pair", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice({}, "128x128_one_pair");

        GEMMProblem                 problem{.m = 128, .n = 128};
        rocRollerTest::Graphs::GEMM gemm(DataType::Float);
        gemm.setProblem(problem);

        AddressCalculationTest kernel(context.get(), problem, gemm);
        kernel.test_equal_one();
    }

    TEST_CASE("address calculation test implicit workitemcount", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice({}, "impl_workitemcnt");

        GEMMProblem                 problem{.m = 128, .n = 128};
        rocRollerTest::Graphs::GEMM gemm(DataType::Float);
        gemm.setProblem(problem);

        AddressCalculationTest kernel(context.get(), problem, gemm);
        kernel.test_implicit_workitemcount();
    }

    TEST_CASE("address calculation test sanity check", "[expression][gpu]")
    {
        auto context = TestContext::ForTestDevice({}, "128x128_sanity_indices");

        GEMMProblem                 problem{.m = 128, .n = 128};
        rocRollerTest::Graphs::GEMM gemm(DataType::Float);
        gemm.setProblem(problem);

        AddressCalculationTest kernel(context.get(), problem, gemm);
        kernel.test_sanity_indices();
    }
}
