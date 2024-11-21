#include <algorithm>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"
#include <common/F8Values.hpp>

using namespace rocRoller;

namespace rocRollerTest
{
    struct F8Problem
    {
        std::shared_ptr<Command>            command;
        rocRoller::Operations::OperationTag resultTag, aTag;
    };

    class F8TestGPU : public GPUContextFixtureParam<rocRoller::DataType>
    {
    };

    const size_t numF8PerElement = 4;

    /**
     * Loads F8x4 to GPU, unpacks to individual F8s, convert to float, store to CPU
    */
    void genF8x4LoadToFloatStore(rocRoller::ContextPtr context,
                                 F8Problem&            prob,
                                 int                   N,
                                 rocRoller::DataType   F8Type)
    {
        auto k = context->kernel();

        k->setKernelDimensions(1);
        prob.command = std::make_shared<Command>();

        prob.resultTag  = prob.command->allocateTag();
        auto result_exp = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Float, PointerType::PointerGlobal}, prob.resultTag, ArgumentType::Value));

        prob.aTag  = prob.command->allocateTag();
        auto a_exp = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {F8Type, PointerType::PointerGlobal}, prob.aTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", {F8Type, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        context->schedule(k->preamble());
        context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield context->argLoader()->getValue("result", s_result);
            co_yield context->argLoader()->getValue("a", s_a);

            auto result_ptr
                = Register::Value::Placeholder(context,
                                               Register::Type::Vector,
                                               {DataType::Float, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());

            auto unsegmented = DataTypeInfo::Get(F8Type).unsegmentedVariableType();
            AssertFatal(unsegmented, "packed data type not found");
            DataType packedDataType = unsegmented->dataType;

            auto a_ptr
                = Register::Value::Placeholder(context,
                                               Register::Type::Vector,
                                               {packedDataType, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());
            auto v_a
                = Register::Value::Placeholder(context, Register::Type::Vector, packedDataType, 1);

            auto v_temp = Register::Value::Placeholder(context, Register::Type::Vector, F8Type, 1);

            co_yield v_a->allocate();
            co_yield a_ptr->allocate();
            co_yield result_ptr->allocate();

            co_yield context->copier()->copy(result_ptr, s_result, "Move pointer.");
            co_yield context->copier()->copy(a_ptr, s_a, "Move pointer.");

            auto bpi
                = CeilDivide(DataTypeInfo::Get(a_ptr->variableType().dataType).elementBits, 8u);
            auto bpo = CeilDivide(
                DataTypeInfo::Get(result_ptr->variableType().dataType).elementBits, 8u);

            for(int i = 0; i < N; i++)
            {
                co_yield context->mem()->loadFlat(v_a, a_ptr, i * bpi, bpi);

                // Bitmask each F8 of F8x4, convert to float, then store
                for(int f8_idx = 0; f8_idx < numF8PerElement; f8_idx++)
                {
                    co_yield_(Instruction::Comment("Extract f8 from packed F8"));
                    co_yield generateOp<Expression::BitFieldExtract>(
                        v_temp, v_a, Expression::BitFieldExtract{{}, F8Type, f8_idx * 8, 8});

                    co_yield_(Instruction::Comment("Convert to float"));
                    co_yield generateOp<Expression::Convert<DataType::Float>>(v_temp, v_temp);

                    co_yield context->mem()->storeFlat(result_ptr,
                                                       v_temp,
                                                       (i * numF8PerElement + f8_idx) * bpo,
                                                       bpo,
                                                       "Store to result");
                }
            }
        };

        context->schedule(kb());
        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());
    }

    /**
     * @param N number of F8x4; so Nx4 float results
     */
    template <typename T>
    void executeF8x4LoadToFloatStore(rocRoller::ContextPtr context, rocRoller::DataType F8Type)
    {
        int N = 256;

        auto rng = RandomGenerator(316473u);
        auto a   = rng.vector<uint>(
            N, std::numeric_limits<uint>::min(), std::numeric_limits<uint>::max());

        std::vector<float> result(a.size() * numF8PerElement);

        F8Problem prob;
        genF8x4LoadToFloatStore(context, prob, a.size(), F8Type);
        CommandKernel commandKernel;
        commandKernel.setContext(context);
        commandKernel.generateKernel();

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<float>(result.size());

        CommandArguments commandArgs = prob.command->createArguments();

        commandArgs.setArgument(prob.resultTag, ArgumentType::Value, d_result.get());
        commandArgs.setArgument(prob.aTag, ArgumentType::Value, d_a.get());

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(
                result.data(), d_result.get(), sizeof(float) * result.size(), hipMemcpyDefault),
            HasHipSuccess(0));

        for(int i = 0; i < a.size(); i++)
        {
            union
            {
                uint32_t word;
                uint8_t  bytes[4];
            } u;
            u.word = a[i];

            for(int f8_idx = 0; f8_idx < numF8PerElement; f8_idx++)
            {
                T expected_f8;
                expected_f8.data = u.bytes[f8_idx];

                float actual   = result.at(i * numF8PerElement + f8_idx);
                float expected = expected_f8.operator float();

                if(std::isnan(expected))
                    EXPECT_TRUE(std::isnan(actual));
                else
                    EXPECT_EQ(actual, expected);
            }
        }
    }

    TEST_P(F8TestGPU, GPU_F8x4LoadToFloatStore)
    {
        auto F8Type = std::get<rocRoller::DataType>(GetParam());
        if(isLocalDevice())
        {
            if(F8Type == rocRoller::DataType::FP8)
                executeF8x4LoadToFloatStore<rocRoller::FP8>(m_context, F8Type);
            else
                executeF8x4LoadToFloatStore<rocRoller::BF8>(m_context, F8Type);
        }
        else
        {
            F8Problem prob;
            genF8x4LoadToFloatStore(m_context, prob, 2, F8Type);
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    /**
     * Loads sparse F8s to GPU, packs into F8x4s, stores to CPU
    */
    void genF8LoadGather(rocRoller::ContextPtr m_context, int N, rocRoller::DataType F8Type)
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto result_exp = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));

        auto aTag  = command->allocateTag();
        auto a_exp = std::make_shared<Expression::Expression>(command->allocateArgument(
            {F8Type, PointerType::PointerGlobal}, aTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::UInt32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", {F8Type, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto result_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());

            auto a_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {F8Type, PointerType::PointerGlobal},
                                               1,
                                               Register::AllocationOptions::FullyContiguous());
            auto v_a = Register::Value::Placeholder(m_context, Register::Type::Vector, F8Type, 1);

            auto v_temp
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               F8Type,
                                               4,
                                               Register::AllocationOptions::FullyContiguous());

            co_yield v_a->allocate();
            co_yield a_ptr->allocate();
            co_yield result_ptr->allocate();
            co_yield v_temp->allocate();

            co_yield m_context->copier()->copy(result_ptr, s_result, "Move pointer.");
            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");

            auto bpi
                = CeilDivide(DataTypeInfo::Get(a_ptr->variableType().dataType).elementBits, 8u);
            auto bpo = CeilDivide(
                DataTypeInfo::Get(result_ptr->variableType().dataType).elementBits, 8u);

            auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
            co_yield bufDesc->setup();
            co_yield bufDesc->setSize(Register::Value::Literal(N));
            co_yield bufDesc->setOptions(Register::Value::Literal(131072)); //0x00020000

            auto bufInstOpts = rocRoller::BufferInstructionOptions();

            auto vgprSerial = m_context->kernel()->workitemIndex()[0];

            co_yield bufDesc->setBasePointer(s_a);
            for(int i = 0; i < N; ++i)
            {
                co_yield m_context->mem()->loadBuffer(
                    v_temp->element({i}), vgprSerial, i, bufDesc, bufInstOpts, 1);
            }
            co_yield bufDesc->setBasePointer(s_result);
            co_yield m_context->mem()->storeBuffer(v_temp, vgprSerial, 0, bufDesc, bufInstOpts, N);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    /**
     * @param N number of F8x4; so Nx4 float results
     */
    void executeF8LoadGather(rocRoller::ContextPtr context, int N, rocRoller::DataType F8Type)
    {
        genF8LoadGather(context, N, F8Type);

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = context->instructions()->getExecutableKernel();

        std::vector<uint8_t> a(N);
        for(int i = 0; i < N; i++)
            a[i] = i + 10;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<uint32_t>(N / 4);

        KernelArguments kargs;
        kargs.append<void*>("result", d_result.get());
        kargs.append<void*>("a", d_a.get());

        KernelInvocation invocation;

        executableKernel->executeKernel(kargs, invocation);

        std::vector<uint32_t> result(N / 4);
        ASSERT_THAT(
            hipMemcpy(result.data(), d_result.get(), sizeof(uint32_t) * N / 4, hipMemcpyDefault),
            HasHipSuccess(0));

        auto bpi = 1;
        auto bpo = 4;
        for(int i = 0; i < N / 4; i++)
        {
            uint32_t expected = a[i] | (a[i + 1] << 8) | (a[i + 2] << 16) | (a[i + 3] << 24);
            EXPECT_EQ(result[i], expected) << std::hex << result[i] << " " << expected;
        }
    }

    TEST_P(F8TestGPU, GPU_F8LoadGather)
    {
        constexpr int N = 4;
        if(isLocalDevice())
        {
            executeF8LoadGather(m_context, N, std::get<rocRoller::DataType>(GetParam()));
        }
        else
        {
            genF8LoadGather(m_context, N, std::get<rocRoller::DataType>(GetParam()));
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    template <typename F8Type>
    void numberConversion(double fp64)
    {
        // F8 to FP32
        F8Type f8(fp64);
        float  fp32(f8);
        if(!std::isnan(fp64))
            EXPECT_FLOAT_EQ(fp32, fp64);
        else
            EXPECT_TRUE(std::isnan(fp32));

        // FP32 to F8
        f8 = F8Type(fp32);
        if(!std::isnan(fp64))
            EXPECT_FLOAT_EQ((double)f8, fp64);
        else
            EXPECT_TRUE(std::isnan(f8));
    }

    TEST_P(F8TestGPU, GPU_CPUConversions)
    {
        auto const& FP8Values = FloatReference<rocRoller::FP8>::Values;
        std::for_each(FP8Values.begin(), FP8Values.end(), numberConversion<rocRoller::FP8>);

        auto const& BF8Values = FloatReference<rocRoller::BF8>::Values;
        std::for_each(BF8Values.begin(), BF8Values.end(), numberConversion<rocRoller::BF8>);
    }

    template <typename F8Type>
    void checkSpecialValues(float& f32_inf, float& f32_nan, float& f32_zero)
    {
        F8Type f8_inf(f32_inf);
        // FP8/BF8 use the same value for NaN and Inf, so we are unable to know
        // if the value is NaN or Inf, and we choose to return NaN in both cases.
        EXPECT_TRUE(std::isnan(f8_inf));
        EXPECT_FALSE(std::isinf(f8_inf));

        F8Type f8_nan(f32_nan);
        EXPECT_TRUE(std::isnan(f8_nan));

        F8Type f8_zero(f32_zero);
        EXPECT_TRUE(std::iszero(f8_zero));
    }

    TEST_P(F8TestGPU, GPU_SpecialValues)
    {
        union
        {
            uint32_t bits;
            float    val;
        } f32_inf, f32_nan, f32_zero;

        // For single-precision, if all exponent bits are 1 and
        //  - if mantissa is zero     => Inf
        //  - if mantissa is non-zero => NaN
        f32_inf.bits  = 0x7F800000;
        f32_nan.bits  = 0x7F800001;
        f32_zero.bits = 0x0;

        EXPECT_TRUE(std::isinf(f32_inf.val));
        EXPECT_TRUE(std::isnan(f32_nan.val));

        checkSpecialValues<rocRoller::FP8>(f32_inf.val, f32_nan.val, f32_zero.val);
        checkSpecialValues<rocRoller::BF8>(f32_inf.val, f32_nan.val, f32_zero.val);
    }

    INSTANTIATE_TEST_SUITE_P(F8TestGPU,
                             F8TestGPU,
                             ::testing::Combine(::testing::Values(GPUArchitectureTarget{
                                                    GPUArchitectureGFX::GFX942, {.sramecc = true}}),
                                                ::testing::Values(rocRoller::DataType::FP8,
                                                                  rocRoller::DataType::BF8)));

}
