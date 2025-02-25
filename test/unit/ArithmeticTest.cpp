#include <cmath>
#include <memory>

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TestValues.hpp"
#include "Utilities.hpp"

namespace ArithmeticTest
{

    template <typename T>
    struct identity
    {
        using type = T;
    };

    template <typename T>
    using try_make_unsigned =
        typename std::conditional_t<std::is_integral_v<T>, std::make_unsigned<T>, identity<T>>;

    template <typename T>
    using try_make_signed =
        typename std::conditional_t<std::is_integral_v<T>, std::make_signed<T>, identity<T>>;

    using namespace rocRoller;

    const int LITERAL_TEST = 227;

    struct IntegralArithmeticTest : public GPUContextFixtureParam<Register::Type>
    {
        template <typename T>
        void testBody()
        {
            auto dataType = TypeInfo<T>::Var.dataType;
            auto regType  = std::get<1>(GetParam());

            auto k = m_context->kernel();

            if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            {
                GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
            }

            auto numBoolRegs = k->wavefront_size() / 32;

            k->setKernelName("IntegralArithmeticTest");
            k->setKernelDimensions(1);

            auto command = std::make_shared<Command>();

            auto resultTag  = command->allocateTag();
            auto resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
                {dataType, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));
            auto comparisonResultTag  = command->allocateTag();
            auto comparisonResultExpr = std::make_shared<Expression::Expression>(
                command->allocateArgument({DataType::UInt32, PointerType::PointerGlobal},
                                          comparisonResultTag,
                                          ArgumentType::Value));
            auto aTag   = command->allocateTag();
            auto aExpr  = std::make_shared<Expression::Expression>(command->allocateArgument(
                {dataType, PointerType::Value}, aTag, ArgumentType::Value));
            auto bTag   = command->allocateTag();
            auto bExpr  = std::make_shared<Expression::Expression>(command->allocateArgument(
                {dataType, PointerType::Value}, bTag, ArgumentType::Value));
            auto shTag  = command->allocateTag();
            auto shExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
                {DataType::UInt32, PointerType::Value}, shTag, ArgumentType::Value));

            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            k->addArgument({"result",
                            {dataType, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            resultExpr});
            k->addArgument({"comparisonResult",
                            {DataType::UInt32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            comparisonResultExpr});
            k->addArgument({"a", dataType, DataDirection::ReadOnly, aExpr});
            k->addArgument({"b", dataType, DataDirection::ReadOnly, bExpr});
            k->addArgument({"shift", DataType::UInt32, DataDirection::ReadOnly, shExpr});

            k->setWorkgroupSize({1, 1, 1});
            k->setWorkitemCount({one, one, one});
            k->setDynamicSharedMemBytes(zero);

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr resultArg, comparisonResultArg, aArg, bArg, shiftArg;
                co_yield m_context->argLoader()->getValue("result", resultArg);
                co_yield m_context->argLoader()->getValue("comparisonResult", comparisonResultArg);
                co_yield m_context->argLoader()->getValue("a", aArg);
                co_yield m_context->argLoader()->getValue("b", bArg);
                co_yield m_context->argLoader()->getValue("shift", shiftArg);

                auto resultPtr = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, {dataType, PointerType::PointerGlobal}, 1);
                auto resultReg
                    = Register::Value::Placeholder(m_context, Register::Type::Vector, dataType, 1);

                auto a     = Register::Value::Placeholder(m_context, regType, dataType, 1);
                auto b     = Register::Value::Placeholder(m_context, regType, dataType, 1);
                auto c     = Register::Value::Placeholder(m_context, regType, dataType, 1);
                auto shift = Register::Value::Placeholder(m_context, regType, DataType::UInt32, 1);
                auto boolean = (regType == Register::Type::Vector)
                                   ? Register::Value::WavefrontPlaceholder(m_context)
                                   : Register::Value::Placeholder(
                                       m_context, Register::Type::Scalar, DataType::Bool, 1);

                co_yield a->allocate();
                co_yield b->allocate();
                co_yield c->allocate();
                co_yield shift->allocate();
                co_yield boolean->allocate();
                co_yield resultPtr->allocate();
                co_yield resultReg->allocate();

                co_yield m_context->copier()->copy(resultPtr, resultArg, "Move pointer");
                co_yield m_context->copier()->copy(a, aArg, "Move value");
                co_yield m_context->copier()->copy(b, bArg, "Move value");
                co_yield m_context->copier()->copy(shift, shiftArg, "Move value");

                auto store = [&](size_t idx, bool logical = false) -> Generator<Instruction> {
                    co_yield m_context->copier()->copy(
                        resultReg, c, "Move result to VGPR to store.");
                    if(logical && regType == Register::Type::Scalar)
                    {
                        co_yield m_context->mem()->store(
                            MemoryInstructions::Flat,
                            resultPtr,
                            resultReg,
                            Register::Value::Literal(idx * sizeof(uint32_t)),
                            sizeof(uint32_t));
                    }
                    else if(logical && regType == Register::Type::Vector)
                    {
                        co_yield m_context->mem()->store(
                            MemoryInstructions::Flat,
                            resultPtr,
                            resultReg,
                            Register::Value::Literal(idx * sizeof(uint32_t) * numBoolRegs),
                            sizeof(uint32_t) * numBoolRegs);
                    }
                    else
                    {
                        co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                                         resultPtr,
                                                         resultReg,
                                                         Register::Value::Literal(idx * sizeof(T)),
                                                         sizeof(T));
                    }
                };

                co_yield generateOp<Expression::Add>(c, a, b);
                co_yield store(0);

                co_yield generateOp<Expression::Subtract>(c, a, b);
                co_yield store(1);

                co_yield generateOp<Expression::Multiply>(c, a, b);
                co_yield store(2);

                co_yield generateOp<Expression::Divide>(c, a, b);
                co_yield store(3);

                co_yield generateOp<Expression::Modulo>(c, a, b);
                co_yield store(4);

                co_yield generateOp<Expression::ShiftL>(c, a, b);
                co_yield store(5);

                co_yield generateOp<Expression::LogicalShiftR>(c, a, b);
                co_yield store(6);

                co_yield generateOp<Expression::ArithmeticShiftR>(c, a, b);
                co_yield store(7);

                co_yield generateOp<Expression::BitwiseAnd>(c, a, b);
                co_yield store(8);

                co_yield generateOp<Expression::MultiplyHigh>(c, a, b);
                co_yield store(9);

                co_yield generateOp<Expression::Negate>(c, a);
                co_yield store(10);

                co_yield generateOp<Expression::BitwiseXor>(c, a, b);
                co_yield store(11);

                co_yield generateOp<Expression::AddShiftL>(c, a, b, shift);
                co_yield store(12);

                co_yield generateOp<Expression::ShiftLAdd>(c, a, shift, b);
                co_yield store(13);

                co_yield generateOp<Expression::BitwiseOr>(c, a, b);
                co_yield store(14);

                co_yield generateOp<Expression::Divide>(
                    c, a, Register::Value::Literal(LITERAL_TEST));
                co_yield store(15);

                co_yield generateOp<Expression::Divide>(
                    c, Register::Value::Literal(LITERAL_TEST), b);
                co_yield store(16);

                co_yield generateOp<Expression::Modulo>(
                    c, a, Register::Value::Literal(LITERAL_TEST));
                co_yield store(17);

                co_yield generateOp<Expression::Modulo>(
                    c, Register::Value::Literal(LITERAL_TEST), b);
                co_yield store(18);

                co_yield generateOp<Expression::BitwiseNegate>(c, a);
                co_yield store(19);

                co_yield generateOp<Expression::GreaterThanEqual>(boolean, a, b);
                co_yield generateOp<Expression::Conditional>(c, boolean, a, b);
                co_yield store(20);

                //
                // Logical / boolean
                //
                // Change result and c
                //

                resultPtr
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Raw32, PointerType::PointerGlobal},
                                                   1);
                co_yield resultPtr->allocate();
                co_yield m_context->copier()->copy(resultPtr, comparisonResultArg, "Move pointer");

                resultReg = boolean->placeholder(Register::Type::Vector,
                                                 Register::AllocationOptions::FullyContiguous());
                co_yield resultReg->allocate();

                c = boolean;

                co_yield generateOp<Expression::GreaterThan>(c, a, b);
                co_yield store(0, true);

                co_yield generateOp<Expression::GreaterThanEqual>(c, a, b);
                co_yield store(1, true);

                co_yield generateOp<Expression::LessThan>(c, a, b);
                co_yield store(2, true);

                co_yield generateOp<Expression::LessThanEqual>(c, a, b);
                co_yield store(3, true);

                co_yield generateOp<Expression::Equal>(c, a, b);
                co_yield store(4, true);

                co_yield generateOp<Expression::NotEqual>(c, a, b);
                co_yield store(5, true);

                auto aExpr = a->expression();
                auto bExpr = b->expression();

                co_yield generate(c,
                                  (aExpr < Expression::literal(0, dataType))
                                      && (bExpr < Expression::literal(0, dataType)),
                                  m_context);
                co_yield store(6, true);

                co_yield generate(c,
                                  (aExpr < Expression::literal(0, dataType))
                                      || (bExpr < Expression::literal(0, dataType)),
                                  m_context);
                co_yield store(7, true);

                co_yield generate(c, Expression::logicalNot(aExpr < bExpr), m_context);
                co_yield store(8, true);

                co_yield generate(
                    c, Expression::logicalNot(Expression::logicalNot(aExpr > bExpr)), m_context);
                co_yield store(9, true);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            if(isLocalDevice())
            {
                CommandKernel commandKernel(m_context);

                size_t const resultSize           = 21;
                size_t const comparisonResultSize = 10;

                auto d_result = make_shared_device<T>(resultSize);
                auto d_comparisonResult
                    = make_shared_device<uint32_t>(comparisonResultSize * numBoolRegs);

                for(T a : TestValues::ByType<T>::values)
                {
                    for(T b : TestValues::ByType<T>::values)
                    {
                        using T_unsigned = typename try_make_unsigned<T>::type;
                        using T_signed   = typename try_make_signed<T>::type;

                        // Signed division/modulo is used for unsigned
                        auto withinDivisionDomain = b != 0
                                                    && a <= std::numeric_limits<T_signed>::max()
                                                    && b <= std::numeric_limits<T_signed>::max();

                        for(uint32_t shift : TestValues::shiftValues)
                        {
                            CommandArguments commandArgs = command->createArguments();

                            commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
                            commandArgs.setArgument(
                                comparisonResultTag, ArgumentType::Value, d_comparisonResult.get());
                            commandArgs.setArgument(aTag, ArgumentType::Value, a);
                            commandArgs.setArgument(bTag, ArgumentType::Value, b);
                            commandArgs.setArgument(shTag, ArgumentType::Value, shift);

                            commandKernel.launchKernel(commandArgs.runtimeArguments());

                            std::vector<T> result(resultSize);
                            ASSERT_THAT(hipMemcpy(result.data(),
                                                  d_result.get(),
                                                  result.size() * sizeof(T),
                                                  hipMemcpyDefault),
                                        HasHipSuccess(0));

                            std::vector<uint32_t> comparisonResult(comparisonResultSize
                                                                   * numBoolRegs);
                            ASSERT_THAT(hipMemcpy(comparisonResult.data(),
                                                  d_comparisonResult.get(),
                                                  comparisonResult.size() * sizeof(uint32_t),
                                                  hipMemcpyDefault),
                                        HasHipSuccess(0));

                            EXPECT_EQ(result[0], a + b);
                            EXPECT_EQ(result[1], a - b);
                            EXPECT_EQ(result[2], a * b);
                            if(withinDivisionDomain)
                            {
                                EXPECT_EQ(result[3], a / b) << "a: " << a << " b: " << b;
                                EXPECT_EQ(result[4], a % b);
                            }
                            if(b < 32 && b >= 0)
                            {
                                EXPECT_EQ(result[5], a << b);
                                EXPECT_EQ(result[6], static_cast<T_unsigned>(a) >> b);
                                EXPECT_EQ(result[7], a >> b);
                            }
                            EXPECT_EQ(result[8], a & b);
                            if constexpr(std::is_same_v<T_signed, int32_t>)
                            {
                                EXPECT_EQ(result[9],
                                          (a * (int64_t)b)
                                              >> std::numeric_limits<T_unsigned>::digits);
                            }
                            else if constexpr(std::is_same_v<T, int64_t>)
                            {
                                EXPECT_EQ(result[9],
                                          (int64_t)(((__int128_t)a * (__int128_t)b)
                                                    >> std::numeric_limits<T_unsigned>::digits));
                            }
                            EXPECT_EQ(result[10], -a);
                            EXPECT_EQ(result[11], a ^ b);
                            EXPECT_EQ(result[12], (a + b) << shift);
                            EXPECT_EQ(result[13], (a << shift) + b);
                            EXPECT_EQ(result[14], a | b);
                            if(withinDivisionDomain)
                            {
                                EXPECT_EQ(result[15], a / LITERAL_TEST);
                                EXPECT_EQ(result[16], LITERAL_TEST / b);
                                EXPECT_EQ(result[17], a % LITERAL_TEST);
                                EXPECT_EQ(result[18], LITERAL_TEST % b);
                            }
                            EXPECT_EQ(result[19], ~a);
                            EXPECT_EQ(result[20], (a >= b) ? a : b);

                            int wm = regType == Register::Type::Vector ? numBoolRegs : 1;

                            EXPECT_EQ(comparisonResult[0 * wm], (a > b ? 1 : 0));
                            EXPECT_EQ(comparisonResult[1 * wm], (a >= b ? 1 : 0));
                            EXPECT_EQ(comparisonResult[2 * wm], (a < b ? 1 : 0));
                            EXPECT_EQ(comparisonResult[3 * wm], (a <= b ? 1 : 0));
                            EXPECT_EQ(comparisonResult[4 * wm], (a == b ? 1 : 0));
                            EXPECT_EQ(comparisonResult[5 * wm], (a != b ? 1 : 0));
                            EXPECT_EQ(comparisonResult[6 * wm], ((a < 0) && (b < 0)) ? 1 : 0);
                            EXPECT_EQ(comparisonResult[7 * wm], ((a < 0) || (b < 0)) ? 1 : 0);
                            EXPECT_EQ(comparisonResult[8 * wm], !(a < b) ? 1 : 0);
                            EXPECT_EQ(comparisonResult[9 * wm], !!(a > b) ? 1 : 0);
                        }
                    }
                }
            }
            else
            {
                std::vector<char> assembledKernel = m_context->instructions()->assemble();
                EXPECT_GT(assembledKernel.size(), 0);
            }
        }
    };

    TEST_P(IntegralArithmeticTest, Int32)
    {
        testBody<int32_t>();
    }

    TEST_P(IntegralArithmeticTest, UInt32)
    {
        testBody<uint32_t>();
    }

    TEST_P(IntegralArithmeticTest, Int64)
    {
        testBody<int64_t>();
    }

    TEST_P(IntegralArithmeticTest, UInt64)
    {
        testBody<uint64_t>();
    }

    INSTANTIATE_TEST_SUITE_P(NewArithmeticTests,
                             IntegralArithmeticTest,
                             ::testing::Combine(supportedISAValues(),
                                                ::testing::Values(Register::Type::Vector,
                                                                  Register::Type::Scalar)));

    class FPArithmeticTest : public GPUContextFixture
    {
    public:
        FPArithmeticTest() {}
    };

    TEST_P(FPArithmeticTest, ArithFloat)
    {
        auto k = m_context->kernel();

        k->setKernelName("ArithFloat");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto resultTag       = command->allocateTag();
        auto resultExpr      = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));
        auto condResultTag   = command->allocateTag();
        auto cond_resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::PointerGlobal}, condResultTag, ArgumentType::Value));
        auto aTag            = command->allocateTag();
        auto aExpr           = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value));
        auto bTag            = command->allocateTag();
        auto bExpr           = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::Value}, bTag, ArgumentType::Value));
        auto cTag            = command->allocateTag();
        auto cExpr           = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::Value}, cTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        resultExpr});
        k->addArgument({"cond_result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        cond_resultExpr});

        k->addArgument({"a", DataType::Float, DataDirection::ReadOnly, aExpr});
        k->addArgument({"b", DataType::Float, DataDirection::ReadOnly, bExpr});
        k->addArgument({"c", DataType::Float, DataDirection::ReadOnly, cExpr});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_cond_result, s_a, s_b, s_c;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("cond_result", s_cond_result);

            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);
            co_yield m_context->argLoader()->getValue("c", s_c);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Float, PointerType::PointerGlobal},
                                               1);

            auto v_cond_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Float, PointerType::PointerGlobal},
                                               1);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_r = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto s_r = Register::Value::Placeholder(m_context,
                                                    Register::Type::Scalar,
                                                    {DataType::Float, PointerType::PointerGlobal},
                                                    1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield v_r->allocate();
            co_yield s_r->allocate();
            co_yield v_result->allocate();
            co_yield v_cond_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_cond_result, s_cond_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move value");
            co_yield m_context->copier()->copy(v_b, s_b, "Move value");
            co_yield m_context->copier()->copy(v_c, s_c, "Move value");

            co_yield generateOp<Expression::Add>(v_r, v_a, v_b);
            co_yield m_context->mem()->storeFlat(v_result, v_r, 0, 4);

            co_yield generateOp<Expression::Subtract>(v_r, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(4), 4);

            co_yield generateOp<Expression::Multiply>(v_r, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(8), 4);

            co_yield generateOp<Expression::Negate>(v_r, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(12), 4);

            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_a, v_b, v_c);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(16), 4);

            auto vcc = m_context->getVCC();
            co_yield generate(vcc, v_c->expression() >= v_a->expression(), m_context);
            co_yield generateOp<Expression::Conditional>(v_r, vcc, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(20), 4);

            co_yield generateOp<Expression::GreaterThan>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 0, 4);

            co_yield generateOp<Expression::GreaterThanEqual>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 4, 4);

            co_yield generateOp<Expression::LessThan>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 8, 4);

            co_yield generateOp<Expression::LessThanEqual>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 12, 4);

            co_yield generateOp<Expression::Equal>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 16, 4);

            co_yield generateOp<Expression::NotEqual>(s_r, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_r, s_r->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_r->subset({0}), 20, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result      = make_shared_device<float>(6);
            auto d_cond_result = make_shared_device<int>(6);

            for(float a : TestValues::floatValues)
            {
                for(float b : TestValues::floatValues)
                {
                    for(float c : TestValues::floatValues)
                    {

                        CommandArguments commandArgs = command->createArguments();

                        commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
                        commandArgs.setArgument(
                            condResultTag, ArgumentType::Value, d_cond_result.get());
                        commandArgs.setArgument(aTag, ArgumentType::Value, a);
                        commandArgs.setArgument(bTag, ArgumentType::Value, b);
                        commandArgs.setArgument(cTag, ArgumentType::Value, c);

                        commandKernel.launchKernel(commandArgs.runtimeArguments());

                        std::vector<float> result(6);
                        ASSERT_THAT(hipMemcpy(result.data(),
                                              d_result.get(),
                                              result.size() * sizeof(float),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        std::vector<int> cond_result(6);
                        ASSERT_THAT(hipMemcpy(cond_result.data(),
                                              d_cond_result.get(),
                                              cond_result.size() * sizeof(int),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        EXPECT_EQ(result[0], a + b);
                        EXPECT_EQ(result[1], a - b);
                        EXPECT_EQ(result[2], a * b);
                        EXPECT_EQ(result[3], -a);
                        float fma = std::fma(a, b, c);
                        if(fma != 0.0)
                        {
                            EXPECT_LT((result[4] - fma) / fma, 5.e-7)
                                << "a: " << a << ", b: " << b << ", c: " << c;
                            ;
                        }
                        EXPECT_EQ(result[5], c >= a ? a : b)
                            << "a: " << a << ", b: " << b << ", c: " << c;
                        ;
                        EXPECT_EQ(cond_result[0], (a > b ? 1 : 0));
                        EXPECT_EQ(cond_result[1], (a >= b ? 1 : 0));
                        EXPECT_EQ(cond_result[2], (a < b ? 1 : 0));
                        EXPECT_EQ(cond_result[3], (a <= b ? 1 : 0));
                        EXPECT_EQ(cond_result[4], (a == b ? 1 : 0));
                        EXPECT_EQ(cond_result[5], (a != b ? 1 : 0));
                    }
                }
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FPArithmeticTest, ArithUnaryFloat)
    {
        auto k = m_context->kernel();

        k->setKernelName("ArithUnaryFloat");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));

        auto aTag  = command->allocateTag();
        auto aExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        resultExpr});

        k->addArgument({"a", DataType::Float, DataDirection::ReadOnly, aExpr});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Float, PointerType::PointerGlobal},
                                               1);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_r = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            co_yield v_a->allocate();
            co_yield v_r->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move value");

            co_yield generateOp<Expression::Exponential2>(v_r, v_a);
            co_yield m_context->mem()->store(MemoryInstructions::Flat, v_result, v_r, 0, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result = make_shared_device<float>(1);

            for(float a : TestValues::floatValues)
            {

                CommandArguments commandArgs = command->createArguments();

                commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
                commandArgs.setArgument(aTag, ArgumentType::Value, a);

                commandKernel.launchKernel(commandArgs.runtimeArguments());

                std::vector<float> result(1);
                ASSERT_THAT(hipMemcpy(result.data(),
                                      d_result.get(),
                                      result.size() * sizeof(float),
                                      hipMemcpyDefault),
                            HasHipSuccess(0));

                EXPECT_EQ(result[0], (a > -126.0) ? std::exp2(a) : 0) << "a: " << a;
                ;
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FPArithmeticTest, ArithFMAMixed)
    {
        auto k = m_context->kernel();

        k->setKernelName("ArithFMAMixed");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));
        auto aTag       = command->allocateTag();
        auto aExpr      = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value));
        auto bTag       = command->allocateTag();
        auto bExpr      = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, bTag, ArgumentType::Value));
        auto cTag       = command->allocateTag();
        auto cExpr      = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::PointerGlobal}, cTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        resultExpr});
        k->addArgument({"a", DataType::Float, DataDirection::ReadOnly, aExpr});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, bExpr});
        k->addArgument(
            {"c", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, cExpr});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b, s_c;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);
            co_yield m_context->argLoader()->getValue("c", s_c);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Int64, PointerType::PointerGlobal},
                                               1);

            auto vbPtr = Register::Value::Placeholder(m_context,
                                                      Register::Type::Vector,
                                                      {DataType::Int64, PointerType::PointerGlobal},
                                                      1);

            auto vcPtr = Register::Value::Placeholder(m_context,
                                                      Register::Type::Vector,
                                                      {DataType::Int64, PointerType::PointerGlobal},
                                                      1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            auto v_c = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::Float,
                                                    2,
                                                    Register::AllocationOptions::FullyContiguous());

            auto v_r = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::Float,
                                                    2,
                                                    Register::AllocationOptions::FullyContiguous());

            co_yield vbPtr->allocate();
            co_yield vcPtr->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield v_r->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(vbPtr, s_b, "Move Pointer");
            co_yield m_context->copier()->copy(vcPtr, s_c, "Move Pointer");
            co_yield m_context->mem()->loadFlat(v_b, vbPtr, 0, 4);
            co_yield m_context->mem()->loadFlat(v_c, vcPtr, 0, 8);

            // fp32 = fp32 * fp16 + fp32
            co_yield generateOp<Expression::MultiplyAdd>(v_r, s_a, v_b, v_c);
            co_yield m_context->mem()->store(MemoryInstructions::Flat, v_result, v_r, 0, 8);

            // fp32 = fp16 * fp16 + fp32
            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_b, v_b, v_c);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(8), 8);

            // fp32 = fp16 * fp32 + fp16
            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_b, v_c, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(16), 8);

            // fp32 = fp32 * fp16 + fp16
            co_yield generateOp<Expression::MultiplyAdd>(v_r, s_a, v_b, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(24), 8);

            // fp32 = fp32 * fp32 + fp16
            co_yield generateOp<Expression::MultiplyAdd>(v_r, s_a, v_c, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(32), 8);

            // fp32 = fp16 * fp32 + fp32
            co_yield generateOp<Expression::MultiplyAdd>(v_r, v_b, v_c, v_c);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_r, Register::Value::Literal(40), 8);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9
           || m_context->targetArchitecture().target().getVersionString() == "gfx900")
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            float               a       = 2.f;
            std::vector<__half> b       = {static_cast<__half>(1.), static_cast<__half>(2.)};
            std::vector<float>  c       = {1.f, 2.f};
            auto                bDevice = make_shared_device<__half>(b);
            auto                cDevice = make_shared_device<float>(c);
            auto                dResult = make_shared_device<float>(12);

            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(resultTag, ArgumentType::Value, dResult.get());
            commandArgs.setArgument(aTag, ArgumentType::Value, a);
            commandArgs.setArgument(bTag, ArgumentType::Value, bDevice.get());
            commandArgs.setArgument(cTag, ArgumentType::Value, cDevice.get());

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            //6 different options
            std::vector<float> result(12);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), dResult.get(), result.size() * sizeof(float), hipMemcpyDefault),
                HasHipSuccess(0));

            // fp32 * fp16 + f32
            float fma1 = a * static_cast<float>(b[0]) + c[0];
            float fma2 = a * static_cast<float>(b[1]) + c[1];

            // fp16 * fp16 + fp32
            float fma3 = static_cast<float>(b[0]) * static_cast<float>(b[0]) + c[0];
            float fma4 = static_cast<float>(b[1]) * static_cast<float>(b[1]) + c[1];

            // fp16 * fp32 + fp16
            float fma5 = static_cast<float>(b[0]) * c[0] + static_cast<float>(b[0]);
            float fma6 = static_cast<float>(b[1]) * c[1] + static_cast<float>(b[1]);

            // fp32 * fp16 + fp16
            float fma7 = a * static_cast<float>(b[0]) + static_cast<float>(b[0]);
            float fma8 = a * static_cast<float>(b[1]) + static_cast<float>(b[1]);

            // fp32 * fp32 + fp16
            float fma9  = a * c[0] + static_cast<float>(b[0]);
            float fma10 = a * c[1] + static_cast<float>(b[1]);

            // fp16 * fp32 + fp32
            float fma11 = static_cast<float>(b[0]) * c[0] + c[0];
            float fma12 = static_cast<float>(b[1]) * c[1] + c[1];

            EXPECT_LT(std::abs(result[0] - fma1) / fma1, 5.e-7);
            EXPECT_LT(std::abs(result[1] - fma2) / fma2, 5.e-7);
            EXPECT_LT(std::abs(result[2] - fma3) / fma3, 5.e-7);
            EXPECT_LT(std::abs(result[3] - fma4) / fma4, 5.e-7);
            EXPECT_LT(std::abs(result[4] - fma5) / fma5, 5.e-7);
            EXPECT_LT(std::abs(result[5] - fma6) / fma6, 5.e-7);
            EXPECT_LT(std::abs(result[6] - fma7) / fma7, 5.e-7);
            EXPECT_LT(std::abs(result[7] - fma8) / fma8, 5.e-7);
            EXPECT_LT(std::abs(result[8] - fma9) / fma9, 5.e-7);
            EXPECT_LT(std::abs(result[9] - fma10) / fma10, 5.e-7);
            EXPECT_LT(std::abs(result[10] - fma11) / fma11, 5.e-7);
            EXPECT_LT(std::abs(result[11] - fma12) / fma12, 5.e-7);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FPArithmeticTest, ArithDouble)
    {
        auto k = m_context->kernel();

        k->setKernelName("ArithDouble");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto resultTag       = command->allocateTag();
        auto resultExpr      = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Double, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));
        auto condResultTag   = command->allocateTag();
        auto cond_resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::PointerGlobal}, condResultTag, ArgumentType::Value));
        auto aTag            = command->allocateTag();
        auto aExpr           = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Double, PointerType::Value}, aTag, ArgumentType::Value));
        auto bTag            = command->allocateTag();
        auto bExpr           = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Double, PointerType::Value}, bTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Double, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        resultExpr});
        k->addArgument({"cond_result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        cond_resultExpr});
        k->addArgument({"a", DataType::Double, DataDirection::ReadOnly, aExpr});
        k->addArgument({"b", DataType::Double, DataDirection::ReadOnly, bExpr});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_cond_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("cond_result", s_cond_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Double, PointerType::PointerGlobal},
                                               1);

            auto v_cond_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Double, PointerType::PointerGlobal},
                                               1);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto s_c = Register::Value::Placeholder(m_context,
                                                    Register::Type::Scalar,
                                                    {DataType::Double, PointerType::PointerGlobal},
                                                    1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield s_c->allocate();
            co_yield v_result->allocate();
            co_yield v_cond_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_cond_result, s_cond_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move value");
            co_yield m_context->copier()->copy(v_b, s_b, "Move value");

            co_yield generateOp<Expression::Add>(v_c, v_a, v_b);
            co_yield m_context->mem()->storeFlat(v_result, v_c, 0, 8);

            co_yield generateOp<Expression::Subtract>(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 8);

            co_yield generateOp<Expression::Multiply>(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(16), 8);

            co_yield generateOp<Expression::Negate>(v_c, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(24), 8);

            auto vcc = m_context->getVCC();
            co_yield generate(vcc, v_a->expression() >= v_b->expression(), m_context);
            co_yield generateOp<Expression::Conditional>(v_c, vcc, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(32), 8);

            co_yield generateOp<Expression::GreaterThan>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(0),
                                             4);

            co_yield generateOp<Expression::GreaterThanEqual>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(4),
                                             4);

            co_yield generateOp<Expression::LessThan>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(8),
                                             4);

            co_yield generateOp<Expression::LessThanEqual>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(12),
                                             4);

            co_yield generateOp<Expression::Equal>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(16),
                                             4);

            co_yield generateOp<Expression::NotEqual>(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(20),
                                             4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result      = make_shared_device<double>(5);
            auto d_cond_result = make_shared_device<int>(6);

            for(double a : TestValues::doubleValues)
            {
                for(double b : TestValues::doubleValues)
                {
                    CommandArguments commandArgs = command->createArguments();

                    commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
                    commandArgs.setArgument(
                        condResultTag, ArgumentType::Value, d_cond_result.get());
                    commandArgs.setArgument(aTag, ArgumentType::Value, a);
                    commandArgs.setArgument(bTag, ArgumentType::Value, b);

                    commandKernel.launchKernel(commandArgs.runtimeArguments());

                    std::vector<double> result(5);
                    ASSERT_THAT(hipMemcpy(result.data(),
                                          d_result.get(),
                                          result.size() * sizeof(double),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    std::vector<int> cond_result(6);
                    ASSERT_THAT(hipMemcpy(cond_result.data(),
                                          d_cond_result.get(),
                                          cond_result.size() * sizeof(int),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    EXPECT_EQ(result[0], a + b);
                    EXPECT_EQ(result[1], a - b);
                    EXPECT_EQ(result[2], a * b);
                    EXPECT_EQ(result[3], -a);
                    EXPECT_EQ(result[4], a >= b ? a : b);
                    EXPECT_EQ(cond_result[0], (a > b ? 1 : 0));
                    EXPECT_EQ(cond_result[1], (a >= b ? 1 : 0));
                    EXPECT_EQ(cond_result[2], (a < b ? 1 : 0));
                    EXPECT_EQ(cond_result[3], (a <= b ? 1 : 0));
                    EXPECT_EQ(cond_result[4], (a == b ? 1 : 0));
                    EXPECT_EQ(cond_result[5], (a != b ? 1 : 0));
                }
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FPArithmeticTest, NullChecks)
    {
        (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

        auto kb_null = [&]() -> Generator<Instruction> {
            co_yield generateOp<Expression::Subtract>(nullptr, nullptr, nullptr);
        };
        ASSERT_THROW(m_context->schedule(kb_null()), FatalError);

        auto kb32 = [&]() -> Generator<Instruction> {
            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_c->allocate();
            co_yield generateOp<Expression::Subtract>(v_c, nullptr, nullptr);
        };
        ASSERT_THROW(m_context->schedule(kb32()), FatalError);

        auto kb64 = [&]() -> Generator<Instruction> {
            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            co_yield v_c->allocate();
            co_yield generateOp<Expression::Subtract>(v_c, nullptr, nullptr);
        };
        ASSERT_THROW(m_context->schedule(kb64()), FatalError);

        auto kb_dst = [&]() -> Generator<Instruction> {
            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);
            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield generateOp<Expression::Subtract>(nullptr, v_a, v_b);
        };
        ASSERT_THROW(m_context->schedule(kb_dst()), FatalError);
    }

    INSTANTIATE_TEST_SUITE_P(ArithmeticTests, FPArithmeticTest, supportedISATuples());
}
