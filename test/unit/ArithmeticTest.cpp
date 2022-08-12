
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/Double.hpp>
#include <rocRoller/CodeGen/Arithmetic/Float.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int64.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TestValues.hpp"

using namespace rocRoller;

namespace ArithmeticTest
{

    class ArithmeticTest : public GPUContextFixture
    {
    public:
        ArithmeticTest() {}
    };

    TEST_P(ArithmeticTest, ArithInt32)
    {
        auto arith = Component::Get<Arithmetic>(m_context, Register::Type::Vector, DataType::Int32);

        auto k = m_context->kernel();

        k->setKernelName("ArithInt32");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));
        auto sh_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::UInt32, PointerType::Value}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", DataType::Int32, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Int32, DataDirection::ReadOnly, b_exp});
        k->addArgument({"shift", DataType::UInt32, DataDirection::ReadOnly, sh_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b, s_shift;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);
            co_yield m_context->argLoader()->getValue("shift", s_shift);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto v_shift = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);

            auto s_c = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Raw32, 2);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield v_shift->allocate();
            co_yield s_c->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move value");
            co_yield m_context->copier()->copy(v_b, s_b, "Move value");
            co_yield m_context->copier()->copy(v_shift, s_shift, "Move value");

            co_yield generateOp<Expression::Add>(v_c, v_a, v_b);
            co_yield m_context->mem()->storeFlat(v_result, v_c, "0", 4);

            co_yield arith->sub(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(4), 4);

            co_yield arith->mul(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 4);

            co_yield arith->div(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(12), 4);

            co_yield arith->mod(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(16), 4);

            co_yield arith->shiftL(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(20), 4);

            co_yield arith->shiftR(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(24), 4);

            co_yield generateOp<Expression::SignedShiftR>(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(28), 4);

            co_yield arith->gt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(32),
                                             4);

            co_yield arith->ge(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(36),
                                             4);

            co_yield arith->lt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(40),
                                             4);

            co_yield arith->le(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(44),
                                             4);

            co_yield arith->eq(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(48),
                                             4);

            co_yield arith->bitwiseAnd(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(52), 4);

            co_yield arith->mulHi(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(56), 4);

            co_yield arith->negate(v_c, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(60), 4);

            co_yield arith->bitwiseXor(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(64), 4);

            co_yield arith->addShiftL(v_c, v_a, v_b, v_shift);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(68), 4);

            co_yield arith->shiftLAdd(v_c, v_a, v_shift, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(72), 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result = make_shared_device<int>(19);

            for(int a : TestValues::int32Values)
            {
                for(int b : TestValues::int32Values)
                {
                    for(unsigned int shift : TestValues::shiftValues)
                    {
                        KernelArguments runtimeArgs;
                        runtimeArgs.append("result", d_result.get());
                        runtimeArgs.append("a", a);
                        runtimeArgs.append("b", b);
                        runtimeArgs.append("shift", shift);

                        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                        std::vector<int> result(19);
                        ASSERT_THAT(hipMemcpy(result.data(),
                                              d_result.get(),
                                              result.size() * sizeof(int),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        EXPECT_EQ(result[0], a + b);
                        EXPECT_EQ(result[1], a - b);
                        EXPECT_EQ(result[2], a * b);
                        if(b != 0)
                        {
                            EXPECT_EQ(result[3], a / b);
                            EXPECT_EQ(result[4], a % b);
                        }
                        if(b < 32 && b >= 0)
                        {
                            EXPECT_EQ(result[5], a << b) << a << " " << b;
                            EXPECT_EQ(result[6], static_cast<unsigned int>(a) >> b);
                            EXPECT_EQ(result[7], a >> b);
                        }
                        EXPECT_EQ(result[8], (a > b ? 1 : 0));
                        EXPECT_EQ(result[9], (a >= b ? 1 : 0));
                        EXPECT_EQ(result[10], (a < b ? 1 : 0));
                        EXPECT_EQ(result[11], (a <= b ? 1 : 0));
                        EXPECT_EQ(result[12], (a == b ? 1 : 0));
                        EXPECT_EQ(result[13], a & b);
                        EXPECT_EQ(result[14], (a * (int64_t)b) >> 32);
                        EXPECT_EQ(result[15], -a);
                        EXPECT_EQ(result[16], a ^ b);
                        EXPECT_EQ(result[17], (a + b) << shift);
                        EXPECT_EQ(result[18], (a << shift) + b);
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

    TEST_P(ArithmeticTest, ArithInt32Scalar)
    {
        auto arith = Component::Get<Arithmetic>(m_context, Register::Type::Scalar, DataType::Int32);

        auto k = m_context->kernel();

        k->setKernelName("ArithInt32Scalar");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));
        auto sh_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::UInt32, PointerType::Value}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", DataType::Int32, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Int32, DataDirection::ReadOnly, b_exp});
        k->addArgument({"shift", DataType::UInt32, DataDirection::ReadOnly, sh_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b, s_shift;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);
            co_yield m_context->argLoader()->getValue("shift", s_shift);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto s_c = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield s_c->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield generateOp<Expression::Add>(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_result, v_c, "", 4);

            co_yield arith->sub(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(4), 4);

            co_yield arith->mul(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 4);

            co_yield arith->div(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(12), 4);

            co_yield arith->mod(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(16), 4);

            co_yield arith->shiftL(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(20), 4);

            co_yield arith->shiftR(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(24), 4);

            co_yield generateOp<Expression::SignedShiftR>(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(28), 4);

            co_yield arith->gt(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(32), 4);

            co_yield arith->ge(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(36), 4);

            co_yield arith->lt(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(40), 4);

            co_yield arith->le(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(44), 4);

            co_yield arith->eq(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(48), 4);

            co_yield arith->bitwiseAnd(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(52), 4);

            co_yield arith->mulHi(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(56), 4);

            co_yield arith->negate(s_c, s_a);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(60), 4);

            co_yield arith->bitwiseXor(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(64), 4);

            co_yield arith->addShiftL(s_c, s_a, s_b, s_shift);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(68), 4);

            co_yield arith->shiftLAdd(s_c, s_a, s_shift, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(72), 4);

            auto shift1 = Register::Value::Literal(1u);
            co_yield arith->shiftLAdd(s_c, s_a, shift1, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(76), 4);

            auto shift2 = Register::Value::Literal(2u);
            co_yield arith->shiftLAdd(s_c, s_a, shift2, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(80), 4);

            auto shift3 = Register::Value::Literal(3u);
            co_yield arith->shiftLAdd(s_c, s_a, shift3, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(84), 4);

            auto shift4 = Register::Value::Literal(4u);
            co_yield arith->shiftLAdd(s_c, s_a, shift4, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(88), 4);

            auto shift5 = Register::Value::Literal(5u);
            co_yield arith->shiftLAdd(s_c, s_a, shift5, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(92), 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result = make_shared_device<int>(24);

            for(int a : TestValues::int32Values)
            {
                for(int b : TestValues::int32Values)
                {
                    for(unsigned int shift : TestValues::shiftValues)
                    {
                        KernelArguments runtimeArgs;
                        runtimeArgs.append("result", d_result.get());
                        runtimeArgs.append("a", a);
                        runtimeArgs.append("b", b);
                        runtimeArgs.append("shift", shift);

                        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                        std::vector<int> result(24);
                        ASSERT_THAT(hipMemcpy(result.data(),
                                              d_result.get(),
                                              result.size() * sizeof(int),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        EXPECT_EQ(result[0], a + b);
                        EXPECT_EQ(result[1], a - b);
                        EXPECT_EQ(result[2], a * b);
                        if(b != 0)
                        {
                            EXPECT_EQ(result[3], a / b);
                            EXPECT_EQ(result[4], a % b);
                        }
                        if(b < 32 && b >= 0)
                        {
                            EXPECT_EQ(result[5], a << b) << a << " " << b;
                            EXPECT_EQ(result[6], static_cast<unsigned int>(a) >> b);
                            EXPECT_EQ(result[7], a >> b);
                        }
                        EXPECT_EQ(result[8], (a > b ? 1 : 0));
                        EXPECT_EQ(result[9], (a >= b ? 1 : 0));
                        EXPECT_EQ(result[10], (a < b ? 1 : 0));
                        EXPECT_EQ(result[11], (a <= b ? 1 : 0));
                        EXPECT_EQ(result[12], (a == b ? 1 : 0));
                        EXPECT_EQ(result[13], a & b);
                        EXPECT_EQ(result[14], (a * (int64_t)b) >> 32);
                        EXPECT_EQ(result[15], -a);
                        EXPECT_EQ(result[16], a ^ b);
                        EXPECT_EQ(result[17], (a + b) << shift);
                        EXPECT_EQ(result[18], (a << shift) + b);
                        if(a >= 2)
                        {
                            EXPECT_EQ(result[19], (a << 1u) + b);
                        }
                        if(a >= 4)
                        {
                            EXPECT_EQ(result[20], (a << 2u) + b);
                        }
                        if(a >= 8)
                        {
                            EXPECT_EQ(result[21], (a << 3u) + b);
                        }
                        if(a >= 16)
                        {
                            EXPECT_EQ(result[22], (a << 4u) + b);
                        }
                        if(a >= 32)
                        {
                            EXPECT_EQ(result[23], (a << 5u) + b);
                        }
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

    TEST_P(ArithmeticTest, ArithInt64)
    {
        auto arith = Component::Get<Arithmetic>(m_context, Register::Type::Vector, DataType::Int64);

        auto k = m_context->kernel();

        k->setKernelName("ArithInt64");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        VariableType Int64Value(DataType::Int64, PointerType::Value);
        VariableType UInt64Value(DataType::UInt64, PointerType::Value);
        VariableType Int64Pointer(DataType::Int64, PointerType::PointerGlobal);

        auto result_exp = command->allocateArgument(Int64Pointer)->expression();
        auto a_exp      = command->allocateArgument(Int64Value)->expression();
        auto b_exp      = command->allocateArgument(Int64Value)->expression();
        auto sh_exp     = command->allocateArgument(UInt64Value)->expression();

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result", Int64Pointer, DataDirection::WriteOnly, result_exp});
        k->addArgument({"a", DataType::Int64, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Int64, DataDirection::ReadOnly, b_exp});
        k->addArgument({"shift", DataType::UInt64, DataDirection::ReadOnly, sh_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b, s_sh;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);
            co_yield m_context->argLoader()->getValue("shift", s_sh);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_shift = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt64, 1);

            auto s_c = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Raw32, 2);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield v_shift->allocate();
            co_yield s_c->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move value");

            co_yield m_context->copier()->copy(v_b, s_b, "Move value");

            co_yield m_context->copier()->copy(v_shift, s_sh, "Move value");

            co_yield generateOp<Expression::Add>(v_c, v_a, v_b);
            co_yield m_context->mem()->storeFlat(v_result, v_c, "", 8);

            co_yield arith->sub(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 8);

            co_yield arith->mul(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(16), 8);

            co_yield arith->div(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(24), 8);

            co_yield arith->mod(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(32), 8);

            co_yield arith->shiftL(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(40), 8);

            co_yield arith->shiftR(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(48), 8);

            co_yield arith->gt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(56), 8);

            co_yield arith->ge(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(64), 8);

            co_yield arith->lt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(72), 8);

            co_yield arith->le(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(80), 8);

            co_yield arith->eq(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(88), 8);

            co_yield arith->bitwiseAnd(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(96), 8);

            co_yield arith->negate(v_c, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(104), 8);

            co_yield arith->bitwiseXor(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(112), 8);

            co_yield arith->addShiftL(v_c, v_a, v_b, v_shift);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(120), 8);

            co_yield arith->shiftLAdd(v_c, v_a, v_shift, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(128), 8);

            co_yield generateOp<Expression::SignedShiftR>(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(136), 8);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            static_assert(sizeof(int64_t) == 8);
            auto long_constant = 1l << 30;
            static_assert(sizeof(long_constant) == 8);

            for(int64_t a : TestValues::int64Values)
            {
                for(int64_t b : TestValues::int64Values)
                {
                    for(uint64_t shift : TestValues::shiftValues)
                    {
                        std::vector<int64_t> result(18);
                        auto                 d_result = make_shared_device<int64_t>(result.size());

                        KernelArguments runtimeArgs;
                        runtimeArgs.append("result", d_result.get());
                        runtimeArgs.append("a", a);
                        runtimeArgs.append("b", b);
                        runtimeArgs.append("shift", shift);

                        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                        ASSERT_THAT(hipMemcpy(result.data(),
                                              d_result.get(),
                                              result.size() * sizeof(int64_t),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        EXPECT_EQ(result[0], a + b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[1], a - b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[2], a * b) << "a: " << a << ", b: " << b;
                        if(b != 0)
                        {
                            EXPECT_EQ(result[3], a / b) << "a: " << a << ", b: " << b;
                            EXPECT_EQ(result[4], a % b) << "a: " << a << ", b: " << b;
                        }
                        if(b < 64 && b >= 0)
                        {
                            EXPECT_EQ(result[5], a << b) << "a: " << a << ", b: " << b;
                            EXPECT_EQ(result[6], a >> b) << "a: " << a << ", b: " << b;
                        }
                        EXPECT_EQ(result[7], (a > b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[8], (a >= b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[9], (a < b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[10], (a <= b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[11], (a == b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[12], a & b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[13], -a) << "a: " << a;
                        EXPECT_EQ(result[14], a ^ b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[15], (a + b) << shift)
                            << "a: " << a << ", b: " << b << ", shift: " << shift;
                        EXPECT_EQ(result[16], (a << shift) + b)
                            << "a: " << a << ", shift: " << shift << ", b: " << b;
                        if(b < 64 && b >= 0)
                        {
                            EXPECT_EQ(result[17], a >> b) << "a: " << a << ", b: " << b;
                        }
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

    TEST_P(ArithmeticTest, ArithInt64Scalar)
    {
        auto arithVector
            = Component::Get<Arithmetic>(m_context, Register::Type::Vector, DataType::Int64);
        auto arith = Component::Get<Arithmetic>(m_context, Register::Type::Scalar, DataType::Int64);

        auto k = m_context->kernel();

        k->setKernelName("ArithInt64Scalar");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int64, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int64, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int64, PointerType::Value}));
        auto sh_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::UInt64, PointerType::Value}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Int64, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", DataType::Int64, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Int64, DataDirection::ReadOnly, b_exp});
        k->addArgument({"shift", DataType::UInt64, DataDirection::ReadOnly, sh_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b, s_shift;

            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);
            co_yield m_context->argLoader()->getValue("shift", s_shift);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto s_c = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Raw32, 2);

            co_yield v_c->allocate();
            co_yield s_c->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield generateOp<Expression::Add>(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_result, v_c, "", 8);

            co_yield arith->sub(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 8);

            co_yield arith->mul(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(16), 8);

            co_yield arith->div(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(24), 8);

            co_yield arith->mod(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(32), 8);

            co_yield arith->shiftL(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(40), 8);

            co_yield arith->shiftR(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(48), 8);

            co_yield arith->gt(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(56), 8);

            co_yield arith->ge(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(64), 8);

            co_yield arith->lt(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(72), 8);

            co_yield arith->le(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(80), 8);

            co_yield arith->eq(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(88), 8);

            co_yield arith->bitwiseAnd(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(96), 8);

            co_yield arith->negate(s_c, s_a);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(104), 8);

            co_yield arith->bitwiseXor(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(112), 8);

            co_yield arith->addShiftL(s_c, s_a, s_b, s_shift);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(120), 8);

            co_yield arith->shiftLAdd(s_c, s_a, s_shift, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(128), 8);

            co_yield generateOp<Expression::SignedShiftR>(s_c, s_a, s_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(136), 8);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result = make_shared_device<int64_t>(18);
            static_assert(sizeof(int64_t) == 8);

            for(int64_t a : TestValues::int64Values)
            {
                for(int64_t b : TestValues::int64Values)
                {
                    for(uint64_t shift : TestValues::shiftValues)
                    {
                        KernelArguments runtimeArgs;
                        runtimeArgs.append("result", d_result.get());
                        runtimeArgs.append("a", a);
                        runtimeArgs.append("b", b);
                        runtimeArgs.append("shift", shift);

                        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                        std::vector<int64_t> result(18);
                        ASSERT_THAT(hipMemcpy(result.data(),
                                              d_result.get(),
                                              result.size() * sizeof(int64_t),
                                              hipMemcpyDefault),
                                    HasHipSuccess(0));

                        EXPECT_EQ(result[0], a + b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[1], a - b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[2], a * b) << "a: " << a << ", b: " << b;
                        if(b != 0)
                        {
                            EXPECT_EQ(result[3], a / b) << "a: " << a << ", b: " << b;
                            EXPECT_EQ(result[4], a % b) << "a: " << a << ", b: " << b;
                        }
                        if(b < 64 && b >= 0)
                        {
                            EXPECT_EQ(result[5], a << b) << "a: " << a << ", b: " << b;
                            EXPECT_EQ(result[6], a >> b) << "a: " << a << ", b: " << b;
                        }
                        EXPECT_EQ(result[7], (a > b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[8], (a >= b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[9], (a < b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[10], (a <= b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[11], (a == b ? 1 : 0)) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[12], a & b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[13], -a) << "a: " << a;
                        EXPECT_EQ(result[14], a ^ b) << "a: " << a << ", b: " << b;
                        EXPECT_EQ(result[15], (a + b) << shift)
                            << "a: " << a << ", b: " << b << ", shift: " << shift;
                        EXPECT_EQ(result[16], (a << shift) + b)
                            << "a: " << a << ", shift: " << shift << ", b: " << b;
                        if(b < 64 && b >= 0)
                        {
                            EXPECT_EQ(result[17], a >> b) << "a: " << a << ", b: " << b;
                        }
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

    TEST_P(ArithmeticTest, ArithFloat)
    {
        auto arith = Component::Get<Arithmetic>(m_context, Register::Type::Vector, DataType::Float);

        auto k = m_context->kernel();

        k->setKernelName("ArithFloat");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::PointerGlobal}));
        auto cond_result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Float, PointerType::Value}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"cond_result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        cond_result_exp});
        k->addArgument({"a", DataType::Float, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Float, DataDirection::ReadOnly, b_exp});

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

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_cond_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto s_c = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Raw32, 2);

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
            co_yield m_context->mem()->storeFlat(v_result, v_c, "", 4);

            co_yield arith->sub(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(4), 4);

            co_yield arith->mul(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 4);

            co_yield arith->negate(v_c, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(12), 4);

            co_yield arith->gt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_c->subset({0}), "", 4);

            co_yield arith->ge(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_c->subset({0}), "4", 4);

            co_yield arith->lt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_c->subset({0}), "8", 4);

            co_yield arith->le(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_c->subset({0}), "12", 4);

            co_yield arith->eq(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(
                v_c, s_c->subset({0}), "Move result to vgpr to store.");
            co_yield m_context->mem()->storeFlat(v_cond_result, v_c->subset({0}), "16", 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result      = make_shared_device<float>(4);
            auto d_cond_result = make_shared_device<int>(5);

            for(float a : TestValues::floatValues)
            {
                for(float b : TestValues::floatValues)
                {

                    KernelArguments runtimeArgs;
                    runtimeArgs.append("result", d_result.get());
                    runtimeArgs.append("cond_result", d_cond_result.get());
                    runtimeArgs.append("a", a);
                    runtimeArgs.append("b", b);

                    commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                    std::vector<float> result(4);
                    ASSERT_THAT(hipMemcpy(result.data(),
                                          d_result.get(),
                                          result.size() * sizeof(float),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    std::vector<int> cond_result(5);
                    ASSERT_THAT(hipMemcpy(cond_result.data(),
                                          d_cond_result.get(),
                                          cond_result.size() * sizeof(int),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    EXPECT_EQ(result[0], a + b);
                    EXPECT_EQ(result[1], a - b);
                    EXPECT_EQ(result[2], a * b);
                    EXPECT_EQ(result[3], -a);
                    EXPECT_EQ(cond_result[0], (a > b ? 1 : 0));
                    EXPECT_EQ(cond_result[1], (a >= b ? 1 : 0));
                    EXPECT_EQ(cond_result[2], (a < b ? 1 : 0));
                    EXPECT_EQ(cond_result[3], (a <= b ? 1 : 0));
                    EXPECT_EQ(cond_result[4], (a == b ? 1 : 0));
                }
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(ArithmeticTest, ArithDouble)
    {
        auto arith
            = Component::Get<Arithmetic>(m_context, Register::Type::Vector, DataType::Double);

        auto k = m_context->kernel();

        k->setKernelName("ArithDouble");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Double, PointerType::PointerGlobal}));
        auto cond_result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Double, PointerType::Value}));
        auto b_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Double, PointerType::Value}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Double, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"cond_result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        cond_result_exp});
        k->addArgument({"a", DataType::Double, DataDirection::ReadOnly, a_exp});
        k->addArgument({"b", DataType::Double, DataDirection::ReadOnly, b_exp});

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

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_cond_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Double, 1);

            auto s_c = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Raw32, 2);

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
            co_yield m_context->mem()->storeFlat(v_result, v_c, "", 8);

            co_yield arith->sub(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(8), 8);

            co_yield arith->mul(v_c, v_a, v_b);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(16), 8);

            co_yield arith->negate(v_c, v_a);
            co_yield m_context->mem()->store(
                MemoryInstructions::Flat, v_result, v_c, Register::Value::Literal(24), 8);

            co_yield arith->gt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(0),
                                             4);

            co_yield arith->ge(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");

            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(4),
                                             4);

            co_yield arith->lt(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(8),
                                             4);

            co_yield arith->le(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(12),
                                             4);

            co_yield arith->eq(s_c, v_a, v_b);
            co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr to store.");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_cond_result,
                                             v_c->subset({0}),
                                             Register::Value::Literal(16),
                                             4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU arithmetic tests for " << GetParam();
        }

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            CommandKernel commandKernel(m_context);

            auto d_result      = make_shared_device<double>(4);
            auto d_cond_result = make_shared_device<int>(5);

            for(double a : TestValues::doubleValues)
            {
                for(double b : TestValues::doubleValues)
                {
                    KernelArguments runtimeArgs;
                    runtimeArgs.append("result", d_result.get());
                    runtimeArgs.append("cond_result", d_cond_result.get());
                    runtimeArgs.append("a", a);
                    runtimeArgs.append("b", b);

                    commandKernel.launchKernel(runtimeArgs.runtimeArguments());

                    std::vector<double> result(4);
                    ASSERT_THAT(hipMemcpy(result.data(),
                                          d_result.get(),
                                          result.size() * sizeof(double),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    std::vector<int> cond_result(5);
                    ASSERT_THAT(hipMemcpy(cond_result.data(),
                                          d_cond_result.get(),
                                          cond_result.size() * sizeof(int),
                                          hipMemcpyDefault),
                                HasHipSuccess(0));

                    EXPECT_EQ(result[0], a + b);
                    EXPECT_EQ(result[1], a - b);
                    EXPECT_EQ(result[2], a * b);
                    EXPECT_EQ(result[3], -a);
                    EXPECT_EQ(cond_result[0], (a > b ? 1 : 0));
                    EXPECT_EQ(cond_result[1], (a >= b ? 1 : 0));
                    EXPECT_EQ(cond_result[2], (a < b ? 1 : 0));
                    EXPECT_EQ(cond_result[3], (a <= b ? 1 : 0));
                    EXPECT_EQ(cond_result[4], (a == b ? 1 : 0));
                }
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        ArithmeticTests,
        ArithmeticTest,
        ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getAllSupportedISAs()));

}
