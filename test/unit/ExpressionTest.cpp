
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "CodeGen/Instruction.hpp"
#include "Expression_fwd.hpp"
#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TestValues.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace ExpressionTest
{

    struct ExpressionTest : public GenericContextFixture
    {
        std::string targetArchitecture() override
        {
            // MFMA, 64 lanes per wavefront
            return "gfx90a";
        }

        void SetUp() override
        {
            m_kernelOptions.preloadKernelArguments = false;
            GenericContextFixture::SetUp();
        }

        void testSerialization(Expression::ExpressionPtr expr);
    };

    TEST_F(ExpressionTest, Basic)
    {
        auto a = Expression::literal(1);
        auto b = Expression::literal(2);

        auto rc = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto expr1  = a + b;
        auto expr2  = b * expr1;
        auto expr3  = b * expr1 - rc->expression();
        auto expr4  = expr1 > expr2;
        auto expr5  = expr3 < expr4;
        auto expr6  = expr4 >= expr5;
        auto expr7  = expr5 <= expr6;
        auto expr8  = expr6 == expr7;
        auto expr9  = -expr2;
        auto expr10 = Expression::fuseTernary(expr1 << b);
        auto expr11 = Expression::fuseTernary((a << b) + b);
        auto expr12 = expr6 != expr7;

        auto sexpr1  = Expression::toString(expr1);
        auto sexpr2  = Expression::toString(expr2);
        auto sexpr3  = Expression::toString(expr3);
        auto sexpr4  = Expression::toString(expr4);
        auto sexpr5  = Expression::toString(expr5);
        auto sexpr6  = Expression::toString(expr6);
        auto sexpr7  = Expression::toString(expr7);
        auto sexpr8  = Expression::toString(expr8);
        auto sexpr9  = Expression::toString(expr9);
        auto sexpr10 = Expression::toString(expr10);
        auto sexpr11 = Expression::toString(expr11);
        auto sexpr12 = Expression::toString(expr12);

        EXPECT_EQ(sexpr1, "Add(1i, 2i)");
        EXPECT_EQ(sexpr2, "Multiply(2i, Add(1i, 2i))");
        EXPECT_EQ(sexpr3, "Subtract(Multiply(2i, Add(1i, 2i)), v0:I)");
        EXPECT_EQ(sexpr4, "GreaterThan(" + sexpr1 + ", " + sexpr2 + ")");
        EXPECT_EQ(sexpr5, "LessThan(" + sexpr3 + ", " + sexpr4 + ")");
        EXPECT_EQ(sexpr6, "GreaterThanEqual(" + sexpr4 + ", " + sexpr5 + ")");
        EXPECT_EQ(sexpr7, "LessThanEqual(" + sexpr5 + ", " + sexpr6 + ")");
        EXPECT_EQ(sexpr8, "Equal(" + sexpr6 + ", " + sexpr7 + ")");
        EXPECT_EQ(sexpr9, "Negate(" + sexpr2 + ")");
        EXPECT_EQ(sexpr10, "AddShiftL(1i, 2i, 2i)");
        EXPECT_EQ(sexpr11, "ShiftLAdd(1i, 2i, 2i)");
        EXPECT_EQ(sexpr12, "NotEqual(" + sexpr6 + ", " + sexpr7 + ")");

        Expression::EvaluationTimes expectedTimes{Expression::EvaluationTime::KernelExecute};
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr8));
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr10));
    }

    void ExpressionTest::testSerialization(Expression::ExpressionPtr expr)
    {
        auto yamlText = Expression::toYAML(expr);

        EXPECT_NE("", yamlText) << toString(expr);

        auto deserialized = Expression::fromYAML(yamlText);
        ASSERT_NE(nullptr, deserialized.get()) << yamlText << toString(expr);

        EXPECT_EQ(Expression::toString(deserialized), Expression::toString(expr));
        EXPECT_EQ(true, Expression::identical(deserialized, expr));
    }

    TEST_F(ExpressionTest, Serialization)
    {
        auto a = Expression::literal(1);
        auto b = Expression::literal(2);

        auto c = Register::Value::Literal(4.2f);
        auto d = Register::Value::Literal(Half(4.2f));

        auto kernelArg                   = std::make_shared<AssemblyKernelArgument>();
        kernelArg->name                  = "KernelArg1";
        kernelArg->variableType.dataType = DataType::Int32;
        kernelArg->expression            = Expression::literal(10);
        kernelArg->offset                = 1;
        kernelArg->size                  = 5;

        Expression::DataFlowTag dataFlow;
        dataFlow.tag              = 50;
        dataFlow.regType          = Register::Type::Vector;
        dataFlow.varType.dataType = DataType::Float;

        auto waveTile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        auto expr1  = a + b;
        auto expr2  = b * expr1;
        auto expr3  = b * expr1 - c->expression();
        auto expr4  = expr1 > (expr2 + d->expression());
        auto expr5  = expr3 < expr4;
        auto expr6  = expr4 >= expr5;
        auto expr7  = expr5 <= expr6;
        auto expr8  = expr6 == expr7;
        auto expr9  = -expr2;
        auto expr10 = Expression::fuseTernary(expr1 << b);
        auto expr11 = Expression::fuseTernary((a << b) + b);
        auto expr12 = b >> std::make_shared<Expression::Expression>(kernelArg);
        auto expr13 = std::make_shared<Expression::Expression>(dataFlow) / a;
        auto expr14 = std::make_shared<Expression::Expression>(waveTile) + b;

        testSerialization(expr1);
        testSerialization(expr2);
        testSerialization(expr3);
        testSerialization(expr4);
        testSerialization(expr5);
        testSerialization(expr6);
        testSerialization(expr7);
        testSerialization(expr8);
        testSerialization(expr9);
        testSerialization(expr10);
        testSerialization(expr11);
        // TODO: Enable test when KernelArgumentPtr can be serialized
        //testSerialization(expr12);
        testSerialization(expr13);
        // TODO: Enable test when WaveTilePtr can be serialized
        //testSerialization(expr14);

        auto reg = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        reg->allocateNow();

        EXPECT_ANY_THROW(testSerialization(reg->expression()));
    }

    TEST_F(ExpressionTest, IdenticalTest)
    {
        auto a    = Expression::literal(1u);
        auto ap   = Expression::literal(1);
        auto b    = Expression::literal(2u);
        auto zero = Expression::literal(0u);

        auto rc = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto rd = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        rd->allocateNow();

        auto c = rc->expression();
        auto d = rd->expression();

        auto cve = std::make_shared<CommandArgument>(nullptr, DataType::Float, 0);
        auto cvf = std::make_shared<CommandArgument>(nullptr, DataType::Float, 8);

        auto e = std::make_shared<Expression::Expression>(cve);
        auto f = std::make_shared<Expression::Expression>(cvf);

        auto expr1 = a + b;
        auto expr2 = a + b;

        auto expr3 = a - b;

        EXPECT_TRUE(identical(expr1, expr2));
        EXPECT_FALSE(identical(expr1, expr3));
        EXPECT_FALSE(identical(ap + b, expr3));

        EXPECT_TRUE(equivalent(expr1, expr2));
        EXPECT_FALSE(equivalent(expr1, expr3));
        EXPECT_FALSE(equivalent(ap + b, expr3));

        auto expr4 = c + d;
        auto expr5 = c + d + zero;

        EXPECT_FALSE(identical(expr1, expr4));
        EXPECT_FALSE(identical(expr4, expr5));
        EXPECT_TRUE(identical(expr4, simplify(expr5)));

        EXPECT_FALSE(equivalent(expr1, expr4));
        EXPECT_FALSE(equivalent(expr4, expr5));
        EXPECT_TRUE(equivalent(expr4, simplify(expr5)));

        auto expr6 = e / f % d;
        auto expr7 = a + f;

        EXPECT_FALSE(identical(expr6, expr7));
        EXPECT_FALSE(identical(e, f));

        EXPECT_TRUE(Expression::identical(nullptr, nullptr));
        EXPECT_FALSE(identical(nullptr, a));
        EXPECT_FALSE(identical(a, nullptr));

        EXPECT_FALSE(equivalent(expr6, expr7));
        EXPECT_FALSE(equivalent(e, f));

        EXPECT_TRUE(Expression::equivalent(nullptr, nullptr));
        EXPECT_FALSE(equivalent(nullptr, a));
        EXPECT_FALSE(equivalent(a, nullptr));

        // Commutative tests
        EXPECT_FALSE(identical(a + b, b + a));
        EXPECT_FALSE(identical(a - b, b - a));

        EXPECT_TRUE(equivalent(a + b, b + a));
        EXPECT_FALSE(equivalent(a - b, b - a));
        EXPECT_TRUE(equivalent(a * b, b * a));
        EXPECT_FALSE(equivalent(a / b, b / a));
        EXPECT_FALSE(equivalent(a % b, b % a));
        EXPECT_FALSE(equivalent(a << b, b << a));
        EXPECT_FALSE(equivalent(a >> b, b >> a));
        EXPECT_TRUE(equivalent(a & b, b & a));
        EXPECT_TRUE(equivalent(a | b, b | a));
        EXPECT_TRUE(equivalent(a ^ b, b ^ a));

        // Unallocated
        auto rg = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        // Unallocated
        auto rh = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        EXPECT_TRUE(Expression::identical(rg->expression(), rg->expression()));
        EXPECT_FALSE(Expression::identical(rg->expression(), rh->expression()));

        EXPECT_TRUE(Expression::equivalent(rg->expression(), rg->expression()));
        EXPECT_FALSE(Expression::equivalent(rg->expression(), rh->expression()));

        // Null
        Expression::ExpressionPtr n = nullptr;
        EXPECT_FALSE(Expression::equivalent(n + n, a + n));
        EXPECT_FALSE(Expression::equivalent(n + n, n + a));
    }

    TEST_F(ExpressionTest, BasicInstructions)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        Register::ValuePtr dest;
        m_context->schedule(Expression::generate(dest, expr2, m_context));

        // Explicitly copy the result into another register.
        auto dest2 = dest->placeholder();
        dest2->allocateNow();
        auto regIndexBefore = Generated(dest2->registerIndices())[0];

        m_context->schedule(Expression::generate(dest2, dest->expression(), m_context));
        auto regIndexAfter = Generated(dest2->registerIndices())[0];
        EXPECT_EQ(regIndexBefore, regIndexAfter);

        m_context->schedule(Expression::generate(dest2, expr2, m_context));
        regIndexAfter = Generated(dest2->registerIndices())[0];
        EXPECT_EQ(regIndexBefore, regIndexAfter);

        std::string expected = R"(
            v_add_i32 v2, v0, v1
            v_mul_lo_u32 v3, v1, v2

            // Note that v2 is reused
            v_mov_b32 v2, v3

            // Still storing into v2
            v_add_i32 v4, v0, v1
            v_mul_lo_u32 v2, v1, v4
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_F(ExpressionTest, BasicFMA)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto rc = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rc->setName("rc");
        rc->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();
        auto c = rc->expression();

        auto expr1 = multiplyAdd(a, b, c);

        auto raf = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        raf->allocateNow();

        auto rbf = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        rbf->allocateNow();

        auto rcf = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        rcf->allocateNow();

        auto af = raf->expression();
        auto bf = rbf->expression();
        auto cf = rcf->expression();

        auto expr2 = multiplyAdd(af, bf, cf);

        Register::ValuePtr dest1, dest2;
        m_context->schedule(Expression::generate(dest1, expr1, m_context));
        m_context->schedule(Expression::generate(dest2, expr2, m_context));

        std::string expected = R"(
            // Int32: a * x + y doesn't have FMA, so should see multiply then add
            v_mul_lo_u32 v6, v0, v1
            v_add_i32 v6, v6, v2

            // Float: a * x + y has FMA
            v_fma_f32 v7, v3, v4, v5
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_F(ExpressionTest, ExpressionCommentsBasic)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        setComment(expr1, "The Addition");
        appendComment(expr1, " extra comment");
        setComment(expr2, "The Multiplication");

        EXPECT_EQ("The Addition extra comment", getComment(expr1));

        auto expr3 = simplify(expr2);
        ASSERT_EQ("The Multiplication", getComment(expr3));

        Register::ValuePtr dest;
        m_context->schedule(Expression::generate(dest, expr2, m_context));

        std::string expected = R"(
            // Generate {The Multiplication: Multiply(v1:I, {The Addition extra comment: Add(v0:I, v1:I)})} into nullptr
            // BEGIN: The Addition extra comment
            // {The Addition extra comment: Add(v0:I, v1:I)}
            // Allocated : 1 VGPR (Value: Int32): v2
            v_add_i32 v2, v0, v1
            // END: The Addition extra comment
            // BEGIN: The Multiplication
            // {The Multiplication: Multiply(v1:I, v2:I)}
            // Allocated : 1 VGPR (Value: Int32): v3
            v_mul_lo_u32 v3, v1, v2
            // END: The Multiplication
            // Freeing : 1 VGPR (Value: Int32): v2
        )";

        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    TEST_F(ExpressionTest, ExpressionCommentsErrors)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto a = ra->expression();
        EXPECT_THROW(setComment(a, "The a input"), FatalError);
        EXPECT_THROW(appendComment(a, "extra comment"), FatalError);
        EXPECT_EQ(getComment(a), "ra");

        Expression::ExpressionPtr expr1;
        EXPECT_THROW(setComment(expr1, "The first expression"), FatalError);
        EXPECT_THROW(appendComment(expr1, "extra"), FatalError);
        EXPECT_EQ(getComment(expr1), "");
    }

    TEST_F(ExpressionTest, GenerateInvalid)
    {
        Register::ValuePtr result;

        m_context->schedule(m_context->kernel()->preamble());
        m_context->schedule(m_context->kernel()->prolog());

        auto reg = m_context->kernel()->workitemIndex()[0]->expression();

        {
            auto exp = Expression::magicMultiple(reg);
            EXPECT_ANY_THROW(m_context->schedule(Expression::generate(result, exp, m_context)))
                << output();
        }

        EXPECT_ANY_THROW(m_context->schedule(
            Expression::generate(result, Expression::magicShifts(reg), m_context)))
            << output();
        EXPECT_ANY_THROW(m_context->schedule(
            Expression::generate(result, Expression::magicSign(reg), m_context)))
            << output();

        CommandArgumentPtr arg;
        auto               argExp = std::make_shared<Expression::Expression>(arg);
        EXPECT_ANY_THROW(m_context->schedule(Expression::generate(result, argExp, m_context)));

        Register::ValuePtr nullResult;
        auto               unallocResult
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int32, 1);
        auto allocResult
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int32, 1);
        allocResult->allocateNow();

        for(auto result : {nullResult, unallocResult, allocResult})
        {
            auto unallocated = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);

            EXPECT_ANY_THROW(m_context->schedule(
                Expression::generate(result, unallocated->expression(), m_context)));
            ASSERT_EQ(unallocated->allocationState(), Register::AllocationState::Unallocated);

            EXPECT_ANY_THROW(m_context->schedule(Expression::generate(
                result, unallocated->expression() + Expression::literal(5), m_context)));
            ASSERT_EQ(unallocated->allocationState(), Register::AllocationState::Unallocated);

            EXPECT_ANY_THROW(m_context->schedule(Expression::generate(
                result,
                Expression::multiplyHigh(unallocated->expression(), Expression::literal(5)),
                m_context)));
            ASSERT_EQ(unallocated->allocationState(), Register::AllocationState::Unallocated);

            EXPECT_ANY_THROW(m_context->schedule(
                Expression::generate(unallocated, unallocated->expression(), m_context)));
            ASSERT_EQ(unallocated->allocationState(), Register::AllocationState::Unallocated);
        }
    }

    TEST_F(ExpressionTest, MatrixMultiplyWaveTiles)
    {
        int M       = 32;
        int N       = 32;
        int K       = 2;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * K / 64,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                K * N / 64,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto ic = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Accumulator,
                                                    DataType::Float,
                                                    M * N * batches / 64,
                                                    Register::AllocationOptions::FullyContiguous());
        ic->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);
        auto C = ic->expression();

        auto expr = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, C));

        m_context->schedule(
            Expression::generate(ic, expr, m_context)); //Test using input C as dest.

        Register::ValuePtr rc;
        m_context->schedule(
            Expression::generate(rc, expr, m_context)); //Test using a nullptr as dest.

        EXPECT_EQ(ic->regType(), Register::Type::Accumulator);
        EXPECT_EQ(ic->valueCount(), 16);

        EXPECT_EQ(rc->regType(), Register::Type::Accumulator);
        EXPECT_EQ(rc->valueCount(), 16);

        auto result = R"(
            v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[0:15] //is matmul
            v_mfma_f32_32x32x2f32 a[16:31], v0, v1, a[0:15] //rc matmul
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, ReuseInputVGPRsAsOutputVGPRsInArithmetic)
    {
        int M       = 16;
        int N       = 16;
        int K       = 4;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * K / 64,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                K * N / 64,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto accumD
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        accumD->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);
        auto D = accumD->expression();

        auto mulABExpr
            = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, D));

        m_context->schedule(
            Expression::generate(accumD, mulABExpr, m_context)); //Test using input D as dest.

        EXPECT_EQ(accumD->regType(), Register::Type::Accumulator);
        EXPECT_EQ(accumD->valueCount(), 4);

        auto vecD
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        m_context->schedule(Expression::generate(vecD, D, m_context));

        auto scaleDExpr = Expression::literal(2.0f) * vecD->expression();
        m_context->schedule(Expression::generate(vecD, scaleDExpr, m_context));

        auto vecC
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        vecC->allocateNow();

        auto addCDExpr = vecC->expression() + vecD->expression();
        m_context->schedule(Expression::generate(vecD, addCDExpr, m_context));

        auto result = R"(
            v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[0:3]

            s_nop 10
            v_accvgpr_read v2, a0
            v_accvgpr_read v3, a1
            v_accvgpr_read v4, a2
            v_accvgpr_read v5, a3

            v_mul_f32 v2, 2.00000, v2
            v_mul_f32 v3, 2.00000, v3
            v_mul_f32 v4, 2.00000, v4
            v_mul_f32 v5, 2.00000, v5

            v_add_f32 v2, v6, v2
            v_add_f32 v3, v7, v3
            v_add_f32 v4, v8, v4
            v_add_f32 v5, v9, v5
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, ReuseInputVGPRsAsOutputVGPRsInArithmeticF16)
    {
        int M       = 32;
        int N       = 32;
        int K       = 8;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Halfx2,
                                                M * K / 64 / 2,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Halfx2,
                                                K * N / 64 / 2,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto accumD
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        accumD->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);
        auto D = accumD->expression();

        auto mulABExpr
            = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, D));

        m_context->schedule(
            Expression::generate(accumD, mulABExpr, m_context)); //Test using input D as dest.

        EXPECT_EQ(accumD->regType(), Register::Type::Accumulator);
        EXPECT_EQ(accumD->valueCount(), 16);

        auto vecD
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());

        auto vecC
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Half,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        vecC->allocateNow();

        auto scaleDExpr = Expression::literal(2.0, DataType::Half) * vecD->expression();
        auto addCDExpr  = vecC->expression() + vecD->expression();

        m_context->schedule(Expression::generate(vecD, D, m_context));
        m_context->schedule(Expression::generate(vecD, scaleDExpr, m_context));
        m_context->schedule(Expression::generate(vecD, addCDExpr, m_context));

        auto X = std::make_shared<Register::Value>(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Halfx2,
                                                   M * K / 64 / 2,
                                                   Register::AllocationOptions::FullyContiguous());
        X->allocateNow();

        auto Y = std::make_shared<Register::Value>(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Half,
                                                   M * K / 64,
                                                   Register::AllocationOptions::FullyContiguous());
        Y->allocateNow();

        auto addXYExpr = X->expression() + Y->expression();
        m_context->schedule(Expression::generate(Y, addXYExpr, m_context));

        // TODO If operand being converted is a literal, do one conversion only.
        auto result = R"(
            // A is in v[0:1], B is in v[2:3], C is in v[4:19], D is in a[0:15]

            // Result R will end up in v[20:35].  Steps are:
            // R <- D
            // R <- alpha * R
            // R <- R + C

            v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]

            s_nop 0xf
            s_nop 2
            v_accvgpr_read v20, a0
            v_accvgpr_read v21, a1
            v_accvgpr_read v22, a2
            v_accvgpr_read v23, a3
            v_accvgpr_read v24, a4
            v_accvgpr_read v25, a5
            v_accvgpr_read v26, a6
            v_accvgpr_read v27, a7
            v_accvgpr_read v28, a8
            v_accvgpr_read v29, a9
            v_accvgpr_read v30, a10
            v_accvgpr_read v31, a11
            v_accvgpr_read v32, a12
            v_accvgpr_read v33, a13
            v_accvgpr_read v34, a14
            v_accvgpr_read v35, a15

            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v20, v36, v20
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v21, v36, v21
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v22, v36, v22
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v23, v36, v23
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v24, v36, v24
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v25, v36, v25
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v26, v36, v26
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v27, v36, v27
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v28, v36, v28
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v29, v36, v29
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v30, v36, v30
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v31, v36, v31
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v32, v36, v32
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v33, v36, v33
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v34, v36, v34
            v_cvt_f32_f16 v36, 2.00000
            v_mul_f32 v35, v36, v35

            v_cvt_f32_f16 v36, v4
            v_add_f32 v20, v36, v20
            v_cvt_f32_f16 v36, v5
            v_add_f32 v21, v36, v21
            v_cvt_f32_f16 v36, v6
            v_add_f32 v22, v36, v22
            v_cvt_f32_f16 v36, v7
            v_add_f32 v23, v36, v23
            v_cvt_f32_f16 v36, v8
            v_add_f32 v24, v36, v24
            v_cvt_f32_f16 v36, v9
            v_add_f32 v25, v36, v25
            v_cvt_f32_f16 v36, v10
            v_add_f32 v26, v36, v26
            v_cvt_f32_f16 v36, v11
            v_add_f32 v27, v36, v27
            v_cvt_f32_f16 v36, v12
            v_add_f32 v28, v36, v28
            v_cvt_f32_f16 v36, v13
            v_add_f32 v29, v36, v29
            v_cvt_f32_f16 v36, v14
            v_add_f32 v30, v36, v30
            v_cvt_f32_f16 v36, v15
            v_add_f32 v31, v36, v31
            v_cvt_f32_f16 v36, v16
            v_add_f32 v32, v36, v32
            v_cvt_f32_f16 v36, v17
            v_add_f32 v33, v36, v33
            v_cvt_f32_f16 v36, v18
            v_add_f32 v34, v36, v34
            v_cvt_f32_f16 v36, v19
            v_add_f32 v35, v36, v35

            // X is v[36:37]:2xH and Y is v[38:41]:H (and Z is same as Y)
            // Then Y <- X + Y will be: Add(v[36:37]:2xH, v[38:41]:H)
            v_mov_b32 v42, 65535
            v_and_b32 v43, v42, v36
            v_lshrrev_b32 v44, 16, v36
            v_add_f16 v38, v43, v38
            v_add_f16 v39, v44, v39
            v_mov_b32 v42, 65535
            v_and_b32 v43, v42, v37
            v_lshrrev_b32 v44, 16, v37
            v_add_f16 v40, v43, v40
            v_add_f16 v41, v44, v41
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, ReuseInputVGPRsAsOutputVGPRsInArithmeticF16SmallerPacking)
    {
        int M = 32;
        int K = 8;

        auto X = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Halfx2, M * K / 64 / 2);
        X->allocateNow();

        auto Y = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Half, M * K / 64);
        Y->allocateNow();

        // Since we are asking the result to be stored into X, we
        // currently get a failure.

        // TODO See the "Destination/result packing mismatch" assertion
        // in Expression_generate.cpp.
        auto addXYExpr = X->expression() + Y->expression();
        EXPECT_THROW(m_context->schedule(Expression::generate(X, addXYExpr, m_context)),
                     FatalError);

        // The above should be possible: Y should be packed, and then
        // the v_pk_add_f16 instructions called.
    }

    TEST_F(ExpressionTest, ResultType)
    {
        auto vgprFloat
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 1)
                  ->expression();
        auto vgprDouble
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Double, 1)
                  ->expression();
        auto vgprInt32
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1)
                  ->expression();
        auto vgprInt64
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int64, 1)
                  ->expression();
        auto vgprUInt32
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt32, 1)
                  ->expression();
        auto vgprUInt64
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt64, 1)
                  ->expression();
        auto vgprHalf
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 1)
                  ->expression();
        auto vgprHalfx2
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Halfx2, 1)
                  ->expression();
        auto vgprBool32
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Bool32, 1)
                  ->expression();
        auto vgprBool
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Bool, 1)
                  ->expression();

        auto sgprFloat
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Float, 1)
                  ->expression();
        auto sgprDouble
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Double, 1)
                  ->expression();
        auto sgprInt32
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int32, 1)
                  ->expression();
        auto sgprInt64
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int64, 1)
                  ->expression();
        auto sgprUInt32
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::UInt32, 1)
                  ->expression();
        auto sgprUInt64
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::UInt64, 1)
                  ->expression();
        auto sgprHalf
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Half, 1)
                  ->expression();
        auto sgprHalfx2
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Halfx2, 1)
                  ->expression();
        auto sgprBool64
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Bool64, 1)
                  ->expression();
        auto sgprBool32
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Bool32, 1)
                  ->expression();
        auto sgprBool
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Bool, 1)
                  ->expression();
        auto sgprWavefrontSized
            = Register::Value::Placeholder(
                  m_context,
                  Register::Type::Scalar,
                  m_context->kernel()->wavefront_size() == 64 ? DataType::Bool64 : DataType::Bool32,
                  1)
                  ->expression();

        auto agprFloat = Register::Value::Placeholder(
                             m_context, Register::Type::Accumulator, DataType::Float, 1)
                             ->expression();
        auto agprDouble = Register::Value::Placeholder(
                              m_context, Register::Type::Accumulator, DataType::Double, 1)
                              ->expression();

        auto litInt32  = Expression::literal<int32_t>(5);
        auto litInt64  = Expression::literal<int64_t>(5);
        auto litFloat  = Expression::literal(5.0f);
        auto litDouble = Expression::literal(5.0);

        Expression::ResultType rVgprFloat{Register::Type::Vector, DataType::Float};
        Expression::ResultType rVgprDouble{Register::Type::Vector, DataType::Double};
        Expression::ResultType rVgprInt32{Register::Type::Vector, DataType::Int32};
        Expression::ResultType rVgprInt64{Register::Type::Vector, DataType::Int64};
        Expression::ResultType rVgprUInt32{Register::Type::Vector, DataType::UInt32};
        Expression::ResultType rVgprUInt64{Register::Type::Vector, DataType::UInt64};
        Expression::ResultType rVgprHalf{Register::Type::Vector, DataType::Half};
        Expression::ResultType rVgprHalfx2{Register::Type::Vector, DataType::Halfx2};
        Expression::ResultType rVgprBool32{Register::Type::Vector, DataType::Bool32};

        Expression::ResultType rSgprFloat{Register::Type::Scalar, DataType::Float};
        Expression::ResultType rSgprDouble{Register::Type::Scalar, DataType::Double};
        Expression::ResultType rSgprInt32{Register::Type::Scalar, DataType::Int32};
        Expression::ResultType rSgprInt64{Register::Type::Scalar, DataType::Int64};
        Expression::ResultType rSgprUInt32{Register::Type::Scalar, DataType::UInt32};
        Expression::ResultType rSgprUInt64{Register::Type::Scalar, DataType::UInt64};
        Expression::ResultType rSgprHalf{Register::Type::Scalar, DataType::Half};
        Expression::ResultType rSgprHalfx2{Register::Type::Scalar, DataType::Halfx2};
        Expression::ResultType rSgprBool32{Register::Type::Scalar, DataType::Bool32};
        Expression::ResultType rSgprBool64{Register::Type::Scalar, DataType::Bool64};
        Expression::ResultType rSgprBool{Register::Type::Scalar, DataType::Bool};
        Expression::ResultType rSgprWavefrontSized{
            Register::Type::Scalar,
            m_context->kernel()->wavefront_size() == 64 ? DataType::Bool64 : DataType::Bool32};

        Expression::ResultType rVCC{Register::Type::VCC, DataType::Bool32};
        Expression::ResultType rSCC{Register::Type::SCC, DataType::Bool};

        Expression::ResultType rAgprFloat{Register::Type::Accumulator, DataType::Float};
        Expression::ResultType rAgprDouble{Register::Type::Accumulator, DataType::Double};

        EXPECT_EQ(rSgprInt64, resultType(sgprInt64));

        EXPECT_EQ(rVgprInt32, resultType(vgprInt32));
        EXPECT_EQ(rVgprInt64, resultType(vgprInt64));
        EXPECT_EQ(rVgprFloat, resultType(vgprFloat));
        EXPECT_EQ(rSgprFloat, resultType(sgprFloat));
        EXPECT_EQ(rVgprDouble, resultType(vgprDouble));
        EXPECT_EQ(rSgprDouble, resultType(sgprDouble));
        EXPECT_EQ(rAgprDouble, resultType(agprDouble));
        EXPECT_EQ(rAgprDouble, resultType(agprDouble));

        EXPECT_EQ(rVgprInt32, resultType(vgprInt32 + vgprInt32));
        EXPECT_EQ(rVgprInt32, resultType(vgprInt32 + sgprInt32));
        EXPECT_EQ(rVgprInt32, resultType(sgprInt32 - vgprInt32));
        EXPECT_EQ(rSgprInt32, resultType(sgprInt32 * sgprInt32));

        EXPECT_EQ(rVgprInt64, resultType(vgprInt64 + vgprInt32));
        EXPECT_EQ(rVgprInt64, resultType(vgprInt32 + vgprInt64));
        EXPECT_EQ(rVgprInt64, resultType(vgprInt64 + vgprInt64));

        EXPECT_EQ(rVgprFloat, resultType(vgprFloat + vgprFloat));
        EXPECT_EQ(rVgprFloat, resultType(vgprFloat - sgprFloat));
        EXPECT_EQ(rVgprFloat, resultType(litFloat * vgprFloat));
        EXPECT_EQ(rVgprFloat, resultType(vgprFloat * litFloat));

        EXPECT_EQ(rSgprInt32, resultType(sgprInt32 + sgprInt32));
        EXPECT_EQ(rSgprInt32, resultType(sgprInt32 + litInt32));
        EXPECT_EQ(rSgprInt32, resultType(litInt32 + sgprInt32));
        EXPECT_EQ(rSgprInt64, resultType(litInt32 + sgprInt64));
        EXPECT_EQ(rSgprInt64, resultType(sgprInt64 + litInt32));
        EXPECT_EQ(rSgprInt64, resultType(sgprInt64 + sgprInt32));

        EXPECT_EQ(rSgprWavefrontSized, resultType(vgprFloat > vgprFloat));
        EXPECT_EQ(rSgprWavefrontSized, resultType(sgprFloat < vgprFloat));
        EXPECT_EQ(rSgprWavefrontSized, resultType(sgprDouble <= vgprDouble));
        EXPECT_EQ(rSgprWavefrontSized, resultType(sgprInt32 <= vgprInt32));
        EXPECT_EQ(rSgprWavefrontSized, resultType(litInt32 > vgprInt64));
        EXPECT_EQ(rSgprBool, resultType(litInt32 <= sgprInt64));
        EXPECT_EQ(rSgprBool, resultType(sgprInt32 >= litInt32));

        EXPECT_ANY_THROW(resultType(sgprDouble <= vgprFloat));
        EXPECT_ANY_THROW(resultType(vgprInt32 > vgprFloat));

        constexpr auto arithmeticUnaryOps = std::to_array({
            Expression::operator-, // cppcheck-suppress syntaxError
            Expression::operator~,
            Expression::magicMultiple,
            Expression::magicSign,
        });

        for(auto const& op : arithmeticUnaryOps)
        {
            EXPECT_EQ(rVgprFloat, resultType(op(vgprFloat))) << op(vgprFloat);
            EXPECT_EQ(rVgprDouble, resultType(op(vgprDouble))) << op(vgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprInt32))) << op(vgprInt32);
            EXPECT_EQ(rVgprInt64, resultType(op(vgprInt64))) << op(vgprInt64);
            EXPECT_EQ(rVgprUInt32, resultType(op(vgprUInt32))) << op(vgprUInt32);
            EXPECT_EQ(rVgprUInt64, resultType(op(vgprUInt64))) << op(vgprUInt64);
            EXPECT_EQ(rVgprHalf, resultType(op(vgprHalf))) << op(vgprHalf);
            EXPECT_EQ(rVgprHalfx2, resultType(op(vgprHalfx2))) << op(vgprHalfx2);
            EXPECT_EQ(rVgprBool32, resultType(op(vgprBool32))) << op(vgprBool32);

            EXPECT_EQ(rSgprFloat, resultType(op(sgprFloat))) << op(sgprFloat);
            EXPECT_EQ(rSgprDouble, resultType(op(sgprDouble))) << op(sgprDouble);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprInt32))) << op(sgprInt32);
            EXPECT_EQ(rSgprInt64, resultType(op(sgprInt64))) << op(sgprInt64);
            EXPECT_EQ(rSgprUInt32, resultType(op(sgprUInt32))) << op(sgprUInt32);
            EXPECT_EQ(rSgprUInt64, resultType(op(sgprUInt64))) << op(sgprUInt64);
            EXPECT_EQ(rSgprHalf, resultType(op(sgprHalf))) << op(sgprHalf);
            EXPECT_EQ(rSgprHalfx2, resultType(op(sgprHalfx2))) << op(sgprHalfx2);
            EXPECT_EQ(rSgprBool32, resultType(op(sgprBool32))) << op(sgprBool32);
        }

        {
            auto op = Expression::magicShifts;
            EXPECT_EQ(rVgprInt32, resultType(op(vgprFloat))) << op(vgprFloat);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprDouble))) << op(vgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprInt32))) << op(vgprInt32);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprInt64))) << op(vgprInt64);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprUInt32))) << op(vgprUInt32);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprUInt64))) << op(vgprUInt64);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprHalf))) << op(vgprHalf);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprHalfx2))) << op(vgprHalfx2);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprBool32))) << op(vgprBool32);

            EXPECT_EQ(rSgprInt32, resultType(op(sgprFloat))) << op(sgprFloat);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprDouble))) << op(sgprDouble);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprInt32))) << op(sgprInt32);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprInt64))) << op(sgprInt64);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprUInt32))) << op(sgprUInt32);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprUInt64))) << op(sgprUInt64);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprHalf))) << op(sgprHalf);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprHalfx2))) << op(sgprHalfx2);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprBool32))) << op(sgprBool32);
        }

        constexpr auto comparisonOps = std::to_array({
            Expression::operator>,
            Expression::operator>=,
            Expression::operator<,
            Expression::operator<=,
            Expression::operator==,
        });

        for(auto const& op : comparisonOps)
        {
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprFloat, vgprFloat)))
                << op(vgprFloat, vgprFloat);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprDouble, vgprDouble)))
                << op(vgprDouble, vgprDouble);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprInt32, vgprInt32)))
                << op(vgprInt32, vgprInt32);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprInt64, vgprInt64)))
                << op(vgprInt64, vgprInt64);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprUInt32, vgprUInt32)))
                << op(vgprUInt32, vgprUInt32);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprUInt64, vgprUInt64)))
                << op(vgprUInt64, vgprUInt64);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprHalf, vgprHalf)))
                << op(vgprHalf, vgprHalf);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprHalfx2, vgprHalfx2)))
                << op(vgprHalfx2, vgprHalfx2);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprBool32, vgprBool32)))
                << op(vgprBool32, vgprBool32);
            EXPECT_EQ(rSgprWavefrontSized, resultType(op(vgprBool, vgprBool)))
                << op(vgprBool, vgprBool);

            EXPECT_EQ(rSgprBool, resultType(op(sgprFloat, sgprFloat))) << op(sgprFloat, sgprFloat);
            EXPECT_EQ(rSgprBool, resultType(op(sgprDouble, sgprDouble)))
                << op(sgprDouble, sgprDouble);
            EXPECT_EQ(rSgprBool, resultType(op(sgprInt32, sgprInt32))) << op(sgprInt32, sgprInt32);
            EXPECT_EQ(rSgprBool, resultType(op(sgprInt64, sgprInt64))) << op(sgprInt64, sgprInt64);
            EXPECT_EQ(rSgprBool, resultType(op(sgprUInt32, sgprUInt32)))
                << op(sgprUInt32, sgprUInt32);
            EXPECT_EQ(rSgprBool, resultType(op(sgprUInt64, sgprUInt64)))
                << op(sgprUInt64, sgprUInt64);
            EXPECT_EQ(rSgprBool, resultType(op(sgprHalf, sgprHalf))) << op(sgprHalf, sgprHalf);
            EXPECT_EQ(rSgprBool, resultType(op(sgprHalfx2, sgprHalfx2)))
                << op(sgprHalfx2, sgprHalfx2);
            EXPECT_EQ(rSgprBool, resultType(op(sgprBool32, sgprBool32)))
                << op(sgprBool32, sgprBool32);
            EXPECT_EQ(rSgprBool, resultType(op(sgprBool, sgprBool))) << op(sgprBool, sgprBool);
        }

        constexpr auto arithmeticBinOps = std::to_array({
            Expression::operator+,
            Expression::operator-,
            Expression::operator*,
            Expression::operator/,
            Expression::operator%,
            Expression::operator<<,
            Expression::operator>>,
            Expression::operator&,
            Expression::arithmeticShiftR,
        });

        for(auto const& op : arithmeticBinOps)
        {
            EXPECT_EQ(rVgprFloat, resultType(op(vgprFloat, vgprFloat))) << op(vgprFloat, vgprFloat);
            EXPECT_EQ(rVgprDouble, resultType(op(vgprDouble, vgprDouble)))
                << op(vgprDouble, vgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprInt32, vgprInt32))) << op(vgprInt32, vgprInt32);
            EXPECT_EQ(rVgprInt64, resultType(op(vgprInt64, vgprInt64))) << op(vgprInt64, vgprInt64);
            EXPECT_EQ(rVgprUInt32, resultType(op(vgprUInt32, vgprUInt32)))
                << op(vgprUInt32, vgprUInt32);
            EXPECT_EQ(rVgprUInt64, resultType(op(vgprUInt64, vgprUInt64)))
                << op(vgprUInt64, vgprUInt64);
            EXPECT_EQ(rVgprHalf, resultType(op(vgprHalf, vgprHalf))) << op(vgprHalf, vgprHalf);
            EXPECT_EQ(rVgprHalfx2, resultType(op(vgprHalfx2, vgprHalfx2)))
                << op(vgprHalfx2, vgprHalfx2);
            EXPECT_EQ(rVgprBool32, resultType(op(vgprBool32, vgprBool32)))
                << op(vgprBool32, vgprBool32);

            EXPECT_EQ(rSgprFloat, resultType(op(sgprFloat, sgprFloat))) << op(sgprFloat, sgprFloat);
            EXPECT_EQ(rSgprDouble, resultType(op(sgprDouble, sgprDouble)))
                << op(sgprDouble, sgprDouble);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprInt32, sgprInt32))) << op(sgprInt32, sgprInt32);
            EXPECT_EQ(rSgprInt64, resultType(op(sgprInt64, sgprInt64))) << op(sgprInt64, sgprInt64);
            EXPECT_EQ(rSgprUInt32, resultType(op(sgprUInt32, sgprUInt32)))
                << op(sgprUInt32, sgprUInt32);
            EXPECT_EQ(rSgprUInt64, resultType(op(sgprUInt64, sgprUInt64)))
                << op(sgprUInt64, sgprUInt64);
            EXPECT_EQ(rSgprHalf, resultType(op(sgprHalf, sgprHalf))) << op(sgprHalf, sgprHalf);
            EXPECT_EQ(rSgprHalfx2, resultType(op(sgprHalfx2, sgprHalfx2)))
                << op(sgprHalfx2, sgprHalfx2);
            EXPECT_EQ(rSgprBool32, resultType(op(sgprBool32, sgprBool32)))
                << op(sgprBool32, sgprBool32);
        }

        constexpr auto logicalOps = std::to_array({
            Expression::operator&&,
            Expression::operator||,
        });

        for(auto const& op : logicalOps)
        {
            EXPECT_ANY_THROW(resultType(op(vgprFloat, vgprFloat))) << op(vgprFloat, vgprFloat);
            EXPECT_ANY_THROW(resultType(op(vgprDouble, vgprDouble))) << op(vgprDouble, vgprDouble);
            EXPECT_ANY_THROW(resultType(op(vgprInt32, vgprInt32))) << op(vgprInt32, vgprInt32);
            EXPECT_ANY_THROW(resultType(op(vgprInt64, vgprInt64))) << op(vgprInt64, vgprInt64);
            EXPECT_ANY_THROW(resultType(op(vgprUInt32, vgprUInt32))) << op(vgprUInt32, vgprUInt32);
            EXPECT_ANY_THROW(resultType(op(vgprUInt64, vgprUInt64))) << op(vgprUInt64, vgprUInt64);
            EXPECT_ANY_THROW(resultType(op(vgprHalf, vgprHalf))) << op(vgprHalf, vgprHalf);
            EXPECT_ANY_THROW(resultType(op(vgprHalfx2, vgprHalfx2))) << op(vgprHalfx2, vgprHalfx2);
            EXPECT_ANY_THROW(resultType(op(vgprBool32, vgprBool32))) << op(vgprBool32, vgprBool32);
            EXPECT_ANY_THROW(resultType(op(vgprBool, vgprBool))) << op(vgprBool, vgprBool);

            EXPECT_ANY_THROW(resultType(op(sgprFloat, sgprFloat))) << op(sgprFloat, sgprFloat);
            EXPECT_ANY_THROW(resultType(op(sgprDouble, sgprDouble))) << op(sgprDouble, sgprDouble);
            EXPECT_ANY_THROW(resultType(op(sgprInt32, sgprInt32))) << op(sgprInt32, sgprInt32);
            EXPECT_ANY_THROW(resultType(op(sgprInt64, sgprInt64))) << op(sgprInt64, sgprInt64);
            EXPECT_ANY_THROW(resultType(op(sgprUInt32, sgprUInt32))) << op(sgprUInt32, sgprUInt32);

            EXPECT_EQ(rSgprBool, resultType(op(sgprBool64, sgprBool64)))
                << op(sgprBool64, sgprBool64);
            EXPECT_ANY_THROW(resultType(op(sgprHalf, sgprHalf))) << op(sgprHalf, sgprHalf);
            EXPECT_ANY_THROW(resultType(op(sgprHalfx2, sgprHalfx2))) << op(sgprHalfx2, sgprHalfx2);
            EXPECT_EQ(rSgprBool, resultType(op(sgprBool32, sgprBool32)))
                << op(sgprBool32, sgprBool32);
            EXPECT_EQ(rSgprBool, resultType(op(sgprBool, sgprBool))) << op(sgprBool, sgprBool);
        }

        constexpr auto bitwiseBinOps = std::to_array({
            Expression::operator<<,
            Expression::logicalShiftR,
            Expression::operator&,
            Expression::operator^,
            Expression::operator|
        });

        for(auto const& op : bitwiseBinOps)
        {
            EXPECT_EQ(rVgprFloat, resultType(op(vgprFloat, vgprFloat))) << op(vgprFloat, vgprFloat);
            EXPECT_EQ(rVgprDouble, resultType(op(vgprDouble, vgprDouble)))
                << op(vgprDouble, vgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprInt32, vgprInt32))) << op(vgprInt32, vgprInt32);
            EXPECT_EQ(rVgprInt64, resultType(op(vgprInt64, vgprInt64))) << op(vgprInt64, vgprInt64);
            EXPECT_EQ(rVgprUInt32, resultType(op(vgprUInt32, vgprUInt32)))
                << op(vgprUInt32, vgprUInt32);
            EXPECT_EQ(rVgprHalf, resultType(op(vgprHalf, vgprHalf))) << op(vgprHalf, vgprHalf);
            EXPECT_EQ(rVgprHalfx2, resultType(op(vgprHalfx2, vgprHalfx2)))
                << op(vgprHalfx2, vgprHalfx2);
            EXPECT_EQ(rVgprBool32, resultType(op(vgprBool32, vgprBool32)))
                << op(vgprBool32, vgprBool32);

            EXPECT_EQ(rSgprFloat, resultType(op(sgprFloat, sgprFloat))) << op(sgprFloat, sgprFloat);
            EXPECT_EQ(rSgprDouble, resultType(op(sgprDouble, sgprDouble)))
                << op(sgprDouble, sgprDouble);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprInt32, sgprInt32))) << op(sgprInt32, sgprInt32);
            EXPECT_EQ(rSgprInt64, resultType(op(sgprInt64, sgprInt64))) << op(sgprInt64, sgprInt64);
            EXPECT_EQ(rSgprUInt32, resultType(op(sgprUInt32, sgprUInt32)))
                << op(sgprUInt32, sgprUInt32);
            EXPECT_EQ(rSgprHalf, resultType(op(sgprHalf, sgprHalf))) << op(sgprHalf, sgprHalf);
            EXPECT_EQ(rSgprHalfx2, resultType(op(sgprHalfx2, sgprHalfx2)))
                << op(sgprHalfx2, sgprHalfx2);
            EXPECT_EQ(rSgprBool32, resultType(op(sgprBool32, sgprBool32)))
                << op(sgprBool32, sgprBool32);
        }

        constexpr auto arithmeticTernaryOps = std::to_array(
            {Expression::multiplyAdd, Expression::addShiftL, Expression::shiftLAdd});

        for(auto const& op : arithmeticTernaryOps)
        {
            EXPECT_EQ(rVgprFloat, resultType(op(vgprFloat, vgprFloat, vgprFloat)))
                << op(vgprFloat, vgprFloat, vgprFloat);
            EXPECT_EQ(rVgprDouble, resultType(op(vgprDouble, vgprDouble, vgprDouble)))
                << op(vgprDouble, vgprDouble, vgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprInt32, vgprInt32, vgprInt32)))
                << op(vgprInt32, vgprInt32, vgprInt32);
            EXPECT_EQ(rVgprInt64, resultType(op(vgprInt64, vgprInt64, vgprInt64)))
                << op(vgprInt64, vgprInt64, vgprInt64);
            EXPECT_EQ(rVgprUInt32, resultType(op(vgprUInt32, vgprUInt32, vgprUInt32)))
                << op(vgprUInt32, vgprUInt32, vgprUInt32);
            EXPECT_EQ(rVgprHalf, resultType(op(vgprHalf, vgprHalf, vgprHalf)))
                << op(vgprHalf, vgprHalf, vgprHalf);
            EXPECT_EQ(rVgprHalfx2, resultType(op(vgprHalfx2, vgprHalfx2, vgprHalfx2)))
                << op(vgprHalfx2, vgprHalfx2, vgprHalfx2);
            EXPECT_EQ(rVgprBool32, resultType(op(vgprBool32, vgprBool32, vgprBool32)))
                << op(vgprBool32, vgprBool32, vgprBool32);

            EXPECT_EQ(rSgprFloat, resultType(op(sgprFloat, sgprFloat, sgprFloat)))
                << op(sgprFloat, sgprFloat, sgprFloat);
            EXPECT_EQ(rSgprDouble, resultType(op(sgprDouble, sgprDouble, sgprDouble)))
                << op(sgprDouble, sgprDouble, sgprDouble);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprInt32, sgprInt32, sgprInt32)))
                << op(sgprInt32, sgprInt32, sgprInt32);
            EXPECT_EQ(rSgprInt64, resultType(op(sgprInt64, sgprInt64, sgprInt64)))
                << op(sgprInt64, sgprInt64, sgprInt64);
            EXPECT_EQ(rSgprUInt32, resultType(op(sgprUInt32, sgprUInt32, sgprUInt32)))
                << op(sgprUInt32, sgprUInt32, sgprUInt32);
            EXPECT_EQ(rSgprHalf, resultType(op(sgprHalf, sgprHalf, sgprHalf)))
                << op(sgprHalf, sgprHalf, sgprHalf);
            EXPECT_EQ(rSgprHalfx2, resultType(op(sgprHalfx2, sgprHalfx2, sgprHalfx2)))
                << op(sgprHalfx2, sgprHalfx2, sgprHalfx2);
            EXPECT_EQ(rSgprBool32, resultType(op(sgprBool32, sgprBool32, sgprBool32)))
                << op(sgprBool32, sgprBool32, sgprBool32);
        }

        {
            auto op = Expression::conditional;
            EXPECT_EQ(rVgprFloat, resultType(op(sgprBool, vgprFloat, vgprFloat)))
                << op(sgprBool, vgprFloat, vgprFloat);
            EXPECT_EQ(rVgprDouble, resultType(op(sgprBool, vgprDouble, vgprDouble)))
                << op(sgprBool, vgprDouble, vgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(sgprBool, vgprInt32, vgprInt32)))
                << op(sgprBool, vgprInt32, vgprInt32);
            EXPECT_EQ(rVgprInt64, resultType(op(sgprBool, vgprInt64, vgprInt64)))
                << op(sgprBool, vgprInt64, vgprInt64);
            EXPECT_EQ(rVgprUInt32, resultType(op(sgprBool, vgprUInt32, vgprUInt32)))
                << op(sgprBool, vgprUInt32, vgprUInt32);
            EXPECT_EQ(rVgprHalf, resultType(op(sgprBool, vgprHalf, vgprHalf)))
                << op(sgprBool, vgprHalf, vgprHalf);
            EXPECT_EQ(rVgprHalfx2, resultType(op(sgprBool, vgprHalfx2, vgprHalfx2)))
                << op(sgprBool, vgprHalfx2, vgprHalfx2);
            EXPECT_EQ(rVgprBool32, resultType(op(sgprBool, vgprBool32, vgprBool32)))
                << op(sgprBool, vgprBool32, vgprBool32);

            EXPECT_EQ(rVgprFloat, resultType(op(vgprBool, vgprFloat, vgprFloat)))
                << op(vgprBool, vgprFloat, vgprFloat);
            EXPECT_EQ(rVgprDouble, resultType(op(vgprBool, vgprDouble, vgprDouble)))
                << op(vgprBool, vgprDouble, vgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprBool, vgprInt32, vgprInt32)))
                << op(vgprBool, vgprInt32, vgprInt32);
            EXPECT_EQ(rVgprInt64, resultType(op(vgprBool, vgprInt64, vgprInt64)))
                << op(vgprBool, vgprInt64, vgprInt64);
            EXPECT_EQ(rVgprUInt32, resultType(op(vgprBool, vgprUInt32, vgprUInt32)))
                << op(vgprBool, vgprUInt32, vgprUInt32);
            EXPECT_EQ(rVgprHalf, resultType(op(vgprBool, vgprHalf, vgprHalf)))
                << op(vgprBool, vgprHalf, vgprHalf);
            EXPECT_EQ(rVgprHalfx2, resultType(op(vgprBool, vgprHalfx2, vgprHalfx2)))
                << op(vgprBool, vgprHalfx2, vgprHalfx2);
            EXPECT_EQ(rVgprBool32, resultType(op(vgprBool, vgprBool32, vgprBool32)))
                << op(vgprBool, vgprBool32, vgprBool32);

            EXPECT_EQ(rSgprFloat, resultType(op(sgprBool, sgprFloat, sgprFloat)))
                << op(sgprBool, sgprFloat, sgprFloat);
            EXPECT_EQ(rSgprDouble, resultType(op(sgprBool, sgprDouble, sgprDouble)))
                << op(sgprBool, sgprDouble, sgprDouble);
            EXPECT_EQ(rSgprInt32, resultType(op(sgprBool, sgprInt32, sgprInt32)))
                << op(sgprBool, sgprInt32, sgprInt32);
            EXPECT_EQ(rSgprInt64, resultType(op(sgprBool, sgprInt64, sgprInt64)))
                << op(sgprBool, sgprInt64, sgprInt64);
            EXPECT_EQ(rSgprUInt32, resultType(op(sgprBool, sgprUInt32, sgprUInt32)))
                << op(sgprBool, sgprUInt32, sgprUInt32);
            EXPECT_EQ(rSgprHalf, resultType(op(sgprBool, sgprHalf, sgprHalf)))
                << op(sgprBool, sgprHalf, sgprHalf);
            EXPECT_EQ(rSgprHalfx2, resultType(op(sgprBool, sgprHalfx2, sgprHalfx2)))
                << op(sgprBool, sgprHalfx2, sgprHalfx2);
            EXPECT_EQ(rSgprBool32, resultType(op(sgprBool, sgprBool32, sgprBool32)))
                << op(sgprBool, sgprBool32, sgprBool32);

            EXPECT_EQ(rVgprFloat, resultType(op(vgprBool, sgprFloat, sgprFloat)))
                << op(vgprBool, sgprFloat, sgprFloat);
            EXPECT_EQ(rVgprDouble, resultType(op(vgprBool, sgprDouble, sgprDouble)))
                << op(vgprBool, sgprDouble, sgprDouble);
            EXPECT_EQ(rVgprInt32, resultType(op(vgprBool, sgprInt32, sgprInt32)))
                << op(vgprBool, sgprInt32, sgprInt32);
            EXPECT_EQ(rVgprInt64, resultType(op(vgprBool, sgprInt64, sgprInt64)))
                << op(vgprBool, sgprInt64, sgprInt64);
            EXPECT_EQ(rVgprUInt32, resultType(op(vgprBool, sgprUInt32, sgprUInt32)))
                << op(vgprBool, sgprUInt32, sgprUInt32);
            EXPECT_EQ(rVgprHalf, resultType(op(vgprBool, sgprHalf, sgprHalf)))
                << op(vgprBool, sgprHalf, sgprHalf);
            EXPECT_EQ(rVgprHalfx2, resultType(op(vgprBool, sgprHalfx2, sgprHalfx2)))
                << op(vgprBool, sgprHalfx2, sgprHalfx2);
            EXPECT_EQ(rVgprBool32, resultType(op(vgprBool, sgprBool32, sgprBool32)))
                << op(vgprBool, sgprBool32, sgprBool32);
        }
    }

    TEST_F(ExpressionTest, EvaluateNoArgs)
    {
        auto a = std::make_shared<Expression::Expression>(1.0);
        auto b = std::make_shared<Expression::Expression>(2.0);

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        auto expectedTimes = Expression::EvaluationTimes::All();
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr2));

        EXPECT_TRUE(Expression::canEvaluateTo(3.0, expr1));
        EXPECT_TRUE(Expression::canEvaluateTo(6.0, expr2));
        EXPECT_EQ(3.0, std::get<double>(Expression::evaluate(expr1)));
        EXPECT_EQ(6.0, std::get<double>(Expression::evaluate(expr2)));
    }

    TEST_F(ExpressionTest, EvaluateArgs)
    {
        VariableType doubleVal{DataType::Double, PointerType::Value};
        auto         ca = std::make_shared<CommandArgument>(nullptr, doubleVal, 0);
        auto         cb = std::make_shared<CommandArgument>(nullptr, doubleVal, 8);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto expr1 = a + b;
        auto expr2 = b * expr1;
        auto expr3 = -expr2;

        struct
        {
            double a = 1.0;
            double b = 2.0;
        } args;
        RuntimeArguments runtimeArgs((uint8_t*)&args, sizeof(args));

        Expression::ResultType expected{Register::Type::Literal, DataType::Double};
        EXPECT_EQ(expected, resultType(expr2));
        EXPECT_EQ(6.0, std::get<double>(Expression::evaluate(expr2, runtimeArgs)));

        args.a = 2.0;
        EXPECT_EQ(8.0, std::get<double>(Expression::evaluate(expr2, runtimeArgs)));
        EXPECT_EQ(-8.0, std::get<double>(Expression::evaluate(expr3, runtimeArgs)));

        args.b = 1.5;
        EXPECT_EQ(5.25, std::get<double>(Expression::evaluate(expr2, runtimeArgs)));

        EXPECT_FALSE(Expression::canEvaluateTo(5.25, expr2));
        // Don't send in the runtimeArgs, can't evaluate the arguments.
        EXPECT_THROW(Expression::evaluate(expr2), std::runtime_error);

        Expression::EvaluationTimes expectedTimes{Expression::EvaluationTime::KernelLaunch};
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr2));
    }

    TEST_F(ExpressionTest, EvaluateMixedTypes)
    {
        auto one          = std::make_shared<Expression::Expression>(1.0);
        auto two          = std::make_shared<Expression::Expression>(2.0f);
        auto twoPoint5    = std::make_shared<Expression::Expression>(2.5f);
        auto five         = std::make_shared<Expression::Expression>(5);
        auto seven        = std::make_shared<Expression::Expression>(7.0);
        auto eightPoint75 = std::make_shared<Expression::Expression>(8.75);

        auto ptrNull = std::make_shared<Expression::Expression>((float*)nullptr);

        float x        = 3.0f;
        auto  ptrValid = std::make_shared<Expression::Expression>(&x);

        double y              = 9.0;
        auto   ptrDoubleValid = std::make_shared<Expression::Expression>(&y);

        // double + float -> double
        auto expr1 = one + two;
        // float * double -> double
        auto exprSix = two * expr1;

        // double - int -> double
        auto exprOne = exprSix - five;

        // float + int -> float
        auto exprSeven = two + five;

        EXPECT_EQ(6.0, std::get<double>(Expression::evaluate(exprSix)));
        EXPECT_EQ(1.0, std::get<double>(Expression::evaluate(exprOne)));
        EXPECT_EQ(7.0f, std::get<float>(Expression::evaluate(exprSeven)));

        auto twoDouble = convert(DataType::Double, two);
        EXPECT_EQ(2.0, std::get<double>(Expression::evaluate(twoDouble)));

        auto twoInt = convert(DataType::Int32, twoPoint5);
        EXPECT_EQ(2, std::get<int>(Expression::evaluate(twoInt)));

        auto fiveDouble = seven - twoInt;
        EXPECT_EQ(5.0, std::get<double>(Expression::evaluate(fiveDouble)));

        auto minusThree64 = convert(DataType::Int64, twoInt - five);
        EXPECT_EQ(-3l, std::get<int64_t>(Expression::evaluate(minusThree64)));

        auto minusThreeU64 = convert(DataType::UInt64, twoInt - five);
        EXPECT_EQ(18446744073709551613ul, std::get<uint64_t>(Expression::evaluate(minusThreeU64)));

        auto eight75Half = convert(DataType::Half, eightPoint75);
        EXPECT_EQ(Half(8.75), std::get<Half>(evaluate(eight75Half)));

        Expression::ResultType litDouble{Register::Type::Literal, DataType::Double};
        Expression::ResultType litFloat{Register::Type::Literal, DataType::Float};
        Expression::ResultType litBool{Register::Type::Literal, DataType::Bool};

        EXPECT_EQ(litDouble, resultType(exprSix));
        // Result type not (yet?) defined for mixed integral/floating point types.
        EXPECT_ANY_THROW(resultType(exprOne));
        EXPECT_ANY_THROW(resultType(exprSeven));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix > exprOne)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix >= exprOne)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix < exprOne)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix <= exprOne)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix != exprOne)));

        EXPECT_ANY_THROW(resultType(exprSix > exprOne));
        EXPECT_ANY_THROW(resultType(exprSix >= exprOne));
        EXPECT_ANY_THROW(resultType(exprSix < exprOne));
        EXPECT_ANY_THROW(resultType(exprSix <= exprOne));
        EXPECT_EQ(litBool, resultType(one > seven));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix < exprSeven)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix <= exprSeven)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix > exprSeven)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix >= exprSeven)));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(one <= exprOne)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(one == exprOne)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(one >= exprOne)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(one != exprOne)));

        auto trueExp = std::make_shared<Expression::Expression>(true);
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(trueExp == (one >= exprOne))));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(trueExp == (one < exprOne))));

        // Pointer + double -> error.
        {
            auto exprThrow = ptrValid + exprOne;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
            EXPECT_ANY_THROW(resultType(exprThrow));
        }

        // Pointer * int -> error.
        {
            auto exprThrow = ptrValid * five;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        // Pointer + pointer -> error
        {
            auto exprThrow = ptrValid + ptrDoubleValid;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
            EXPECT_ANY_THROW(resultType(exprThrow));
        }

        // (float *) -  (double *) -> error
        {
            auto exprThrow = ptrValid - ptrDoubleValid;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
            EXPECT_ANY_THROW(resultType(exprThrow));
        }

        {
            auto exprThrow = ptrNull + five;
            // nullptr + int -> error;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = -ptrNull;
            // -pointer -> error;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = five + ptrNull;
            // Nullptr + int -> error;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        auto   exprXPlus5          = ptrValid + five;
        float* dontDereferenceThis = std::get<float*>(Expression::evaluate(exprXPlus5));
        auto   ptrDifference       = dontDereferenceThis - (&x);
        EXPECT_EQ(5, ptrDifference);

        auto expr10PlusX    = five + exprXPlus5;
        dontDereferenceThis = std::get<float*>(Expression::evaluate(expr10PlusX));
        ptrDifference       = dontDereferenceThis - (&x);
        EXPECT_EQ(10, ptrDifference);

        auto expr5PtrDiff = expr10PlusX - exprXPlus5;
        EXPECT_EQ(5, std::get<int64_t>(Expression::evaluate(expr5PtrDiff)));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(expr10PlusX > ptrValid)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(expr10PlusX < ptrValid)));
    }

    TEST_F(ExpressionTest, EqualityTest)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * a;
        auto expr3 = expr1 == expr2;

        Register::ValuePtr destReg;
        m_context->schedule(Expression::generate(destReg, expr3, m_context));

        auto result = R"(
            v_add_i32 v2, v0, v1
            v_mul_lo_u32 v3, v1, v0
            v_cmp_eq_i32 s[0:1], v2, v3
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, EvaluateConditional)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Double, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Double, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_gt = a > b;
        auto vals_lt = a < b;
        auto vals_ge = a >= b;
        auto vals_le = a <= b;
        auto vals_eq = a == b;

        auto expr_gt = a > (a + b);
        auto expr_lt = a < (a + b);
        auto expr_ge = a >= (a + b);
        auto expr_le = a <= (a + b);
        auto expr_eq = a == (a + b);

        for(double aVal : TestValues::doubleValues)
        {
            for(double bVal : TestValues::doubleValues)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ(aVal > bVal, std::get<bool>(Expression::evaluate(vals_gt, args)));
                EXPECT_EQ(aVal < bVal, std::get<bool>(Expression::evaluate(vals_lt, args)));
                EXPECT_EQ(aVal >= bVal, std::get<bool>(Expression::evaluate(vals_ge, args)));
                EXPECT_EQ(aVal <= bVal, std::get<bool>(Expression::evaluate(vals_le, args)));
                EXPECT_EQ(aVal == bVal, std::get<bool>(Expression::evaluate(vals_eq, args)));

                EXPECT_EQ(aVal > (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_gt, args)));
                EXPECT_EQ(aVal < (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_lt, args)));
                EXPECT_EQ(aVal >= (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_ge, args)));
                EXPECT_EQ(aVal <= (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_le, args)));
                EXPECT_EQ(aVal == (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_eq, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateLogical)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_negate        = logicalNot(a);
        auto vals_double_negate = logicalNot(logicalNot(a));
        auto vals_and           = a && b;
        auto vals_or            = a || b;

        for(auto aVal : TestValues::int32Values)
        {
            for(auto bVal : TestValues::int32Values)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ(!aVal, std::get<bool>(Expression::evaluate(vals_negate, args)));
                EXPECT_EQ(!!aVal, std::get<bool>(Expression::evaluate(vals_double_negate, args)));
                EXPECT_EQ(aVal && bVal, std::get<bool>(Expression::evaluate(vals_and, args)));
                EXPECT_EQ(aVal || bVal, std::get<bool>(Expression::evaluate(vals_or, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateShifts)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_shiftL       = a << b;
        auto vals_shiftR       = logicalShiftR(a, b);
        auto vals_signedShiftR = a >> b;

        auto expr_shiftL       = (a + b) << b;
        auto expr_shiftR       = logicalShiftR(a + b, b);
        auto expr_signedShiftR = (a + b) >> b;

        for(auto aVal : TestValues::int32Values)
        {
            for(auto bVal : TestValues::shiftValues)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ(aVal << bVal, std::get<int>(Expression::evaluate(vals_shiftL, args)));
                EXPECT_EQ(static_cast<unsigned int>(aVal) >> bVal,
                          std::get<int>(Expression::evaluate(vals_shiftR, args)));
                EXPECT_EQ(aVal >> bVal,
                          std::get<int>(Expression::evaluate(vals_signedShiftR, args)));

                EXPECT_EQ((aVal + bVal) << bVal,
                          std::get<int>(Expression::evaluate(expr_shiftL, args)));
                EXPECT_EQ(static_cast<unsigned int>(aVal + bVal) >> bVal,
                          std::get<int>(Expression::evaluate(expr_shiftR, args)));
                EXPECT_EQ((aVal + bVal) >> bVal,
                          std::get<int>(Expression::evaluate(expr_signedShiftR, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateConditionalTernary)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_shiftL = conditional(a >= b, a, b);

        for(auto aVal : TestValues::int32Values)
        {
            for(auto bVal : TestValues::int32Values)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                // At kernel launch time
                EXPECT_EQ(aVal >= bVal ? aVal : bVal,
                          std::get<int>(Expression::evaluate(vals_shiftL, args)));

                // At translate time
                auto a_static = std::make_shared<Expression::Expression>(aVal);
                auto b_static = std::make_shared<Expression::Expression>(bVal);
                EXPECT_EQ(aVal >= bVal ? aVal : bVal,
                          std::get<int>(Expression::evaluate(
                              conditional(a_static >= b_static, a_static, b_static))));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateBitwiseOps)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_and    = a & b;
        auto vals_or     = a | b;
        auto vals_negate = ~a;

        auto expr_and = (a + b) & b;
        auto expr_or  = (a + b) | b;

        for(auto aVal : TestValues::int32Values)
        {
            for(auto bVal : TestValues::int32Values)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ(aVal & bVal, std::get<int>(Expression::evaluate(vals_and, args)));
                EXPECT_EQ(aVal | bVal, std::get<int>(Expression::evaluate(vals_or, args)));
                EXPECT_EQ(~aVal, std::get<int>(Expression::evaluate(vals_negate, args)));

                EXPECT_EQ((aVal + bVal) & bVal,
                          std::get<int>(Expression::evaluate(expr_and, args)));
                EXPECT_EQ((aVal + bVal) | bVal, std::get<int>(Expression::evaluate(expr_or, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateMultiplyHigh)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto expr1 = multiplyHigh(a, b);

        auto expr2 = multiplyHigh(a + b, b);

        std::vector<int> a_values = {-21474836,
                                     -146000,
                                     -1,
                                     0,
                                     1,
                                     2,
                                     4,
                                     5,
                                     7,
                                     12,
                                     19,
                                     33,
                                     63,
                                     906,
                                     3017123,
                                     800000,
                                     1234456,
                                     4022112};
        for(auto aVal : a_values)
        {
            for(auto bVal : a_values)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ((aVal * (int64_t)bVal) >> 32,
                          std::get<int>(Expression::evaluate(expr1, args)));

                EXPECT_EQ(((aVal + bVal) * (int64_t)bVal) >> 32,
                          std::get<int>(Expression::evaluate(expr2, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateMultiplyHighUnsigned)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::UInt32, PointerType::Value}, aTag, ArgumentType::Value);
        auto bTag = command->allocateTag();
        auto cb   = command->allocateArgument(
            {DataType::UInt32, PointerType::Value}, bTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto expr1 = multiplyHigh(a, b);

        auto expr2 = multiplyHigh(a + b, b);

        std::vector<unsigned int> a_values = {
            0, 1, 2, 4, 5, 7, 12, 19, 33, 63, 906, 3017123, 800000, 1234456, 4022112,
            //2863311531u // Can cause overflow
        };
        for(auto aVal : a_values)
        {
            for(auto bVal : a_values)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ((aVal * (uint64_t)bVal) >> 32,
                          std::get<unsigned int>(Expression::evaluate(expr1, args)))
                    << ShowValue(aVal) << ShowValue(bVal);

                EXPECT_EQ(((aVal + (uint64_t)bVal) * (uint64_t)bVal) >> 32,
                          std::get<unsigned int>(Expression::evaluate(expr2, args)))
                    << ShowValue(aVal) << ShowValue(bVal);
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateExponential2)
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);

        auto expr = exp2(a);

        for(auto aVal : TestValues::floatValues)
        {

            KernelArguments runtimeArgs;
            runtimeArgs.append("a", aVal);
            auto args = runtimeArgs.runtimeArguments();

            EXPECT_EQ(std::exp2(aVal), std::get<float>(Expression::evaluate(expr, args)));
        }
    }

    TEST_F(ExpressionTest, EvaluateConvertExpressions)
    {
        using namespace Expression;

        float  a = 1.25f;
        Half   b = 1.1111;
        double c = 5.2619;

        auto a_exp = literal(a);
        auto b_exp = literal(b);
        auto c_exp = literal(c);

        auto exp1 = convert<DataType::Half>(a_exp);
        auto exp2 = convert<DataType::Half>(b_exp);
        auto exp3 = convert<DataType::Half>(c_exp);

        EXPECT_EQ(resultVariableType(exp1).dataType, DataType::Half);
        EXPECT_EQ(resultVariableType(exp2).dataType, DataType::Half);
        EXPECT_EQ(resultVariableType(exp3).dataType, DataType::Half);

        EXPECT_EQ(std::get<Half>(evaluate(exp1)), static_cast<Half>(a));
        EXPECT_EQ(std::get<Half>(evaluate(exp2)), b);
        EXPECT_EQ(std::get<Half>(evaluate(exp3)), static_cast<Half>(c));

        auto exp4 = convert<DataType::Float>(a_exp);
        auto exp5 = convert<DataType::Float>(b_exp);
        auto exp6 = convert<DataType::Float>(c_exp);

        EXPECT_EQ(resultVariableType(exp4).dataType, DataType::Float);
        EXPECT_EQ(resultVariableType(exp5).dataType, DataType::Float);
        EXPECT_EQ(resultVariableType(exp6).dataType, DataType::Float);

        EXPECT_EQ(std::get<float>(evaluate(exp4)), a);
        EXPECT_EQ(std::get<float>(evaluate(exp5)), static_cast<float>(b));
        EXPECT_EQ(std::get<float>(evaluate(exp6)), static_cast<float>(c));
    }

    TEST_F(ExpressionTest, GenerateDF)
    {
        Register::AllocationOptions allocOptions{.contiguousChunkWidth
                                                 = Register::FULLY_CONTIGUOUS};

        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 4, allocOptions);
        ra->allocateNow();
        auto dfa = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{1, Register::Type::Vector, DataType::None});
        m_context->registerTagManager()->addRegister(1, ra);

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 4, allocOptions);
        rb->allocateNow();
        auto dfb = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{2, Register::Type::Vector, DataType::None});
        m_context->registerTagManager()->addRegister(2, rb);

        auto rc = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 4, allocOptions);
        rc->allocateNow();
        auto dfc = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{3, Register::Type::Vector, DataType::None});
        m_context->registerTagManager()->addRegister(3, rc);

        Register::ValuePtr rr1;
        m_context->schedule(Expression::generate(rr1, dfa * dfb, m_context));

        Register::ValuePtr rr2;
        m_context->schedule(
            Expression::generate(rr2, Expression::fuseTernary(dfa * dfb + dfc), m_context));

        auto result = R"(
            v_mul_f32 v12, v0, v4
            v_mul_f32 v13, v1, v5
            v_mul_f32 v14, v2, v6
            v_mul_f32 v15, v3, v7

            v_fma_f32 v16, v0, v4, v8
            v_fma_f32 v17, v1, v5, v9
            v_fma_f32 v18, v2, v6, v10
            v_fma_f32 v19, v3, v7, v11
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, LiteralTest)
    {
        std::vector<VariableType> dataTypes = {{DataType::Int32},
                                               {DataType::UInt32},
                                               {DataType::Int64},
                                               {DataType::UInt64},
                                               {DataType::Float},
                                               {DataType::Half},
                                               {DataType::Double},
                                               {DataType::Bool}};

        for(auto& dataType : dataTypes)
        {
            EXPECT_EQ(dataType, Expression::resultVariableType(Expression::literal(1, dataType)));
        }
    }

    TEST_F(ExpressionTest, SwapLiteralIntTest)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto expr1 = ra->expression();
        auto expr2 = Expression::literal(-5);

        Register::ValuePtr destReg;

        m_context->schedule(Expression::generate(destReg, expr1 + expr2, m_context));

        m_context->schedule(Expression::generate(destReg, expr1 & expr2, m_context));
        m_context->schedule(Expression::generate(destReg, expr1 | expr2, m_context));
        m_context->schedule(Expression::generate(destReg, expr1 ^ expr2, m_context));

        auto result = R"(
            v_add_i32 v1, -5, v0
            v_and_b32 v1, -5, v0
            v_or_b32 v1, -5, v0
            v_xor_b32 v1, -5, v0
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, VariantTest)
    {
        int32_t  x1          = 3;
        auto     intPtr      = Expression::literal(&x1);
        int64_t  x2          = 3L;
        auto     intLongPtr  = Expression::literal(&x2);
        uint32_t x3          = 3u;
        auto     uintPtr     = Expression::literal(&x3);
        uint64_t x4          = 3UL;
        auto     uintLongPtr = Expression::literal(&x4);
        float    x5          = 3.0f;
        auto     floatPtr    = Expression::literal(&x5);
        double   x6          = 3.0;
        auto     doublePtr   = Expression::literal(&x6);

        auto intExpr    = Expression::literal(1);
        auto uintExpr   = Expression::literal(1u);
        auto floatExpr  = Expression::literal(1.0f);
        auto doubleExpr = Expression::literal(1.0);
        auto boolExpr   = Expression::literal(true);

        auto v_a
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Double, 1);
        v_a->allocateNow();

        Expression::Expression    value    = Register::Value::Literal(1);
        Expression::ExpressionPtr valuePtr = std::make_shared<Expression::Expression>(value);

        Expression::Expression    tag    = Expression::DataFlowTag();
        Expression::ExpressionPtr tagPtr = std::make_shared<Expression::Expression>(tag);
        Expression::Expression    waveTile
            = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        Expression::ExpressionPtr waveTilePtr = std::make_shared<Expression::Expression>(waveTile);

        std::vector<Expression::ExpressionPtr> exprs = {
            intExpr,
            uintExpr,
            floatExpr,
            doubleExpr,
            boolExpr,
            intExpr + intExpr,
            intExpr - intExpr,
            intExpr * intExpr,
            intExpr / intExpr,
            intExpr % intExpr,
            intExpr << intExpr,
            intExpr >> intExpr,
            logicalShiftR(intExpr, intExpr),
            intExpr & intExpr,
            intExpr ^ intExpr,
            intExpr > intExpr,
            intExpr < intExpr,
            intExpr >= intExpr,
            intExpr <= intExpr,
            intExpr == intExpr,
            -intExpr,
            intPtr,
            intLongPtr,
            uintPtr,
            uintLongPtr,
            floatPtr,
            doublePtr,
            valuePtr,
        };

        auto testFunc = [](auto const& expr) {
            EXPECT_NO_THROW(Expression::toString(expr));
            EXPECT_NO_THROW(Expression::evaluationTimes(expr));
        };

        for(auto const& expr : exprs)
        {
            testFunc(expr);
            EXPECT_NO_THROW(Expression::evaluate(expr));
            EXPECT_NO_THROW(Expression::fastDivision(expr, m_context));
        }

        testFunc(v_a);
        EXPECT_THROW(Expression::evaluate(v_a), FatalError);

        testFunc(tag);
        EXPECT_THROW(Expression::evaluate(tag), FatalError);

        testFunc(tagPtr);
        EXPECT_THROW(Expression::evaluate(tagPtr), FatalError);
        EXPECT_NO_THROW(Expression::fastDivision(tagPtr, m_context));

        testFunc(value);
        EXPECT_NO_THROW(Expression::evaluate(value));

        testFunc(waveTile);
        EXPECT_THROW(Expression::evaluate(waveTile), FatalError);

        testFunc(waveTilePtr);
        EXPECT_THROW(Expression::evaluate(waveTilePtr), FatalError);
        EXPECT_NO_THROW(Expression::fastDivision(waveTilePtr, m_context));

        EXPECT_NO_THROW(Expression::convert(DataType::Float, intExpr));
        EXPECT_NO_THROW(Expression::convert(DataType::Double, intExpr));
        EXPECT_THROW(Expression::convert(DataType::ComplexFloat, intExpr), FatalError);
        EXPECT_THROW(Expression::convert(DataType::ComplexDouble, intExpr), FatalError);
        EXPECT_NO_THROW(Expression::convert(DataType::Half, intExpr));
        EXPECT_NO_THROW(Expression::convert(DataType::Halfx2, intExpr));
        EXPECT_THROW(Expression::convert(DataType::Int8x4, intExpr), FatalError);
        EXPECT_NO_THROW(Expression::convert(DataType::Int32, intExpr));
        EXPECT_NO_THROW(Expression::convert(DataType::Int64, intExpr));
        EXPECT_THROW(Expression::convert(DataType::BFloat16, intExpr), FatalError);
        EXPECT_THROW(Expression::convert(DataType::Int8, intExpr), FatalError);
        EXPECT_THROW(Expression::convert(DataType::Raw32, intExpr), FatalError);
        EXPECT_NO_THROW(Expression::convert(DataType::UInt32, intExpr));
        EXPECT_NO_THROW(Expression::convert(DataType::UInt64, intExpr));
        EXPECT_THROW(Expression::convert(DataType::Bool, intExpr), FatalError);
        EXPECT_THROW(Expression::convert(DataType::Bool32, intExpr), FatalError);
        EXPECT_THROW(Expression::convert(DataType::Count, intExpr), FatalError);
        EXPECT_THROW(Expression::convert(static_cast<DataType>(200), intExpr), FatalError);
    }

    TEST_F(ExpressionTest, ComplexityTest)
    {
        auto intExpr = Expression::literal(1);

        EXPECT_EQ(Expression::complexity(intExpr), 0);
        EXPECT_GT(Expression::complexity(intExpr + intExpr), Expression::complexity(intExpr));
        EXPECT_GT(Expression::complexity(intExpr + intExpr + intExpr),
                  Expression::complexity(intExpr + intExpr));

        EXPECT_GT(Expression::complexity(intExpr / intExpr),
                  Expression::complexity(intExpr + intExpr));
    }

    class ARCH_ExpressionTest : public GPUContextFixture
    {
    public:
        template <typename TA, typename TB, typename TR>
        void basicBinaryExpression(std::function<Expression::ExpressionPtr(
                                       Expression::ExpressionPtr, Expression::ExpressionPtr)> f,
                                   TA aValue,
                                   TB bValue,
                                   TR resultValue)
        {
            DataType aDType = TypeInfo<TA>::Var.dataType;
            DataType bDType = TypeInfo<TB>::Var.dataType;
            DataType rDType = TypeInfo<TR>::Var.dataType;

            auto k    = m_context->kernel();
            auto v_a  = Register::Value::Placeholder(m_context, Register::Type::Vector, aDType, 1);
            auto v_b  = Register::Value::Placeholder(m_context, Register::Type::Vector, bDType, 1);
            auto a    = v_a->expression();
            auto b    = v_b->expression();
            auto expr = f(a, b);

            k->addArgument(
                {"result", {rDType, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument({"a", aDType});
            k->addArgument({"b", bDType});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);
                co_yield m_context->argLoader()->getValue("b", s_b);

                auto v_result = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, {rDType, PointerType::PointerGlobal}, 1);

                co_yield v_a->allocate();
                co_yield v_b->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                co_yield m_context->copier()->copy(v_a, s_a, "Move pointer");
                co_yield m_context->copier()->copy(v_b, s_b, "Move pointer");

                Register::ValuePtr v_c;
                co_yield Expression::generate(v_c, expr, m_context);

                co_yield m_context->mem()->storeFlat(v_result, v_c, 0, sizeof(TR));
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            if(m_context->targetArchitecture().target().getMajorVersion() != 9)
                GTEST_SKIP() << "Skipping GPU tests for " << GetParam();

            // Only execute the kernels if running on the architecture that the kernel was built for,
            // otherwise just assemble the instructions.
            if(isLocalDevice())
            {
                std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                    = m_context->instructions()->getExecutableKernel();

                auto d_result = make_shared_device<TR>();

                KernelArguments kargs;
                kargs.append("result", d_result.get());
                kargs.append("a", aValue);
                kargs.append("b", bValue);
                KernelInvocation invocation;

                executableKernel->executeKernel(kargs, invocation);

                std::vector<TR> result(1);
                ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(TR), hipMemcpyDefault),
                            HasHipSuccess(0));

                EXPECT_EQ(result[0], resultValue);
            }
            else
            {
                std::vector<char> assembledKernel = m_context->instructions()->assemble();
                EXPECT_GT(assembledKernel.size(), 0);
            }
        }

        template <typename TA, typename TB, typename TC, typename TR>
        void basicTernaryExpression(
            std::function<Expression::ExpressionPtr(
                Expression::ExpressionPtr, Expression::ExpressionPtr, Expression::ExpressionPtr)> f,
            TA             aValue,
            TB             bValue,
            TC             cValue,
            TR             resultValue,
            bool           resultPlaceholder = false,
            Register::Type regType           = Register::Type::Vector)
        {
            DataType aDType = TypeInfo<TA>::Var.dataType;
            DataType bDType = TypeInfo<TB>::Var.dataType;
            DataType cDType = TypeInfo<TC>::Var.dataType;
            DataType rDType = TypeInfo<TR>::Var.dataType;

            auto k    = m_context->kernel();
            auto v_a  = Register::Value::Placeholder(m_context, regType, aDType, 1);
            auto v_b  = Register::Value::Placeholder(m_context, regType, bDType, 1);
            auto v_c  = Register::Value::Placeholder(m_context, regType, cDType, 1);
            auto a    = v_a->expression();
            auto b    = v_b->expression();
            auto c    = v_c->expression();
            auto expr = f(a, b, c);

            k->addArgument(
                {"result", {rDType, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument({"a", aDType});
            k->addArgument({"b", bDType});
            k->addArgument({"c", cDType});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b, s_c;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);
                co_yield m_context->argLoader()->getValue("b", s_b);
                co_yield m_context->argLoader()->getValue("c", s_c);

                auto v_result_ptr = Register::Value::Placeholder(
                    m_context, regType, {rDType, PointerType::PointerGlobal}, 1);

                co_yield v_a->allocate();
                co_yield v_b->allocate();
                co_yield v_c->allocate();
                co_yield v_result_ptr->allocate();

                co_yield m_context->copier()->copy(v_result_ptr, s_result, "Move pointer");

                co_yield m_context->copier()->copy(v_a, s_a, "Move value");
                co_yield m_context->copier()->copy(v_b, s_b, "Move value");
                co_yield m_context->copier()->copy(v_c, s_c, "Move value");

                Register::ValuePtr v_r;
                if(resultPlaceholder)
                    v_r = Register::Value::Placeholder(m_context, regType, TypeInfo<TR>::Var, 1);
                co_yield Expression::generate(v_r, expr, m_context);

                if(regType == Register::Type::Vector)
                {
                    co_yield m_context->mem()->storeFlat(v_result_ptr, v_r, 0, sizeof(TR));
                }
                else
                {
                    co_yield m_context->mem()->storeScalar(v_result_ptr, v_r, 0, sizeof(TR));
                }
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            if(m_context->targetArchitecture().target().getMajorVersion() != 9)
                GTEST_SKIP() << "Skipping GPU tests for " << GetParam();

            // Only execute the kernels if running on the architecture that the kernel was built for,
            // otherwise just assemble the instructions.
            if(isLocalDevice())
            {
                std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                    = m_context->instructions()->getExecutableKernel();

                auto d_result = make_shared_device<TR>();

                KernelArguments kargs;
                kargs.append("result", d_result.get());
                kargs.append("a", aValue);
                kargs.append("b", bValue);
                kargs.append("c", cValue);
                KernelInvocation invocation;

                executableKernel->executeKernel(kargs, invocation);

                std::vector<TR> result(1);
                ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(TR), hipMemcpyDefault),
                            HasHipSuccess(0));

                EXPECT_EQ(result[0], resultValue);
            }
            else
            {
                std::vector<char> assembledKernel = m_context->instructions()->assemble();
                EXPECT_GT(assembledKernel.size(), 0);
            }
        }

        void MFMA_F8F6F4(DataType          typeA,
                         DataType          typeB,
                         int               M,
                         int               N,
                         int               K,
                         int               batches,
                         int               regCountA,
                         int               regCountB,
                         std::string const result)
        {
            if(m_context->targetArchitecture().target().getVersionString() != "gfx950")
            {
                GTEST_SKIP() << "Skipping MFMA F8F6F4 tests for "
                             << m_context->targetArchitecture().target().getVersionString();
            }

            auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
            auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

            A_tile->sizes = {M, K};
            A_tile->vgpr
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Vector,
                                                    typeA,
                                                    regCountA,
                                                    Register::AllocationOptions::FullyContiguous());
            A_tile->vgpr->allocateNow();

            B_tile->sizes = {K, N};
            B_tile->vgpr
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Vector,
                                                    typeB,
                                                    regCountB,
                                                    Register::AllocationOptions::FullyContiguous());
            B_tile->vgpr->allocateNow();

            auto ic
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Accumulator,
                                                    DataType::Float,
                                                    M * N * batches / 64,
                                                    Register::AllocationOptions::FullyContiguous());
            ic->allocateNow();

            auto A = std::make_shared<Expression::Expression>(A_tile);
            auto B = std::make_shared<Expression::Expression>(B_tile);
            auto C = ic->expression();

            auto expr
                = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, C));

            m_context->schedule(Expression::generate(ic, expr, m_context));

            EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
        }

        void MFMA_SCALE_F8F6F4(DataType          typeA,
                               DataType          typeB,
                               int               M,
                               int               N,
                               int               K,
                               int               regCountA,
                               int               regCountB,
                               std::string const result)
        {
            if(m_context->targetArchitecture().target().getVersionString() != "gfx950")
            {
                GTEST_SKIP() << "Skipping MFMA F8F6F4 tests for "
                             << m_context->targetArchitecture().target().getVersionString();
            }

            auto ATile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
            auto BTile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

            ATile->sizes = {M, K};
            ATile->vgpr
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Vector,
                                                    typeA,
                                                    regCountA,
                                                    Register::AllocationOptions::FullyContiguous());
            ATile->vgpr->allocateNow();

            BTile->sizes = {K, N};
            BTile->vgpr
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Vector,
                                                    typeB,
                                                    regCountB,
                                                    Register::AllocationOptions::FullyContiguous());
            BTile->vgpr->allocateNow();

            auto rC
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Accumulator,
                                                    DataType::Float,
                                                    M * N / 64,
                                                    Register::AllocationOptions::FullyContiguous());
            rC->allocateNow();

            // M * K is the number of elements in the A wavetile.
            // (M * K) / 32 is the number of scale elements of A wavetile.
            // For a wave with 64 threads, each thread will process
            // (M * K) / (32 * 64 = 2048) scale elements of A wavetile.
            auto rScaleA
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Vector,
                                                    DataType::UInt32,
                                                    M * K / 2048,
                                                    Register::AllocationOptions::FullyContiguous());
            rScaleA->allocateNow();

            auto rScaleB
                = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Vector,
                                                    DataType::UInt32,
                                                    K * N / 2048,
                                                    Register::AllocationOptions::FullyContiguous());
            rScaleB->allocateNow();

            auto A      = std::make_shared<Expression::Expression>(ATile);
            auto B      = std::make_shared<Expression::Expression>(BTile);
            auto C      = rC->expression();
            auto scaleA = rScaleA->expression();
            auto scaleB = rScaleB->expression();

            auto expr = std::make_shared<Expression::Expression>(
                Expression::ScaledMatrixMultiply(A, B, C, scaleA, scaleB));

            m_context->schedule(Expression::generate(rC, expr, m_context));

            EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
        }
    };

    TEST_P(ARCH_ExpressionTest, BasicExpression01)
    {
        double a = 192.0;
        double b = 12981.0;
        double r = -b * (a + b);

        auto expr
            = [](Expression::ExpressionPtr a, Expression::ExpressionPtr b) { return -b * (a + b); };

        basicBinaryExpression(expr, a, b, r);
    }

    TEST_P(ARCH_ExpressionTest, BasicExpression02)
    {
        int          a = 12;
        unsigned int b = 2u;
        int          r = (a + b) << b;

        auto expr = [](Expression::ExpressionPtr a, Expression::ExpressionPtr b) {
            return Expression::fuseTernary((a + b) << b);
        };

        basicBinaryExpression(expr, a, b, r);
    }

    TEST_P(ARCH_ExpressionTest, BasicExpression03)
    {
        float a     = 192.0;
        float b     = -182.1;
        float alpha = 0.5;

        float r = alpha * a + b;

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return Expression::fuseTernary(a * b + c); };

        basicTernaryExpression(expr, alpha, a, b, r);

        auto assembly = m_context->instructions()->toString();
        EXPECT_THAT(assembly, testing::HasSubstr("v_fma_f32"));
    }

    TEST_P(ARCH_ExpressionTest, Conversions1)
    {
        namespace Ex = Expression;
        int32_t  a   = -192;
        int64_t  b   = 2235530478080l;
        uint64_t c   = 40534632177074688l;

        auto r = static_cast<uint64_t>(((b << 3) * a) >> 4) + static_cast<int64_t>(c);

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            return convert<DataType::UInt64>(((b << Ex::literal(3)) * a) >> Ex::literal(4))
                   + convert(DataType::Int64, c);
        };

        basicTernaryExpression(expr, a, b, c, r);
    }

    TEST_P(ARCH_ExpressionTest, Conversions2)
    {
        namespace Ex = Expression;
        uint32_t a   = 2149597184u;
        uint32_t b   = 3223339008u;
        int64_t  c   = 4611703644973170816l;

        auto r = ((static_cast<uint64_t>(a) * 23) + b) - static_cast<uint64_t>(c);

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            return ((convert<DataType::UInt64>(a) * Ex::literal(23)) + b)
                   - convert(DataType::UInt64, c);
        };

        basicTernaryExpression(expr, a, b, c, r);
    }

    TEST_P(ARCH_ExpressionTest, Conversions3)
    {
        namespace Ex = Expression;
        int32_t  a   = -192;
        int64_t  b   = 2235530478080l;
        uint32_t c   = 40588;

        auto r = ((static_cast<int32_t>(b) + a) >> 4) + static_cast<int32_t>(c);

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            return ((convert(DataType::Int32, b) + a) >> Ex::literal(4))
                   + convert(DataType::Int32, c);
        };

        basicTernaryExpression(expr, a, b, c, r);
    }
    TEST_P(ARCH_ExpressionTest, Conversions4)
    {
        namespace Ex = Expression;
        uint32_t a   = 192;
        uint64_t b   = 2235530478080l;
        int32_t  c   = 40588;

        auto r = ((static_cast<uint32_t>(b) + a) >> 4) + static_cast<uint32_t>(c);

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            return logicalShiftR(convert(DataType::UInt32, b) + a, Ex::literal(4))
                   + convert(DataType::UInt32, c);
        };

        basicTernaryExpression(expr, a, b, c, r);
    }

    TEST_P(ARCH_ExpressionTest, ImplicitlyConvertShiftL1)
    {
        namespace Ex = Expression;
        uint32_t a   = 2149597184u;
        uint32_t b   = 12326;
        int      c   = 9;

        uint64_t r = static_cast<uint64_t>(a + b) << c;

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return (a + b) << c; };

        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ImplicitlyConvertShiftL2)
    {
        namespace Ex = Expression;
        int32_t a    = -9597184;
        int32_t b    = 12326;
        int     c    = 9;

        int64_t r = static_cast<int64_t>(a + b) << c;

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return (a + b) << c; };

        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ImplicitlyConvertShiftR1)
    {
        namespace Ex = Expression;
        uint32_t a   = 2149597184u;
        uint32_t b   = 12326;
        int      c   = 9;

        uint64_t r = static_cast<uint64_t>(a + b) >> c;

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return Ex::logicalShiftR((a + b), c); };

        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ImplicitlyConvertShiftR2)
    {
        namespace Ex = Expression;
        uint32_t a   = 2149597184u;
        uint32_t b   = 12326;
        int      c   = 9;

        uint64_t r = static_cast<uint64_t>(a + b) >> c;

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return (a + b) >> c; };

        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ImplicitlyConvertShiftR3)
    {
        namespace Ex = Expression;
        int32_t a    = -9597184;
        int32_t b    = 12326;
        int     c    = 9;

        int64_t r = static_cast<int64_t>(a + b) >> c;

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return (a + b) >> c; };

        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalUInt32Scalar_0)
    {
        uint32_t a   = std::numeric_limits<uint32_t>::max();
        uint32_t b   = 12326;
        uint32_t c   = 9;
        namespace Ex = Expression;

        auto r    = a > b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a > b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true, Register::Type::Scalar);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalUInt32Scalar_1)
    {
        uint32_t a   = std::numeric_limits<uint32_t>::max();
        uint32_t b   = 12326;
        uint32_t c   = 9;
        namespace Ex = Expression;

        auto r    = a < b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a < b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true, Register::Type::Scalar);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt32Scalar_0)
    {
        int32_t a    = std::numeric_limits<int32_t>::max();
        int32_t b    = -12326;
        int32_t c    = 9;
        namespace Ex = Expression;

        auto r    = a > b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a > b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true, Register::Type::Scalar);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt32Scalar_1)
    {
        int32_t a    = std::numeric_limits<int32_t>::max();
        int32_t b    = -12326;
        int32_t c    = 9;
        namespace Ex = Expression;

        auto r    = a < b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a < b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true, Register::Type::Scalar);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt64Scalar_0)
    {
        int64_t a    = std::numeric_limits<int64_t>::max();
        int64_t b    = -12326;
        int64_t c    = 9;
        namespace Ex = Expression;

        auto r    = a ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return Ex::conditional(a, b, c); };
        basicTernaryExpression(expr, a, b, c, r, true, Register::Type::Scalar);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt64Scalar_1)
    {
        int64_t a    = std::numeric_limits<int64_t>::max();
        int64_t b    = -12326;
        int64_t c    = 9;
        namespace Ex = Expression;

        auto r    = a ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) { return Ex::conditional(a, b, c); };
        basicTernaryExpression(expr, a, b, c, r, true, Register::Type::Scalar);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt32_0)
    {
        int32_t a    = std::numeric_limits<int32_t>::max();
        int32_t b    = -12326;
        int32_t c    = 9;
        namespace Ex = Expression;

        auto r    = a > b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a > b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt32_1)
    {
        int32_t a    = std::numeric_limits<int32_t>::max();
        int32_t b    = -12326;
        int32_t c    = 9;
        namespace Ex = Expression;

        auto r    = a < b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a < b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt64_0)
    {
        int64_t a    = std::numeric_limits<int64_t>::max();
        int64_t b    = -12326;
        int64_t c    = 9;
        namespace Ex = Expression;

        auto r    = a > b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a > b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalInt64_1)
    {
        int64_t a    = std::numeric_limits<int64_t>::max();
        int64_t b    = -12326;
        int64_t c    = 9;
        namespace Ex = Expression;

        auto r    = a < b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a < b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalFloat_0)
    {
        float a      = std::numeric_limits<float>::max();
        float b      = -12326.156;
        float c      = 9.894;
        namespace Ex = Expression;

        auto r    = a > b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a > b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalFloat_1)
    {
        float a      = std::numeric_limits<float>::max();
        float b      = -12326.156;
        float c      = 9.894;
        namespace Ex = Expression;

        auto r    = a < b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a < b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalDouble_0)
    {
        double a     = std::numeric_limits<double>::max();
        double b     = -12326.156;
        double c     = 9.894;
        namespace Ex = Expression;

        auto r    = a > b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a > b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ConditionalDouble_1)
    {
        double a     = std::numeric_limits<double>::max();
        double b     = -12326.156;
        double c     = 9.894;
        namespace Ex = Expression;

        auto r    = a < b ? b : c;
        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            auto d = a < b;
            return Ex::conditional(d, b, c);
        };
        basicTernaryExpression(expr, a, b, c, r, true);
    }

    TEST_P(ARCH_ExpressionTest, ComplexExpressionScalar)
    {
        auto s_a
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_b
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::UInt32, 1);

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument({"a", DataType::Int32});
        k->addArgument({"b", DataType::UInt32});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b, s_c, temp;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto a = s_a->expression();
            auto b = s_b->expression();

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Int32, PointerType::PointerGlobal},
                                               1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            auto expr1 = b > Expression::literal(0);
            co_yield Expression::generate(temp, expr1, m_context);

            auto expr2 = Expression::fuseTernary((a + (a < Expression::literal(5))) << b)
                         + temp->expression();
            co_yield Expression::generate(s_c, expr2, m_context);
            co_yield m_context->copier()->copy(v_c, s_c, "Copy result");

            co_yield m_context->mem()->storeFlat(v_result, v_c, 0, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping GPU tests for " << GetParam();

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<int>();

            for(int a = -10; a < 10; a++)
            {
                for(unsigned int b = 0; b < 5; b++)
                {

                    KernelArguments kargs;
                    kargs.append("result", d_result.get());
                    kargs.append("a", a);
                    kargs.append("b", b);
                    KernelInvocation invocation;

                    executableKernel->executeKernel(kargs, invocation);

                    int result;
                    ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(int), hipMemcpyDefault),
                                HasHipSuccess(0));

                    auto expectedResult = ((a + (a < 5)) << b) + (b > 0);
                    EXPECT_EQ(result, expectedResult);
                }
            }
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(ARCH_ExpressionTest, MFMA_SCALE_F32_16x16x128_F8F6F4)
    {
        MFMA_SCALE_F8F6F4(
            DataType::FP8x4,
            DataType::FP8x4,
            16,
            16,
            128,
            16 * 128 / 64 / 4, // M * K / 64 / 4
            128 * 16 / 64 / 4, // K * N / 64 / 4
            R"(v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 cbsz:0 abid:1 blgp:0)");
    }

    TEST_P(ARCH_ExpressionTest, MFMA_SCALE_F32_32x32x64_F8F6F4)
    {
        MFMA_SCALE_F8F6F4(
            DataType::FP8x4,
            DataType::FP8x4,
            32,
            32,
            64,
            32 * 64 / 64 / 4, // M * K / 64 / 4
            64 * 32 / 64 / 4, // K * N / 64 / 4
            R"(v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[0:7], v[8:15], a[0:15], v16, v17 cbsz:0 abid:1 blgp:0)");
    }

    TEST_P(ARCH_ExpressionTest, MFMA_F32_32x32x64_F8F6F4)
    {
        MFMA_F8F6F4(
            DataType::FP8x4,
            DataType::FP8x4,
            32,
            32,
            64,
            1,
            32 * 64 / 64 / 4, // M * K / 64 / 4
            64 * 32 / 64 / 4, // K * N / 64 / 4
            R"(v_mfma_f32_32x32x64_f8f6f4 a[0:15], v[0:7], v[8:15], a[0:15] cbsz:0b000 blgp:0b000)");
    }

    TEST_P(ARCH_ExpressionTest, MFMA_F32_16x16x128_F8F6F4)
    {
        MFMA_F8F6F4(
            DataType::FP8x4,
            DataType::FP8x4,
            16,
            16,
            128,
            1,
            16 * 128 / 64 / 4, // M * K / 64 / 4
            128 * 16 / 64 / 4, // K * N / 64 / 4
            R"(v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3] cbsz:0b000 blgp:0b000)");
    }

    INSTANTIATE_TEST_SUITE_P(ARCH_ExpressionTests, ARCH_ExpressionTest, supportedISATuples());

    class GPU_ExpressionTest : public CurrentGPUContextFixture
    {
    };
}
