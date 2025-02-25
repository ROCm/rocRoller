#include <cmath>
#include <memory>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    TEST_CASE("Create expressions and convert to string", "[expression][toString]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto context = TestContext::ForTarget(arch);

            auto a = Expression::literal(1);
            auto b = Expression::literal(2);

            auto rc = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::Int32, 1);
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

            SECTION("toString()")
            {
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

                CHECK(sexpr1 == "Add(1i, 2i)");
                CHECK(sexpr2 == "Multiply(2i, Add(1i, 2i))");
                CHECK(sexpr3 == "Subtract(Multiply(2i, Add(1i, 2i)), v0:I)");
                CHECK(sexpr4 == "GreaterThan(Add(1i, 2i), Multiply(2i, Add(1i, 2i)))");
                CHECK(sexpr5 == "LessThan(" + sexpr3 + ", " + sexpr4 + ")");
                CHECK(sexpr6 == "GreaterThanEqual(" + sexpr4 + ", " + sexpr5 + ")");
                CHECK(sexpr7 == "LessThanEqual(" + sexpr5 + ", " + sexpr6 + ")");
                CHECK(sexpr8 == "Equal(" + sexpr6 + ", " + sexpr7 + ")");
                CHECK(sexpr9 == "Negate(" + sexpr2 + ")");
                CHECK(sexpr10 == "AddShiftL(1i, 2i, 2i)");
                CHECK(sexpr11 == "ShiftLAdd(1i, 2i, 2i)");
                CHECK(sexpr12 == "NotEqual(" + sexpr6 + ", " + sexpr7 + ")");
            }

            SECTION("evaluationTimes()")
            {
                Expression::EvaluationTimes expectedTimes{
                    Expression::EvaluationTime::KernelExecute};
                CHECK(expectedTimes == Expression::evaluationTimes(expr8));
                CHECK(expectedTimes == Expression::evaluationTimes(expr10));
            }
        }
    }

    TEST_CASE("Expression serialization", "[expression][serialization]")
    {
        auto a = Expression::literal(1);
        auto b = Expression::literal(2);
        SECTION("Serializable expressions")
        {

            auto c = Register::Value::Literal(4.2f);
            auto d = Register::Value::Literal(Half(4.2f));

            Expression::DataFlowTag dataFlow;
            dataFlow.tag              = 50;
            dataFlow.regType          = Register::Type::Vector;
            dataFlow.varType.dataType = DataType::Float;

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
            auto expr12 = std::make_shared<Expression::Expression>(dataFlow) / a;

            auto expr = GENERATE_COPY(expr1,
                                      expr2,
                                      expr3,
                                      expr4,
                                      expr5,
                                      expr6,
                                      expr7,
                                      expr8,
                                      expr9,
                                      expr10,
                                      expr11,
                                      expr12);

            CAPTURE(expr);

            auto yamlText = Expression::toYAML(expr);
            INFO(yamlText);

            CHECK(yamlText != "");

            auto deserialized = Expression::fromYAML(yamlText);
            REQUIRE(deserialized.get() != nullptr);

            CHECK(Expression::toString(deserialized) == Expression::toString(expr));
            CHECK(Expression::identical(deserialized, expr));
        }

        SECTION("Unserializable expressions")
        {
            SECTION("Kernel arg")
            {
                auto kernelArg                   = std::make_shared<AssemblyKernelArgument>();
                kernelArg->name                  = "KernelArg1";
                kernelArg->variableType.dataType = DataType::Int32;
                kernelArg->expression            = Expression::literal(10);
                kernelArg->offset                = 1;
                kernelArg->size                  = 5;

                auto expr = b >> std::make_shared<Expression::Expression>(kernelArg);
                CHECK_THROWS(Expression::fromYAML(Expression::toYAML(expr)));
            }

            SECTION("WaveTile")
            {
                auto waveTile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
                auto expr     = std::make_shared<Expression::Expression>(waveTile) + b;
                CHECK_THROWS(Expression::fromYAML(Expression::toYAML(expr)));
            }

            SUPPORTED_ARCH_SECTION(arch)
            {
                auto context = TestContext::ForTarget(arch);

                auto reg = std::make_shared<Register::Value>(
                    context.get(), Register::Type::Vector, DataType::Int32, 1);
                reg->allocateNow();

                CHECK_THROWS(Expression::toYAML(reg->expression()));
            }
        }
    }

    TEST_CASE("Expression identical and equivalent", "[expression]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto a    = Expression::literal(1u);
        auto ap   = Expression::literal(1);
        auto b    = Expression::literal(2u);
        auto zero = Expression::literal(0u);

        auto rc = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto rd = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
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

        CHECK(identical(expr1, expr2));
        CHECK_FALSE(identical(expr1, expr3));
        CHECK_FALSE(identical(ap + b, expr3));

        CHECK(equivalent(expr1, expr2));
        CHECK_FALSE(equivalent(expr1, expr3));
        CHECK_FALSE(equivalent(ap + b, expr3));

        auto expr4 = c + d;
        auto expr5 = c + d + zero;

        CHECK_FALSE(identical(expr1, expr4));
        CHECK_FALSE(identical(expr4, expr5));
        CHECK(identical(expr4, simplify(expr5)));

        CHECK_FALSE(equivalent(expr1, expr4));
        CHECK_FALSE(equivalent(expr4, expr5));
        CHECK(equivalent(expr4, simplify(expr5)));

        auto expr6 = e / f % d;
        auto expr7 = a + f;

        CHECK_FALSE(identical(expr6, expr7));
        CHECK_FALSE(identical(e, f));

        CHECK(Expression::identical(nullptr, nullptr));
        CHECK_FALSE(identical(nullptr, a));
        CHECK_FALSE(identical(a, nullptr));

        CHECK_FALSE(equivalent(expr6, expr7));
        CHECK_FALSE(equivalent(e, f));

        CHECK(Expression::equivalent(nullptr, nullptr));
        CHECK_FALSE(equivalent(nullptr, a));
        CHECK_FALSE(equivalent(a, nullptr));

        // Commutative tests
        CHECK_FALSE(identical(a + b, b + a));
        CHECK_FALSE(identical(a - b, b - a));

        CHECK(equivalent(a + b, b + a));
        CHECK_FALSE(equivalent(a - b, b - a));
        CHECK(equivalent(a * b, b * a));
        CHECK_FALSE(equivalent(a / b, b / a));
        CHECK_FALSE(equivalent(a % b, b % a));
        CHECK_FALSE(equivalent(a << b, b << a));
        CHECK_FALSE(equivalent(a >> b, b >> a));
        CHECK(equivalent(a & b, b & a));
        CHECK(equivalent(a | b, b | a));
        CHECK(equivalent(a ^ b, b ^ a));

        // Unallocated
        auto rg = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);

        // Unallocated
        auto rh = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);

        CHECK(Expression::identical(rg->expression(), rg->expression()));
        CHECK_FALSE(Expression::identical(rg->expression(), rh->expression()));

        CHECK(Expression::equivalent(rg->expression(), rg->expression()));
        CHECK_FALSE(Expression::equivalent(rg->expression(), rh->expression()));

        // Null
        Expression::ExpressionPtr n = nullptr;
        CHECK_FALSE(Expression::equivalent(n + n, a + n));
        CHECK_FALSE(Expression::equivalent(n + n, n + a));
    }

    TEST_CASE("Basic expression code generation", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        Register::ValuePtr dest;
        context.get()->schedule(Expression::generate(dest, expr2, context.get()));

        // Explicitly copy the result into another register.
        auto dest2 = dest->placeholder();
        dest2->allocateNow();
        auto regIndexBefore = Generated(dest2->registerIndices())[0];

        context.get()->schedule(Expression::generate(dest2, dest->expression(), context.get()));
        auto regIndexAfter = Generated(dest2->registerIndices())[0];
        CHECK(regIndexBefore == regIndexAfter);

        context.get()->schedule(Expression::generate(dest2, expr2, context.get()));
        regIndexAfter = Generated(dest2->registerIndices())[0];
        CHECK(regIndexBefore == regIndexAfter);

        std::string expected = R"(
            v_add_i32 v2, v0, v1
            v_mul_lo_u32 v3, v1, v2

            // Note that v2 is reused
            v_mov_b32 v2, v3

            // Still storing into v2
            v_add_i32 v4, v0, v1
            v_mul_lo_u32 v2, v1, v4
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(expected));
    }

    TEST_CASE("FMA Expressions", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto rc = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rc->setName("rc");
        rc->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();
        auto c = rc->expression();

        auto expr1 = multiplyAdd(a, b, c);

        auto raf = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        raf->allocateNow();

        auto rbf = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        rbf->allocateNow();

        auto rcf = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        rcf->allocateNow();

        auto af = raf->expression();
        auto bf = rbf->expression();
        auto cf = rcf->expression();

        auto expr2 = multiplyAdd(af, bf, cf);

        Register::ValuePtr dest1, dest2;
        context.get()->schedule(Expression::generate(dest1, expr1, context.get()));
        context.get()->schedule(Expression::generate(dest2, expr2, context.get()));

        std::string expected = R"(
            // Int32: a * x + y doesn't have FMA, so should see multiply then add
            v_mul_lo_u32 v6, v0, v1
            v_add_i32 v6, v6, v2

            // Float: a * x + y has FMA
            v_fma_f32 v7, v3, v4, v5
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(expected));
    }

    TEST_CASE("Expression comments", "[expression][comments][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        setComment(expr1, "The Addition");
        appendComment(expr1, " extra comment");
        setComment(expr2, "The Multiplication");

        CHECK(getComment(expr1) == "The Addition extra comment");

        auto expr3 = simplify(expr2);
        CHECK(getComment(expr3) == "The Multiplication");

        Register::ValuePtr dest;
        context.get()->schedule(Expression::generate(dest, expr2, context.get()));

        std::string expected = R"(
            // Generate {The Multiplication: Multiply(v1:I, {The Addition extra comment: Add(v0:I, v1:I)})} into nullptr
            // BEGIN: The Addition extra comment
            // {The Addition extra comment: Add(v0:I, v1:I)}
            // Allocated : 1 VGPR (Value: Int32): v2
            v_add_i32 v2, v0, v1
            // END: The Addition extra comment
            // BEGIN: The Multiplication
            // {The Multiplication: Multiply(v1:I, {The Addition extra comment: v2:I})}
            // Allocated : 1 VGPR (Value: Int32): v3
            v_mul_lo_u32 v3, v1, v2
            // END: The Multiplication
            // Freeing The Addition extra comment: 1 VGPR (Value: Int32): v2
        )";

        CHECK(NormalizedSource(context.output(), true) == NormalizedSource(expected, true));

        BENCHMARK("Generate expression")
        {
            context.get()->schedule(Expression::generate(dest, expr2, context.get()));
        };
    }

    TEST_CASE("Expression comment exceptions", "[expression][comments][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto a = ra->expression();
        CHECK_THROWS_AS(setComment(a, "The a input"), FatalError);
        CHECK_THROWS_AS(appendComment(a, "extra comment"), FatalError);
        CHECK(getComment(a) == "ra");

        Expression::ExpressionPtr expr1;
        CHECK_THROWS_AS(setComment(expr1, "The first expression"), FatalError);
        CHECK_THROWS_AS(appendComment(expr1, "extra"), FatalError);
        CHECK(getComment(expr1) == "");
    }

    TEST_CASE("Expression generation exceptions", "[expression][comments][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        Register::ValuePtr result;

        context.get()->schedule(context.get()->kernel()->preamble());
        context.get()->schedule(context.get()->kernel()->prolog());

        SECTION("Magic Numbers need generate time operands")
        {
            auto reg = context.get()->kernel()->workitemIndex()[0]->expression();

            auto exp = Expression::magicMultiple(reg);
            CHECK_THROWS(context.get()->schedule(Expression::generate(result, exp, context.get())));

            CHECK_THROWS(context.get()->schedule(
                Expression::generate(result, Expression::magicShifts(reg), context.get())));
            CHECK_THROWS(context.get()->schedule(
                Expression::generate(result, Expression::magicSign(reg), context.get())));
        }

        SECTION("CommandArgument needs the user args")
        {
            CommandArgumentPtr arg;
            auto               argExp = std::make_shared<Expression::Expression>(arg);
            CHECK_THROWS(
                context.get()->schedule(Expression::generate(result, argExp, context.get())));
        }

        SECTION("More complex expressions")
        {
            Register::ValuePtr nullResult;
            auto               unallocResult = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::Int32, 1);
            auto allocResult = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::Int32, 1);
            allocResult->allocateNow();

            auto result = GENERATE_COPY(nullResult, unallocResult, allocResult);
            CAPTURE(result);

            auto unallocated = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::Int32, 1);

            CHECK_THROWS(context.get()->schedule(
                Expression::generate(result, unallocated->expression(), context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);

            CHECK_THROWS(context.get()->schedule(Expression::generate(
                result, unallocated->expression() + Expression::literal(5), context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);

            CHECK_THROWS(context.get()->schedule(Expression::generate(
                result,
                Expression::multiplyHigh(unallocated->expression(), Expression::literal(5)),
                context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);

            CHECK_THROWS(context.get()->schedule(
                Expression::generate(unallocated, unallocated->expression(), context.get())));
            REQUIRE(unallocated->allocationState() == Register::AllocationState::Unallocated);
        }
    }

    TEST_CASE("Matrix Multiply Expressions", "[expression][codegen][mfma]")
    {
        auto context = TestContext::ForDefaultTarget();

        int M       = 32;
        int N       = 32;
        int K       = 2;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * K / 64,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                K * N / 64,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto ic = std::make_shared<Register::Value>(context.get(),
                                                    Register::Type::Accumulator,
                                                    DataType::Float,
                                                    M * N * batches / 64,
                                                    Register::AllocationOptions::FullyContiguous());
        ic->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);
        auto C = ic->expression();

        auto expr = std::make_shared<Expression::Expression>(Expression::MatrixMultiply(A, B, C));

        context.get()->schedule(
            Expression::generate(ic, expr, context.get())); //Test using input C as dest.

        Register::ValuePtr rc;
        context.get()->schedule(
            Expression::generate(rc, expr, context.get())); //Test using a nullptr as dest.

        CHECK(ic->regType() == Register::Type::Accumulator);
        CHECK(ic->valueCount() == 16);

        CHECK(rc->regType() == Register::Type::Accumulator);
        CHECK(rc->valueCount() == 16);

        auto result = R"(
            v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[0:15] //is matmul
            v_mfma_f32_32x32x2f32 a[16:31], v0, v1, a[0:15] //rc matmul
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expressions reuse input vgprs as output vgprs in arithmetic",
              "[expression][codegen][optimization][fp32]")
    {
        auto context = TestContext::ForDefaultTarget();

        int M       = 16;
        int N       = 16;
        int K       = 4;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * K / 64,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                K * N / 64,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto accumD
            = std::make_shared<Register::Value>(context.get(),
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

        context.get()->schedule(
            Expression::generate(accumD, mulABExpr, context.get())); //Test using input D as dest.

        CHECK(accumD->regType() == Register::Type::Accumulator);
        CHECK(accumD->valueCount() == 4);

        auto vecD
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        context.get()->schedule(Expression::generate(vecD, D, context.get()));

        auto scaleDExpr = Expression::literal(2.0f) * vecD->expression();
        context.get()->schedule(Expression::generate(vecD, scaleDExpr, context.get()));

        auto vecC
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        vecC->allocateNow();

        auto addCDExpr = vecC->expression() + vecD->expression();
        context.get()->schedule(Expression::generate(vecD, addCDExpr, context.get()));

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

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expressions reuse input vgprs as output vgprs in arithmetic f16",
              "[expression][codegen][optimization][fp16]")
    {
        auto context = TestContext::ForDefaultTarget();

        int M       = 32;
        int N       = 32;
        int K       = 8;
        int batches = 1;

        auto A_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
        auto B_tile = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();

        A_tile->sizes = {M, K};
        A_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Halfx2,
                                                M * K / 64 / 2,
                                                Register::AllocationOptions::FullyContiguous());
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Halfx2,
                                                K * N / 64 / 2,
                                                Register::AllocationOptions::FullyContiguous());
        B_tile->vgpr->allocateNow();

        auto accumD
            = std::make_shared<Register::Value>(context.get(),
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

        context.get()->schedule(
            Expression::generate(accumD, mulABExpr, context.get())); //Test using input D as dest.

        CHECK(accumD->regType() == Register::Type::Accumulator);
        CHECK(accumD->valueCount() == 16);

        auto vecD
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Float,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());

        auto vecC
            = std::make_shared<Register::Value>(context.get(),
                                                Register::Type::Vector,
                                                DataType::Half,
                                                M * N * batches / 64,
                                                Register::AllocationOptions::FullyContiguous());
        vecC->allocateNow();

        auto scaleDExpr = Expression::literal(2.0, DataType::Half) * vecD->expression();
        auto addCDExpr  = vecC->expression() + vecD->expression();

        context.get()->schedule(Expression::generate(vecD, D, context.get()));
        context.get()->schedule(Expression::generate(vecD, scaleDExpr, context.get()));
        context.get()->schedule(Expression::generate(vecD, addCDExpr, context.get()));

        auto X = std::make_shared<Register::Value>(context.get(),
                                                   Register::Type::Vector,
                                                   DataType::Halfx2,
                                                   M * K / 64 / 2,
                                                   Register::AllocationOptions::FullyContiguous());
        X->allocateNow();

        auto Y = std::make_shared<Register::Value>(context.get(),
                                                   Register::Type::Vector,
                                                   DataType::Half,
                                                   M * K / 64,
                                                   Register::AllocationOptions::FullyContiguous());
        Y->allocateNow();

        auto addXYExpr = X->expression() + Y->expression();
        context.get()->schedule(Expression::generate(Y, addXYExpr, context.get()));

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

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE(
        "Expressions reuse input vgprs as output vgprs in arithmetic f16 with smaller packing",
        "[expression][codegen][optimization][future][fp16]")
    {
        auto context = TestContext::ForDefaultTarget();
        int  M       = 32;
        int  K       = 8;

        auto X = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Halfx2, M * K / 64 / 2);
        X->allocateNow();

        auto Y = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Half, M * K / 64);
        Y->allocateNow();

        // Since we are asking the result to be stored into X, we
        // currently get a failure.

        // TODO See the "Destination/result packing mismatch" assertion
        // in Expression_generate.cpp.
        auto addXYExpr = X->expression() + Y->expression();
        CHECK_THROWS_AS(context.get()->schedule(Expression::generate(X, addXYExpr, context.get())),
                        FatalError);

        // The above should be possible: Y should be packed, and then
        // the v_pk_add_f16 instructions called.
    }

    TEST_CASE("Expression result types", "[expression]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto vgprFloat = Register::Value::Placeholder(
                             context.get(), Register::Type::Vector, DataType::Float, 1)
                             ->expression();
        auto vgprDouble = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::Double, 1)
                              ->expression();
        auto vgprInt32 = Register::Value::Placeholder(
                             context.get(), Register::Type::Vector, DataType::Int32, 1)
                             ->expression();
        auto vgprInt64 = Register::Value::Placeholder(
                             context.get(), Register::Type::Vector, DataType::Int64, 1)
                             ->expression();
        auto vgprUInt32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::UInt32, 1)
                              ->expression();
        auto vgprUInt64 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::UInt64, 1)
                              ->expression();
        auto vgprHalf
            = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Half, 1)
                  ->expression();
        auto vgprHalfx2 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::Halfx2, 1)
                              ->expression();
        auto vgprBool32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Vector, DataType::Bool32, 1)
                              ->expression();
        auto vgprBool
            = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Bool, 1)
                  ->expression();

        auto sgprFloat = Register::Value::Placeholder(
                             context.get(), Register::Type::Scalar, DataType::Float, 1)
                             ->expression();
        auto sgprDouble = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Double, 1)
                              ->expression();
        auto sgprInt32 = Register::Value::Placeholder(
                             context.get(), Register::Type::Scalar, DataType::Int32, 1)
                             ->expression();
        auto sgprInt64 = Register::Value::Placeholder(
                             context.get(), Register::Type::Scalar, DataType::Int64, 1)
                             ->expression();
        auto sgprUInt32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::UInt32, 1)
                              ->expression();
        auto sgprUInt64 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::UInt64, 1)
                              ->expression();
        auto sgprHalf
            = Register::Value::Placeholder(context.get(), Register::Type::Scalar, DataType::Half, 1)
                  ->expression();
        auto sgprHalfx2 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Halfx2, 1)
                              ->expression();
        auto sgprBool64 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Bool64, 1)
                              ->expression();
        auto sgprBool32 = Register::Value::Placeholder(
                              context.get(), Register::Type::Scalar, DataType::Bool32, 1)
                              ->expression();
        auto sgprBool
            = Register::Value::Placeholder(context.get(), Register::Type::Scalar, DataType::Bool, 1)
                  ->expression();
        auto sgprWavefrontSized
            = Register::Value::Placeholder(context.get(),
                                           Register::Type::Scalar,
                                           context.get()->kernel()->wavefront_size() == 64
                                               ? DataType::Bool64
                                               : DataType::Bool32,
                                           1)
                  ->expression();

        auto agprFloat = Register::Value::Placeholder(
                             context.get(), Register::Type::Accumulator, DataType::Float, 1)
                             ->expression();
        auto agprDouble = Register::Value::Placeholder(
                              context.get(), Register::Type::Accumulator, DataType::Double, 1)
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
            context.get()->kernel()->wavefront_size() == 64 ? DataType::Bool64 : DataType::Bool32};

        Expression::ResultType rVCC{Register::Type::VCC, DataType::Bool32};
        Expression::ResultType rSCC{Register::Type::SCC, DataType::Bool};

        Expression::ResultType rAgprFloat{Register::Type::Accumulator, DataType::Float};
        Expression::ResultType rAgprDouble{Register::Type::Accumulator, DataType::Double};

        SECTION("Value expressions")
        {
            CHECK(rSgprInt64 == resultType(sgprInt64));

            CHECK(rVgprInt32 == resultType(vgprInt32));
            CHECK(rVgprInt64 == resultType(vgprInt64));
            CHECK(rVgprFloat == resultType(vgprFloat));
            CHECK(rSgprFloat == resultType(sgprFloat));
            CHECK(rVgprDouble == resultType(vgprDouble));
            CHECK(rSgprDouble == resultType(sgprDouble));
            CHECK(rAgprDouble == resultType(agprDouble));
            CHECK(rAgprDouble == resultType(agprDouble));
        }

        SECTION("Binary expressions")
        {
            CHECK(rVgprInt32 == resultType(vgprInt32 + vgprInt32));
            CHECK(rVgprInt32 == resultType(vgprInt32 + sgprInt32));
            CHECK(rVgprInt32 == resultType(sgprInt32 - vgprInt32));
            CHECK(rSgprInt32 == resultType(sgprInt32 * sgprInt32));

            CHECK(rVgprInt64 == resultType(vgprInt64 + vgprInt32));
            CHECK(rVgprInt64 == resultType(vgprInt32 + vgprInt64));
            CHECK(rVgprInt64 == resultType(vgprInt64 + vgprInt64));

            CHECK(rVgprFloat == resultType(vgprFloat + vgprFloat));
            CHECK(rVgprFloat == resultType(vgprFloat - sgprFloat));
            CHECK(rVgprFloat == resultType(litFloat * vgprFloat));
            CHECK(rVgprFloat == resultType(vgprFloat * litFloat));

            CHECK(rSgprInt32 == resultType(sgprInt32 + sgprInt32));
            CHECK(rSgprInt32 == resultType(sgprInt32 + litInt32));
            CHECK(rSgprInt32 == resultType(litInt32 + sgprInt32));
            CHECK(rSgprInt64 == resultType(litInt32 + sgprInt64));
            CHECK(rSgprInt64 == resultType(sgprInt64 + litInt32));
            CHECK(rSgprInt64 == resultType(sgprInt64 + sgprInt32));

            CHECK(rSgprWavefrontSized == resultType(vgprFloat > vgprFloat));
            CHECK(rSgprWavefrontSized == resultType(sgprFloat < vgprFloat));
            CHECK(rSgprWavefrontSized == resultType(sgprDouble <= vgprDouble));
            CHECK(rSgprWavefrontSized == resultType(sgprInt32 <= vgprInt32));
            CHECK(rSgprWavefrontSized == resultType(litInt32 > vgprInt64));
            CHECK(rSgprBool == resultType(litInt32 <= sgprInt64));
            CHECK(rSgprBool == resultType(sgprInt32 >= litInt32));
        }

        CHECK_THROWS(resultType(sgprDouble <= vgprFloat));
        CHECK_THROWS(resultType(vgprInt32 > vgprFloat));

        SECTION("Arithmetic unary ops")
        {
            // auto ops = ;
            auto op = GENERATE_COPY(
                from_range(std::to_array({Expression::operator-, // cppcheck-suppress syntaxError
                                          Expression::operator~,
                                          Expression::magicMultiple,
                                          Expression::magicSign})));

            CAPTURE(op(vgprFloat));
            CHECK(rVgprFloat == resultType(op(vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32)));
            CHECK(rVgprUInt64 == resultType(op(vgprUInt64)));
            CHECK(rVgprHalf == resultType(op(vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32)));
            CHECK(rSgprUInt64 == resultType(op(sgprUInt64)));
            CHECK(rSgprHalf == resultType(op(sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32)));
        }

        SECTION("Magic shifts")
        {
            auto op = Expression::magicShifts;
            CHECK(rVgprInt32 == resultType(op(vgprFloat)));
            CHECK(rVgprInt32 == resultType(op(vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32)));
            CHECK(rVgprInt32 == resultType(op(vgprInt64)));
            CHECK(rVgprInt32 == resultType(op(vgprUInt32)));
            CHECK(rVgprInt32 == resultType(op(vgprUInt64)));
            CHECK(rVgprInt32 == resultType(op(vgprHalf)));
            CHECK(rVgprInt32 == resultType(op(vgprHalfx2)));
            CHECK(rVgprInt32 == resultType(op(vgprBool32)));

            CHECK(rSgprInt32 == resultType(op(sgprFloat)));
            CHECK(rSgprInt32 == resultType(op(sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32)));
            CHECK(rSgprInt32 == resultType(op(sgprInt64)));
            CHECK(rSgprInt32 == resultType(op(sgprUInt32)));
            CHECK(rSgprInt32 == resultType(op(sgprUInt64)));
            CHECK(rSgprInt32 == resultType(op(sgprHalf)));
            CHECK(rSgprInt32 == resultType(op(sgprHalfx2)));
            CHECK(rSgprInt32 == resultType(op(sgprBool32)));
        }

        SECTION("Comparisons")
        {
            auto op = GENERATE(Expression::operator>,
                               Expression::operator>=,
                               Expression::operator<,
                               Expression::operator<=,
                               Expression::operator==);

            CAPTURE(op(vgprFloat, vgprFloat));

            CHECK(rSgprWavefrontSized == resultType(op(vgprFloat, vgprFloat)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprDouble, vgprDouble)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprInt32, vgprInt32)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprInt64, vgprInt64)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprUInt32, vgprUInt32)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprUInt64, vgprUInt64)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprHalf, vgprHalf)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprBool32, vgprBool32)));
            CHECK(rSgprWavefrontSized == resultType(op(vgprBool, vgprBool)));

            CHECK(rSgprBool == resultType(op(sgprFloat, sgprFloat)));
            CHECK(rSgprBool == resultType(op(sgprDouble, sgprDouble)));
            CHECK(rSgprBool == resultType(op(sgprInt32, sgprInt32)));
            CHECK(rSgprBool == resultType(op(sgprInt64, sgprInt64)));
            CHECK(rSgprBool == resultType(op(sgprUInt32, sgprUInt32)));
            CHECK(rSgprBool == resultType(op(sgprUInt64, sgprUInt64)));
            CHECK(rSgprBool == resultType(op(sgprHalf, sgprHalf)));
            CHECK(rSgprBool == resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool == resultType(op(sgprBool32, sgprBool32)));
            CHECK(rSgprBool == resultType(op(sgprBool, sgprBool)));
        }

        SECTION("Arithmetic binary")
        {
            auto op = GENERATE(from_range(std::to_array({Expression::operator+,
                                                         Expression::operator-,
                                                         Expression::operator*,
                                                         Expression::operator/,
                                                         Expression::operator%,
                                                         Expression::operator<<,
                                                         Expression::operator>>,
                                                         Expression::operator&,
                                                         Expression::arithmeticShiftR})));

            CAPTURE(op(vgprFloat, vgprFloat));

            CHECK(rVgprFloat == resultType(op(vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32, vgprUInt32)));
            CHECK(rVgprUInt64 == resultType(op(vgprUInt64, vgprUInt64)));
            CHECK(rVgprHalf == resultType(op(vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32, vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32, sgprUInt32)));
            CHECK(rSgprUInt64 == resultType(op(sgprUInt64, sgprUInt64)));
            CHECK(rSgprHalf == resultType(op(sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool32)));
        }

        SECTION("Logical")
        {
            auto op = GENERATE(Expression::operator&&, Expression::operator||);

            CAPTURE(op(sgprBool64, sgprBool64));

            CHECK_THROWS(resultType(op(vgprFloat, vgprFloat)));
            CHECK_THROWS(resultType(op(vgprDouble, vgprDouble)));
            CHECK_THROWS(resultType(op(vgprInt32, vgprInt32)));
            CHECK_THROWS(resultType(op(vgprInt64, vgprInt64)));
            CHECK_THROWS(resultType(op(vgprUInt32, vgprUInt32)));
            CHECK_THROWS(resultType(op(vgprUInt64, vgprUInt64)));
            CHECK_THROWS(resultType(op(vgprHalf, vgprHalf)));
            CHECK_THROWS(resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK_THROWS(resultType(op(vgprBool32, vgprBool32)));
            CHECK_THROWS(resultType(op(vgprBool, vgprBool)));

            CHECK_THROWS(resultType(op(sgprFloat, sgprFloat)));
            CHECK_THROWS(resultType(op(sgprDouble, sgprDouble)));
            CHECK_THROWS(resultType(op(sgprInt32, sgprInt32)));
            CHECK_THROWS(resultType(op(sgprInt64, sgprInt64)));
            CHECK_THROWS(resultType(op(sgprUInt32, sgprUInt32)));

            CHECK(rSgprBool == resultType(op(sgprBool64, sgprBool64)));
            CHECK_THROWS(resultType(op(sgprHalf, sgprHalf)));
            CHECK_THROWS(resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool == resultType(op(sgprBool32, sgprBool32)));
            CHECK(rSgprBool == resultType(op(sgprBool, sgprBool)));
        }

        SECTION("Bitwise binary")
        {

            auto op = GENERATE(from_range(std::to_array({Expression::operator<<,
                                    Expression::logicalShiftR,
                                    Expression::operator&,
                                    Expression::operator^,
                                    Expression::operator|})));

            CAPTURE(op(vgprFloat, vgprFloat));

            CHECK(rVgprFloat == resultType(op(vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32, vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32, sgprUInt32)));
            CHECK(rSgprHalf == resultType(op(sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool32)));
        }

        SECTION("Arithmetic ternary")
        {
            auto op = GENERATE_COPY(
                Expression::multiplyAdd, Expression::addShiftL, Expression::shiftLAdd);

            CAPTURE(op(vgprFloat, vgprFloat, vgprFloat));

            CHECK(rVgprFloat == resultType(op(vgprFloat, vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprDouble, vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprInt32, vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprInt64, vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprUInt32, vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprHalf, vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprHalfx2, vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool32, vgprBool32, vgprBool32)));
            CHECK(rSgprFloat == resultType(op(sgprFloat, sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprDouble, sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprInt32, sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprInt64, sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprUInt32, sgprUInt32, sgprUInt32)));
            CHECK(rSgprHalf == resultType(op(sgprHalf, sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprHalfx2, sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool32, sgprBool32, sgprBool32)));
        }

        SECTION("Conditional")
        {
            auto op = Expression::conditional;
            CHECK(rVgprFloat == resultType(op(sgprBool, vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(sgprBool, vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(sgprBool, vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(sgprBool, vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(sgprBool, vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(sgprBool, vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(sgprBool, vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(sgprBool, vgprBool32, vgprBool32)));

            CHECK(rVgprFloat == resultType(op(vgprBool, vgprFloat, vgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprBool, vgprDouble, vgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprBool, vgprInt32, vgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprBool, vgprInt64, vgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprBool, vgprUInt32, vgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprBool, vgprHalf, vgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprBool, vgprHalfx2, vgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool, vgprBool32, vgprBool32)));

            CHECK(rSgprFloat == resultType(op(sgprBool, sgprFloat, sgprFloat)));
            CHECK(rSgprDouble == resultType(op(sgprBool, sgprDouble, sgprDouble)));
            CHECK(rSgprInt32 == resultType(op(sgprBool, sgprInt32, sgprInt32)));
            CHECK(rSgprInt64 == resultType(op(sgprBool, sgprInt64, sgprInt64)));
            CHECK(rSgprUInt32 == resultType(op(sgprBool, sgprUInt32, sgprUInt32)));
            CHECK(rSgprHalf == resultType(op(sgprBool, sgprHalf, sgprHalf)));
            CHECK(rSgprHalfx2 == resultType(op(sgprBool, sgprHalfx2, sgprHalfx2)));
            CHECK(rSgprBool32 == resultType(op(sgprBool, sgprBool32, sgprBool32)));

            CHECK(rVgprFloat == resultType(op(vgprBool, sgprFloat, sgprFloat)));
            CHECK(rVgprDouble == resultType(op(vgprBool, sgprDouble, sgprDouble)));
            CHECK(rVgprInt32 == resultType(op(vgprBool, sgprInt32, sgprInt32)));
            CHECK(rVgprInt64 == resultType(op(vgprBool, sgprInt64, sgprInt64)));
            CHECK(rVgprUInt32 == resultType(op(vgprBool, sgprUInt32, sgprUInt32)));
            CHECK(rVgprHalf == resultType(op(vgprBool, sgprHalf, sgprHalf)));
            CHECK(rVgprHalfx2 == resultType(op(vgprBool, sgprHalfx2, sgprHalfx2)));
            CHECK(rVgprBool32 == resultType(op(vgprBool, sgprBool32, sgprBool32)));
        }
    }

    TEST_CASE("Expression evaluate", "[expression]")
    {
        SECTION("No arguments")
        {
            auto a = std::make_shared<Expression::Expression>(1.0);
            auto b = std::make_shared<Expression::Expression>(2.0);

            auto expr1 = a + b;
            auto expr2 = b * expr1;

            auto expectedTimes = Expression::EvaluationTimes::All();
            CHECK(expectedTimes == Expression::evaluationTimes(expr2));

            CHECK(Expression::canEvaluateTo(3.0, expr1));
            CHECK(Expression::canEvaluateTo(6.0, expr2));
            CHECK(3.0 == std::get<double>(Expression::evaluate(expr1)));
            CHECK(6.0 == std::get<double>(Expression::evaluate(expr2)));
        }

        SECTION("Arguments")
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
            CHECK(expected == resultType(expr2));
            CHECK(6.0 == std::get<double>(Expression::evaluate(expr2, runtimeArgs)));

            args.a = 2.0;
            CHECK(8.0 == std::get<double>(Expression::evaluate(expr2, runtimeArgs)));
            CHECK(-8.0 == std::get<double>(Expression::evaluate(expr3, runtimeArgs)));

            args.b = 1.5;
            CHECK(5.25 == std::get<double>(Expression::evaluate(expr2, runtimeArgs)));

            CHECK_FALSE(Expression::canEvaluateTo(5.25, expr2));
            // Don't send in the runtimeArgs, can't evaluate the arguments.
            CHECK_THROWS_AS(Expression::evaluate(expr2), std::runtime_error);

            Expression::EvaluationTimes expectedTimes{Expression::EvaluationTime::KernelLaunch};
            CHECK(expectedTimes == Expression::evaluationTimes(expr2));
        }
    }

    TEST_CASE("Expression test evaluate mixed types", "[expression]")
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

        CHECK(6.0 == std::get<double>(Expression::evaluate(exprSix)));
        CHECK(1.0 == std::get<double>(Expression::evaluate(exprOne)));
        CHECK(7.0f == std::get<float>(Expression::evaluate(exprSeven)));

        auto twoDouble = convert(DataType::Double, two);
        CHECK(2.0 == std::get<double>(Expression::evaluate(twoDouble)));

        auto twoInt = convert(DataType::Int32, twoPoint5);
        CHECK(2 == std::get<int>(Expression::evaluate(twoInt)));

        auto fiveDouble = seven - twoInt;
        CHECK(5.0 == std::get<double>(Expression::evaluate(fiveDouble)));

        auto minusThree64 = convert(DataType::Int64, twoInt - five);
        CHECK(-3l == std::get<int64_t>(Expression::evaluate(minusThree64)));

        auto minusThreeU64 = convert(DataType::UInt64, twoInt - five);
        CHECK(18446744073709551613ul == std::get<uint64_t>(Expression::evaluate(minusThreeU64)));

        auto eight75Half = convert(DataType::Half, eightPoint75);
        CHECK(Half(8.75) == std::get<Half>(evaluate(eight75Half)));

        Expression::ResultType litDouble{Register::Type::Literal, DataType::Double};
        Expression::ResultType litFloat{Register::Type::Literal, DataType::Float};
        Expression::ResultType litBool{Register::Type::Literal, DataType::Bool};

        CHECK(litDouble == resultType(exprSix));
        // Result type not (yet?) defined for mixed integral/floating point types.
        CHECK_THROWS(resultType(exprOne));
        CHECK_THROWS(resultType(exprSeven));

        CHECK(true == std::get<bool>(Expression::evaluate(exprSix > exprOne)));
        CHECK(true == std::get<bool>(Expression::evaluate(exprSix >= exprOne)));
        CHECK(false == std::get<bool>(Expression::evaluate(exprSix < exprOne)));
        CHECK(false == std::get<bool>(Expression::evaluate(exprSix <= exprOne)));
        CHECK(true == std::get<bool>(Expression::evaluate(exprSix != exprOne)));

        CHECK_THROWS(resultType(exprSix > exprOne));
        CHECK_THROWS(resultType(exprSix >= exprOne));
        CHECK_THROWS(resultType(exprSix < exprOne));
        CHECK_THROWS(resultType(exprSix <= exprOne));
        CHECK(litBool == resultType(one > seven));

        CHECK(true == std::get<bool>(Expression::evaluate(exprSix < exprSeven)));
        CHECK(true == std::get<bool>(Expression::evaluate(exprSix <= exprSeven)));
        CHECK(false == std::get<bool>(Expression::evaluate(exprSix > exprSeven)));
        CHECK(false == std::get<bool>(Expression::evaluate(exprSix >= exprSeven)));

        CHECK(true == std::get<bool>(Expression::evaluate(one <= exprOne)));
        CHECK(true == std::get<bool>(Expression::evaluate(one == exprOne)));
        CHECK(true == std::get<bool>(Expression::evaluate(one >= exprOne)));
        CHECK(false == std::get<bool>(Expression::evaluate(one != exprOne)));

        auto trueExp = std::make_shared<Expression::Expression>(true);
        CHECK(true == std::get<bool>(Expression::evaluate(trueExp == (one >= exprOne))));
        CHECK(false == std::get<bool>(Expression::evaluate(trueExp == (one < exprOne))));

        // Pointer + double -> error.
        {
            auto exprThrow = ptrValid + exprOne;
            CHECK_THROWS_AS(Expression::evaluate(exprThrow), std::runtime_error);
            CHECK_THROWS(resultType(exprThrow));
        }

        // Pointer * int -> error.
        {
            auto exprThrow = ptrValid * five;
            CHECK_THROWS_AS(Expression::evaluate(exprThrow), std::runtime_error);
        }

        // Pointer + pointer -> error
        {
            auto exprThrow = ptrValid + ptrDoubleValid;
            CHECK_THROWS_AS(Expression::evaluate(exprThrow), std::runtime_error);
            CHECK_THROWS(resultType(exprThrow));
        }

        // (float *) -  (double *) -> error
        {
            auto exprThrow = ptrValid - ptrDoubleValid;
            CHECK_THROWS_AS(Expression::evaluate(exprThrow), std::runtime_error);
            CHECK_THROWS(resultType(exprThrow));
        }

        {
            auto exprThrow = ptrNull + five;
            // nullptr + int -> error;
            CHECK_THROWS_AS(Expression::evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = -ptrNull;
            // -pointer -> error;
            CHECK_THROWS_AS(Expression::evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = five + ptrNull;
            // Nullptr + int -> error;
            CHECK_THROWS_AS(Expression::evaluate(exprThrow), std::runtime_error);
        }

        auto   exprXPlus5          = ptrValid + five;
        float* dontDereferenceThis = std::get<float*>(Expression::evaluate(exprXPlus5));
        auto   ptrDifference       = dontDereferenceThis - (&x);
        CHECK(5 == ptrDifference);

        auto expr10PlusX    = five + exprXPlus5;
        dontDereferenceThis = std::get<float*>(Expression::evaluate(expr10PlusX));
        ptrDifference       = dontDereferenceThis - (&x);
        CHECK(10 == ptrDifference);

        auto expr5PtrDiff = expr10PlusX - exprXPlus5;
        CHECK(5 == std::get<int64_t>(Expression::evaluate(expr5PtrDiff)));

        CHECK(true == std::get<bool>(Expression::evaluate(expr10PlusX > ptrValid)));
        CHECK(false == std::get<bool>(Expression::evaluate(expr10PlusX < ptrValid)));
    }

    TEST_CASE("Expression equality", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * a;
        auto expr3 = expr1 == expr2;

        Register::ValuePtr destReg;
        context.get()->schedule(Expression::generate(destReg, expr3, context.get()));

        auto result = R"(
            v_add_i32 v2, v0, v1
            v_mul_lo_u32 v3, v1, v0
            v_cmp_eq_i32 s[0:1], v2, v3
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expression evaluate comparisons", "[expression]")
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

        auto aVal = GENERATE(from_range(TestValues::doubleValues));
        auto bVal = GENERATE(from_range(TestValues::doubleValues));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<bool>(Expression::evaluate(vals_gt, args)) == (aVal > bVal));
        CHECK(std::get<bool>(Expression::evaluate(vals_lt, args)) == (aVal < bVal));
        CHECK(std::get<bool>(Expression::evaluate(vals_ge, args)) == (aVal >= bVal));
        CHECK(std::get<bool>(Expression::evaluate(vals_le, args)) == (aVal <= bVal));
        CHECK(std::get<bool>(Expression::evaluate(vals_eq, args)) == (aVal == bVal));

        CHECK(std::get<bool>(Expression::evaluate(expr_gt, args)) == (aVal > (aVal + bVal)));
        CHECK(std::get<bool>(Expression::evaluate(expr_lt, args)) == (aVal < (aVal + bVal)));
        CHECK(std::get<bool>(Expression::evaluate(expr_ge, args)) == (aVal >= (aVal + bVal)));
        CHECK(std::get<bool>(Expression::evaluate(expr_le, args)) == (aVal <= (aVal + bVal)));
        CHECK(std::get<bool>(Expression::evaluate(expr_eq, args)) == (aVal == (aVal + bVal)));
    }

    TEST_CASE("Expression evaluate logical", "[expression]")
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

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::int32Values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<bool>(Expression::evaluate(vals_negate, args)) == (!aVal));
        CHECK(std::get<bool>(Expression::evaluate(vals_double_negate, args)) == (!!aVal));
        CHECK(std::get<bool>(Expression::evaluate(vals_and, args)) == (aVal && bVal));
        CHECK(std::get<bool>(Expression::evaluate(vals_or, args)) == (aVal || bVal));
    }

    TEST_CASE("Expression evaluate shifts", "[expression]")
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

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::shiftValues));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<int>(Expression::evaluate(vals_shiftL, args)) == (aVal << bVal));

        CHECK(std::get<int>(Expression::evaluate(vals_shiftR, args))
              == (static_cast<unsigned int>(aVal) >> bVal));
        CHECK(std::get<int>(Expression::evaluate(vals_signedShiftR, args)) == (aVal >> bVal));

        CHECK(std::get<int>(Expression::evaluate(expr_shiftL, args)) == ((aVal + bVal) << bVal));
        CHECK(std::get<int>(Expression::evaluate(expr_shiftR, args))
              == (static_cast<unsigned int>(aVal + bVal) >> bVal));
        CHECK(std::get<int>(Expression::evaluate(expr_signedShiftR, args))
              == ((aVal + bVal) >> bVal));
    }

    TEST_CASE("Expression evaluate conditional operator", "[expression]")
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

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::int32Values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        // At kernel launch time
        CHECK(std::get<int>(Expression::evaluate(vals_shiftL, args))
              == (aVal >= bVal ? aVal : bVal));

        // At translate time
        auto a_static = std::make_shared<Expression::Expression>(aVal);
        auto b_static = std::make_shared<Expression::Expression>(bVal);
        CHECK(std::get<int>(
                  Expression::evaluate(conditional(a_static >= b_static, a_static, b_static)))
              == (aVal >= bVal ? aVal : bVal));
    }

    TEST_CASE("Expression evaluate bitwise ops", "[expression]")
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

        auto aVal = GENERATE(from_range(TestValues::int32Values));
        auto bVal = GENERATE(from_range(TestValues::int32Values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<int>(Expression::evaluate(vals_and, args)) == (aVal & bVal));
        CHECK(std::get<int>(Expression::evaluate(vals_or, args)) == (aVal | bVal));
        CHECK(std::get<int>(Expression::evaluate(vals_negate, args)) == (~aVal));

        CHECK(std::get<int>(Expression::evaluate(expr_and, args)) == ((aVal + bVal) & bVal));
        CHECK(std::get<int>(Expression::evaluate(expr_or, args)) == ((aVal + bVal) | bVal));
    }

    TEST_CASE("Expression evaluate multiplyHigh signed", "[expression]")
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

        auto aVal = GENERATE_COPY(from_range(a_values));
        auto bVal = GENERATE_COPY(from_range(a_values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<int>(Expression::evaluate(expr1, args)) == ((aVal * (int64_t)bVal) >> 32));

        CHECK(std::get<int>(Expression::evaluate(expr2, args))
              == (((aVal + bVal) * (int64_t)bVal) >> 32));
    }

    TEST_CASE("Expression evaluate multiplyHigh unsigned", "[expression]")
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
        auto aVal = GENERATE_COPY(from_range(a_values));
        auto bVal = GENERATE_COPY(from_range(a_values));

        CAPTURE(aVal, bVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        runtimeArgs.append("b", bVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::get<unsigned int>(Expression::evaluate(expr1, args))
              == ((aVal * (uint64_t)bVal) >> 32));

        CHECK(std::get<unsigned int>(Expression::evaluate(expr2, args))
              == (((aVal + (uint64_t)bVal) * (uint64_t)bVal) >> 32));
    }

    TEST_CASE("Expression evaluate exp2", "[expression]")
    {
        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto ca      = command->allocateArgument(
            {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value);

        auto a = std::make_shared<Expression::Expression>(ca);

        auto expr = exp2(a);

        auto aVal = GENERATE(from_range(TestValues::floatValues));
        CAPTURE(aVal);

        KernelArguments runtimeArgs;
        runtimeArgs.append("a", aVal);
        auto args = runtimeArgs.runtimeArguments();

        CHECK(std::exp2(aVal) == std::get<float>(Expression::evaluate(expr, args)));
    }

    TEST_CASE("Expression evaluate convert expressions", "[expression]")
    {
        using namespace Expression;

        float    a = 1.25f;
        Half     b = 1.1111;
        double   c = 5.2619;
        BFloat16 d(1.0f);

        auto a_exp = literal(a);
        auto b_exp = literal(b);
        auto c_exp = literal(c);
        auto d_exp = literal(d);

        auto exp1 = convert<DataType::Half>(a_exp);
        auto exp2 = convert<DataType::Half>(b_exp);
        auto exp3 = convert<DataType::Half>(c_exp);

        CHECK(resultVariableType(exp1).dataType == DataType::Half);
        CHECK(resultVariableType(exp2).dataType == DataType::Half);
        CHECK(resultVariableType(exp3).dataType == DataType::Half);

        CHECK(std::get<Half>(evaluate(exp1)) == static_cast<Half>(a));
        CHECK(std::get<Half>(evaluate(exp2)) == b);
        CHECK(std::get<Half>(evaluate(exp3)) == static_cast<Half>(c));

        auto exp4 = convert<DataType::Float>(a_exp);
        auto exp5 = convert<DataType::Float>(b_exp);
        auto exp6 = convert<DataType::Float>(c_exp);
        auto exp7 = convert<DataType::Float>(d_exp);

        CHECK(resultVariableType(exp4).dataType == DataType::Float);
        CHECK(resultVariableType(exp5).dataType == DataType::Float);
        CHECK(resultVariableType(exp6).dataType == DataType::Float);
        CHECK(resultVariableType(exp7).dataType == DataType::Float);

        CHECK(std::get<float>(evaluate(exp4)) == a);
        CHECK(std::get<float>(evaluate(exp5)) == static_cast<float>(b));
        CHECK(std::get<float>(evaluate(exp6)) == static_cast<float>(c));
        CHECK(std::get<float>(evaluate(exp7)) == static_cast<float>(d));
    }

    TEST_CASE("Expression generate dataflow tags", "[expression][codegen]")
    {
        auto context = TestContext::ForDefaultTarget();

        Register::AllocationOptions allocOptions{.contiguousChunkWidth
                                                 = Register::FULLY_CONTIGUOUS};

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 4, allocOptions);
        ra->allocateNow();
        auto dfa = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{1, Register::Type::Vector, DataType::None});
        context.get()->registerTagManager()->addRegister(1, ra);

        auto rb = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 4, allocOptions);
        rb->allocateNow();
        auto dfb = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{2, Register::Type::Vector, DataType::None});
        context.get()->registerTagManager()->addRegister(2, rb);

        auto rc = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 4, allocOptions);
        rc->allocateNow();
        auto dfc = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{3, Register::Type::Vector, DataType::None});
        context.get()->registerTagManager()->addRegister(3, rc);

        Register::ValuePtr rr1;
        context.get()->schedule(Expression::generate(rr1, dfa * dfb, context.get()));

        Register::ValuePtr rr2;
        context.get()->schedule(
            Expression::generate(rr2, Expression::fuseTernary(dfa * dfb + dfc), context.get()));

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

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expression literal datatypes", "[expression]")
    {
        std::vector<VariableType> dataTypes = {{DataType::Int32},
                                               {DataType::UInt32},
                                               {DataType::Int64},
                                               {DataType::UInt64},
                                               {DataType::Float},
                                               {DataType::Half},
                                               {DataType::Double},
                                               {DataType::Bool}};

        auto dataType = GENERATE_COPY(from_range(dataTypes));

        CAPTURE(dataType);
        CHECK(dataType == Expression::resultVariableType(Expression::literal(1, dataType)));
    }

    TEST_CASE("Expression codegen literal int swap", "[expression][codegen][optimization]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto ra = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto expr1 = ra->expression();
        auto expr2 = Expression::literal(-5);

        Register::ValuePtr destReg;

        context.get()->schedule(Expression::generate(destReg, expr1 + expr2, context.get()));

        context.get()->schedule(Expression::generate(destReg, expr1 & expr2, context.get()));
        context.get()->schedule(Expression::generate(destReg, expr1 | expr2, context.get()));
        context.get()->schedule(Expression::generate(destReg, expr1 ^ expr2, context.get()));

        auto result = R"(
            v_add_i32 v1, -5, v0
            v_and_b32 v1, -5, v0
            v_or_b32 v1, -5, v0
            v_xor_b32 v1, -5, v0
        )";

        CHECK(NormalizedSource(context.output()) == NormalizedSource(result));
    }

    TEST_CASE("Expression variant test", "[expression]")
    {
        auto context = TestContext::ForDefaultTarget();

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

        auto v_a = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Double, 1);
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
            CAPTURE(expr);
            CHECK_NOTHROW(Expression::toString(expr));
            CHECK_NOTHROW(Expression::evaluationTimes(expr));
        };

        for(auto const& expr : exprs)
        {
            testFunc(expr);
            CHECK_NOTHROW(Expression::evaluate(expr));
            CHECK_NOTHROW(Expression::fastDivision(expr, context.get()));
        }

        testFunc(v_a);
        CHECK_THROWS_AS(Expression::evaluate(v_a), FatalError);

        testFunc(tag);
        CHECK_THROWS_AS(Expression::evaluate(tag), FatalError);

        testFunc(tagPtr);
        CHECK_THROWS_AS(Expression::evaluate(tagPtr), FatalError);
        CHECK_NOTHROW(Expression::fastDivision(tagPtr, context.get()));

        testFunc(value);
        CHECK_NOTHROW(Expression::evaluate(value));

        testFunc(waveTile);
        CHECK_THROWS_AS(Expression::evaluate(waveTile), FatalError);

        testFunc(waveTilePtr);
        CHECK_THROWS_AS(Expression::evaluate(waveTilePtr), FatalError);
        CHECK_NOTHROW(Expression::fastDivision(waveTilePtr, context.get()));

        CHECK_NOTHROW(Expression::convert(DataType::Float, intExpr));
        CHECK_NOTHROW(Expression::convert(DataType::Double, intExpr));
        CHECK_THROWS_AS(Expression::convert(DataType::ComplexFloat, intExpr), FatalError);
        CHECK_THROWS_AS(Expression::convert(DataType::ComplexDouble, intExpr), FatalError);
        CHECK_NOTHROW(Expression::convert(DataType::Half, intExpr));
        CHECK_NOTHROW(Expression::convert(DataType::Halfx2, intExpr));
        CHECK_THROWS_AS(Expression::convert(DataType::Int8x4, intExpr), FatalError);
        CHECK_NOTHROW(Expression::convert(DataType::Int32, intExpr));
        CHECK_NOTHROW(Expression::convert(DataType::Int64, intExpr));
        CHECK_NOTHROW(Expression::convert(DataType::BFloat16, intExpr));
        CHECK_NOTHROW(Expression::convert(DataType::BFloat16x2, intExpr));
        CHECK_THROWS_AS(Expression::convert(DataType::Int8, intExpr), FatalError);
        CHECK_THROWS_AS(Expression::convert(DataType::Raw32, intExpr), FatalError);
        CHECK_NOTHROW(Expression::convert(DataType::UInt32, intExpr));
        CHECK_NOTHROW(Expression::convert(DataType::UInt64, intExpr));
        CHECK_THROWS_AS(Expression::convert(DataType::Bool, intExpr), FatalError);
        CHECK_THROWS_AS(Expression::convert(DataType::Bool32, intExpr), FatalError);
        CHECK_THROWS_AS(Expression::convert(DataType::Count, intExpr), FatalError);
        CHECK_THROWS_AS(Expression::convert(static_cast<DataType>(200), intExpr), FatalError);
    }

    TEST_CASE("Expression complexity values", "[expression][optimization]")
    {
        auto intExpr = Expression::literal(1);

        CHECK(Expression::complexity(intExpr) == 0);
        CHECK(Expression::complexity(intExpr + intExpr) > Expression::complexity(intExpr));
        CHECK(Expression::complexity(intExpr + intExpr + intExpr)
              > Expression::complexity(intExpr + intExpr));

        CHECK(Expression::complexity(intExpr / intExpr)
              > Expression::complexity(intExpr + intExpr));
    }

}
