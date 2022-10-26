
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include "GenericContextFixture.hpp"

class ExpressionTransformationTest : public GenericContextFixture
{
};

using namespace rocRoller;

TEST_F(ExpressionTransformationTest, Simplify)
{
    auto r = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    // multiply
    EXPECT_EQ(Expression::toString(simplify(zero * one)), "0i");
    EXPECT_EQ(Expression::toString(simplify(c * zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(c * one)), "12.0000f");
    EXPECT_EQ(Expression::toString(simplify(v * zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v * zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(a * b)), "3300i");

    // add
    EXPECT_EQ(Expression::toString(simplify(zero + one)), "1i");
    EXPECT_EQ(Expression::toString(simplify(zero + one)), "1i");
    EXPECT_EQ(Expression::toString(simplify(v + zero)), "v0:I");
    EXPECT_EQ(Expression::toString(simplify(a + b)), "133i");

    // divide
    EXPECT_EQ(Expression::toString(simplify(a / one)), "33i");
    EXPECT_EQ(Expression::toString(simplify(v / one)), "v0:I");
    EXPECT_EQ(Expression::toString(simplify(v / a)), "Divide(v0:I, 33i)");
    EXPECT_EQ(Expression::toString(simplify(b / v)), "Divide(100i, v0:I)");

    // mod
    EXPECT_EQ(Expression::toString(simplify(a % one)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v % one)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v % a)), "Modulo(v0:I, 33i)");
    EXPECT_EQ(Expression::toString(simplify(b % v)), "Modulo(100i, v0:I)");

    // bitwiseAnd
    EXPECT_EQ(Expression::toString(simplify(one & b)), "0i");
    EXPECT_EQ(Expression::toString(simplify(one & a)), "1i");
    EXPECT_EQ(Expression::toString(simplify(v & zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v & (zero + zero))), "0i");

    // shiftL
    EXPECT_EQ(Expression::toString(simplify(one << zero)), "1i");
}

TEST_F(ExpressionTransformationTest, FuseAssociative)
{
    auto r = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    EXPECT_EQ(Expression::toString(fuseAssociative(v & one)), "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative((v & one) & a)), "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative(simplify((one & a) & v))),
              "BitwiseAnd(1i, v0:I)");
    EXPECT_EQ(Expression::toString(fuseAssociative(a & (v & one))), "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative((one & v) & a)), "BitwiseAnd(1i, v0:I)");
    EXPECT_EQ(Expression::toString(fuseAssociative(a & (one & v))), "BitwiseAnd(1i, v0:I)");
    EXPECT_EQ(Expression::toString(fuseAssociative(((((v & one) & a) & a) & a) & a)),
              "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative(((((v & one) & a) + a) & a) & one)),
              "BitwiseAnd(Add(BitwiseAnd(v0:I, 1i), 33i), 1i)");

    EXPECT_EQ(Expression::toString(fuseAssociative((v + one) + a)), "Add(v0:I, 34i)");

    EXPECT_EQ(
        Expression::toString(fuseAssociative((v - one) - a)),
        "Subtract(Subtract(v0:I, 1i), 33i)"); // fuseAssociative does not affect non-associative ops
}
