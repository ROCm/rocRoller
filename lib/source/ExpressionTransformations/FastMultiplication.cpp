#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>

#include <bit>

template <typename T>
constexpr auto cast_to_unsigned(T val)
{
    return static_cast<typename std::make_unsigned<T>::type>(val);
}

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Fast Multiplication
         *
         * Attempt to replace division operations found within an expression with faster
         * operations.
         */

        struct MultiplicationByConstant
        {
            ExpressionPtr m_lhs;

            // Fast Multiplication for when rhs is power of two
            template <typename T>
            requires(std::integral<T> && !std::same_as<bool, T>) ExpressionPtr operator()(T rhs)
            {
                if(rhs == 0)
                {
                    return literal(0);
                }
                else if(rhs == 1)
                {
                    return m_lhs;
                }
                // Power of 2 Multiplication
                else if(rhs > 0 && std::has_single_bit(cast_to_unsigned(rhs)))
                {
                    auto rhs_exp
                        = literal(cast_to_unsigned(std::countr_zero(cast_to_unsigned(rhs))));

                    return m_lhs << rhs_exp;
                }

                return nullptr;
            }

            // If the rhs is not an integer, return a nullptr to indicate we can't optimize.
            template <typename T>
            requires(!std::integral<T> || std::same_as<bool, T>) ExpressionPtr operator()(T rhs)
            {
                return nullptr;
            }

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct FastMultiplicationExpressionVisitor
        {
            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(
                    Expr({(*this)(expr.lhs), (*this)(expr.r1hs), (*this)(expr.r2hs)}));
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(Expr({(*this)(expr.lhs), (*this)(expr.rhs)}));
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(Expr({(*this)(expr.arg)}));
            }

            ExpressionPtr operator()(Multiply const& expr) const
            {
                auto lhs = (*this)(expr.lhs);
                auto rhs = (*this)(expr.rhs);

                if(evaluationTimes(rhs)[EvaluationTime::Translate])
                {
                    auto rhs_val = evaluate(rhs);
                    auto rv      = MultiplicationByConstant()(lhs, rhs_val);
                    if(rv != nullptr)
                        return rv;
                }

                if(evaluationTimes(lhs)[EvaluationTime::Translate])
                {
                    auto lhs_val = evaluate(lhs);
                    // lhs becomes rhs because visitor checks rhs for optimization
                    auto rv = MultiplicationByConstant()(rhs, lhs_val);
                    if(rv != nullptr)
                        return rv;
                }

                return std::make_shared<Expression>(Multiply({lhs, rhs}));
            }

            ExpressionPtr operator()(MatrixMultiply const& expr) const
            {
                return std::make_shared<Expression>(MatrixMultiply(expr));
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

        /**
         * Attempts to use fastMultiplication for all of the multiplications within an Expression.
         */
        ExpressionPtr fastMultiplication(ExpressionPtr expr)
        {
            auto visitor = FastMultiplicationExpressionVisitor();
            return visitor(expr);
        }

    }
}
