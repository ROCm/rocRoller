#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

template <typename T>
constexpr auto cast_to_unsigned(T val)
{
    return static_cast<typename std::make_unsigned<T>::type>(val);
}

namespace rocRoller
{
    namespace Expression
    {
        struct WidenTo64BitVisitor
        {

            ExpressionPtr operator()(Convert const& expr) const
            {
                Convert cpy = expr;
                if(expr.arg)
                {
                    // Here is an assumption that call(cpy.arg) never goes above 64-bit and
                    // input convert's destination types are either int32, uint32, int64 or uint64.
                    // resultVaribleType(expr.arg) is not called intentionally as it will visit the
                    // subtree of expr.arg again.
                    // We can also import similar logic from ExpressionResultTypeVisitor and
                    // make the operator()(...) return a pair of ExpressionPtr and VariableType
                    // in order to avoid repeated visit of the subtree of cpy.arg.
                    cpy.arg = call(expr.arg);
                    if(expr.destinationType == DataType::UInt32)
                        return convert(DataType::UInt64, cpy.arg);
                    else if(expr.destinationType == DataType::Int32)
                        return convert(DataType::Int64, cpy.arg);
                    else if(expr.destinationType == DataType::UInt64
                            || expr.destinationType == DataType::Int64)
                        return convert(expr.destinationType, cpy.arg);
                    else
                    {
                        AssertFatal(
                            false,
                            "Expected Destination type for a convert is either (u)int{32,64}");
                        return nullptr;
                    }
                }

                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Negate const& expr) const
            {
                Negate cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }

                return std::make_shared<Expression>(cpy);
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                // Only Convert and perhaps negates are expected.
                AssertFatal(false, "Unexpected Unary expression", ShowValue(expr));
                return nullptr;
            }

            ExpressionPtr operator()(Divide const& expr) const
            {
                // TODO: Add check that divisor is a CValue within (u)int32-bit
                Log::debug("Divisor: {} ", toString(expr.rhs));

                Divide cpy = expr;
                if(expr.lhs)
                    cpy.lhs = call(expr.lhs);

                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Modulo const& expr) const
            {
                // TODO: Add check that divisor is a CValue within (u)int32-bit
                Log::debug("Modulo: {} ", toString(expr.rhs));

                Modulo cpy = expr;
                if(expr.lhs)
                    cpy.lhs = call(expr.lhs);

                return std::make_shared<Expression>(cpy);
            }

            // TODO: for some shifts, logical, subtract operations these may not be correct.
            //       Not sure yet if address calculation expressions with identity_transducer
            //       those types of Expression or not.
            template <CBinary Expr>
            requires(CArithmetic<Expr>) ExpressionPtr operator()(Expr const& expr) const
            {
                if constexpr(std::same_as<Expr, Subtract>)
                {
                    AssertFatal(false, "Subtracts are not expected");
                    return nullptr;
                }

                if constexpr(CLogical<Expr>)
                {
                    AssertFatal(false, "logicals are not expected");
                    return nullptr;
                }

                Expr cpy = expr;
                if(expr.lhs)
                    cpy.lhs = call(expr.lhs);
                if(expr.rhs)
                    cpy.rhs = call(expr.rhs);

                return std::make_shared<Expression>(cpy);
            }

            // Even with catch-all operator()(ExpressionPtr) without following,
            // compilation fails.
            template <typename Expr>
            requires(CBinary<Expr>) ExpressionPtr operator()(Expr const& expr) const
            {
                AssertFatal(false, "Not expected expr : ", ShowValue(expr));
                return nullptr;
            }

            template <CTernary Expr>
            requires(CArithmetic<Expr>) ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                    cpy.lhs = call(expr.lhs);
                if(expr.r1hs)
                    cpy.r1hs = call(expr.r1hs);
                if(expr.r2hs)
                    cpy.r2hs = call(expr.r2hs);

                return std::make_shared<Expression>(cpy);
            }

            template <typename Expr>
            requires(CTernary<Expr>) ExpressionPtr operator()(Expr const& expr) const
            {
                AssertFatal(false, "Not expected expr : ", ShowValue(expr));
                return nullptr;
            }

            // leaves
            ExpressionPtr operator()(CommandArgumentPtr const& expr) const
            {
                Log::debug("CommandArgumentPtr {}", toString(expr));

                auto varType = expr->variableType();

                assertIfNotExpectedType(varType.dataType, toString(expr));

                CommandArgumentPtr cpy = expr;
                return widenTo64(varType.dataType, cpy);
            }

            ExpressionPtr operator()(CommandArgumentValue const& expr) const
            {
                Log::debug("CommandArgumentValue {} Type {} ",
                           toString(expr),
                           toString(variableType(expr)));

                auto varType = variableType(expr);

                assertIfNotExpectedType(varType.dataType, toString(expr));

                CommandArgumentValue cpy = expr;
                return widenTo64(varType.dataType, cpy);
            }

            ExpressionPtr operator()(Register::ValuePtr const& expr) const
            {
                Log::debug("Register::ValuePtr {}", toString(expr));

                auto varType = expr->variableType();

                assertIfNotExpectedType(varType.dataType, toString(expr));

                Register::ValuePtr cpy = expr;
                return widenTo64(varType.dataType, cpy);
            }

            ExpressionPtr operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                Log::debug("AssemblyKernelArgumentPtr {} its expression is {}",
                           toString(expr),
                           toString(expr->expression));

                auto varType = expr->variableType;

                assertIfNotExpectedType(varType.dataType, toString(expr));

                AssemblyKernelArgumentPtr cpy = expr;
                return widenTo64(varType.dataType, cpy);
            }

            // catch the rest CValue
            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                AssertFatal(
                    false, "No expectation to meet WaveTilePtr or DataFlowTag : ", ShowValue(expr));
                return nullptr;
            }

            ExpressionPtr operator()(Expression const& expr) const
            {
                AssertFatal(
                    false, "No expectation to meet this type of Expression: ", ShowValue(expr));
                return nullptr;
            }

            template <typename T>
            ExpressionPtr widenTo64(DataType srcType, T const& expr) const
            {
                if(srcType == DataType::UInt32)
                    return convert(DataType::UInt64, std::make_shared<Expression>(expr));
                else if(srcType == DataType::Int32)
                    return convert(DataType::Int64, std::make_shared<Expression>(expr));

                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr const& expr) const
            {
                return std::visit(*this, *expr);
            }

            void assertIfNotExpectedType(DataType dt, std::string const& showValue) const
            {
                AssertFatal(dt == DataType::Int32 || dt == DataType::UInt32
                                || dt == DataType::UInt64 || dt == DataType::Int64,
                            "Unexpected DataType for Command/Kernel arguments or "
                            "workgroup/item indices ",
                            showValue);
            }
        };

        ExpressionPtr widenTo64bit(ExpressionPtr expr)
        {
            auto origVarType = resultVariableType(expr);

            auto visitor = WidenTo64BitVisitor();
            auto widened = visitor.call(expr);

            auto finalVarType = resultVariableType(widened);

            AssertFatal(origVarType.dataType == finalVarType.dataType,
                        "Original and final data types should be the same",
                        ShowValue(origVarType.dataType),
                        ShowValue(finalVarType.dataType));

            return widened;
        }
    }
}
