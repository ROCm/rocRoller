
#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        /*
         * to string
         */

        struct ExpressionToStringVisitor
        {
            template <CTernary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(),
                                   "(",
                                   call(expr.lhs),
                                   ", ",
                                   call(expr.r1hs),
                                   ", ",
                                   call(expr.r2hs),
                                   ")");
            }

            template <CBinary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(
                    ExpressionInfo<Expr>::name(), "(", call(expr.lhs), ", ", call(expr.rhs), ")");
            }

            template <CUnary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(), "(", call(expr.arg), ")");
            }

            std::string operator()(BitFieldExtract const& expr) const
            {
                return concatenate(ExpressionInfo<BitFieldExtract>::name(),
                                   "(",
                                   call(expr.arg),
                                   ", width:",
                                   expr.width,
                                   ", offset:",
                                   expr.offset,
                                   ")");
            }

            std::string operator()(Register::ValuePtr const& expr) const
            {
                // This allows an unallocated register value to be rendered into a string which
                // improves debugging by allowing the string representation of that expression
                // to be put into the source file as a comment.
                // Trying to generate the code for the expression will throw an exception.

                std::string tostr = "UNALLOCATED";
                if(expr->canUseAsOperand())
                    tostr = expr->toString();

                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return tostr + ":";
            }

            std::string operator()(ScaledMatrixMultiply const& expr) const
            {
                return concatenate("ScaledMatrixMultiply(",
                                   call(expr.matA),
                                   ", ",
                                   call(expr.matB),
                                   ", ",
                                   call(expr.matC),
                                   ", ",
                                   call(expr.scaleA),
                                   ", ",
                                   call(expr.scaleB),
                                   ")");
            }

            std::string operator()(CommandArgumentPtr const& expr) const
            {
                if(expr)
                    return concatenate("CommandArgument(", expr->name(), ")");
                else
                    return "CommandArgument(nullptr)";
            }

            std::string operator()(CommandArgumentValue const& expr) const
            {
                return std::visit([](auto const& val) { return concatenate(val, ":"); }, expr);
            }

            std::string operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return expr->name + ":";
            }

            std::string operator()(WaveTilePtr const& expr) const
            {
                return "WaveTile";
            }

            std::string operator()(DataFlowTag const& expr) const
            {
                return concatenate("DataFlowTag(", expr.tag, ")");
            }

            std::string call(Expression const& expr) const
            {
                auto functionalPart = std::visit(*this, expr);
                auto vt             = resultVariableType(expr);
                functionalPart += TypeAbbrev(vt);

                std::string comment = getComment(expr);
                if(comment.length() > 0)
                {
                    return concatenate("{", comment, ": ", functionalPart, "}");
                }

                return functionalPart;
            }

            std::string call(ExpressionPtr expr) const
            {
                if(!expr)
                    return "nullptr";

                return call(*expr);
            }
        };

        std::string toString(Expression const& expr)
        {
            auto visitor = ExpressionToStringVisitor();
            return visitor.call(expr);
        }

        std::string toString(ExpressionPtr const& expr)
        {
            auto visitor = ExpressionToStringVisitor();
            return visitor.call(expr);
        }

        struct ExpressionToShortStringVisitor
        {

            template <CTernary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(),
                                   "(",
                                   call(expr.lhs),
                                   ", ",
                                   call(expr.r1hs),
                                   ", ",
                                   call(expr.r2hs),
                                   ")");
            }

            template <CBinary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(
                    ExpressionInfo<Expr>::name(), "(", call(expr.lhs), ", ", call(expr.rhs), ")");
            }
            template <CUnary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(), "(", call(expr.arg), ")");
            }
            std::string operator()(Register::ValuePtr const& expr) const
            {
                // This allows an unallocated register value to be rendered into a string which
                // improves debugging by allowing the string representation of that expression
                // to be put into the source file as a comment.
                // Trying to generate the code for the expression will throw an exception.

                std::string tostr = "UNALLOCATED";
                if(expr->canUseAsOperand())
                    tostr = expr->toString();

                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return tostr;
            }
            std::string operator()(CommandArgumentPtr const& expr) const
            {
                if(expr)
                    return concatenate("CommandArgument(", expr->name(), ")");
                else
                    return "CommandArgument(nullptr)";
            }

            std::string operator()(CommandArgumentValue const& expr) const
            {
                return std::visit([](auto const& val) { return concatenate(val); }, expr);
            }

            std::string operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return expr->name;
            }

            std::string operator()(WaveTilePtr const& expr) const
            {
                return "WaveTile";
            }

            std::string operator()(DataFlowTag const& expr) const
            {
                return concatenate("DataFlowTag(", expr.tag, ")");
            }

#define HANDLE_INFIX_OP(TYPE, INFIX)                                         \
    std::string operator()(TYPE const& expr) const                           \
    {                                                                        \
        return concatenate("(", call(expr.lhs), INFIX, call(expr.rhs), ")"); \
    }
            HANDLE_INFIX_OP(Add, "+");
            HANDLE_INFIX_OP(Subtract, "-");
            HANDLE_INFIX_OP(Multiply, "*");
            HANDLE_INFIX_OP(Divide, "/");
            HANDLE_INFIX_OP(Modulo, "%");
            HANDLE_INFIX_OP(ShiftL, "<<");
            HANDLE_INFIX_OP(ArithmeticShiftR, ">>");
            HANDLE_INFIX_OP(BitwiseAnd, "&");
            HANDLE_INFIX_OP(BitwiseOr, "|");
            HANDLE_INFIX_OP(BitwiseXor, "^");
            HANDLE_INFIX_OP(GreaterThan, ">");
            HANDLE_INFIX_OP(GreaterThanEqual, ">=");
            HANDLE_INFIX_OP(LessThan, "<");
            HANDLE_INFIX_OP(LessThanEqual, "<=");
            HANDLE_INFIX_OP(Equal, "==");
            HANDLE_INFIX_OP(NotEqual, "!=");
            HANDLE_INFIX_OP(LogicalAnd, "&&");
            HANDLE_INFIX_OP(LogicalOr, "||");

            std::string operator()(AddShiftL const& expr) const
            {
                return concatenate(
                    "((", call(expr.lhs), "+", call(expr.r1hs), ")<<", call(expr.r2hs), ")");
            }

            std::string operator()(ShiftLAdd const& expr) const
            {
                return concatenate(
                    "((", call(expr.lhs), "<<", call(expr.r1hs), ")+", call(expr.r2hs), ")");
            }

            std::string operator()(MultiplyAdd const& expr) const
            {
                return concatenate(
                    "((", call(expr.lhs), "*", call(expr.r1hs), ")+", call(expr.r2hs), ")");
            }

            std::string operator()(Conditional const& expr) const
            {
                return concatenate(
                    "(", call(expr.lhs), "?", call(expr.r1hs), ":", call(expr.r2hs), ")");
            }

            std::string operator()(Negate const& expr) const
            {
                return concatenate("(-", call(expr.arg), ")");
            }

            std::string operator()(BitwiseNegate const& expr) const
            {
                return concatenate("(~", call(expr.arg), ")");
            }

            std::string operator()(Convert const& expr) const
            {
                return concatenate(
                    "((", TypeAbbrev(resultVariableType(expr)), ")", call(expr.arg), ")");
            }

            std::string call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            std::string call(ExpressionPtr expr) const
            {
                if(!expr)
                    return "nullptr";

                return call(*expr);
            }
        };

        std::string toShortString(Expression const& expr)
        {
            auto visitor = ExpressionToShortStringVisitor();
            return visitor.call(expr);
        }

        std::string toShortString(ExpressionPtr const& expr)
        {
            auto visitor = ExpressionToShortStringVisitor();
            return visitor.call(expr);
        }
    }
}
