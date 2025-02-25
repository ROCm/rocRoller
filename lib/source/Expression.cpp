
#include <variant>

#include "DataTypes/DataTypes.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "Utilities/Generator.hpp"

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
        std::string toString(EvaluationTime t)
        {
            switch(t)
            {
            case EvaluationTime::Translate:
                return "Translate";
            case EvaluationTime::KernelLaunch:
                return "KernelLaunch";
            case EvaluationTime::KernelExecute:
                return "KernelExecute";
            case EvaluationTime::Count:
            default:
                break;
            }
            Throw<FatalError>("Invalid EvaluationTime");
        }

        std::string toString(AlgebraicProperty t)
        {
            switch(t)
            {
            case AlgebraicProperty::Commutative:
                return "Commutative";
            case AlgebraicProperty::Associative:
                return "Associative";
            case AlgebraicProperty::Count:
            default:
                break;
            }
            Throw<FatalError>("Invalid EvaluationTime");
        }

        /*
         * to string
         */

        struct ExpressionToStringVisitor
        {
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

                return tostr + ":" + TypeAbbrev(expr->variableType());
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
                return std::visit(
                    [](auto const& val) { return concatenate(val) + typeid(val).name(); }, expr);
            }

            std::string operator()(AssemblyKernelArgumentPtr const& expr) const
            {
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

            std::string call(Expression const& expr) const
            {
                std::string comment = getComment(expr);
                if(comment.length() > 0)
                {
                    return concatenate("{", comment, ": ", std::visit(*this, expr), "}");
                }

                return std::visit(*this, expr);
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

        ExpressionPtr fromKernelArgument(AssemblyKernelArgument const& arg)
        {
            return std::make_shared<Expression>(std::make_shared<AssemblyKernelArgument>(arg));
        }

        /*
         * result type
         */

        class ExpressionResultTypeVisitor
        {
            std::weak_ptr<Context> m_context;

        public:
            template <typename T>
            requires(CBinary<T>&& CArithmetic<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);

                auto regType = Register::PromoteType(lhsVal.regType, rhsVal.regType);

                VariableType varType;

                if constexpr(std::same_as<T, ArithmeticShiftR>)
                {
                    varType = lhsVal.varType;
                }
                else
                {
                    varType = VariableType::Promote(lhsVal.varType, rhsVal.varType);
                }

                return {regType, varType};
            }

            ResultType operator()(ScaledMatrixMultiply const& expr)
            {
                auto matAVal = call(expr.matA);
                auto matBVal = call(expr.matB);
                auto matCVal = call(expr.matC);

                auto regType = Register::PromoteType(matAVal.regType, matBVal.regType);
                regType      = Register::PromoteType(regType, matCVal.regType);

                auto varType = VariableType::Promote(matAVal.varType, matBVal.varType);
                varType      = VariableType::Promote(varType, matCVal.varType);

                return {regType, varType};
            }

            template <typename T>
            requires(CTernary<T>&& CArithmetic<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);
                auto r2hsVal = call(expr.r2hs);

                auto regType = Register::PromoteType(lhsVal.regType, r1hsVal.regType);
                regType      = Register::PromoteType(regType, r2hsVal.regType);

                auto varType = VariableType::Promote(lhsVal.varType, r1hsVal.varType);
                varType      = VariableType::Promote(varType, r2hsVal.varType);

                return {regType, varType};
            }

            template <typename T>
            requires(CUnary<T>&& CArithmetic<T>) ResultType operator()(T const& expr)
            {
                auto argVal = call(expr.arg);

                if constexpr(std::same_as<T, MagicShifts>)
                    return {argVal.regType, DataType::Int32};

                return argVal;
            }

            template <DataType DATATYPE>
            ResultType operator()(Convert<DATATYPE> const& expr)
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, DATATYPE};
            }

            template <DataType DATATYPE>
            ResultType operator()(SRConvert<DATATYPE> const& expr)
            {
                // SR conversion currently only supports FP8 and BF8
                static_assert(DATATYPE == DataType::FP8 || DATATYPE == DataType::BF8);
                auto argVal = call(expr.lhs);
                return {argVal.regType, DATATYPE};
            }

            ResultType operator()(BitFieldExtract const& expr)
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, expr.outputDataType};
            }

            template <typename T>
            requires(CBinary<T>&& CComparison<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);

                // Can't compare between two different types on the GPU.
                AssertFatal(lhsVal.regType == Register::Type::Literal
                                || rhsVal.regType == Register::Type::Literal
                                || lhsVal.varType == rhsVal.varType,
                            ShowValue(lhsVal.varType),
                            ShowValue(rhsVal.varType),
                            ShowValue(expr));

                auto inputRegType = Register::PromoteType(lhsVal.regType, rhsVal.regType);
                auto inputVarType = VariableType::Promote(lhsVal.varType, rhsVal.varType);

                switch(inputRegType)
                {
                case Register::Type::Literal:
                    return {Register::Type::Literal, DataType::Bool};
                case Register::Type::Scalar:
                    return {Register::Type::Scalar, DataType::Bool};
                case Register::Type::Vector:
                    if(auto context = m_context.lock(); context)
                    {
                        if(context->kernel()->wavefront_size() == 32)
                            return {Register::Type::Scalar, DataType::Bool32};
                        return {Register::Type::Scalar, DataType::Bool64};
                    }
                    // If you are reading this, it probably means that this visitor
                    // was called on an expression with registers that didn't have
                    // a context.
                    Throw<FatalError>("Need context to determine wavefront size", ShowValue(expr));
                default:
                    break;
                }
                Throw<FatalError>("Invalid register types for comparison: ",
                                  ShowValue(lhsVal.regType),
                                  ShowValue(rhsVal.regType));
            }

            template <typename T>
            requires(CBinary<T>&& CLogical<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);
                return logical(lhsVal, rhsVal);
            }

            ResultType logical(ResultType lhsVal, ResultType rhsVal)
            {
                if(lhsVal.varType == DataType::Bool
                   && (rhsVal.varType == DataType::Bool32 || rhsVal.varType == DataType::Bool64))
                {
                    std::swap(lhsVal, rhsVal);
                }

                // Can't compare between two different types on the GPU.
                AssertFatal(
                    lhsVal.regType == Register::Type::Literal
                        || rhsVal.regType == Register::Type::Literal
                        || lhsVal.varType == rhsVal.varType
                        || (lhsVal.varType == DataType::Bool32 && rhsVal.varType == DataType::Bool)
                        || (lhsVal.varType == DataType::Bool64 && rhsVal.varType == DataType::Bool),
                    ShowValue(lhsVal.varType),
                    ShowValue(rhsVal.varType));

                auto inputRegType = Register::PromoteType(lhsVal.regType, rhsVal.regType);
                auto inputVarType = VariableType::Promote(lhsVal.varType, rhsVal.varType);

                switch(inputRegType)
                {
                case Register::Type::Scalar:
                    if(inputVarType == DataType::Bool || inputVarType == DataType::Bool32
                       || inputVarType == DataType::Bool64)
                    {
                        return {Register::Type::Scalar, DataType::Bool};
                    }
                default:
                    break;
                }
                Throw<FatalError>("Invalid register types for logical: ",
                                  ShowValue(lhsVal.regType),
                                  ShowValue(lhsVal.varType),
                                  ShowValue(rhsVal.regType),
                                  ShowValue(rhsVal.varType),
                                  ShowValue(inputRegType),
                                  ShowValue(inputVarType));
            }

            template <typename T>
            requires(CUnary<T>&& CLogical<T>) ResultType operator()(T const& expr)
            {
                auto val = call(expr.arg);
                switch(val.regType)
                {
                case Register::Type::Scalar:
                {
                    if(!(val.varType == DataType::Bool || val.varType == DataType::Bool32
                         || val.varType == DataType::Bool64 || val.varType == DataType::Raw32))
                    {
                        Throw<FatalError>("Invalid variable type for unary logical: ",
                                          ShowValue(val.varType));
                    }
                    return val;
                }
                default:
                    Throw<FatalError>("Invalid register type for unary logical: ",
                                      ShowValue(val.regType));
                }
            }

            ResultType operator()(Conditional const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);
                auto r2hsVal = call(expr.r2hs);

                AssertFatal(r2hsVal.varType == r1hsVal.varType,
                            ShowValue(r1hsVal.varType),
                            ShowValue(r2hsVal.varType));
                auto varType = r2hsVal.varType;

                if(lhsVal.varType == DataType::Bool32 || lhsVal.varType == DataType::Bool64
                   || lhsVal.regType == Register::Type::Vector
                   || r1hsVal.regType == Register::Type::Vector
                   || r2hsVal.regType == Register::Type::Vector)
                {
                    return {Register::Type::Vector, varType};
                }
                return {Register::Type::Scalar, varType};
            }

            ResultType operator()(CommandArgumentPtr const& expr)
            {
                AssertFatal(expr != nullptr, "Null subexpression!");
                return {Register::Type::Literal, expr->variableType()};
            }

            ResultType operator()(AssemblyKernelArgumentPtr const& expr)
            {
                AssertFatal(expr != nullptr, "Null subexpression!");
                return {Register::Type::Scalar, expr->variableType};
            }

            ResultType operator()(CommandArgumentValue const& expr)
            {
                return {Register::Type::Literal, variableType(expr)};
            }

            ResultType operator()(Register::ValuePtr const& expr)
            {
                AssertFatal(expr != nullptr, "Null subexpression!");
                m_context = expr->context();
                return {expr->regType(), expr->variableType()};
            }

            ResultType operator()(DataFlowTag const& expr)
            {
                return {expr.regType, expr.varType};
            }

            ResultType operator()(WaveTilePtr const& expr)
            {
                return call(expr->vgpr);
            }

            ResultType call(Expression const& expr)
            {
                return std::visit(*this, expr);
            }

            ResultType call(ExpressionPtr const& expr)
            {
                return call(*expr);
            }
        };

        VariableType resultVariableType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).varType;
        }

        Register::Type resultRegisterType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).regType;
        }

        ResultType resultType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr);
        }

        ResultType resultType(Expression const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr);
        }

        /*
         * identical
         */

        struct ExpressionIdenticalVisitor
        {
            bool operator()(ScaledMatrixMultiply const& a, ScaledMatrixMultiply const& b)
            {
                bool matA   = false;
                bool matB   = false;
                bool matC   = false;
                bool scaleA = false;
                bool scaleB = false;

                matA = call(a.matA, b.matA);
                if(a.matA == nullptr && b.matA == nullptr)
                {
                    matA = true;
                }

                matB = call(a.matB, b.matB);
                if(a.matB == nullptr && b.matB == nullptr)
                {
                    matB = true;
                }

                matC = call(a.matC, b.matC);
                if(a.matC == nullptr && b.matC == nullptr)
                {
                    matC = true;
                }

                scaleA = call(a.scaleA, b.scaleA);
                if(a.scaleA == nullptr && b.scaleA == nullptr)
                {
                    scaleA = true;
                }

                scaleB = call(a.scaleB, b.scaleB);
                if(a.scaleB == nullptr && b.scaleB == nullptr)
                {
                    scaleB = true;
                }

                return matA && matB && matC && scaleA && scaleB;
            }

            template <CTernary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs  = false;
                bool r1hs = false;
                bool r2hs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                r1hs = call(a.r1hs, b.r1hs);
                if(a.r1hs == nullptr && b.r1hs == nullptr)
                {
                    r1hs = true;
                }

                r2hs = call(a.r2hs, b.r2hs);

                if(a.r2hs == nullptr && b.r2hs == nullptr)
                {
                    r2hs = true;
                }
                return lhs && r1hs && r2hs;
            }

            template <CBinary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs = false;
                bool rhs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                rhs = call(a.rhs, b.rhs);
                if(a.rhs == nullptr && b.rhs == nullptr)
                {
                    rhs = true;
                }

                return lhs && rhs;
            }

            template <CUnary T>
            bool operator()(T const& a, T const& b)
            {
                if(a.arg == nullptr && b.arg == nullptr)
                {
                    return true;
                }
                return call(a.arg, b.arg);
            }

            constexpr bool operator()(CommandArgumentValue const& a, CommandArgumentValue const& b)
            {
                return a == b;
            }

            bool operator()(CommandArgumentPtr const& a, CommandArgumentPtr const& b)
            {
                // Need to be careful not to invoke the overloaded operators, we want to compare
                // the pointers directly.
                // a->expression && b->expression -> logical and of both expressions
                if(a.get() == b.get())
                    return true;

                if(a == nullptr || b == nullptr)
                    return false;

                return (*a) == (*b);
            }

            bool operator()(AssemblyKernelArgumentPtr const& a, AssemblyKernelArgumentPtr const& b)
            {
                if(a->name == b->name)
                    return true;

                if((a->expression != nullptr) && (b->expression != nullptr))
                    return call(a->expression, b->expression);

                return false;
            }

            bool operator()(Register::ValuePtr const& a, Register::ValuePtr const& b)
            {
                return a->sameAs(b);
            }

            constexpr bool operator()(DataFlowTag const& a, DataFlowTag const& b)
            {
                return a == b;
            }

            bool operator()(WaveTilePtr const& a, WaveTilePtr const& b)
            {
                return a == b;
            }

            // a & b are different operator/value classes
            template <class T, class U>
            requires(!std::same_as<T, U>) constexpr bool operator()(T const& a, U const& b)
            {
                return false;
            }

            bool call(ExpressionPtr const& a, ExpressionPtr const& b)
            {
                if(a == nullptr)
                {
                    return b == nullptr;
                }
                else if(b == nullptr)
                {
                    return false;
                }
                return std::visit(*this, *a, *b);
            }
        };

        bool identical(ExpressionPtr const& a, ExpressionPtr const& b)
        {
            auto visitor = ExpressionIdenticalVisitor();
            return visitor.call(a, b);
        }

        struct ExpressionEquivalentVisitor
        {
            ExpressionEquivalentVisitor(AlgebraicProperties properties)
                : m_properties(properties)
            {
            }

            bool operator()(ScaledMatrixMultiply const& a, ScaledMatrixMultiply const& b)
            {
                bool matA   = false;
                bool matB   = false;
                bool matC   = false;
                bool scaleA = false;
                bool scaleB = false;

                matA = call(a.matA, b.matA);
                if(a.matA == nullptr && b.matA == nullptr)
                {
                    matA = true;
                }

                matB = call(a.matB, b.matB);
                if(a.matB == nullptr && b.matB == nullptr)
                {
                    matB = true;
                }

                matC = call(a.matC, b.matC);
                if(a.matC == nullptr && b.matC == nullptr)
                {
                    matC = true;
                }

                scaleA = call(a.scaleA, b.scaleA);
                if(a.scaleA == nullptr && b.scaleA == nullptr)
                {
                    scaleA = true;
                }

                scaleB = call(a.scaleB, b.scaleB);
                if(a.scaleB == nullptr && b.scaleB == nullptr)
                {
                    scaleB = true;
                }

                return matA && matB && matC && scaleA && scaleB;
            }

            template <CTernary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs  = false;
                bool r1hs = false;
                bool r2hs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                r1hs = call(a.r1hs, b.r1hs);
                if(a.r1hs == nullptr && b.r1hs == nullptr)
                {
                    r1hs = true;
                }

                r2hs = call(a.r2hs, b.r2hs);

                if(a.r2hs == nullptr && b.r2hs == nullptr)
                {
                    r2hs = true;
                }
                return lhs && r1hs && r2hs;
            }

            template <CBinary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs = false;
                bool rhs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                rhs = call(a.rhs, b.rhs);
                if(a.rhs == nullptr && b.rhs == nullptr)
                {
                    rhs = true;
                }

                bool result = lhs && rhs;

                // Test if equivalent if expression is commutative
                if(!result && CCommutativeBinary<T> && m_properties[AlgebraicProperty::Commutative])
                {
                    lhs = call(a.lhs, b.rhs);
                    if(a.lhs == nullptr && b.rhs == nullptr)
                    {
                        lhs = true;
                    }

                    rhs = call(a.rhs, b.lhs);
                    if(a.rhs == nullptr && b.lhs == nullptr)
                    {
                        rhs = true;
                    }

                    result = lhs && rhs;
                }

                return result;
            }

            template <CUnary T>
            bool operator()(T const& a, T const& b)
            {
                if(a.arg == nullptr && b.arg == nullptr)
                {
                    return true;
                }
                return call(a.arg, b.arg);
            }

            constexpr bool operator()(CommandArgumentValue const& a, CommandArgumentValue const& b)
            {
                return a == b;
            }

            bool operator()(CommandArgumentPtr const& a, CommandArgumentPtr const& b)
            {
                return (*a) == (*b);
            }

            bool operator()(AssemblyKernelArgumentPtr const& a, AssemblyKernelArgumentPtr const& b)
            {
                if(a->name == b->name)
                    return true;

                if((a->expression != nullptr) && (b->expression != nullptr))
                    return call(a->expression, b->expression);

                return false;
            }

            bool operator()(Register::ValuePtr const& a, Register::ValuePtr const& b)
            {
                return a->sameAs(b);
            }

            constexpr bool operator()(DataFlowTag const& a, DataFlowTag const& b)
            {
                return a == b;
            }

            bool operator()(WaveTilePtr const& a, WaveTilePtr const& b)
            {
                return a == b;
            }

            // a & b are different operator/value classes
            template <class T, class U>
            requires(!std::same_as<T, U>) bool operator()(T const& a, U const& b)
            {
                return false;
            }

            bool call(ExpressionPtr const& a, ExpressionPtr const& b)
            {
                if(a == nullptr)
                {
                    return b == nullptr;
                }
                else if(b == nullptr)
                {
                    return false;
                }
                return std::visit(*this, *a, *b);
            }

        private:
            AlgebraicProperties const m_properties;
        };

        bool equivalent(ExpressionPtr const& a,
                        ExpressionPtr const& b,
                        AlgebraicProperties  properties)
        {
            auto visitor = ExpressionEquivalentVisitor(properties);
            return visitor.call(a, b);
        }

        /*
         * comments
         */

        struct ExpressionSetCommentVisitor
        {
            std::string comment;
            bool        throwIfNotSupported = true;

            template <typename Expr>
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) void operator()(Expr& expr)
            {
                expr.comment = std::move(comment);
            }

            void operator()(auto& expr)
            {
                if(throwIfNotSupported)
                    Throw<FatalError>("Cannot set a comment for a base expression.");
            }

            void call(Expression& expr)
            {
                return std::visit(*this, expr);
            }
        };

        void setComment(Expression& expr, std::string comment)
        {
            auto visitor = ExpressionSetCommentVisitor{std::move(comment)};
            return visitor.call(expr);
        }

        void setComment(ExpressionPtr& expr, std::string comment)
        {
            if(expr)
            {
                setComment(*expr, std::move(comment));
            }
            else
            {
                Throw<FatalError>("Cannot set the comment for a null expression pointer.");
            }
        }

        void copyComment(ExpressionPtr const& dst, ExpressionPtr const& src)
        {
            if(!dst || !src)
                return;
            copyComment(*dst, *src);
        }

        void copyComment(Expression& dst, ExpressionPtr const& src)
        {
            if(!src)
                return;
            copyComment(dst, *src);
        }

        void copyComment(ExpressionPtr const& dst, Expression const& src)
        {
            if(!dst)
                return;
            copyComment(*dst, src);
        }

        void copyComment(Expression& dst, Expression const& src)
        {

            if(&src == &dst)
                return;

            auto comment = getComment(src);
            if(comment.empty())
                return;

            comment = getComment(dst) + std::move(comment);

            ExpressionSetCommentVisitor vis{std::move(comment), false};
            vis.call(dst);
        }

        struct ExpressionGetCommentVisitor
        {
            bool includeRegisterComments = true;

            template <typename Expr>
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) std::string
                operator()(Expr const& expr) const
            {
                return expr.comment;
            }

            std::string operator()(Register::ValuePtr const& expr) const
            {
                if(includeRegisterComments && expr)
                    return expr->name();

                return "";
            }

            std::string operator()(auto const& expr) const
            {
                return "";
            }

            std::string call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }
        };

        std::string getComment(Expression const& expr, bool includeRegisterComments)
        {
            auto visitor = ExpressionGetCommentVisitor{includeRegisterComments};
            return visitor.call(expr);
        }

        std::string getComment(ExpressionPtr const& expr, bool includeRegisterComments)
        {
            if(!expr)
            {
                return "";
            }
            return getComment(*expr, includeRegisterComments);
        }

        std::string getComment(ExpressionPtr const& expr)
        {
            return getComment(expr, true);
        }

        std::string getComment(Expression const& expr)
        {
            return getComment(expr, true);
        }

        void appendComment(Expression& expr, std::string comment)
        {
            setComment(expr, getComment(expr) + comment);
        }

        void appendComment(ExpressionPtr& expr, std::string comment)
        {
            setComment(expr, getComment(expr) + comment);
        }

        /*
         * stream operators
         */

        std::ostream& operator<<(std::ostream& stream, ResultType const& obj)
        {
            return stream << "{" << obj.regType << ", " << obj.varType << "}";
        }

        std::ostream& operator<<(std::ostream& stream, ExpressionPtr const& expr)
        {
            return stream << toString(expr);
        }

        std::ostream& operator<<(std::ostream& stream, Expression const& expr)
        {
            return stream << toString(expr);
        }

        std::ostream& operator<<(std::ostream& stream, std::vector<ExpressionPtr> const& exprs)
        {
            auto iter = exprs.begin();
            stream << "[";
            if(iter != exprs.end())
                stream << *iter;
            iter++;

            for(; iter != exprs.end(); iter++)
                stream << ", " << *iter;

            stream << "]";

            return stream;
        }

        struct ExpressionComplexityVisitor
        {

            template <CUnary Expr>
            int operator()(Expr const& expr) const
            {
                return Expr::Complexity + call(expr.arg);
            }

            template <CBinary Expr>
            int operator()(Expr const& expr) const
            {
                return Expr::Complexity + call(expr.lhs) + call(expr.rhs);
            }

            template <CTernary Expr>
            int operator()(Expr const& expr) const
            {
                return Expr::Complexity + call(expr.lhs) + call(expr.r1hs) + call(expr.r2hs);
            }

            int operator()(ScaledMatrixMultiply const& expr) const
            {
                return ScaledMatrixMultiply::Complexity + call(expr.matA) + call(expr.matB)
                       + call(expr.matC) + call(expr.scaleA) + call(expr.scaleB);
            }

            template <CValue Value>
            constexpr int operator()(Value const& expr) const
            {
                return 0;
            }

            int call(ExpressionPtr expr) const
            {
                if(!expr)
                    return 0;

                return call(*expr);
            }

            int call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

        private:
        };

        int complexity(ExpressionPtr expr)
        {
            return ExpressionComplexityVisitor().call(expr);
        }

        int complexity(Expression const& expr)
        {
            return ExpressionComplexityVisitor().call(expr);
        }
    }
}
