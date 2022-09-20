/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <memory>
#include <stack>
#include <variant>

#include "CodeGen/Arithmetic.hpp"
#include "CodeGen/Instruction.hpp"
#include "InstructionValues/Register.hpp"
#include "Operations/CommandArgument.hpp"
#include "Utilities/Component.hpp"

namespace rocRoller
{

    namespace Expression
    {
        inline ExpressionPtr operator+(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Add{a, b});
        }

        inline ExpressionPtr operator-(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Subtract{a, b});
        }

        inline ExpressionPtr operator*(ExpressionPtr a, ExpressionPtr b)
        {
            if(std::holds_alternative<WaveTilePtr>(*a) && std::holds_alternative<WaveTilePtr>(*b))
            {
                // sizes are deferred
                return std::make_shared<Expression>(MatrixMultiply(a, b, 0, 0, 0, 0));
            }
            return std::make_shared<Expression>(Multiply{a, b});
        }

        inline ExpressionPtr operator/(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Divide{a, b});
        }

        inline ExpressionPtr operator%(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Modulo{a, b});
        }

        inline ExpressionPtr operator<<(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(ShiftL{a, b});
        }

        inline ExpressionPtr operator>>(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(SignedShiftR{a, b});
        }

        inline ExpressionPtr shiftR(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(ShiftR{a, b});
        }

        inline ExpressionPtr operator&(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(BitwiseAnd{a, b});
        }

        inline ExpressionPtr operator|(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(BitwiseOr{a, b});
        }

        inline ExpressionPtr operator^(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(BitwiseXor{a, b});
        }

        inline ExpressionPtr operator>(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(GreaterThan{a, b});
        }

        inline ExpressionPtr operator>=(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(GreaterThanEqual{a, b});
        }

        inline ExpressionPtr operator<(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(LessThan{a, b});
        }

        inline ExpressionPtr operator<=(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(LessThanEqual{a, b});
        }

        inline ExpressionPtr operator==(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(Equal{a, b});
        }

        inline ExpressionPtr operator-(ExpressionPtr a)
        {
            return std::make_shared<Expression>(Negate{a});
        }

        inline ExpressionPtr multiplyHigh(ExpressionPtr a, ExpressionPtr b)
        {
            return std::make_shared<Expression>(MultiplyHigh{a, b});
        }

        inline ExpressionPtr magicMultiple(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicMultiple{a});
        }

        inline ExpressionPtr magicShifts(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicShifts{a});
        }

        inline ExpressionPtr magicSign(ExpressionPtr a)
        {
            return std::make_shared<Expression>(MagicSign{a});
        }

        inline ExpressionPtr convert(DataType dt, ExpressionPtr a)
        {
            switch(dt)
            {
            case DataType::Half:
                return convert<DataType::Half>(a);
                break;
            case DataType::Float:
                return convert<DataType::Float>(a);
                break;
            default:
                Throw<FatalError>("Unsupported datatype conversion");
            }
        }

        template <DataType DATATYPE>
        inline ExpressionPtr convert(ExpressionPtr a)
        {
            return std::make_shared<Expression>(Convert<DATATYPE>{a});
        }

        template <CCommandArgumentValue T>
        inline ExpressionPtr literal(T value)
        {
            return std::make_shared<Expression>(value);
        }

        static_assert(CExpression<Add>);
        static_assert(!CExpression<Register::Value>,
                      "ValuePtr can be an Expression but Value cannot.");

        template <typename Expr>
        struct ExpressionInfo
        {
        };

#define EXPRESSION_INFO_CUSTOM(cls, cls_name) \
    template <>                               \
    struct ExpressionInfo<cls>                \
    {                                         \
        constexpr static auto name()          \
        {                                     \
            return cls_name;                  \
        }                                     \
    }

#define EXPRESSION_INFO(cls) EXPRESSION_INFO_CUSTOM(cls, #cls)

        EXPRESSION_INFO(Add);
        EXPRESSION_INFO(Subtract);
        EXPRESSION_INFO(MatrixMultiply);
        EXPRESSION_INFO(Multiply);
        EXPRESSION_INFO(MultiplyHigh);

        EXPRESSION_INFO(Divide);
        EXPRESSION_INFO(Modulo);

        EXPRESSION_INFO(ShiftL);
        EXPRESSION_INFO(ShiftR);
        EXPRESSION_INFO(SignedShiftR);
        EXPRESSION_INFO(BitwiseAnd);
        EXPRESSION_INFO(BitwiseOr);
        EXPRESSION_INFO(BitwiseXor);

        EXPRESSION_INFO(FusedShiftAdd);
        EXPRESSION_INFO(FusedAddShift);

        EXPRESSION_INFO(GreaterThan);
        EXPRESSION_INFO(GreaterThanEqual);
        EXPRESSION_INFO(LessThan);
        EXPRESSION_INFO(LessThanEqual);
        EXPRESSION_INFO(Equal);

        EXPRESSION_INFO(MagicMultiple);
        EXPRESSION_INFO(MagicShifts);
        EXPRESSION_INFO(MagicSign);

        EXPRESSION_INFO(Negate);

        EXPRESSION_INFO(Convert<DataType::Float>);
        EXPRESSION_INFO(Convert<DataType::Half>);

        EXPRESSION_INFO_CUSTOM(Register::ValuePtr, "Register Value");
        EXPRESSION_INFO_CUSTOM(CommandArgumentPtr, "Command Argument");
        EXPRESSION_INFO_CUSTOM(CommandArgumentValue, "Literal Value");
        EXPRESSION_INFO_CUSTOM(AssemblyKernelArgumentPtr, "Kernel Argument");
        EXPRESSION_INFO_CUSTOM(WaveTilePtr, "WaveTile");

        EXPRESSION_INFO(DataFlowTag);

#undef EXPRESSION_INFO
#undef EXPRESSION_INFO_CUSTOM
        struct ExpressionNameVisitor
        {
            template <CExpression Expr>
            std::string operator()(Expr const& expr) const
            {
                return ExpressionInfo<Expr>::name();
            }

            std::string operator()(ExpressionPtr const& expr) const
            {
                return std::visit(*this, *expr);
            }
        };

        inline std::string name(ExpressionPtr expr)
        {
            return ExpressionNameVisitor()(expr);
        }

        struct ExpressionEvaluationTimesVisitor
        {
            EvaluationTimes operator()(WaveTilePtr const& expr) const
            {
                return {EvaluationTime::KernelExecute};
            }

            template <CTernary Expr>
            EvaluationTimes operator()(Expr const& expr) const
            {
                auto lhs  = call(expr.lhs);
                auto r1hs = call(expr.r1hs);
                auto r2hs = call(expr.r2hs);

                return lhs & r1hs & r2hs & Expr::EvalTimes;
            }

            template <CBinary Expr>
            EvaluationTimes operator()(Expr const& expr) const
            {
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                return lhs & rhs & Expr::EvalTimes;
            }

            template <CUnary Expr>
            EvaluationTimes operator()(Expr const& expr) const
            {
                return call(expr.arg) & Expr::EvalTimes;
            }

            EvaluationTimes operator()(std::shared_ptr<Register::Value> const& expr) const
            {
                if(expr->regType() == Register::Type::Literal)
                    return EvaluationTimes::All();

                return {EvaluationTime::KernelExecute};
            }

            EvaluationTimes operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                return {EvaluationTime::KernelExecute};
            }

            EvaluationTimes operator()(DataFlowTag const& expr) const
            {
                return {EvaluationTime::KernelExecute};
            }

            EvaluationTimes operator()(std::shared_ptr<CommandArgument> const& expr) const
            {
                return {EvaluationTime::KernelLaunch};
            }

            EvaluationTimes operator()(CommandArgumentValue const& expr) const
            {
                return EvaluationTimes::All();
            }

            EvaluationTimes call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            EvaluationTimes call(ExpressionPtr const& expr) const
            {
                return call(*expr);
            }
        };

        inline EvaluationTimes evaluationTimes(Expression const& expr)
        {
            return ExpressionEvaluationTimesVisitor().call(expr);
        }

        inline EvaluationTimes evaluationTimes(ExpressionPtr const& expr)
        {
            return ExpressionEvaluationTimesVisitor().call(expr);
        }
    }
}
