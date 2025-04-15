/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

/**
 *
 */

#pragma once

#include <rocRoller/rocRoller.hpp>

#include <concepts>
#include <functional>
#include <memory>
#include <variant>

#include <rocRoller/AssemblyKernelArgument_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension_fwd.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct ROCROLLER_DECLSPEC Add;
        struct ROCROLLER_DECLSPEC MatrixMultiply;
        struct ROCROLLER_DECLSPEC ScaledMatrixMultiply;
        struct ROCROLLER_DECLSPEC Multiply;
        struct ROCROLLER_DECLSPEC MultiplyAdd;
        struct ROCROLLER_DECLSPEC MultiplyHigh;
        struct ROCROLLER_DECLSPEC Subtract;
        struct ROCROLLER_DECLSPEC Divide;
        struct ROCROLLER_DECLSPEC Modulo;
        struct ROCROLLER_DECLSPEC ShiftL;
        struct ROCROLLER_DECLSPEC LogicalShiftR;
        struct ROCROLLER_DECLSPEC ArithmeticShiftR;
        struct ROCROLLER_DECLSPEC BitwiseNegate;
        struct ROCROLLER_DECLSPEC BitwiseAnd;
        struct ROCROLLER_DECLSPEC BitwiseOr;
        struct ROCROLLER_DECLSPEC BitwiseXor;
        struct ROCROLLER_DECLSPEC GreaterThan;
        struct ROCROLLER_DECLSPEC GreaterThanEqual;
        struct ROCROLLER_DECLSPEC LessThan;
        struct ROCROLLER_DECLSPEC LessThanEqual;
        struct ROCROLLER_DECLSPEC Equal;
        struct ROCROLLER_DECLSPEC NotEqual;
        struct ROCROLLER_DECLSPEC LogicalAnd;
        struct ROCROLLER_DECLSPEC LogicalOr;
        struct ROCROLLER_DECLSPEC LogicalNot;

        struct ROCROLLER_DECLSPEC Exponential2;
        struct ROCROLLER_DECLSPEC Exponential;

        struct ROCROLLER_DECLSPEC MagicMultiple;
        struct ROCROLLER_DECLSPEC MagicShifts;
        struct ROCROLLER_DECLSPEC MagicSign;
        struct ROCROLLER_DECLSPEC Negate;

        struct ROCROLLER_DECLSPEC RandomNumber;

        struct ROCROLLER_DECLSPEC BitFieldExtract;

        struct ROCROLLER_DECLSPEC AddShiftL;
        struct ROCROLLER_DECLSPEC ShiftLAdd;
        struct ROCROLLER_DECLSPEC Conditional;

        struct ROCROLLER_DECLSPEC Convert;

        template <DataType DATATYPE>
        struct ROCROLLER_DECLSPEC SRConvert;

        struct ROCROLLER_DECLSPEC DataFlowTag;
        using WaveTilePtr = std::shared_ptr<KernelGraph::CoordinateGraph::WaveTile>;

        using Expression = std::variant<
            // --- Binary Operations ---
            Add,
            Subtract,
            Multiply,
            MultiplyHigh,
            Divide,
            Modulo,
            ShiftL,
            LogicalShiftR,
            ArithmeticShiftR,
            BitwiseAnd,
            BitwiseOr,
            BitwiseXor,
            GreaterThan,
            GreaterThanEqual,
            LessThan,
            LessThanEqual,
            Equal,
            NotEqual,
            LogicalAnd,
            LogicalOr,

            // --- Unary Operations ---
            MagicMultiple,
            MagicShifts,
            MagicSign,
            Negate,
            BitwiseNegate,
            LogicalNot,
            Exponential2,
            Exponential,
            RandomNumber,
            BitFieldExtract,

            // --- Ternary Operations ---
            AddShiftL,
            ShiftLAdd,
            MatrixMultiply,
            Conditional,

            // --- TernaryMixed Operations ---
            MultiplyAdd,

            // ---Quinary Operation(s) ---
            ScaledMatrixMultiply,

            // --- Convert Operations ---
            Convert,

            // --- Stochastic Rounding Convert ---
            SRConvert<DataType::FP8>,
            SRConvert<DataType::BF8>,

            // --- Values ---
            // Literal: Always available
            CommandArgumentValue,

            // Available at kernel launch
            CommandArgumentPtr,

            // Available at kernel execute (i.e. on the GPU), and at kernel launch.
            AssemblyKernelArgumentPtr,

            // Available at kernel execute (i.e. on the GPU)
            Register::ValuePtr,
            DataFlowTag,
            WaveTilePtr>;
        using ExpressionPtr = std::shared_ptr<Expression>;

        using ExpressionTransducer = std::function<ExpressionPtr(ExpressionPtr)>;

        template <typename T>
        concept CExpression = std::constructible_from<Expression, T>;

        enum class EvaluationTime : int
        {
            Translate = 0, //< An expression where all the values come from CommandArgumentValues.
            KernelLaunch, //< An expression where values may come from CommandArguments or CommandArgumentValues.
            KernelExecute, // An expression that depends on at least one Register::Value.
            Count
        };

        enum class Category : int;

    }
}
