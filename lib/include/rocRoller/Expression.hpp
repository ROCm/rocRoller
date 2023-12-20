/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

#include <bitset>
#include <memory>
#include <stack>

#include "Expression_fwd.hpp"

#include "InstructionValues/Register_fwd.hpp"
#include "Operations/CommandArgument_fwd.hpp"

#include "CodeGen/Instruction.hpp"
#include "Utilities/Component.hpp"
#include "Utilities/EnumBitset.hpp"

namespace rocRoller
{
    namespace Expression
    {
        enum class EvaluationTime : int
        {
            Translate = 0, //< An expression where all the values come from CommandArgumentValues.
            KernelLaunch, //< An expression where values may come from CommandArguments or CommandArgumentValues.
            KernelExecute, // An expression that depends on at least one Register::Value.
            Count
        };
        std::string toString(EvaluationTime t);

        using EvaluationTimes = EnumBitset<EvaluationTime>;

        enum class AlgebraicProperty : int
        {
            Commutative = 0,
            Associative,
            Count
        };
        std::string toString(AlgebraicProperty t);

        using AlgebraicProperties = EnumBitset<AlgebraicProperty>;

        enum class Category : int
        {
            Arithmetic = 0,
            Comparison,
            Logical,
            Conversion,
            Value,
            Count
        };

        // Expression: type alias for std::variant of all expression subtypes.
        // Defined in Expression_fwd.hpp.

        struct Binary
        {
            ExpressionPtr lhs, rhs;
            std::string   comment = "";
        };

        template <typename T>
        concept CBinary = requires
        {
            requires std::derived_from<T, Binary>;
        };

        struct Add : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
        };

        struct Subtract : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct Multiply : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
        };

        struct MultiplyHigh : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Commutative};
        };

        struct Divide : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct Modulo : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct ShiftL : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct LogicalShiftR : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct ArithmeticShiftR : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct BitwiseAnd : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
        };

        struct BitwiseOr : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
        };

        struct BitwiseXor : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
        };

        struct GreaterThan : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct GreaterThanEqual : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct LessThan : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct LessThanEqual : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
        };

        struct Equal : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Commutative};
        };

        struct LogicalAnd : Binary
        {
            constexpr static inline auto                Type      = Category::Logical;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
        };

        struct LogicalOr : Binary
        {
            constexpr static inline auto                Type      = Category::Logical;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
        };

        struct Ternary
        {
            ExpressionPtr lhs, r1hs, r2hs;
            std::string   comment = "";
        };

        struct TernaryMixed : Ternary
        {
        };

        template <typename T>
        concept CTernaryMixed = requires
        {
            requires std::derived_from<T, TernaryMixed>;
        };

        template <typename T>
        concept CTernary = requires
        {
            requires std::derived_from<T, Ternary> || CTernaryMixed<T>;
        };

        /*
         * AddShiftL performs a fusion of Add expression followed by
         * ShiftL expression, lowering to the fused instruction if possible.
         * result = (lhs + r1hs) << r2hs
         */
        struct AddShiftL : Ternary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::KernelExecute};
        };

        /*
         * ShiftLAdd performs a fusion of ShiftL expression followed by
         * Add expression, lowering to the fused instruction if possible.
         * result = (lhs << r1hs) + r2hs
         */
        struct ShiftLAdd : Ternary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::KernelExecute};
        };

        /**
         * Represents DEST = MatA * MatB + MatC.
         *
         * MatA is M x K, with B batches.  MatB is K x N, with B batches.  MatC is M x N, with B batches.
         */
        struct MatrixMultiply : Ternary
        {
            MatrixMultiply() = default;

            /**
             * @brief Construct a new Matrix Multiply object
             *
             * @param matA WaveTile. M x K, B batches
             * @param matB WaveTile. K x N, B batches
             * @param matC WaveTile. M x N, B batches
             */
            MatrixMultiply(ExpressionPtr matA, ExpressionPtr matB, ExpressionPtr matC)
                : Ternary{matA, matB, matC}
            {
            }

            DataType accumulationPrecision = DataType::Float;

            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::KernelExecute};
        };

        /**
         * Represents DEST = LHS ? R1HS : R2HS.
         * Utilizes cselect
        */
        struct Conditional : Ternary
        {
            constexpr static inline auto Type      = Category::Arithmetic;
            constexpr static inline auto EvalTimes = EvaluationTimes::All();
        };

        /**
         * Represents DEST = LHS * R1HS + R2HS.
         * Utilizes TernaryMixed instead of Ternary
         * allows for mixed precision arithmetic
         */
        struct MultiplyAdd : TernaryMixed
        {
            constexpr static inline auto Type        = Category::Arithmetic;
            constexpr static inline auto EvalTimes   = EvaluationTimes::All();
            constexpr static inline bool Associative = false;
            constexpr static inline bool Commutative = false;
        };

        struct Unary
        {
            ExpressionPtr arg;
            std::string   comment = "";
        };

        template <typename T>
        concept CUnary = requires
        {
            requires std::derived_from<T, Unary>;
        };

        struct MagicMultiple : Unary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate,
                                                              EvaluationTime::KernelLaunch};
        };

        struct MagicShifts : Unary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate,
                                                              EvaluationTime::KernelLaunch};
        };

        struct MagicSign : Unary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate,
                                                              EvaluationTime::KernelLaunch};
        };

        struct Negate : Unary
        {
            constexpr static inline auto Type      = Category::Arithmetic;
            constexpr static inline auto EvalTimes = EvaluationTimes::All();
        };

        template <DataType DATATYPE>
        struct Convert : Unary
        {
            constexpr static inline auto Type      = Category::Conversion;
            constexpr static inline auto EvalTimes = EvaluationTimes::All();
        };

        /**
         * @brief Register value from the coordinate graph.
         *
         * If the register associated with the `tag` hasn't been
         * allocated yet, a new register is created based on `regType`
         * and `varType`.
         *
         * If `varType` is `DataType::None`, the data type is
         * "deferred".
         */
        struct DataFlowTag
        {
            int tag;

            Register::Type regType;
            VariableType   varType;

            bool operator==(DataFlowTag const&) const = default;
        };

        ExpressionPtr operator+(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator-(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator*(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator/(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator%(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator<<(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator>>(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator&(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator>(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator>=(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator<(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator<=(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator==(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr operator&&(ExpressionPtr a, ExpressionPtr b);

        ExpressionPtr operator-(ExpressionPtr a);

        ExpressionPtr multiplyHigh(ExpressionPtr a, ExpressionPtr b);

        // arithmeticShiftR is the same as >>
        ExpressionPtr arithmeticShiftR(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr logicalShiftR(ExpressionPtr a, ExpressionPtr b);

        ExpressionPtr magicMultiple(ExpressionPtr a);
        ExpressionPtr magicShifts(ExpressionPtr a);
        ExpressionPtr magicSign(ExpressionPtr a);

        ExpressionPtr convert(DataType dt, ExpressionPtr a);

        template <DataType DATATYPE>
        ExpressionPtr convert(ExpressionPtr a);

        template <CCommandArgumentValue T>
        ExpressionPtr literal(T value);

        /**
         * @brief Create an Expression representing a literal value with a
         *        specific datatype. Does not accept pointer variable types.
         *
         * @tparam T
         * @param value The value to represent.
         * @param v The datatype of value.
         * @return ExpressionPtr
         */
        template <CCommandArgumentValue T>
        ExpressionPtr literal(T value, VariableType v);

        template <typename T>
        concept CValue = requires
        {
            // clang-format off
            requires std::same_as<AssemblyKernelArgumentPtr, T> ||
                     std::same_as<CommandArgumentPtr, T>        ||
                     std::same_as<CommandArgumentValue, T>      ||
                     std::same_as<DataFlowTag, T>               ||
                     std::same_as<Register::ValuePtr, T>        ||
                     std::same_as<WaveTilePtr, T>;
            // clang-format on
        };

        template <Category cat, typename T>
        concept COpCategory = requires
        {
            requires static_cast<Category>(T::Type) == cat;
        };

        template <typename T>
        concept CArithmetic = requires
        {
            requires static_cast<Category>(T::Type) == Category::Arithmetic;
        };

        template <typename T>
        concept CComparison = requires
        {
            requires static_cast<Category>(T::Type) == Category::Comparison;
        };

        template <typename T>
        concept CLogical = requires
        {
            requires static_cast<Category>(T::Type) == Category::Logical;
        };

        template <typename T>
        concept CShift
            = std::same_as<ShiftL, T> || std::same_as<LogicalShiftR,
                                                      T> || std::same_as<ArithmeticShiftR, T>;

        template <typename T>
        concept CAssociativeBinary = requires
        {
            requires CBinary<T> && T::Properties[AlgebraicProperty::Associative] == true;
        };

        template <typename T>
        concept CCommutativeBinary = requires
        {
            requires CBinary<T> && T::Properties[AlgebraicProperty::Commutative] == true;
        };

        static_assert(CBinary<Add>);
        static_assert(CArithmetic<Add>);
        static_assert(!CComparison<Add>);
        static_assert(!CBinary<Register::ValuePtr>);
        static_assert(CAssociativeBinary<Add>);
        static_assert(!CAssociativeBinary<Subtract>);

        template <typename T>
        concept CTranslateTimeValue = requires
        {
            requires std::same_as<CommandArgumentValue, T>;
        };

        template <typename T>
        concept CTranslateTimeOperation = requires
        {
            requires T::EvalTimes[EvaluationTime::Translate] == true;
        };

        template <typename T>
        concept CTranslateTime = requires
        {
            requires CTranslateTimeValue<T> || CTranslateTimeOperation<T>;
        };

        template <typename T>
        concept CKernelLaunchTimeValue = requires
        {
            requires std::same_as<CommandArgumentValue, T> || std::same_as<CommandArgumentPtr, T>;
        };

        template <typename T>
        concept CKernelLaunchTimeOperation = requires
        {
            requires T::EvalTimes[EvaluationTime::KernelLaunch] == true;
        };

        template <typename T>
        concept CKernelLaunchTime = requires
        {
            requires CKernelLaunchTimeValue<T> || CKernelLaunchTimeOperation<T>;
        };

        template <typename T>
        concept CKernelExecuteTimeValue = requires
        {
            // clang-format off
            requires std::same_as<AssemblyKernelArgumentPtr, T> ||
                     std::same_as<CommandArgumentValue, T>      ||
                     std::same_as<DataFlowTag, T>               ||
                     std::same_as<Register::ValuePtr, T>        ||
                     std::same_as<WaveTilePtr, T>;
            // clang-format on
        };

        template <typename T>
        concept CKernelExecuteTimeOperation = requires
        {
            requires(T::EvalTimes[EvaluationTime::KernelExecute] == true);
        };

        template <typename T>
        concept CKernelExecuteTime = requires
        {
            requires CKernelExecuteTimeValue<T> || CKernelExecuteTimeOperation<T>;
        };

        static_assert(CTranslateTime<Add>);
        static_assert(CTranslateTime<MagicMultiple>);

        static_assert(CKernelLaunchTime<Add>);
        static_assert(CKernelLaunchTime<MagicMultiple>);

        static_assert(CKernelExecuteTime<Add>);
        static_assert(CKernelExecuteTime<Multiply>);
        static_assert(!CKernelExecuteTime<MagicMultiple>);

        //
        // Other visitors
        //

        std::string   toString(ExpressionPtr const& expr);
        std::string   toString(Expression const& expr);
        std::ostream& operator<<(std::ostream&, ExpressionPtr const&);
        std::ostream& operator<<(std::ostream&, Expression const&);
        std::ostream& operator<<(std::ostream&, std::vector<ExpressionPtr> const&);

        std::string name(ExpressionPtr const& expr);
        std::string name(Expression const& expr);

        // EvaluationTime max(EvaluationTime lhs, EvaluationTime rhs);

        EvaluationTimes evaluationTimes(ExpressionPtr const& expr);
        EvaluationTimes evaluationTimes(Expression const& expr);

        VariableType   resultVariableType(ExpressionPtr const& expr);
        Register::Type resultRegisterType(ExpressionPtr const& expr);

        struct ResultType
        {
            Register::Type regType;
            VariableType   varType;
            bool           operator==(ResultType const&) const = default;
        };
        ResultType    resultType(ExpressionPtr const& expr);
        ResultType    resultType(Expression const& expr);
        std::ostream& operator<<(std::ostream&, ResultType const&);

        /**
         * True when two expressions are identical.
         *
         * NOTE: Never considers commutativity or associativity.
         */
        bool identical(ExpressionPtr const&, ExpressionPtr const&);

        /**
         * True when two expressions are equivalent.
         * Optionally considers algebraic properties like commutativity.
         */
        bool equivalent(ExpressionPtr const&,
                        ExpressionPtr const&,
                        AlgebraicProperties = AlgebraicProperties::All());

        /**
         * Comment accessors.
         */
        void setComment(ExpressionPtr& expr, std::string comment);
        void setComment(Expression& expr, std::string comment);

        std::string getComment(ExpressionPtr const& expr);
        std::string getComment(Expression const& expr);

        void appendComment(ExpressionPtr& expr, std::string comment);
        void appendComment(Expression& expr, std::string comment);

        /**
         * Evaluate an expression whose evaluationTime is Translate.  Will throw an exception otherwise.
         */
        CommandArgumentValue evaluate(ExpressionPtr const& expr);
        CommandArgumentValue evaluate(Expression const& expr);

        bool canEvaluateTo(CommandArgumentValue val, ExpressionPtr const& expr);

        /**
         * Evaluate an expression whose evaluationTime is Translate or KernelLaunch.  Will throw an exception if it contains any Register values.
         */
        CommandArgumentValue evaluate(ExpressionPtr const& expr, RuntimeArguments const& args);
        CommandArgumentValue evaluate(Expression const& expr, RuntimeArguments const& args);

        Generator<Instruction>
            generate(Register::ValuePtr& dest, ExpressionPtr expr, ContextPtr context);

        std::string   toYAML(ExpressionPtr const& expr);
        ExpressionPtr fromYAML(std::string const& str);

    } // namespace Expression
} // namespace rocRoller

#include "Expression_impl.hpp"
