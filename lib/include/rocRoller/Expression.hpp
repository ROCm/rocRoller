/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#include <rocRoller/rocRoller.hpp>

#include <bitset>
#include <memory>
#include <stack>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/EnumBitset.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ROCROLLER_DECLSPEC std::string toString(EvaluationTime t);
        ROCROLLER_DECLSPEC std::ostream& operator<<(std::ostream&, EvaluationTime const&);

        using EvaluationTimes = EnumBitset<EvaluationTime>;

        enum class AlgebraicProperty : int
        {
            Commutative = 0,
            Associative,
            Count
        };
        ROCROLLER_DECLSPEC std::string toString(AlgebraicProperty t);
        ROCROLLER_DECLSPEC std::ostream& operator<<(std::ostream&, AlgebraicProperty const&);

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
        ROCROLLER_DECLSPEC std::string toString(Category c);
        ROCROLLER_DECLSPEC std::ostream& operator<<(std::ostream&, Category const&);

        // Expression: type alias for std::variant of all expression subtypes.
        // Defined in Expression_fwd.hpp.

        struct ROCROLLER_DECLSPEC Binary
        {
            ExpressionPtr lhs, rhs;
            std::string   comment = "";

            template <typename T>
            requires std::derived_from<T, Binary>
            inline T& copyParams(const T& other)
            {
                return static_cast<T&>(*this);
            }
        };

        template <typename T>
        concept CBinary = requires
        {
            requires std::derived_from<T, Binary>;
        };

        // Complexity is a heuristic that estimates the relative cost of computing different
        // expressions. See the KernelOption minLaunchTimeExpressionComplexity for a more
        // in-depth description.

        struct ROCROLLER_DECLSPEC Add : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC Subtract : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC Multiply : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 4;
        };

        struct ROCROLLER_DECLSPEC MultiplyHigh : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 4;
        };

        struct ROCROLLER_DECLSPEC Divide : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 50;
        };

        struct ROCROLLER_DECLSPEC Modulo : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 50;
        };

        struct ROCROLLER_DECLSPEC ShiftL : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC LogicalShiftR : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC ArithmeticShiftR : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC BitwiseAnd : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC BitwiseOr : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC BitwiseXor : Binary
        {
            constexpr static inline auto                Type      = Category::Arithmetic;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC GreaterThan : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC GreaterThanEqual : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC LessThan : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC LessThanEqual : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{};
            constexpr static inline int                 Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC Equal : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC NotEqual : Binary
        {
            constexpr static inline auto                Type      = Category::Comparison;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC LogicalAnd : Binary
        {
            constexpr static inline auto                Type      = Category::Logical;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC LogicalOr : Binary
        {
            constexpr static inline auto                Type      = Category::Logical;
            constexpr static inline auto                EvalTimes = EvaluationTimes::All();
            constexpr static inline AlgebraicProperties Properties{AlgebraicProperty::Associative,
                                                                   AlgebraicProperty::Commutative};
            constexpr static inline int                 Complexity = 1;
        };

        /*
         * SRConversion performs a stochastic rounding conversion.
         * The lhs is the value to be converted, the rhs is the seed
         * for stochastic rounding.
         */
        template <DataType DATATYPE>
        struct ROCROLLER_DECLSPEC SRConvert : Binary
        {
            constexpr static inline auto DestinationType = DATATYPE;
            constexpr static inline auto Type            = Category::Conversion;
            constexpr static inline auto EvalTimes       = EvaluationTimes::All();
            constexpr static inline int  Complexity      = 2;
        };

        struct ROCROLLER_DECLSPEC Ternary
        {
            ExpressionPtr lhs, r1hs, r2hs;
            std::string   comment = "";

            template <typename T>
            requires std::derived_from<T, Ternary>
            inline T& copyParams(const T& other)
            {
                return static_cast<T&>(*this);
            }
        };

        struct ROCROLLER_DECLSPEC TernaryMixed : Ternary
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
        struct ROCROLLER_DECLSPEC AddShiftL : Ternary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::KernelExecute};
            constexpr static inline int             Complexity = 2;
        };

        /*
         * ShiftLAdd performs a fusion of ShiftL expression followed by
         * Add expression, lowering to the fused instruction if possible.
         * result = (lhs << r1hs) + r2hs
         */
        struct ROCROLLER_DECLSPEC ShiftLAdd : Ternary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::KernelExecute};
            constexpr static inline int             Complexity = 2;
        };

        /**
         * Represents DEST = MatA * MatB + MatC.
         *
         * MatA is M x K, with B batches.  MatB is K x N, with B batches.  MatC is M x N, with B batches.
         */
        struct ROCROLLER_DECLSPEC MatrixMultiply : Ternary
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
            constexpr static inline int             Complexity = 20;
        };

        struct ROCROLLER_DECLSPEC ScaledMatrixMultiply
        {
            ExpressionPtr matA, matB, matC, scaleA, scaleB;
            DataType      accumulationPrecision = DataType::Float;
            std::string   comment               = "";

            ScaledMatrixMultiply() = default;
            ScaledMatrixMultiply(ExpressionPtr a,
                                 ExpressionPtr b,
                                 ExpressionPtr c,
                                 ExpressionPtr sA,
                                 ExpressionPtr sB)
                : matA(a)
                , matB(b)
                , matC(c)
                , scaleA(sA)
                , scaleB(sB)
            {
            }

            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::KernelExecute};
            constexpr static inline int             Complexity = 20;
        };

        /**
         * Represents DEST = LHS ? R1HS : R2HS.
         * Utilizes cselect
        */
        struct ROCROLLER_DECLSPEC Conditional : Ternary
        {
            constexpr static inline auto Type       = Category::Arithmetic;
            constexpr static inline auto EvalTimes  = EvaluationTimes::All();
            constexpr static inline int  Complexity = 4;
        };

        /**
         * Represents DEST = LHS * R1HS + R2HS.
         * Utilizes TernaryMixed instead of Ternary
         * allows for mixed precision arithmetic
         */
        struct ROCROLLER_DECLSPEC MultiplyAdd : TernaryMixed
        {
            constexpr static inline auto Type        = Category::Arithmetic;
            constexpr static inline auto EvalTimes   = EvaluationTimes::All();
            constexpr static inline bool Associative = false;
            constexpr static inline bool Commutative = false;
            constexpr static inline int  Complexity  = 4;
        };

        struct ROCROLLER_DECLSPEC Unary
        {
            ExpressionPtr arg;
            std::string   comment = "";

            template <typename T>
            requires std::derived_from<T, Unary>
            inline T& copyParams(const T& other)
            {
                return static_cast<T&>(*this);
            }
        };

        template <typename T>
        concept CUnary = requires
        {
            requires std::derived_from<T, Unary>;
        };

        struct ROCROLLER_DECLSPEC MagicMultiple : Unary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate,
                                                              EvaluationTime::KernelLaunch};
            constexpr static inline int             Complexity = 50;
        };

        struct ROCROLLER_DECLSPEC MagicShifts : Unary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate,
                                                              EvaluationTime::KernelLaunch};
            constexpr static inline int             Complexity = 50;
        };

        struct ROCROLLER_DECLSPEC MagicSign : Unary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate,
                                                              EvaluationTime::KernelLaunch};
            constexpr static inline int             Complexity = 50;
        };

        struct ROCROLLER_DECLSPEC Negate : Unary
        {
            constexpr static inline auto Type       = Category::Arithmetic;
            constexpr static inline auto EvalTimes  = EvaluationTimes::All();
            constexpr static inline int  Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC BitwiseNegate : Unary
        {
            constexpr static inline auto Type       = Category::Arithmetic;
            constexpr static inline auto EvalTimes  = EvaluationTimes::All();
            constexpr static inline int  Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC Convert : Unary
        {
            inline Convert& copyParams(const Convert& other)
            {
                destinationType = other.destinationType;

                return *this;
            }

            constexpr static inline auto Type       = Category::Conversion;
            constexpr static inline auto EvalTimes  = EvaluationTimes::All();
            constexpr static inline int  Complexity = 2;

            DataType destinationType = DataType::None;
        };

        struct ROCROLLER_DECLSPEC LogicalNot : Unary
        {
            constexpr static inline auto Type       = Category::Logical;
            constexpr static inline auto EvalTimes  = EvaluationTimes::All();
            constexpr static inline int  Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC Exponential2 : Unary
        {
            constexpr static inline auto Type       = Category::Arithmetic;
            constexpr static inline auto EvalTimes  = EvaluationTimes::All();
            constexpr static inline int  Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC Exponential : Unary
        {
            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate,
                                                              EvaluationTime::KernelLaunch};
            constexpr static inline int             Complexity = 2;
        };

        struct ROCROLLER_DECLSPEC RandomNumber : Unary
        {
            constexpr static inline auto Type       = Category::Arithmetic;
            constexpr static inline auto EvalTimes  = EvaluationTimes::All();
            constexpr static inline int  Complexity = 1;
        };

        struct ROCROLLER_DECLSPEC BitFieldExtract : Unary
        {
            inline BitFieldExtract& copyParams(const BitFieldExtract& other)
            {
                outputDataType = other.outputDataType;
                offset         = other.offset;
                width          = other.width;

                return *this;
            }

            constexpr static inline auto            Type = Category::Arithmetic;
            constexpr static inline EvaluationTimes EvalTimes{EvaluationTime::Translate};
            constexpr static inline int             Complexity = 1;

            DataType outputDataType = DataType::None;
            int      offset         = 0;
            int      width          = 0;
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
        struct ROCROLLER_DECLSPEC DataFlowTag
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
        ExpressionPtr operator||(ExpressionPtr a, ExpressionPtr b);

        ExpressionPtr operator-(ExpressionPtr a);
        ExpressionPtr logicalNot(ExpressionPtr a);

        ExpressionPtr multiplyHigh(ExpressionPtr a, ExpressionPtr b);

        // arithmeticShiftR is the same as >>
        ExpressionPtr arithmeticShiftR(ExpressionPtr a, ExpressionPtr b);
        ExpressionPtr logicalShiftR(ExpressionPtr a, ExpressionPtr b);

        ExpressionPtr magicMultiple(ExpressionPtr a);
        ExpressionPtr magicShifts(ExpressionPtr a);
        ExpressionPtr magicSign(ExpressionPtr a);

        ExpressionPtr convert(VariableType vt, ExpressionPtr a);
        ExpressionPtr convert(DataType dt, ExpressionPtr a);

        template <DataType DATATYPE>
        ExpressionPtr convert(ExpressionPtr a);

        ExpressionPtr bfe(DataType dt, ExpressionPtr a, uint8_t offset, uint8_t width);
        ExpressionPtr bfe(ExpressionPtr a, uint8_t offset, uint8_t width);

        template <CCommandArgumentValue T>
        ExpressionPtr literal(T value);

        ExpressionPtr fromKernelArgument(AssemblyKernelArgument const& arg);

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
        concept CValue = CIsAnyOf<T,
                                  AssemblyKernelArgumentPtr,
                                  CommandArgumentPtr,
                                  CommandArgumentValue,
                                  DataFlowTag,
                                  Register::ValuePtr,
                                  WaveTilePtr>;

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
        concept CConversion = requires
        {
            requires static_cast<Category>(T::Type) == Category::Conversion;
        };

        template <typename T>
        concept CShift = CIsAnyOf<T, ShiftL, LogicalShiftR, ArithmeticShiftR>;

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
        concept CTranslateTimeValue = std::same_as<T, CommandArgumentValue>;

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
        concept CKernelLaunchTimeValue = CIsAnyOf<T, CommandArgumentValue, CommandArgumentPtr>;

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
        concept CKernelExecuteTimeValue = CIsAnyOf<T,
                                                   AssemblyKernelArgumentPtr,
                                                   CommandArgumentValue,
                                                   DataFlowTag,
                                                   Register::ValuePtr,
                                                   WaveTilePtr>;

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

        ROCROLLER_DECLSPEC std::string toString(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC std::string toString(Expression const& expr);
        ROCROLLER_DECLSPEC std::ostream& operator<<(std::ostream&, ExpressionPtr const&);
        ROCROLLER_DECLSPEC std::ostream& operator<<(std::ostream&, Expression const&);
        ROCROLLER_DECLSPEC std::ostream& operator<<(std::ostream&,
                                                    std::vector<ExpressionPtr> const&);

        ROCROLLER_DECLSPEC std::string name(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC std::string name(Expression const& expr);

        ROCROLLER_DECLSPEC std::string argumentName(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC std::string argumentName(Expression const& expr);

        // EvaluationTime max(EvaluationTime lhs, EvaluationTime rhs);

        ROCROLLER_DECLSPEC EvaluationTimes evaluationTimes(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC EvaluationTimes evaluationTimes(Expression const& expr);

        ROCROLLER_DECLSPEC VariableType resultVariableType(Expression const& expr);
        ROCROLLER_DECLSPEC VariableType resultVariableType(ExpressionPtr const& expr);

        ROCROLLER_DECLSPEC Register::Type resultRegisterType(Expression const& expr);
        ROCROLLER_DECLSPEC Register::Type resultRegisterType(ExpressionPtr const& expr);

        struct ROCROLLER_DECLSPEC ResultType
        {
            Register::Type regType;
            VariableType   varType;
            bool           operator==(ResultType const&) const = default;
        };
        ROCROLLER_DECLSPEC ResultType resultType(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC ResultType resultType(Expression const& expr);

        ROCROLLER_DECLSPEC std::string toString(ResultType const& obj);
        ROCROLLER_DECLSPEC std::ostream& operator<<(std::ostream&, ResultType const&);

        /**
         * True when two expressions are identical.
         *
         * NOTE: Never considers commutativity or associativity.
         */
        ROCROLLER_DECLSPEC bool identical(ExpressionPtr const&, ExpressionPtr const&);
        ROCROLLER_DECLSPEC bool identical(Expression const&, Expression const&);

        /**
         * True when two expressions are equivalent.
         * Optionally considers algebraic properties like commutativity.
         */
        ROCROLLER_DECLSPEC bool equivalent(ExpressionPtr const&,
                                           ExpressionPtr const&,
                                           AlgebraicProperties = AlgebraicProperties::All());

        /**
         * Comment accessors.
         */
        ROCROLLER_DECLSPEC void setComment(ExpressionPtr& expr, std::string comment);
        ROCROLLER_DECLSPEC void setComment(Expression& expr, std::string comment);

        ROCROLLER_DECLSPEC std::string getComment(Expression const& expr,
                                                  bool              includeRegisterComments);
        ROCROLLER_DECLSPEC std::string getComment(ExpressionPtr const& expr,
                                                  bool                 includeRegisterComments);

        ROCROLLER_DECLSPEC std::string getComment(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC std::string getComment(Expression const& expr);
        ROCROLLER_DECLSPEC std::string getComment(ExpressionPtr const& expr,
                                                  bool                 includeRegisterComments);
        ROCROLLER_DECLSPEC std::string getComment(Expression const& expr,
                                                  bool              includeRegisterComments);

        /**
         * Copies any comments from src into dst.  If dst is not of a type that allows
         * comments, does nothing.
         */
        ROCROLLER_DECLSPEC void copyComment(ExpressionPtr const& dst, ExpressionPtr const& src);
        ROCROLLER_DECLSPEC void copyComment(Expression& dst, ExpressionPtr const& src);
        ROCROLLER_DECLSPEC void copyComment(ExpressionPtr const& dst, Expression const& src);
        ROCROLLER_DECLSPEC void copyComment(Expression& dst, Expression const& src);

        ROCROLLER_DECLSPEC void appendComment(ExpressionPtr& expr, std::string comment);
        ROCROLLER_DECLSPEC void appendComment(Expression& expr, std::string comment);

        /**
         * Evaluate an expression whose evaluationTime is Translate.  Will throw an exception
         * otherwise.
         */
        ROCROLLER_DECLSPEC CommandArgumentValue evaluate(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC CommandArgumentValue evaluate(Expression const& expr);

        /**
         * Evaluate an expression if its evaluationTime is Translate, returns nullopt
         * otherwise.
         */
        ROCROLLER_DECLSPEC std::optional<CommandArgumentValue>
                           tryEvaluate(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC std::optional<CommandArgumentValue> tryEvaluate(Expression const& expr);

        ROCROLLER_DECLSPEC bool canEvaluateTo(CommandArgumentValue val, ExpressionPtr const& expr);

        /**
         * Evaluate an expression whose evaluationTime is Translate or KernelLaunch.  Will throw
         * an exception if it contains any Register values.
         */
        ROCROLLER_DECLSPEC CommandArgumentValue evaluate(ExpressionPtr const&    expr,
                                                         RuntimeArguments const& args);
        ROCROLLER_DECLSPEC CommandArgumentValue evaluate(Expression const&       expr,
                                                         RuntimeArguments const& args);

        /**
         * Splits an expression and returns its operands in a tuple.
         *
         * Return type:
         * std::tuple<ExpressionPtr> for unary expressions
         * std::tuple<ExpressionPtr, ExpressionPtr> for binary expressions
         * std::tuple<ExpressionPtr, ExpressionPtr, ExpressionPtr> for ternary expressions
         *
         * Throws if expr is not of type Expr.
         */
        template <typename Expr>
        requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) ROCROLLER_DECLSPEC
            auto split(ExpressionPtr expr);

        /**
         * Returns an approximate total complexity for an expression, to be used as a heuristic.
         * See the KernelOption minLaunchTimeExpressionComplexity for a more in-depth
         * description.
         */
        ROCROLLER_DECLSPEC int complexity(ExpressionPtr expr);
        ROCROLLER_DECLSPEC int complexity(Expression const& expr);

        ROCROLLER_DECLSPEC Generator<Instruction>
            generate(Register::ValuePtr& dest, ExpressionPtr expr, ContextPtr context);

        ROCROLLER_DECLSPEC std::string   toYAML(ExpressionPtr const& expr);
        ROCROLLER_DECLSPEC ExpressionPtr fromYAML(std::string const& str);

        /**
         * Returns true if expr is of type T or if expr contains a subexpression of type T.
         */
        template <CExpression T>
        ROCROLLER_DECLSPEC bool contains(ExpressionPtr expr);

        /**
         * Returns true if expr is of type T or if expr contains a subexpression of type T.
         */
        template <CExpression T>
        ROCROLLER_DECLSPEC bool contains(Expression const& expr);

        /**
         * Returns true if expr contains a sub-expression
         */
        ROCROLLER_DECLSPEC bool containsSubExpression(ExpressionPtr const& expr,
                                                      ExpressionPtr const& subExpr);
        ROCROLLER_DECLSPEC bool containsSubExpression(Expression const& expr,
                                                      Expression const& subExpr);

    } // namespace Expression
} // namespace rocRoller

#include <rocRoller/Expression_impl.hpp>
