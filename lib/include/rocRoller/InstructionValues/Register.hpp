#pragma once

#include <bit>
#include <cassert>
#include <concepts>

#include "LDSAllocator.hpp"
#include "RegisterAllocator_fwd.hpp"
#include "Register_fwd.hpp"

#include "../CodeGen/Instruction_fwd.hpp"
#include "../Context_fwd.hpp"
#include "../DataTypes/DataTypes.hpp"
#include "../Expression_fwd.hpp"
#include "../Operations/CommandArgument_fwd.hpp"
#include "../Scheduling/Scheduling.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    /**
     * @brief
     *
     * TODO: Figure out how to handle multi-dimensional arrays of registers
     */
    namespace Register
    {
        enum
        {
            /// Contiguity equal to element count
            FULLY_CONTIGUOUS = -3,

            /// Suffucient contiguity to fit the datatype
            VALUE_CONTIGUOUS = -2,

            /// Won't be using allocator
            /// (e.g. taking allocation from elsewhere or assigning particular register numbers)
            MANUAL = -1,
        };
        struct AllocationOptions
        {
            /// In units of registers
            int contiguousChunkWidth = VALUE_CONTIGUOUS;

            /// Allocation x must have (x % alignment) == alignmentPhase. -1 means to use default for register type.
            int alignment      = -1;
            int alignmentPhase = 0;

            static AllocationOptions FullyContiguous();
        };

        struct RegisterId
        {
            RegisterId(Type regType, int index)
                : regType(regType)
                , regIndex(index)
            {
            }
            Type        regType;
            int         regIndex;
            auto        operator<=>(RegisterId const&) const = default;
            std::string toString() const;
        };

        std::string toString(RegisterId const& regId);

        // For some reason, GCC will not find the operator declared in Utils.hpp.
        std::ostream& operator<<(std::ostream& stream, RegisterId const& regId);

        struct RegisterIdHash
        {
            size_t operator()(RegisterId const& regId) const noexcept
            {
                size_t h1 = static_cast<size_t>(regId.regType);
                size_t h2 = static_cast<size_t>(regId.regIndex);
                return h1 | (h2 << std::bit_width(static_cast<unsigned int>(Type::Count)));
            }
        };

        std::string TypePrefix(Type t);

        /**
         * Returns a register type suitable for holding the result of an arithmetic
         * operation between two types.  Generally Literal -> Scalar -> Vector.
         */
        constexpr Type PromoteType(Type lhs, Type rhs);

        /**
         * @brief Says whether a Register::Type represents an actual register, as opposed
         * to a literal, or other non-register value.
         *
         * @param t Register::Type to test
         * @return true If the type represents an actual register
         * @return false If the type does not represent an actual register
         */
        constexpr bool IsRegister(Type t);

        /**
         * @brief Says whether a Register::Type represents a special register.
         *
         * @param t Register::Type to test
         * @return true If the type represents a special
         * @return false If the type does not represent a special register.
         */
        constexpr bool IsSpecial(Type t);

        /**
         * Represents a single value (or single value per lane) stored in one or more registers,
         * or a literal value, or a one-dimensional array of registers suitable for use in a
         * MFMA or similar instruction.
         *
         * Maintains a `shared_ptr` reference to the `Allocation` object.
         *
         */
        struct Value : public std::enable_shared_from_this<Value>
        {
        public:
            Value();

            ~Value();

            template <CCommandArgumentValue T>
            static ValuePtr Literal(T const& value);

            static ValuePtr Literal(CommandArgumentValue const& value);

            static ValuePtr Label(const std::string& label);

            /**
             * Placeholder value to be filled in later.
             */
            static ValuePtr Placeholder(ContextPtr        ctx,
                                        Type              regType,
                                        VariableType      varType,
                                        int               count,
                                        AllocationOptions allocOptions = {});

            static ValuePtr WavefrontPlaceholder(ContextPtr context);

            static ValuePtr AllocateLDS(ContextPtr   ctx,
                                        VariableType varType,
                                        int          count,
                                        unsigned int alignment = 4);

            AllocationState allocationState() const;

            AllocationPtr allocation() const;

            std::vector<int> allocationCoord() const;

            //> Returns a new instruction that only allocates registers for this value.
            Instruction allocate();
            void        allocate(Instruction& inst);

            bool canAllocateNow() const;
            void allocateNow();

            bool isPlaceholder() const;
            bool isZeroLiteral() const;

            bool isSpecial() const;
            bool isSCC() const;
            bool isVCC() const;
            bool isExec() const;

            /**
             * Asserts that `this` is in a valid state to be used as an operand to an instruction.
             */
            void assertCanUseAsOperand() const;
            bool canUseAsOperand() const;

            /**
             * Returns a new unallocated RegisterValue with the same characteristics (register type,
             * data type, count, etc.)
             */
            ValuePtr placeholder() const;

            /**
             * Returns a new unallocated Value with the specified register type but
             * the same other properties.
             */
            ValuePtr placeholder(Type regType, AllocationOptions allocOptions) const;

            Type         regType() const;
            VariableType variableType() const;

            void setVariableType(VariableType value);

            void        toStream(std::ostream& os) const;
            std::string toString() const;
            std::string description() const;

            Value(ContextPtr        ctx,
                  Type              regType,
                  VariableType      variableType,
                  int               count,
                  AllocationOptions options = {});

            Value(ContextPtr         ctx,
                  Type               regType,
                  VariableType       variableType,
                  std::vector<int>&& coord);

            template <std::ranges::input_range T>
            Value(ContextPtr ctx, Type regType, VariableType variableType, T const& coord);

            Value(AllocationPtr alloc, Type regType, VariableType variableType, int count);

            Value(AllocationPtr      alloc,
                  Type               regType,
                  VariableType       variableType,
                  std::vector<int>&& coord);

            template <std::ranges::input_range T>
            Value(AllocationPtr alloc, Type regType, VariableType variableType, T& coord);

            std::string name() const;
            void        setName(std::string name);

            /**
             * Return negated copy.
             */
            ValuePtr negate() const;

            /**
             * Return subset of 32bit registers from multi-register values; always DataType::Raw32.
             *
             * For example,
             *
             *   auto v = Value(Register::Type::Vector, DataType::Double, count=4);
             *
             * represents 4 64-bit floating point numbers and spans 8
             * 32-bit registers.  Then
             *
             *   v->subset({1})
             *
             * would give v1, a single 32-bit register.
             */
            template <std::ranges::forward_range T>
            ValuePtr subset(T const& indices) const;

            template <std::integral T>
            ValuePtr subset(std::initializer_list<T> indices) const;

            /**
             * Splits the registers allocated into individual values.
             *
             * For each entry in `indices`, will return a `Value` that now has ownership
             * over those individual registers.
             *
             * The indices may not have any overlap, and any registers assigned to `this`
             * that are not used will be freed.

             * `this` will be in an unallocated state.
             *
             */
            std::vector<ValuePtr> split(std::vector<std::vector<int>> const& indices);

            bool intersects(Register::ValuePtr) const;

            /**
             * Return sub-elements of multi-value values.
             *
             * For example,
             *
             *   auto v = Value(Register::Type::Vector, DataType::Double, count=4);
             *
             * represents 4 64-bit floating point numbers and spans 8
             * 32-bit registers.  Then
             *
             *   v->element({1})
             *
             * would give v[2:3], a single 64-bit floating point value
             * (that spans two 32-bit registers).
             */
            template <std::ranges::forward_range T>
            ValuePtr element(T const& indices) const;
            template <typename T>
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, ValuePtr>
                element(std::initializer_list<T> indices) const;

            size_t registerCount() const;
            size_t valueCount() const;

            bool           hasContiguousIndices() const;
            Generator<int> registerIndices() const;

            Generator<RegisterId> getRegisterIds() const;

            std::string getLiteral() const;

            /**
             * Return a literal's actual value.
             */
            CommandArgumentValue getLiteralValue() const;

            std::string getLabel() const;

            std::shared_ptr<LDSAllocation> getLDSAllocation() const;

            rocRoller::Expression::ExpressionPtr expression();

            ContextPtr context() const;

            /**
             * @brief Determine if the other register is the same as this one
             *
             */
            bool sameAs(ValuePtr) const;

        private:
            /**
             * Implementation of toString() for general-purpose registers.
             */
            void gprString(std::ostream& os) const;

            /**
             * Implementation of toString() for special registers.
             */
            void specialString(std::ostream& os) const;

            /**
             * Must only be called during `split()`.
             * Creates a new Allocation which consists of only the registers assigned to `this`.
             * The parent `Value` and `Allocation` will be left in an invalid state.
             */
            void takeAllocation();

            friend class Allocation;

            std::weak_ptr<Context> m_context;

            std::string m_name;

            std::string m_label;

            CommandArgumentValue m_literalValue;

            AllocationPtr                  m_allocation;
            std::shared_ptr<LDSAllocation> m_ldsAllocation;

            Type         m_regType = Type::Count;
            VariableType m_varType;
            bool         m_negate = false;

            /**
             * Pulls values from the allocation
             */
            std::vector<int> m_allocationCoord;
            /**
             * If true, m_indices contains a contiguous set of
             * numbers, so we can represent as a range, e.g. v[0:3]
             * If false, we must be represented as a list, e.g. [v0, v2, v5]
             * If no value, state is unknown and we will check when converting to string.
             */
            mutable std::optional<bool> m_contiguousIndices;

            void updateContiguousIndices() const;
        };

        ValuePtr Representative(std::initializer_list<ValuePtr> values);

        /**
         * Represents one (possible) allocation of register(s) that are thought of collectively.
         *
         * TODO: Make not copyable, enforce construction through shared_ptr
         */
        struct Allocation : public std::enable_shared_from_this<Allocation>
        {
            Allocation(ContextPtr        context,
                       Type              regType,
                       VariableType      variableType,
                       int               count   = 1,
                       AllocationOptions options = {});

            ~Allocation();

            static AllocationPtr
                SameAs(Value const& val, std::string name, AllocationOptions const& options);

            Instruction allocate();

            bool canAllocateNow() const;
            void allocateNow();

            Type regType() const;

            AllocationState allocationState() const;

            ValuePtr operator*();

            std::string descriptiveComment(std::string const& prefix) const;

            int               registerCount() const;
            AllocationOptions options() const;

            std::vector<int> const& registerIndices() const;

            void setAllocation(std::shared_ptr<Allocator> allocator,
                               std::vector<int> const&    registers);
            void setAllocation(std::shared_ptr<Allocator> allocator, std::vector<int>&& registers);

            void free();

            std::string name() const;
            void        setName(std::string name);

        private:
            friend class Value;
            friend class Allocator;

            std::weak_ptr<Context> m_context;

            Type         m_regType;
            VariableType m_variableType;

            AllocationOptions m_options;

            int m_valueCount;
            int m_registerCount;

            AllocationState m_allocationState = AllocationState::Unallocated;

            std::shared_ptr<Allocator> m_allocator;
            std::vector<int>           m_registerIndices;

            std::string m_name;

            void setRegisterCount();
        };

        std::string   toString(AllocationOptions const& opts);
        std::ostream& operator<<(std::ostream& stream, AllocationOptions const& opts);
    }
}

#include "Register_impl.hpp"
