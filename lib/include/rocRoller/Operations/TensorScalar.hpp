#pragma once

#include "../DataTypes/DataTypes.hpp"
#include "CommandArgument.hpp"
#include "CommandArguments_fwd.hpp"
#include "Operation.hpp"
#include "TensorScalar_fwd.hpp"

namespace rocRoller
{
    namespace Operations
    {
        class Literal : public BaseOperation
        {
        public:
            Literal() = delete;
            Literal(CommandArgumentValue value);

            std::string          toString() const;
            CommandArgumentValue value() const;

            bool operator==(Literal const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            CommandArgumentValue m_value;
        };

        std::ostream& operator<<(std::ostream& stream, Literal const& val);

        class Scalar : public BaseOperation
        {
        public:
            Scalar() = delete;
            Scalar(VariableType variableType);

            std::string toString() const;
            std::string toString(const unsigned char*) const;

            void allocateArguments();

            CommandArgumentPtr data() const;
            VariableType       variableType() const;

            bool operator==(Scalar const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            std::string getArgumentString(const unsigned char*) const;

            CommandArgumentPtr m_pointer;
            VariableType       m_variableType;
        };

        std::ostream& operator<<(std::ostream& stream, Scalar const& val);

        class Tensor : public BaseOperation
        {
        public:
            Tensor() = delete;
            Tensor(int numDims, VariableType variableType);
            Tensor(int                        numDims,
                   VariableType               variableType,
                   std::vector<size_t> const& literalStrides);

            std::string toString() const;
            std::string toString(const unsigned char*) const;

            void allocateArguments();

            std::vector<size_t> const& literalStrides() const;

            std::vector<CommandArgumentPtr> const& strides() const;
            std::vector<CommandArgumentPtr> const& sizes() const;

            CommandArgumentPtr limit() const;
            CommandArgumentPtr data() const;

            VariableType variableType() const;
            DataType     dataType() const;

            bool operator==(Tensor const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            std::string getArgumentString(const unsigned char*) const;

            VariableType m_variableType;
            int          m_numDims = -1;

            CommandArgumentPtr m_pointer;
            CommandArgumentPtr m_extent;

            std::vector<CommandArgumentPtr> m_sizes;
            std::vector<CommandArgumentPtr> m_strides;

            std::vector<size_t> m_literalStrides;
        };

        std::ostream& operator<<(std::ostream& stream, Tensor const& val);
    }
}

#include "TensorScalar_impl.hpp"
