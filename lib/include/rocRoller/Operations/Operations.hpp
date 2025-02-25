#pragma once

#include <unordered_set>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Operations/Operations_fwd.hpp>
#include <rocRoller/Operations/RandomNumberGenerator.hpp>
#include <rocRoller/Operations/T_Execute_fwd.hpp>

#include <rocRoller/Operations/BlockScale.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Operations/T_LoadStore.hpp>
#include <rocRoller/Operations/T_Mul.hpp>
#include <rocRoller/Operations/TensorScalar.hpp>

namespace rocRoller
{
    namespace Operations
    {
        struct Nop
        {
            Nop() {}
            template <typename... Args>
            Nop(Args&&... i)
            {
            }

            auto operator<=>(Nop const&) const = default;
        };

        struct Inputs
        {
            std::unordered_set<OperationTag> call(Operation const&);

            std::unordered_set<OperationTag> operator()(Tensor const&);
            std::unordered_set<OperationTag> operator()(Scalar const&);
            std::unordered_set<OperationTag> operator()(Literal const&);
            std::unordered_set<OperationTag> operator()(BlockScale const&);
            std::unordered_set<OperationTag> operator()(T_Load_Linear const&);
            std::unordered_set<OperationTag> operator()(T_Load_Scalar const&);
            std::unordered_set<OperationTag> operator()(T_Load_Tiled const&);
            std::unordered_set<OperationTag> operator()(T_Mul const&);
            std::unordered_set<OperationTag> operator()(T_Store_Linear const&);
            std::unordered_set<OperationTag> operator()(T_Store_Tiled const&);
            std::unordered_set<OperationTag> operator()(T_Execute const&);
            std::unordered_set<OperationTag> operator()(RandomNumberGenerator const&);

            std::unordered_set<OperationTag> call(XOp const&);
            std::unordered_set<OperationTag> operator()(E_Unary const&);
            std::unordered_set<OperationTag> operator()(E_Binary const&);
            std::unordered_set<OperationTag> operator()(E_Ternary const&);
            std::unordered_set<OperationTag> operator()(Nop const&);
        };

        struct Outputs
        {
            std::unordered_set<OperationTag> call(Operation const&);

            std::unordered_set<OperationTag> operator()(Tensor const&);
            std::unordered_set<OperationTag> operator()(Scalar const&);
            std::unordered_set<OperationTag> operator()(Literal const&);
            std::unordered_set<OperationTag> operator()(BlockScale const&);
            std::unordered_set<OperationTag> operator()(T_Load_Linear const&);
            std::unordered_set<OperationTag> operator()(T_Load_Scalar const&);
            std::unordered_set<OperationTag> operator()(T_Load_Tiled const&);
            std::unordered_set<OperationTag> operator()(T_Mul const&);
            std::unordered_set<OperationTag> operator()(T_Store_Linear const&);
            std::unordered_set<OperationTag> operator()(T_Store_Tiled const&);
            std::unordered_set<OperationTag> operator()(T_Execute const&);
            std::unordered_set<OperationTag> operator()(RandomNumberGenerator const&);

            std::unordered_set<OperationTag> call(XOp const&);
            std::unordered_set<OperationTag> operator()(E_Unary const&);
            std::unordered_set<OperationTag> operator()(E_Binary const&);
            std::unordered_set<OperationTag> operator()(E_Ternary const&);
            std::unordered_set<OperationTag> operator()(Nop const&);
        };

        struct TagVisitor
        {
            OperationTag call(XOp const&);
            OperationTag operator()(E_Unary const&);
            OperationTag operator()(E_Binary const&);
            OperationTag operator()(E_Ternary const&);
        };

        struct AssignOutputs
        {
            std::unordered_set<OperationTag> call(Operation&, OperationTag);

            std::unordered_set<OperationTag> operator()(Tensor&);
            std::unordered_set<OperationTag> operator()(Scalar&);
            std::unordered_set<OperationTag> operator()(Literal&);
            std::unordered_set<OperationTag> operator()(BlockScale&);
            std::unordered_set<OperationTag> operator()(T_Load_Linear&);
            std::unordered_set<OperationTag> operator()(T_Load_Scalar&);
            std::unordered_set<OperationTag> operator()(T_Load_Tiled&);
            std::unordered_set<OperationTag> operator()(T_Mul&);
            std::unordered_set<OperationTag> operator()(T_Store_Linear&);
            std::unordered_set<OperationTag> operator()(T_Store_Tiled&);
            std::unordered_set<OperationTag> operator()(T_Execute&);
            std::unordered_set<OperationTag> operator()(RandomNumberGenerator&);

            std::unordered_set<OperationTag> call(XOp&, OperationTag);
            std::unordered_set<OperationTag> operator()(E_Unary&);
            std::unordered_set<OperationTag> operator()(E_Binary&);
            std::unordered_set<OperationTag> operator()(E_Ternary&);
            std::unordered_set<OperationTag> operator()(Nop&);

        private:
            OperationTag m_nextTagValue;
        };

        struct ToStringVisitor
        {
            std::string call(Operation const&, const unsigned char*);

            std::string operator()(Tensor const&);
            std::string operator()(Scalar const&);
            std::string operator()(Literal const&);
            std::string operator()(BlockScale const&);
            std::string operator()(T_Load_Linear const&);
            std::string operator()(T_Load_Scalar const&);
            std::string operator()(T_Load_Tiled const&);
            std::string operator()(T_Mul const&);
            std::string operator()(T_Store_Linear const&);
            std::string operator()(T_Store_Tiled const&);
            std::string operator()(T_Execute const&);
            std::string operator()(RandomNumberGenerator const&);

            std::string call(XOp const&);
            std::string operator()(E_Unary const&);
            std::string operator()(E_Binary const&);
            std::string operator()(E_Ternary const&);
            std::string operator()(Nop const&);

        private:
            const unsigned char* m_runtimeArgs;
        };

        struct SetCommand
        {
            SetCommand(CommandPtr);

            void call(Operation&);

            void operator()(Tensor&);
            void operator()(Scalar&);
            void operator()(Literal&);
            void operator()(BlockScale&);
            void operator()(T_Load_Linear&);
            void operator()(T_Load_Scalar&);
            void operator()(T_Load_Tiled&);
            void operator()(T_Mul&);
            void operator()(T_Store_Linear&);
            void operator()(T_Store_Tiled&);
            void operator()(T_Execute&);
            void operator()(Nop&);
            void operator()(RandomNumberGenerator&);

            CommandPtr command;
        };

        struct AllocateArguments
        {
            void call(Operation&);

            void operator()(Tensor&);
            void operator()(Scalar&);
            void operator()(Literal&);
            void operator()(BlockScale&);
            void operator()(T_Load_Linear&);
            void operator()(T_Load_Scalar&);
            void operator()(T_Load_Tiled&);
            void operator()(T_Mul&);
            void operator()(T_Store_Linear&);
            void operator()(T_Store_Tiled&);
            void operator()(T_Execute&);
            void operator()(Nop&);
            void operator()(RandomNumberGenerator&);
        };

        struct VariableTypeVisitor
        {
            rocRoller::VariableType call(Operation&);

            rocRoller::VariableType operator()(Tensor&);
            rocRoller::VariableType operator()(Scalar&);
            rocRoller::VariableType operator()(Literal&);
            rocRoller::VariableType operator()(BlockScale&);
            rocRoller::VariableType operator()(T_Load_Linear&);
            rocRoller::VariableType operator()(T_Load_Scalar&);
            rocRoller::VariableType operator()(T_Load_Tiled&);
            rocRoller::VariableType operator()(T_Mul&);
            rocRoller::VariableType operator()(T_Store_Linear&);
            rocRoller::VariableType operator()(T_Store_Tiled&);
            rocRoller::VariableType operator()(T_Execute&);
            rocRoller::VariableType operator()(Nop&);
            rocRoller::VariableType operator()(RandomNumberGenerator&);
        };

    }
}

#include <rocRoller/Operations/Operations_impl.hpp>
