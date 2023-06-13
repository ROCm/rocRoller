#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{
    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::LessThanEqual>>
        GetGenerator<Expression::LessThanEqual>(Register::ValuePtr dst,
                                                Register::ValuePtr lhs,
                                                Register::ValuePtr rhs);

    // Templated Generator class based on the register type and datatype.
    template <Register::Type REGISTER_TYPE, DataType DATATYPE>
    class LessThanEqualGenerator : public BinaryArithmeticGenerator<Expression::LessThanEqual>
    {
    public:
        LessThanEqualGenerator<REGISTER_TYPE, DATATYPE>(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::LessThanEqual>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(Argument const& arg)
        {
            ContextPtr     ctx;
            Register::Type registerType;
            DataType       dataType;

            std::tie(ctx, registerType, dataType) = arg;

            return registerType == REGISTER_TYPE && dataType == DATATYPE;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<LessThanEqualGenerator<REGISTER_TYPE, DATATYPE>>(
                std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction>
            generate(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

        static const std::string Name;
    };

    // Specializations for supported Register Type / DataType combinations
    template <>
    Generator<Instruction>
        LessThanEqualGenerator<Register::Type::Scalar, DataType::Int32>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction>
        LessThanEqualGenerator<Register::Type::Vector, DataType::Int32>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction>
        LessThanEqualGenerator<Register::Type::Scalar, DataType::Int64>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction>
        LessThanEqualGenerator<Register::Type::Vector, DataType::Int64>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction>
        LessThanEqualGenerator<Register::Type::Vector, DataType::Float>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction>
        LessThanEqualGenerator<Register::Type::Vector, DataType::Double>::generate(
            Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
}
