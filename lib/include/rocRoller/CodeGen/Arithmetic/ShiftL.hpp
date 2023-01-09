#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::ShiftL>> GetGenerator<Expression::ShiftL>(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

    // Generator for all register types and datatypes.
    class ShiftLGenerator : public BinaryArithmeticGenerator<Expression::ShiftL>
    {
    public:
        ShiftLGenerator(std::shared_ptr<Context> c)
            : BinaryArithmeticGenerator<Expression::ShiftL>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(Argument const& arg)
        {
            std::shared_ptr<Context> ctx;
            Register::Type           registerType;
            DataType                 dataType;

            std::tie(ctx, registerType, dataType) = arg;

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<ShiftLGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount);

        static const std::string Name;
    };
}
