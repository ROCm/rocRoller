#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::LogicalShiftR>>
        GetGenerator<Expression::LogicalShiftR>(Register::ValuePtr dst,
                                                Register::ValuePtr lhs,
                                                Register::ValuePtr rhs,
                                                Expression::LogicalShiftR const&);

    // Generator for all register types and datatypes.
    class LogicalShiftRGenerator : public BinaryArithmeticGenerator<Expression::LogicalShiftR>
    {
    public:
        LogicalShiftRGenerator(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::LogicalShiftR>(c)
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

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<LogicalShiftRGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dest,
                                        Register::ValuePtr value,
                                        Register::ValuePtr shiftAmount,
                                        Expression::LogicalShiftR const&);

        static const std::string Name;
    };
}
