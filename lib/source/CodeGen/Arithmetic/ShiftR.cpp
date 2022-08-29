#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/ShiftR.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(ShiftRGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::ShiftR>> GetGenerator<Expression::ShiftR>(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::ShiftR>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> ShiftRGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                     std::shared_ptr<Register::Value> value,
                                                     std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        auto elementSize = std::max({DataTypeInfo::Get(dest->variableType()).elementSize,
                                     DataTypeInfo::Get(value->variableType()).elementSize});

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("s_lshr_b32", {dest}, {value, shiftAmount}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(
                    Instruction("s_ashr_i64", {dest}, {value, shiftAmount->subset({0})}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift right operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("v_lshrrev_b32", {dest}, {shiftAmount, value}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(Instruction(
                    "v_ashrrev_i64", {dest}, {shiftAmount->subset({0}), value}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift right operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for shift right operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
