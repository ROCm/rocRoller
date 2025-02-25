#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/LessThan.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Scalar, DataType::Int32);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Scalar, DataType::UInt32);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Vector, DataType::Int32);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Vector, DataType::UInt32);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Scalar, DataType::Int64);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Scalar, DataType::UInt64);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Vector, DataType::Int64);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Vector, DataType::UInt64);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Vector, DataType::Float);
    RegisterComponentTemplateSpec(LessThanGenerator, Register::Type::Vector, DataType::Double);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::LessThan>>
        GetGenerator<Expression::LessThan>(Register::ValuePtr dst,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           Expression::LessThan const&)
    {
        // Choose the proper generator, based on the context, register type
        // and datatype.
        return Component::Get<BinaryArithmeticGenerator<Expression::LessThan>>(
            getContextFromValues(dst, lhs, rhs),
            promoteRegisterType(nullptr, lhs, rhs),
            promoteDataType(nullptr, lhs, rhs));
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        co_yield_(Instruction("s_cmp_lt_i32", {}, {lhs, rhs}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::UInt32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC,
                                       "Start Compare writing to non-SCC dest"));
        }

        co_yield_(Instruction("s_cmp_lt_u32", {}, {lhs, rhs}, {}, ""));

        if(dst != nullptr && !dst->isSCC())
        {
            co_yield m_context->copier()->copy(dst, m_context->getSCC(), "");
            co_yield(Instruction::Unlock("End Compare writing to non-SCC dest"));
        }
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_i32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::UInt32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_u32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield scalarCompareThroughVALU("v_cmp_lt_i64", dst, lhs, rhs);
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::UInt64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield scalarCompareThroughVALU("v_cmp_lt_u64", dst, lhs, rhs);
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_i64", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::UInt64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_u64", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_f32", {dst}, {lhs, rhs}, {}, ""));
    }

    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_f64", {dst}, {lhs, rhs}, {}, ""));
    }
}
