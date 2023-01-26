#pragma once

#include "CommandArgument.hpp"

#include "../Utilities/Error.hpp"
#include "Expression.hpp"
#include "Utilities/Utils.hpp"

#include <cstring>

namespace rocRoller
{
    inline CommandArgument::CommandArgument(std::shared_ptr<Command> com,
                                            VariableType             variableType,
                                            size_t                   offset,
                                            DataDirection            direction,
                                            std::string              name)
        : m_command(com)
        , m_variableType(variableType)
        , m_offset(offset)
        , m_size(variableType.getElementSize())
        , m_direction(direction)
        , m_name(name)
    {
    }

    template <typename T>
    requires(!std::is_pointer_v<T>)
        CommandArgumentValue CommandArgument::getValue(RuntimeArguments const& args)
    const
    {
        if(m_variableType.pointerType == PointerType::Value)
        {
            AssertFatal(m_size == sizeof(T), ShowValue(m_size), ShowValue(sizeof(T*)));
            AssertFatal((m_offset + m_size) <= args.size(),
                        ShowValue(m_offset),
                        ShowValue(m_size),
                        ShowValue(args.size()));

            T rv;
            std::memcpy(&rv, &args[m_offset], sizeof(T));
            return rv;
        }

        if(m_variableType.pointerType == PointerType::PointerGlobal)
        {
            if constexpr(CCommandArgumentValue<T*>)
            {
                AssertFatal(m_size == sizeof(T*), ShowValue(m_size), ShowValue(sizeof(T*)));
                AssertFatal((m_offset + m_size) <= args.size(),
                            ShowValue(m_offset),
                            ShowValue(m_size),
                            ShowValue(args.size()));

                T* rv;
                std::memcpy(&rv, &args[m_offset], sizeof(T*));
                return rv;
            }
        }

        throw std::runtime_error(
            concatenate("Invalid variable type for command argument:", m_variableType));
    }

    inline std::string CommandArgument::toString() const
    {
        return concatenate(m_name,
                           ": ",
                           m_variableType,
                           "(offset: ",
                           m_offset,
                           ", size: ",
                           m_size,
                           ", ",
                           m_direction,
                           ")");
    }

    inline CommandArgumentValue CommandArgument::getValue(RuntimeArguments const& args) const
    {
        if(args.data() == nullptr)
            throw std::runtime_error("Can't get runtime argument value.");

        switch(m_variableType.dataType)
        {
        case DataType::Float:
            return getValue<float>(args);
        case DataType::Double:
            return getValue<double>(args);
        // case DataType::ComplexFloat:
        //     return getValue<std::complex<float>>(args);
        // case DataType::ComplexDouble:
        //     return getValue<std::complex<double>>(args);
        case DataType::Half:
            return getValue<Half>(args);
        // case DataType::Int8x4:
        //     return getValue<Int8x4>(args);
        case DataType::Int32:
            return getValue<int32_t>(args);
        case DataType::Int64:
            return getValue<int64_t>(args);
        // case DataType::BFloat16:
        //     return getValue<BFloat16>(args);
        // case DataType::Int8:
        //     return getValue<int8_t>(args);
        case DataType::Raw32:
            return getValue<uint32_t>(args);
        case DataType::UInt32:
            return getValue<uint32_t>(args);
        case DataType::UInt64:
            return getValue<uint64_t>(args);
        case DataType::Bool:
            return getValue<bool>(args);
        case DataType::Count:
        default:
            throw std::runtime_error("Unsupported argument type.");
        }
    }

    inline Expression::ExpressionPtr CommandArgument::expression()
    {
        return std::make_shared<Expression::Expression>(shared_from_this());
    }

    inline bool CommandArgument::operator==(const CommandArgument& rhs) const
    {
        return m_size == rhs.m_size && m_offset == rhs.m_offset
               && m_variableType == rhs.m_variableType && m_direction == rhs.m_direction
               && m_name == rhs.m_name;
    }

    inline unsigned int getUnsignedInt(CommandArgumentValue val)
    {
        return visit(
            [](auto value) -> unsigned int {
                using T = std::decay_t<decltype(value)>;

                if constexpr(std::integral<T>)
                {
                    return static_cast<unsigned int>(value);
                }
                else
                {
                    throw std::runtime_error("Not integer type");
                }
            },
            val);
    }

    inline bool isInteger(CommandArgumentValue val)
    {
        return visit(
            [](auto value) -> bool {
                using T = std::decay_t<decltype(value)>;

                return (std::integral<T>) ? true : false;
            },
            val);
    }

    inline int CommandArgument::offset() const
    {
        return m_offset;
    }

    inline int CommandArgument::size() const
    {
        return m_size;
    }

    inline VariableType CommandArgument::variableType() const
    {
        return m_variableType;
    }

    inline DataDirection CommandArgument::direction() const
    {
        return m_direction;
    }

    inline std::string CommandArgument::name() const
    {
        return m_name;
    }

    struct CommandArgumentValueVariableTypeVisitor
    {
        template <CCommandArgumentValue Value>
        requires(!std::is_pointer<Value>::value) VariableType value()
        const
        {
            return TypeInfo<Value>::Var;
        }

        template <CCommandArgumentValue Value>
        requires(std::is_pointer<Value>::value) VariableType value()
        const
        {
            using Pointed = typename std::remove_pointer<Value>::type;
            // no pointers-to-pointer (yet)
            static_assert(!std::is_pointer<Pointed>::value);

            auto pointedType = TypeInfo<Pointed>::Var.dataType;
            return VariableType(pointedType, PointerType::PointerGlobal);
        }

        template <CCommandArgumentValue Value>
        VariableType operator()(Value const&) const
        {
            return value<Value>();
        }

        VariableType call(CommandArgumentValue const& val) const
        {
            return std::visit(*this, val);
        }
    };

    inline VariableType variableType(CommandArgumentValue const& val)
    {
        CommandArgumentValueVariableTypeVisitor visitor;
        return visitor.call(val);
    }

    struct CommandArgumentValueNameVisitor
    {
        template <CCommandArgumentValue Value>
        requires(!std::is_pointer<Value>::value) std::string value()
        const
        {
            return TypeInfo<Value>::Name();
        }

        template <CCommandArgumentValue Value>
        requires(std::is_pointer<Value>::value) std::string value()
        const
        {
            using Pointed = typename std::remove_pointer<Value>::type;
            static_assert(!std::is_same<Value, Pointed>::value);

            return value<Pointed>() + " *";
        }

        template <CCommandArgumentValue Value>
        std::string operator()(Value const&) const
        {
            return value<Value>();
        }

        std::string call(CommandArgumentValue const& val) const
        {
            return std::visit(*this, val);
        }
    };

    inline std::string name(CommandArgumentValue const& val)
    {
        CommandArgumentValueNameVisitor v;
        return v.call(val);
    }

    inline std::string toString(CommandArgumentValue const& val)
    {
        return std::visit([](auto const& value) { return concatenate(value); }, val);
    }

    inline std::ostream& operator<<(std::ostream& stream, CommandArgumentValue const& val)
    {
        return stream << toString(val);
    }

    inline std::ostream& operator<<(std::ostream& stream, CommandArgument const& arg)
    {
        return stream << arg.toString();
    }

    inline std::ostream& operator<<(std::ostream&                           stream,
                                    std::shared_ptr<CommandArgument> const& arg)
    {
        if(arg)
            return stream << *arg;
        else
            return stream << "nullptr";
    }

    inline std::ostream& operator<<(std::ostream&                                        stream,
                                    std::vector<std::shared_ptr<CommandArgument>> const& arg)
    {
        stream << "[\n";
        streamJoin(stream, arg, ", \n");
        return stream << "\n]";
    }
}
