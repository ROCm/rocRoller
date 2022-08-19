#pragma once

#include <memory>
#include <span>
#include <variant>

#include "DataTypes/DataTypes.hpp"

namespace rocRoller
{
    class CommandArgument;
    using CommandArgumentPtr = std::shared_ptr<CommandArgument>;

    using CommandArgumentValue = std::variant<
        // int16_t,
        int32_t,
        int64_t,
        // uint16_t,
        uint32_t,
        uint64_t,
        float,
        double,
        Half,
        bool,
        // int16_t*,
        int32_t*,
        int64_t*,
        // uint16_t*,
        uint32_t*,
        uint64_t*,
        float*,
        double*,
        Half*>;

    template <typename T>
    concept CCommandArgumentValue = requires(T& val)
    {
        {CommandArgumentValue(val)};
    };

    static_assert(!CCommandArgumentValue<bool*>);

    using RuntimeArguments = std::span<uint8_t const>;
}
