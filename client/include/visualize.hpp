
#pragma once

#include <memory>

#include <rocRoller/Operations/Command.hpp>

namespace rocRoller
{
    namespace Client
    {
        /**
         * Function that's designed to be customized to visualize the relationship
         * between different graph dimensions and the memory locations accessed.
         */
        void visualize(CommandPtr command, CommandKernel& kc, CommandArguments const& commandArgs);
    }
}
