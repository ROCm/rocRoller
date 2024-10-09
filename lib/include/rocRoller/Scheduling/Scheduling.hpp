
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "../CodeGen/Instruction_fwd.hpp"
#include "../CodeGen/WaitCount.hpp"
#include "../Context_fwd.hpp"

#include "../Utilities/EnumBitset.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * Struct that describes modeled properties of an instruction if it were to be scheduled now.
         *
         * Note that any new members need to:
         * 1. Have a neutral default value, either in the declaration or in the constructor
         * 2. Be added to the combine() function which is used to merge values from different observers.
         */
        struct InstructionStatus
        {
            unsigned int stallCycles = 0;
            WaitCount    waitCount;
            unsigned int nops = 0;

            /// The new length of each of the queues.
            std::array<int, GPUWaitQueueType::Count> waitLengths;

            /// How many new registers of each type must be allocated?
            std::array<int, static_cast<size_t>(Register::Type::Count)> allocatedRegisters;

            /// How many free registers of each type will remain?
            std::array<int, static_cast<size_t>(Register::Type::Count)> remainingRegisters;

            /// How much does this instruction add to the high water mark of allocated registers?
            std::array<int, static_cast<size_t>(Register::Type::Count)> highWaterMarkRegistersDelta;

            /// Will this cause an out-of-registers error?
            EnumBitset<Register::Type> outOfRegisters;

            std::vector<std::string> errors;

            static InstructionStatus StallCycles(unsigned int const value);
            static InstructionStatus Wait(WaitCount const& value);
            static InstructionStatus Nops(unsigned int const value);
            static InstructionStatus Error(std::string const& msg);

            /**
             * Merge values of members from `other` into `this`.
             */
            void combine(InstructionStatus const& other);

            InstructionStatus();

            std::string toString() const;
        };

        template <typename T>
        concept CObserver = requires(T a, Instruction inst)
        {
            //> Speculatively predict stalls if this instruction were scheduled now.
            {
                a.peek(inst)
                } -> std::convertible_to<InstructionStatus>;

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            {a.modify(inst)};

            //> This instruction _will_ be scheduled now, record any side effects.
            //> This is after all observers have had the opportunity to modify the instruction.
            {a.observe(inst)};
        };

        template <typename T>
        concept CObserverConst = requires(T a, GPUArchitectureTarget const& target)
        {
            requires CObserver<T>;

            //> This observer is required in ctx, checking GPUArchitectureTarget if needed.
            {
                a.required(target)
                } -> std::convertible_to<bool>;
        };

        template <typename T>
        concept CObserverRuntime = requires(T a)
        {
            requires CObserver<T>;

            //> This observer is required in ctx, determined at runtime.
            {
                a.runtimeRequired()
                } -> std::convertible_to<bool>;
        };

        struct IObserver
        {
            virtual ~IObserver();

            //> Speculatively predict stalls if this instruction were scheduled now.
            virtual InstructionStatus peek(Instruction const& inst) const = 0;

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            virtual void modify(Instruction& inst) const = 0;

            //> This instruction _will_ be scheduled now, record any side effects.
            virtual void observe(Instruction const& inst) = 0;
        };
    }
}

#include "Scheduling_impl.hpp"
