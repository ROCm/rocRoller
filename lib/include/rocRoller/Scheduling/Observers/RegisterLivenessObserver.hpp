#pragma once

#include <unordered_map>

#include <rocRoller/Context.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        enum class RegisterLiveState
        {
            Dead = 0,
            Write,
            Read,
            ReadWrite,
            Live,
            Allocated
        };

        struct LivenessHistoryEntry
        {
            std::string instruction;
            std::unordered_map<Register::Type, std::unordered_map<size_t, RegisterLiveState>>
                        registerStates;
            bool        isBranch = false;
            std::string label    = "";
            std::string name;
            size_t      lineNumber = 0;
        };

        class RegisterLivenessObserver
        {
        public:
            RegisterLivenessObserver() {}

            RegisterLivenessObserver(ContextPtr context)
                : m_context(context)
            {
                m_history = {};
            };

            InstructionStatus peek(Instruction const& inst) const
            {
                return {};
            };

            void modify(Instruction& inst) const {}

            /**
             * @brief This function updates the history with the liveness impact of the observed instruction.
             *
             * A new history entry is added for the newly observed instruction, with the current linecount as the line number,
             * allocated registers marked as such, src registers marked as reads, dst registers marked as writes, and registers
             * that are both src and dst marked as read/write.
             *
             * The linecount is incremented by the number of newlines that occur in the string representation of the observed instruction.
             *
             * The past liveness information is updated by, for every src register in the observed instruction, backtracking through the
             * history entries, and marking that register as live until an entry is encountered where it was written to or read from.
             *
             * @param inst The observed instruction.
             */
            void observe(Instruction const& inst);

            static bool runtimeRequired();

        private:
            std::weak_ptr<Context>            m_context;
            std::vector<LivenessHistoryEntry> m_history;
            size_t                            m_lineCount = 1;

            static constexpr std::array<Register::Type, 3> SUPPORTED_REG_TYPES{
                Register::Type::Accumulator, Register::Type::Vector, Register::Type::Scalar};

            static bool isSupported(Register::Type regType);

            /**
             * @brief Get the maximum registers used at any point in the history.
             *
             * @param regType
             * @return size_t
             */
            size_t getMaxRegisters(Register::Type regType) const;

            RegisterLiveState
                getState(Register::Type regType, size_t index, size_t pointInHistory) const;

            /**
             * @brief This function updates past liveness information.
             *
             * The past liveness information is updated by, for the given register index, backtracking through the
             * history entries, from start to stop, and marking that register as live until an entry is encountered
             * where it was written to or read from.
             *
             * @param regType
             * @param index Register to backtrack through and update.
             * @param start Where to start backtracking.
             * @param stop Where to stop backtracking if the a read or write is never encountered.
             */
            void backtrackLiveness(Register::Type regType,
                                   size_t         index,
                                   size_t         start,
                                   size_t         stop = 0);

            /**
             * @brief This function updates the liveness in consideration of branches.
             *
             * Specifically:
             * 1. If a register is live at a label, and there is a branch to that label earlier in the program,
             *    the register has it's liveness recorded starting at the branch and backtracking to the beginning of the program.
             *
             * 2. If a register is live at a label, and there is a branch to that label later in the program,
             *    the register has it's liveness recorded starting at the branch and backtracking to the label.
             *
             */
            void handleBranchLiveness();

            std::string livenessString(size_t                           pointInHistory,
                                       std::map<Register::Type, size_t> maxRegs) const;

            std::string livenessString() const;
        };

        static_assert(CObserverRuntime<RegisterLivenessObserver>);
    }
}
