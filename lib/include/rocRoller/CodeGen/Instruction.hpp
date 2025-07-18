/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <memory>
#include <string>

#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Scheduling/Scheduler_fwd.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

namespace rocRoller
{
    struct Instruction
    {
        enum
        {
            MaxDstRegisters   = 2,
            MaxSrcRegisters   = 5,
            MaxExtraRegisters = 5,
            MaxModifiers      = 10,
            MaxAllocations    = 4,
        };

        Instruction(std::string const&                        opcode,
                    std::initializer_list<Register::ValuePtr> dst,
                    std::initializer_list<Register::ValuePtr> src,
                    std::initializer_list<std::string>        modifiers,
                    std::string const&                        comment);

        Instruction();

        static Generator<std::string> EscapeComment(std::string comment, int indent = 0);

        static Instruction Allocate(Register::ValuePtr reg);
        static Instruction Allocate(std::shared_ptr<Register::Allocation> reg);
        static Instruction
            Allocate(std::initializer_list<std::shared_ptr<Register::Allocation>> regs);

        static Instruction Directive(std::string const& directive);
        static Instruction Directive(std::string const& directive, std::string const& comment);

        static Instruction Comment(std::string const& comment);

        static Instruction Warning(std::string const& warning);

        static Instruction Nop();
        static Instruction Nop(int nopCount);
        static Instruction Nop(std::string const& comment);
        static Instruction Nop(int nopCount, std::string const& comment);

        static Instruction Label(std::string const& name);
        static Instruction Label(std::string&& name);

        static Instruction Label(std::string const& name, std::string const& comment);
        static Instruction Label(std::string&& name, std::string const& comment);

        static Instruction Label(Register::ValuePtr label);
        static Instruction Label(Register::ValuePtr label, std::string const& comment);

        static Instruction InoutInstruction(std::string const&                        opcode,
                                            std::initializer_list<Register::ValuePtr> inout,
                                            std::initializer_list<std::string>        modifiers,
                                            std::string const&                        comment);

        static Instruction Wait(WaitCount const& wait);
        static Instruction Wait(WaitCount&& wait);

        static Instruction Lock(Scheduling::Dependency const& dependency, std::string comment);
        static Instruction Unlock(std::string comment);

        /**
         * This instruction will be considered to write `reg` even though it
         * will not be included in the actual dst operands in the string
         * rendering.  This will include it in the wait queue for the
         * instruction if it's a memory instruction.
         */
        Instruction& addExtraDst(Register::ValuePtr reg);

        /**
         * This instruction will implicitly add a waitcnt for any pending
         * load/stores on `reg`, without including it in the input operands for
         * the string rendering of this instruction.
         */
        Instruction& addExtraSrc(Register::ValuePtr reg);

        /**
         * Returns all the destination operands that will appear in the instruction.
         */
        auto const& getDsts() const;

        /**
         * Returns all the source operands that will appear in the instruction.
         */
        auto const& getSrcs() const;

        /**
         * Returns all the 'extra' destination operands; i.e. ones that will
         * not appear in the instruction.
         */
        auto const& getExtraDsts() const;

        /**
         * Returns all the 'extra' source operands; i.e. ones that will
         * not appear in the instruction.
         */
        auto const& getExtraSrcs() const;

        /**
         * Yields all the destination operands, including ones that will not
         * appear in the instruction.
         */
        Generator<Register::ValuePtr> getAllDsts() const;

        /**
         * Yields all the source operands, including ones that will not
         * appear in the instruction.
         */
        Generator<Register::ValuePtr> getAllSrcs() const;

        /**
         * Yields all destination and source operands, including ones that will
         * not appear in the instruction.
         */
        Generator<Register::ValuePtr> getAllOperands() const;

        bool           hasRegisters() const;
        constexpr bool readsSpecialRegisters() const;
        WaitCount      getWaitCount() const;

        /**
         * @brief Returns |currentSrc n previousDest| > 0 or |currentDest n previousDest| > 0
         *
         * @param previousDest Destination registers of previous instruction.
         * @return Whether this instructions registers intersect with a past
         * instruction's destination registers.
         */
        template <CForwardRangeOf<Register::ValuePtr> T>
        bool isAfterWriteDependency(T const& previousDest) const;

        void        toStream(std::ostream&, LogLevel level) const;
        std::string toString(LogLevel level) const;

        constexpr int          getLockValue() const;
        Scheduling::Dependency getDependency() const;

        std::string const&                                 getOpCode() const;
        std::array<std::string, Instruction::MaxModifiers> getModifiers() const;

        constexpr int getNopCount() const
        {
            return m_nopCount;
        }

        Instruction lock(Scheduling::Dependency const& dependency, std::string comment);
        Instruction unlock(std::string comment);

        void                    addControlOp(int id);
        std::vector<int> const& controlOps() const;
        std::optional<int>      innerControlOp() const;

        std::string const& referencedArg() const;
        void               setReferencedArg(std::string arg);

        void addAllocation(std::shared_ptr<Register::Allocation> alloc);
        void addWaitCount(WaitCount const& wait);
        void addComment(std::string const& comment);
        void addWarning(std::string const& warning);
        void addNop();
        void addNop(int count);
        void setNopMin(int count);

        std::vector<std::string> const& comments() const;

        constexpr int nopCount() const;

        constexpr bool isCommentOnly() const;
        constexpr bool isLabel() const;
        std::string    getLabel() const;

        /**
         * Get the status that was peeked by the Observers
         */
        Scheduling::InstructionStatus const& peekedStatus() const;

        /**
         * Set the status that was peeked by the Observers
         */
        void setPeekedStatus(Scheduling::InstructionStatus status);

        /**
         * How many instructions actually executed by the GPU are included?
         * This includes the main instruction, as well as any s_nop or s_waitcnt
         * instructions attached to it.
         */
        int numExecutedInstructions() const;

        void allocateNow();

        using AllocationArray = std::array<std::shared_ptr<Register::Allocation>, MaxAllocations>;
        AllocationArray allocations() const;

    private:
        /**
         * toString = preamble + functional + coda
         */
        void preambleString(std::ostream& oss, LogLevel level) const;
        void functionalString(std::ostream& oss, LogLevel level) const;
        void codaString(std::ostream& oss, LogLevel level) const;

        /**
         * Attempts to find a pointer to the context by looking at all the operands.
         */
        ContextPtr findContextFromOperands() const;

        /**
         * A comment detailing allocations that happened when scheduling this instruction.
         */
        void allocationString(std::ostream& oss, LogLevel level) const;

        /**
         * Assembler directive(s), if this instruction contains any.
         */
        void directiveString(std::ostream& oss, LogLevel level) const;

        /**
         * Just the main instruction.
         */
        void coreInstructionString(std::ostream& oss) const;

        bool requiresVnopForHazard() const;

        /**
         * When this instruction is scheduled, perform this register allocation.
         */
        AllocationArray m_allocations;

        int m_nopCount = 0;

        std::string m_directive;

        std::vector<std::string> m_warnings;

        std::vector<std::string> m_comments;

        std::vector<int> m_controlOps;

        std::string m_label;

        WaitCount m_waitCount;

        std::string m_opcode;

        std::string m_referencedArg;

        Scheduling::Dependency m_dependency = Scheduling::Dependency::None;

        std::array<Register::ValuePtr, MaxDstRegisters> m_inoutDsts;
        std::array<Register::ValuePtr, MaxDstRegisters> m_dst;
        std::array<Register::ValuePtr, MaxSrcRegisters> m_src;

        std::array<Register::ValuePtr, MaxExtraRegisters> m_extraDsts;
        std::array<Register::ValuePtr, MaxExtraRegisters> m_extraSrcs;

        std::array<std::string, MaxModifiers> m_modifiers;

        bool m_operandsAreInout = false;

        Scheduling::InstructionStatus m_peekedStatus;
    };
}

#include <rocRoller/CodeGen/Instruction_impl.hpp>
