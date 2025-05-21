/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponentBase(Scheduler);

        std::string toString(SchedulerProcedure proc)
        {
            switch(proc)
            {
            case SchedulerProcedure::Sequential:
                return "Sequential";
            case SchedulerProcedure::RoundRobin:
                return "RoundRobin";
            case SchedulerProcedure::Random:
                return "Random";
            case SchedulerProcedure::Cooperative:
                return "Cooperative";
            case SchedulerProcedure::Priority:
                return "Priority";
            case SchedulerProcedure::Count:
                return "Count";
            }

            Throw<FatalError>("Invalid Scheduler Procedure: ", ShowValue(static_cast<int>(proc)));
        }

        std::ostream& operator<<(std::ostream& stream, SchedulerProcedure proc)
        {
            return stream << toString(proc);
        }

        std::string toString(Dependency dep)
        {
            switch(dep)
            {
            case Dependency::None:
                return "None";
            case Dependency::SCC:
                return "SCC";
            case Dependency::VCC:
                return "VCC";
            case Dependency::Branch:
                return "Branch";
            case Dependency::M0:
                return "M0";
            default:
                break;
            }

            Throw<FatalError>("Invalid Dependency: ", ShowValue(static_cast<int>(dep)));
        }

        std::ostream& operator<<(std::ostream& stream, Dependency dep)
        {
            return stream << toString(dep);
        }

        LockState::LockState(ContextPtr ctx)
            : m_ctx(ctx)
        {
        }

        LockState::LockState(ContextPtr ctx, Dependency dependency)
            : m_ctx(ctx)
        {
            lock(dependency, -1);
        }

        void LockState::lock(Dependency dep, int streamId)
        {
            AssertFatal(dep != Dependency::Count);

            if(isExclusive(dep))
            {
                for(auto const& [myDep, myStreamId] : m_lockStack)
                    AssertFatal(myStreamId == streamId,
                                ShowValue(myStreamId),
                                ShowValue(myDep),
                                ShowValue(streamId));
            }

            m_lockStack.push_back({dep, streamId});
            m_locks.insert(dep);
            if(isExclusive(dep))
                m_exclusive = true;
            m_streamIds[dep] = streamId;
        }

        void LockState::unlock(Dependency dep, int streamId)
        {
            AssertFatal(m_lockStack.size() > 0);

            {
                auto [backDep, backId] = m_lockStack.back();

                if(dep != Dependency::None)
                {
                    AssertFatal(backDep == dep);
                }
                else
                {
                    dep = backDep;
                }

                if(streamId >= 0 && backId >= 0)
                    AssertFatal(backId == streamId);
            }

            {
                auto iter = m_streamIds.find(dep);
                AssertFatal(iter != m_streamIds.end()
                                && (streamId < 0 || iter->second < 0 || iter->second == streamId),
                            ShowValue(dep),
                            ShowValue(streamId));
            }

            m_lockStack.pop_back();

            bool foundDep = false;
            m_exclusive   = false;
            for(auto iter = m_lockStack.rbegin(); iter != m_lockStack.rend(); ++iter)
            {
                auto [myDep, myId] = *iter;
                if(!foundDep && myDep == dep)
                {
                    foundDep         = true;
                    m_streamIds[dep] = myId;
                }

                if(isExclusive(myDep))
                    m_exclusive = true;

                if(foundDep && m_exclusive)
                    break;
            }

            {
                auto iter = m_locks.find(dep);
                AssertFatal(iter != m_locks.end());
                m_locks.erase(iter);
            }

            // if(isExclusive(dep))
            // {
            //     m_exclusive = false;
            //     for(int i = 0; i < static_cast<int>(Dependency::Count); i++)
            //     {
            //         auto thisDep = static_cast<Dependency>(i);

            //         if(isExclusive(thisDep) && m_locks.contains(thisDep))
            //         {
            //             m_exclusive = true;
            //             break;
            //         }
            //     }
            // }
        }

        bool LockState::isLockedFrom(Instruction const& instr, int streamId) const
        {
            if(m_lockStack.empty())
                return false;

            if(m_exclusive)
            {
                if(std::get<1>(m_lockStack.back()) == streamId)
                    return false;
                return true;
            }

            auto lockOp = instr.getLockValue();
            if(lockOp != LockOperation::Lock)
                return false;

            auto dep = instr.getDependency();

            if(isExclusive(dep))
            {
                for(auto const& [myDep, myStreamId] : m_lockStack)
                    if(myStreamId != streamId)
                        return true;
            }

            if(m_locks.contains(dep))
            {
                return m_streamIds.at(dep) != streamId;
            }

            return false;
        }

        void LockState::add(Instruction const& instruction, int streamId)
        {
            lockCheck(instruction);

            auto lockOp = instruction.getLockValue();

            switch(lockOp)
            {
            case LockOperation::None:
                break;

            case LockOperation::Lock:
                lock(instruction.getDependency(), streamId);
                break;

            case LockOperation::Unlock:
                unlock(instruction.getDependency(), streamId);
                break;

            case LockOperation::Count:
                Throw<FatalError>("Invalid LockOperation ", static_cast<int>(lockOp));
            }
        }

        bool LockState::isLocked() const
        {
            return m_exclusive;
        }

        constexpr bool isExclusive(Dependency dep)
        {
            return dep != Dependency::M0;
        }

        // Will grow into a function that accepts args and checks the lock is in a valid state against those args
        void LockState::isValid(bool locked) const
        {
            AssertFatal(isLocked() == locked, "Lock in invalid state");
        }

        void LockState::lockCheck(Instruction const& instruction)
        {
            auto               context      = m_ctx.lock();
            const auto&        architecture = context->targetArchitecture();
            GPUInstructionInfo info = architecture.GetInstructionInfo(instruction.getOpCode());

            AssertFatal(
                !info.isBranch() || isLocked(),
                concatenate(instruction.getOpCode(),
                            " is a branch instruction, it should only be used within a lock."));

            AssertFatal(
                !info.hasImplicitAccess() || isLocked(),
                concatenate(instruction.getOpCode(),
                            " implicitly reads a register, it should only be used within a lock."));

            AssertFatal(
                !instruction.readsSpecialRegisters() || isLocked(),
                concatenate(instruction.getOpCode(),
                            " reads a special register, it should only be used within a lock."));
        }

        Dependency LockState::getTopDependency() const
        {
            if(!m_lockStack.empty())
                return std::get<0>(m_lockStack.back());

            return Dependency::None;
        }

        int LockState::getLockDepth() const
        {
            return m_lockStack.size();
        }

        Generator<Instruction> Scheduler::yieldFromStream(Generator<Instruction>::iterator& iter,
                                                          int streamId)
        {
            do
            {
                AssertFatal(iter != std::default_sentinel_t{},
                            "End of instruction stream reached without unlocking");
                m_lockstate.add(*iter, streamId);
                co_yield *iter;
                ++iter;
                co_yield consumeComments(iter, std::default_sentinel_t{});
            } while(m_lockstate.isLocked());
        }

        bool Scheduler::supportsAddingStreams() const
        {
            return false;
        }

        Generator<Instruction>
            Scheduler::handleNewNodes(std::vector<Generator<Instruction>>&           seqs,
                                      std::vector<Generator<Instruction>::iterator>& iterators)
        {
            while(seqs.size() != iterators.size())
            {
                AssertFatal(seqs.size() >= iterators.size(),
                            "Sequences cannot shrink!",
                            ShowValue(seqs.size()),
                            ShowValue(iterators.size()));
                iterators.reserve(seqs.size());
                for(size_t i = iterators.size(); i < seqs.size(); i++)
                {
                    iterators.emplace_back(seqs[i].begin());
                    // Consume any comments at the beginning of the stream.
                    // This has the effect of immediately executing Deallocate nodes.
                    co_yield consumeComments(iterators[i], seqs[i].end());
                }
            }
        }
    }
}
