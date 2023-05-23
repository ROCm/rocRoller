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

        LockState::LockState(ContextPtr ctx)
            : m_dependency(Dependency::None)
            , m_lockdepth(0)
            , m_ctx(ctx)
        {
        }

        LockState::LockState(ContextPtr ctx, Dependency dependency)
            : m_dependency(dependency)
            , m_ctx(ctx)
        {
            AssertFatal(m_dependency != Scheduling::Dependency::Count
                            && m_dependency != Scheduling::Dependency::Unlock,
                        "Can not instantiate LockState with Count or Unlock dependency");

            m_lockdepth = 1;
        }

        void LockState::add(Instruction const& instruction)
        {
            lockCheck(instruction);

            int inst_lockvalue = instruction.getLockValue();

            // Instruction does not lock or unlock, do nothing
            if(inst_lockvalue == 0)
            {
                return;
            }

            // Instruction can only lock (1) or unlock (-1)
            if(inst_lockvalue != -1 && inst_lockvalue != 1)
            {
                Throw<FatalError>("Invalid instruction lockstate: ", ShowValue(inst_lockvalue));
            }

            // Instruction trying to unlock when there is no lock
            if(m_lockdepth == 0 && inst_lockvalue == -1)
            {
                Throw<FatalError>("Trying to unlock when not locked");
            }

            // Instruction initializes the lockstate
            if(m_lockdepth == 0)
            {
                m_dependency = instruction.getDependency();
            }

            m_lockdepth += inst_lockvalue;

            // Instruction releases lock
            if(m_lockdepth == 0)
            {
                m_dependency = Scheduling::Dependency::None;
            }
        }

        bool LockState::isLocked() const
        {
            return m_lockdepth > 0;
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

        Dependency LockState::getDependency() const
        {
            return m_dependency;
        }

        int LockState::getLockDepth() const
        {
            return m_lockdepth;
        }

        Generator<Instruction> Scheduler::yieldFromStream(Generator<Instruction>::iterator& iter)
        {
            do
            {
                AssertFatal(iter != std::default_sentinel_t{},
                            "End of instruction stream reached without unlocking");
                m_lockstate.add(*iter);
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
