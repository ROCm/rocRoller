
#include <rocRoller/Scheduling/RoundRobinScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(RoundRobinScheduler);
        static_assert(Component::Component<RoundRobinScheduler>);

        inline RoundRobinScheduler::RoundRobinScheduler(ContextPtr ctx)
            : Scheduler{ctx}
        {
        }

        inline bool RoundRobinScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::RoundRobin;
        }

        inline std::shared_ptr<Scheduler> RoundRobinScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<RoundRobinScheduler>(std::get<2>(arg));
        }

        inline std::string RoundRobinScheduler::name() const
        {
            return Name;
        }

        inline Generator<Instruction>
            RoundRobinScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            std::vector<Generator<Instruction>::iterator> iterators;

            if(seqs.empty())
                co_return;

            size_t n         = seqs.size();
            bool   yield_seq = true;

            iterators.reserve(n);
            for(auto& seq : seqs)
            {
                iterators.emplace_back(seq.begin());
            }

            while(yield_seq)
            {
                yield_seq = false;

                for(size_t i = 0; i < n; ++i)
                {
                    if(iterators[i] != seqs[i].end())
                    {
                        yield_seq = true;
                        co_yield yieldFromStream(iterators[i]);
                    }
                }
            }

            m_lockstate.isValid(false);
        }
    }
}
