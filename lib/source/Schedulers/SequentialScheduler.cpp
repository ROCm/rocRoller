
#include <rocRoller/Scheduling/SequentialScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(SequentialScheduler);

        static_assert(Component::Component<SequentialScheduler>);

        inline SequentialScheduler::SequentialScheduler(ContextPtr ctx)
            : Scheduler{ctx}
        {
        }

        inline bool SequentialScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Sequential;
        }

        inline std::shared_ptr<Scheduler> SequentialScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<SequentialScheduler>(std::get<2>(arg));
        }

        inline std::string SequentialScheduler::name() const
        {
            return Name;
        }

        bool SequentialScheduler::supportsAddingStreams() const
        {
            return true;
        }

        inline Generator<Instruction>
            SequentialScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            bool yieldedAny = false;

            std::vector<Generator<Instruction>::iterator> iterators;
            co_yield handleNewNodes(seqs, iterators);

            do
            {
                yieldedAny = false;

                for(size_t i = 0; i < seqs.size(); i++)
                {

                    while(iterators[i] != seqs[i].end())
                    {
                        co_yield yieldFromStream(iterators[i]);
                        yieldedAny = true;
                    }

                    if(seqs.size() != iterators.size())
                    {
                        co_yield handleNewNodes(seqs, iterators);
                        yieldedAny = true;
                    }
                }

            } while(yieldedAny);
        }
    }
}
