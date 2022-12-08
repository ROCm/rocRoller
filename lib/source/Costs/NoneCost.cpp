
#include <rocRoller/Scheduling/Costs/NoneCost.hpp>
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(NoneCost);
        static_assert(Component::Component<NoneCost>);

        inline NoneCost::NoneCost(std::shared_ptr<Context> ctx)
            : Cost{ctx}
        {
            AssertFatal(false, "Cannot use None cost.");
        }

        inline bool NoneCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostProcedure::None;
        }

        inline std::shared_ptr<Cost> NoneCost::Build(Argument arg)
        {
            AssertFatal(false, "Cannot use None cost.");
        }

        inline std::string NoneCost::name()
        {
            return Name;
        }

        inline float NoneCost::cost(const InstructionStatus& inst) const
        {
            AssertFatal(false, "Cannot use None cost.");
        }
    }
}
