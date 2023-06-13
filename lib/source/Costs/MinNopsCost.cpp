
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(MinNopsCost);
        static_assert(Component::Component<MinNopsCost>);

        inline MinNopsCost::MinNopsCost(ContextPtr ctx)
            : Cost{ctx}
        {
        }

        inline bool MinNopsCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::MinNops;
        }

        inline std::shared_ptr<Cost> MinNopsCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<MinNopsCost>(std::get<1>(arg));
        }

        inline std::string MinNopsCost::name() const
        {
            return Name;
        }

        inline float MinNopsCost::cost(Instruction const&       inst,
                                       InstructionStatus const& status) const
        {
            return static_cast<float>(status.nops);
        }
    }
}
