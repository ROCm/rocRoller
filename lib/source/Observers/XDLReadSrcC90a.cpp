#include <rocRoller/Scheduling/Observers/WaitState/XDLReadSrcC90a.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        InstructionStatus XDLReadSrcC90a::observe(Instruction const& inst)
        {
            auto instRef = std::make_shared<InstructionRef>(inst);
            if(trigger(instRef))
            {
                auto regMap = m_context.lock()->getRegisterHazardMap();

                auto srcC = inst.getSrcs().at(2);
                AssertFatal(srcC != nullptr, "Empty SrcC");

                for(auto const& regId : srcC->getRegisterIds())
                {
                    if(!regMap->contains(regId))
                    {
                        (*regMap)[regId] = {};
                    }
                    (*regMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(instRef), instRef, writeTrigger()));
                }
            }

            return InstructionStatus::Nops(inst.getNopCount());
        }

        int XDLReadSrcC90a::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return getNopFromLatency(inst->getOpCode(), m_latencyAndNops);
        }

        bool XDLReadSrcC90a::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isMFMA();
        };

        bool XDLReadSrcC90a::writeTrigger() const
        {
            return false;
        }

        int XDLReadSrcC90a::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isVALU() && !instRef.isMFMA())
            {
                // WAR
                return checkDsts(inst).value_or(0);
            }
            return 0;
        }
    }
}
