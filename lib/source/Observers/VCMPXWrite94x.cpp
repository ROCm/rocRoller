#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VCMPXWrite94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void VCMPXWrite94x::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                for(auto const& regId : m_context.lock()->getExec()->getRegisterIds())
                {
                    (*m_hazardMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(inst), writeTrigger()));
                }
            }
        }

        int VCMPXWrite94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VCMPXWrite94x::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVCMPX(inst.getOpCode());
        };

        int VCMPXWrite94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVReadlane(inst.getOpCode())
               || GPUInstructionInfo::isVWritelane(inst.getOpCode()))
            {
                return checkRegister(m_context.lock()->getExec()).value_or(0);
            }

            if(GPUInstructionInfo::isVPermlane(inst.getOpCode()))
            {
                return checkRegister(m_context.lock()->getExec()).value_or(0);
            }

            // Check if VALU reads EXEC as constant
            if(GPUInstructionInfo::isVALU(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0) - 2;
            }
            return 0;
        }
    }
}
