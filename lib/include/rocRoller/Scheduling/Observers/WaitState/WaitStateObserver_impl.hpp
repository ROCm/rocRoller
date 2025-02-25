#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

#include <rocRoller/Context.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        template <class DerivedObserver>
        InstructionStatus WaitStateObserver<DerivedObserver>::peek(Instruction const& inst) const
        {
            return InstructionStatus::Nops(
                std::max(static_cast<DerivedObserver const*>(this)->getNops(inst), 0));
        }

        template <class DerivedObserver>
        void WaitStateObserver<DerivedObserver>::modify(Instruction& inst) const
        {
            auto const* thisDerived  = static_cast<DerivedObserver const*>(this);
            int         requiredNops = std::max(thisDerived->getNops(inst), 0);
            inst.setNopMin(requiredNops);
            if(requiredNops > 0)
            {
                inst.addComment("Wait state hazard: " + thisDerived->getComment());
            }
        }

        template <class DerivedObserver>
        void WaitStateObserver<DerivedObserver>::observe(Instruction const& inst)
        {
            auto* thisDerived = static_cast<DerivedObserver*>(this);
            thisDerived->decrementHazardCounters(inst);
            thisDerived->observeHazard(inst);
        }

        template <class DerivedObserver>
        void WaitStateObserver<DerivedObserver>::observeHazard(Instruction const& inst)
        {
            auto* thisDerived = static_cast<DerivedObserver*>(this);
            if(thisDerived->trigger(inst))
            {
                for(auto iter = (DerivedObserver::writeTrigger() ? inst.getDsts().begin()
                                                                 : inst.getSrcs().begin());
                    iter
                    != (DerivedObserver::writeTrigger() ? inst.getDsts().end()
                                                        : inst.getSrcs().end());
                    iter++)
                {
                    auto reg = *iter;
                    if(reg)
                    {
                        for(auto const& regId : reg->getRegisterIds())
                        {
                            (*m_hazardMap)[regId].push_back(WaitStateHazardCounter(
                                thisDerived->getMaxNops(inst), DerivedObserver::writeTrigger()));
                        }
                    }
                }
            }
        }

        template <class DerivedObserver>
        void WaitStateObserver<DerivedObserver>::decrementHazardCounters(Instruction const& inst)
        {
            // If instruction is not a comment or empty
            if(!inst.getOpCode().empty())
            {
                for(auto mapIt = m_hazardMap->begin(); mapIt != m_hazardMap->end();)
                {
                    for(auto hazardIt = mapIt->second.begin(); hazardIt != mapIt->second.end();)
                    {
                        hazardIt->decrement(inst.nopCount());
                        if(!hazardIt->stillAlive())
                        {
                            hazardIt = mapIt->second.erase(hazardIt);
                        }
                        else
                        {
                            hazardIt++;
                        }
                    }
                    if(mapIt->second.empty())
                    {
                        mapIt = m_hazardMap->erase(mapIt);
                    }
                    else
                    {
                        mapIt++;
                    }
                }
            }
        }

        template <class DerivedObserver>
        int WaitStateObserver<DerivedObserver>::getNopFromLatency(
            std::string const& opCode, std::unordered_map<int, int> const& latencyAndNops) const
        {
            auto const& architecture = m_context.lock()->targetArchitecture();
            int         passes       = architecture.GetInstructionInfo(opCode).getLatency();

            AssertFatal(latencyAndNops.contains(passes),
                        "Unexpected number of passes",
                        ShowValue(architecture.target().toString()),
                        ShowValue(opCode),
                        ShowValue(passes));

            return latencyAndNops.at(passes);
        }

        template <class DerivedObserver>
        std::optional<int>
            WaitStateObserver<DerivedObserver>::checkRegister(Register::ValuePtr const& reg) const
        {
            if(!reg || reg->allocationState() == Register::AllocationState::Unallocated)
            {
                return std::nullopt;
            }

            auto const* thisDerived = static_cast<DerivedObserver const*>(this);

            int requiredNops = -1;
            for(auto const& regId : reg->getRegisterIds())
            {
                if(m_hazardMap->contains(regId))
                {
                    for(auto const& hazard : m_hazardMap->at(regId))
                    {
                        bool isHazardous
                            = (DerivedObserver::writeTrigger() && hazard.regWasWritten())
                              || (!DerivedObserver::writeTrigger() && !hazard.regWasWritten());
                        if(isHazardous)
                        {
                            requiredNops = std::max(hazard.getRequiredNops(), requiredNops);
                        }
                    }
                }
            }
            if(requiredNops != -1)
                return requiredNops;
            return std::nullopt;
        }

        template <class DerivedObserver>
        std::optional<int>
            WaitStateObserver<DerivedObserver>::checkSrcs(Instruction const& inst) const
        {
            int requiredNops = -1;
            for(auto const& src : inst.getSrcs())
            {
                if(auto val = checkRegister(src))
                {
                    requiredNops = std::max(val.value(), requiredNops);
                }
            }
            if(requiredNops != -1)
                return requiredNops;
            return std::nullopt;
        }

        template <class DerivedObserver>
        std::optional<int>
            WaitStateObserver<DerivedObserver>::checkDsts(Instruction const& inst) const
        {
            int requiredNops = -1;
            for(auto const& dst : inst.getDsts())
            {
                if(auto val = checkRegister(dst))
                {
                    requiredNops = std::max(val.value(), requiredNops);
                }
            }
            if(requiredNops != -1)
                return requiredNops;
            return std::nullopt;
        }
    }
}
