#pragma once

#include <sstream>

#include "WaitcntObserver.hpp"

#include "../Scheduling.hpp"

#include "../../Context.hpp"
#include "../../GPUArchitecture/GPUInstructionInfo.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        constexpr inline bool WaitcntObserver::required(GPUArchitectureTarget const& target)
        {
            return true;
        }

        inline InstructionStatus WaitcntObserver::peek(Instruction const& inst) const
        {
            auto rv = InstructionStatus::Wait(computeWaitCount(inst));

            // The new length of each queue is:
            // - The current length of the queue
            // - With the new WaitCount applied.
            // - Plus the contribution from this instruction

            // Get current length of queue
            for(auto const& pair : m_instructionQueues)
            {
                if(pair.second.size() > 0)
                {
                    auto wqType = m_typeInQueue.at(pair.first);

                    AssertFatal(wqType < rv.waitLengths.size(),
                                ShowValue(static_cast<size_t>(wqType)),
                                ShowValue(rv.waitLengths.size()),
                                ShowValue(pair.second.size()));

                    rv.waitLengths.at(wqType) = pair.second.size();
                }
            }

            // Apply the waitcount from this instruction.
            for(int i = 0; i < GPUWaitQueueType::Count; i++)
            {
                auto wqType = GPUWaitQueueType(i);
                auto wq     = GPUWaitQueue(wqType);

                auto count = rv.waitCount.getCount(wq);

                if(count >= 0)
                    rv.waitLengths.at(i) = std::min(rv.waitLengths.at(i), count);
            }

            // Add contribution from this instruction
            GPUInstructionInfo info
                = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());
            auto whichQueues = info.getWaitQueues();
            for(auto q : whichQueues)
            {
                AssertFatal(q < rv.waitLengths.size(),
                            ShowValue(static_cast<size_t>(q)),
                            ShowValue(rv.waitLengths.size()));
                auto waitCount = info.getWaitCount();
                rv.waitLengths.at(q) += waitCount == 0 ? 1 : waitCount;
            }

            return rv;
        };

        inline void WaitcntObserver::modify(Instruction& inst) const
        {
            auto        context      = m_context.lock();
            const auto& architecture = context->targetArchitecture();

            // Handle if manually specified waitcnts are over the sat limits.
            inst.addWaitCount(inst.getWaitCount().getAsSaturatedWaitCount(architecture));

            std::string  explanation;
            std::string* pExplanation = nullptr;
            if(m_includeExplanation)
                pExplanation = &explanation;

            inst.addWaitCount(computeWaitCount(inst, pExplanation));
            if(m_includeExplanation)
                inst.addComment(explanation);

            if(m_displayState && !inst.isCommentOnly())
            {
                inst.addComment(getWaitQueueState());
            }
        }

        inline void WaitcntObserver::applyWaitToQueue(int waitCnt, GPUWaitQueue queue)
        {
            if(waitCnt >= 0 && m_instructionQueues[queue].size() > (size_t)waitCnt)
            {
                if(!(m_needsWaitZero[queue]
                     && waitCnt
                            > 0)) //Do not partially clear the queue if a waitcnt zero is needed.
                {
                    m_instructionQueues[queue].erase(m_instructionQueues[queue].begin(),
                                                     m_instructionQueues[queue].begin()
                                                         + m_instructionQueues[queue].size()
                                                         - waitCnt);
                }
                if(m_instructionQueues[queue].size() == 0)
                {
                    m_needsWaitZero[queue] = false;
                    m_typeInQueue[queue]   = GPUWaitQueueType::None;
                }
            }
        }

        inline WaitCount WaitcntObserver::computeWaitCount(Instruction const& inst,
                                                           std::string*       explanation) const
        {
            auto        context      = m_context.lock();
            const auto& architecture = context->targetArchitecture();

            if(inst.getOpCode() == "s_barrier")
            {
                if(context->kernelOptions().alwaysWaitZeroBeforeBarrier)
                {
                    if(explanation != nullptr)
                    {
                        *explanation += "WaitCnt Needed: alwaysWaitZeroBeforeBarrier is set.";
                    }
                    return WaitCount::Zero(architecture);
                }

                if(!m_instructionQueues.at(GPUWaitQueue::LGKMQueue).empty())
                {
                    if(explanation != nullptr)
                    {
                        *explanation += "WaitCnt Needed: lgkmcnt(0) before an s_barrier since the "
                                        "lgkm queue is not empty.";
                    }
                    return WaitCount::LGKMCnt(0);
                }
            }

            WaitCount retval;

            if(inst.getOpCode().size() > 0 && inst.hasRegisters())
            {
                for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
                {
                    GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);
                    for(int queue_i = m_instructionQueues.at(waitQueue).size() - 1; queue_i >= 0;
                        queue_i--)
                    {
                        if(inst.isAfterWriteDependency(m_instructionQueues.at(waitQueue)[queue_i]))
                        {
                            if(m_needsWaitZero.at(waitQueue))
                            {
                                retval.combine(WaitCount(waitQueue, 0));
                                if(explanation != nullptr)
                                {
                                    *explanation += "WaitCnt Needed: Intersects with registers in '"
                                                    + waitQueue.toString()
                                                    + "', which needs a wait zero.";
                                }
                            }
                            else
                            {
                                int waitval
                                    = m_instructionQueues.at(waitQueue).size() - (queue_i + 1);
                                retval.combine(WaitCount(waitQueue, waitval));
                                if(explanation != nullptr)
                                {
                                    *explanation += "WaitCnt Needed: Intersects with registers in '"
                                                    + waitQueue.toString() + "', at "
                                                    + std::to_string(queue_i)
                                                    + " and the queue size is "
                                                    + std::to_string(
                                                        m_instructionQueues.at(waitQueue).size())
                                                    + ", so a waitcnt of " + std::to_string(waitval)
                                                    + " is required.";
                                }
                            }
                            break;
                        }
                    }
                }
            }
            return retval.getAsSaturatedWaitCount(architecture);
        }

        inline void WaitcntObserver::addLabelState(std::string const& label)
        {
            m_labelStates[label]
                = WaitcntState(m_needsWaitZero, m_typeInQueue, m_instructionQueues);
        }

        inline void WaitcntObserver::addBranchState(std::string const& label)
        {
            if(m_branchStates.find(label) == m_branchStates.end())
            {
                m_branchStates[label] = {};
            }

            m_branchStates[label].emplace_back(
                WaitcntState(m_needsWaitZero, m_typeInQueue, m_instructionQueues));
        }

        inline void WaitcntObserver::assertLabelConsistency()
        {
            for(auto label_state : m_labelStates)
            {
                if(m_branchStates.find(label_state.first) != m_branchStates.end())
                {
                    for(auto branch_state : m_branchStates[label_state.first])
                    {
                        label_state.second.assertSafeToBranchTo(branch_state, label_state.first);
                    }
                }
            }
        }
    };

}
