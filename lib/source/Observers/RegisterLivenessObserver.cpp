#include <algorithm>
#include <sstream>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        inline std::ostream& operator<<(std::ostream& os, const RegisterLiveState& state)
        {
            switch(state)
            {
            case RegisterLiveState::Dead:
                os << " ";
                break;
            case RegisterLiveState::Write:
                os << "^";
                break;
            case RegisterLiveState::Read:
                os << "v";
                break;
            case RegisterLiveState::ReadWrite:
                os << "x";
                break;
            case RegisterLiveState::Live:
                os << ":";
                break;
            case RegisterLiveState::Allocated:
                os << "_";
                break;
            }
            return os;
        }

        void RegisterLivenessObserver::observe(Instruction const& inst)
        {
            auto               context      = m_context.lock();
            auto const&        architecture = context->targetArchitecture();
            GPUInstructionInfo info         = architecture.GetInstructionInfo(inst.getOpCode());

            LivenessHistoryEntry entry;
            entry.lineNumber = m_lineCount;

            // Mark all currently allocated registers as such in the history entry.
            for(auto& regType : SUPPORTED_REG_TYPES)
            {
                entry.registerStates[regType] = {};
                for(size_t i = 0; i < context->allocator(regType)->useCount(); i++)
                {
                    if(!(context->allocator(regType)->isFree(i)))
                    {
                        entry.registerStates[regType][i] = RegisterLiveState::Allocated;
                    }
                }
            }

            // Mark all registers that are dsts in the observed instruction as written in the history entry.
            auto dests = inst.getDsts();
            for(auto& dest : dests)
            {
                if(dest && isSupported(dest->regType()))
                {
                    for(auto& index : dest->registerIndices())
                    {
                        entry.registerStates[dest->regType()][index] = RegisterLiveState::Write;
                    }
                }
            }

            // For the observed instruction, all src registers are marked as reads, and registers that are both
            // src and dst are marked as read/write in the history entry.
            // Past liveness information is updated by backtracking for all src registers in the observed instruction.
            auto srcs = inst.getSrcs();
            for(auto& src : srcs)
            {
                if(src && isSupported(src->regType()))
                {
                    for(auto& index : src->registerIndices())
                    {
                        if(entry.registerStates[src->regType()].find(index)
                               != entry.registerStates[src->regType()].end()
                           && entry.registerStates[src->regType()][index]
                                  == RegisterLiveState::Write)
                        {
                            entry.registerStates[src->regType()][index]
                                = RegisterLiveState::ReadWrite;
                        }
                        else
                        {
                            entry.registerStates[src->regType()][index] = RegisterLiveState::Read;
                        }
                        backtrackLiveness(src->regType(), index, m_history.size() - 1);
                    }
                }
            }

            entry.instruction = inst.toString(context->kernelOptions().logLevel);
            m_lineCount += std::count(entry.instruction.begin(), entry.instruction.end(), '\n');
            std::replace(entry.instruction.begin(), entry.instruction.end(), '\n', ';');
            entry.isBranch = info.isBranch();
            if(info.isBranch())
            {
                entry.label = srcs[0]->toString();
            }
            else if(inst.isLabel())
            {
                entry.label = inst.getLabel();
            }
            m_history.emplace_back(std::move(entry));

            // If the observed instruction was the end of the program, produce the liveness file.
            auto      instWaitQueues = info.getWaitQueues();
            WaitCount waiting        = inst.getWaitCount();
            if(std::find(
                   instWaitQueues.begin(), instWaitQueues.end(), GPUWaitQueueType::FinalInstruction)
               != instWaitQueues.end())
            {
                handleBranchLiveness();
                std::ofstream liveness_file;
                liveness_file.open(context->assemblyFileName() + ".live", std::ios_base::out);
                AssertFatal(liveness_file.is_open(),
                            "Could not open file " + context->assemblyFileName() + " for writing.");
                liveness_file << livenessString() << std::endl;
                liveness_file.flush();
            }
        }

        bool RegisterLivenessObserver::required(std::shared_ptr<Context>)
        {
            return Settings::getInstance()->get(Settings::KernelAnalysis);
        }

        bool RegisterLivenessObserver::isSupported(Register::Type regType)
        {
            return std::find(SUPPORTED_REG_TYPES.begin(), SUPPORTED_REG_TYPES.end(), regType)
                   != SUPPORTED_REG_TYPES.end();
        }

        size_t RegisterLivenessObserver::getMaxRegisters(Register::Type regType) const
        {
            size_t retval = 0;
            bool   zero   = true;
            for(size_t i = 0; i < m_history.size(); i++)
            {
                for(auto& current : m_history[i].registerStates.at(regType))
                {
                    zero = false;
                    if(std::get<0>(current) > retval)
                    {
                        retval = std::get<0>(current);
                    }
                }
            }
            if(zero)
            {
                return 0;
            }
            return retval + 1;
        }

        RegisterLiveState RegisterLivenessObserver::getState(Register::Type regType,
                                                             size_t         index,
                                                             size_t         pointInHistory) const
        {
            if(m_history[pointInHistory].registerStates.at(regType).find(index)
               != m_history[pointInHistory].registerStates.at(regType).end())
            {
                return m_history[pointInHistory].registerStates.at(regType).at(index);
            }
            else
            {
                return RegisterLiveState::Dead;
            }
        }

        void RegisterLivenessObserver::backtrackLiveness(Register::Type regType,
                                                         size_t         index,
                                                         size_t         start,
                                                         size_t         stop)
        {
            for(size_t i = start; i >= stop && i <= start; i--)
            {
                auto state = getState(regType, index, i);
                if(state == RegisterLiveState::Write || state == RegisterLiveState::Read
                   || state == RegisterLiveState::ReadWrite)
                {
                    break;
                }
                else
                {
                    m_history[i].registerStates[regType][index] = RegisterLiveState::Live;
                }
            }
        }

        void RegisterLivenessObserver::handleBranchLiveness()
        {
            for(size_t inst = m_history.size() - 1; inst < m_history.size(); inst--)
            {
                if(!m_history[inst].isBranch && !m_history[inst].label.empty())
                {
                    // inst is a label.
                    for(size_t i = 0; i < m_history.size(); i++)
                    {
                        if(m_history[i].label == m_history[inst].label && m_history[i].isBranch)
                        {
                            // i is a branch that targets label inst.
                            for(auto& regType : SUPPORTED_REG_TYPES)
                            {
                                for(auto& j : m_history[inst].registerStates[regType])
                                {
                                    if(std::get<1>(j) == RegisterLiveState::Live)
                                    {
                                        if(i < inst)
                                        {
                                            /*
                                            * Register j is live at label inst which is branched to from i earlier in the program.
                                            * The register has it's liveness recorded starting at the branch and backtracking to
                                            * the beginning of the program.
                                            */
                                            backtrackLiveness(regType, std::get<0>(j), i);
                                        }
                                        else if(i > inst)
                                        {
                                            /*
                                            * Register j is live at label inst which is branched to from i later in the program.
                                            * The register has it's liveness recorded starting at the branch and backtracking to the label.
                                            */
                                            backtrackLiveness(regType, std::get<0>(j), i, inst);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::string RegisterLivenessObserver::livenessString(size_t pointInHistory) const
        {
            std::stringstream retval;
            for(auto& regType : SUPPORTED_REG_TYPES)
            {
                size_t maxReg = getMaxRegisters(regType);
                if(maxReg > 0)
                {
                    for(size_t i = 0; i < maxReg; i++)
                    {
                        retval << getState(regType, i, pointInHistory);
                    }
                    retval << "|";
                }
            }
            retval << " " << m_history[pointInHistory].lineNumber << ". "
                   << m_history[pointInHistory].instruction;

            return retval.str();
        }

        std::string RegisterLivenessObserver::livenessString() const
        {
            std::stringstream retval;
            for(auto& regType : SUPPORTED_REG_TYPES)
            {
                size_t maxReg = getMaxRegisters(regType);
                if(maxReg > 0)
                {
                    std::stringstream label;
                    label << regType;
                    retval << regType;
                    for(size_t i = label.str().size(); i < maxReg; i++)
                    {
                        retval << " ";
                    }
                    retval << "|";
                }
            }
            retval << "Instruction" << std::endl;
            for(size_t i = 0; i < m_history.size(); i++)
            {
                retval << livenessString(i) << std::endl;
            }
            return retval.str();
        }
    }
}
