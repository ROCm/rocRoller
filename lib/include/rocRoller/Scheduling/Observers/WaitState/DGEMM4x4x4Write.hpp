#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rules for v_mfma_f64_4x4x4f64 Write Hazards
         *
         * | Arch | 1st Inst                  | 2nd Inst                           | NOPs |
         * | ---- | ------------------------- | ---------------------------------- | ---- |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_f64_4x4x4f64 read SrcC same | 4    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcC overlapped | 4    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma* read SrcC overlapped       | 0    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcA/B          | 6    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_* read/write                     | 6    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | buffer* read overlapped            | 9    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | ds* read overlapped                | 9    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | flat* read overlapped              | 9    |
         *
         */
        class DGEMM4x4x4Write : public WaitStateObserver<DGEMM4x4x4Write>
        {
        public:
            DGEMM4x4x4Write() {}
            DGEMM4x4x4Write(std::shared_ptr<Context> context)
                : WaitStateObserver<DGEMM4x4x4Write>(context){};

            InstructionStatus observe(Instruction const& inst)
            {
                return observe_base(inst);
            }

            static bool required(std::shared_ptr<Context> context)
            {
                return context->targetArchitecture().target().getVersionString() == "gfx90a";
            }

            int  getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool trigger(std::shared_ptr<InstructionRef> inst) const;
            bool writeTrigger() const;
            int  getNops(Instruction const& inst) const;

        private:
            std::string m_targetOpCode = "v_mfma_f64_4x4x4f64";
            int const   m_maxNops      = 9;
        };

        static_assert(CWaitStateObserver<DGEMM4x4x4Write>);
    }
}
