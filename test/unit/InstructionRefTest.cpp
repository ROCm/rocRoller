
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/InstructionRef.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

#include "GenericContextFixture.hpp"

using namespace rocRoller;

class InstructionRefTest : public GenericContextFixture
{
    void SetUp()
    {
        Settings::getInstance()->set(Settings::AllowUnkownInstructions, true);
        GenericContextFixture::SetUp();
    }
};

#define EXPECT_CATEGORY_EQ(opcode, category, val)                                           \
    {                                                                                       \
        std::string str_(opcode);                                                           \
        auto        fromString_ = InstructionRef::category(str_);                           \
                                                                                            \
        Instruction inst_(str_, {}, {}, {}, "");                                            \
        auto        fromInstruction_ = InstructionRef::category(inst);                      \
                                                                                            \
        InstructionRef iref_(inst_);                                                        \
        auto           fromRef_ = iref_.category();                                         \
                                                                                            \
        EXPECT_TRUE(fromString_ == fromInstruction_ && fromInstruction_ == fromRef_         \
                    && fromRef_ == (val))                                                   \
            << #category << "(" << opcode << ")\n"                                          \
            << ShowValue(fromString_) << ShowValue(fromInstruction_) << ShowValue(fromRef_) \
            << ShowValue(val);                                                              \
    }

TEST_F(InstructionRefTest, LDS)
{

    for(auto inst : {"ds_write_b128", "ds_write2_b64", "ds_write_b8"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, true);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, true);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"ds_read_b128", "ds_read2_b64", "ds_read_b8"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, true);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, true);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}

TEST_F(InstructionRefTest, Scalar)
{
    for(auto inst : {"s_load_dword", "s_load_dwordx2", "s_load_dwordx16"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, true);
        EXPECT_CATEGORY_EQ(inst, isSMEM, true);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, false);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"s_lshl_b64", "s_add_i32", "s_max_u32", "s_and_b64"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, true);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, true);

        EXPECT_CATEGORY_EQ(inst, isVector, false);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"s_cbranch_vccz", "s_cbranch_vccnz"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, true);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, true);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, false);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}

TEST_F(InstructionRefTest, Vector)
{

    for(auto inst : {"v_mov_b32", "v_add_u32", "v_addc_co_u32", "v_or_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"buffer_load_dword", "buffer_load_dwordx4", "buffer_load_short_d16"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, true);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, true);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"buffer_store_dword", "buffer_store_dwordx4", "buffer_store_short"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, true);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, true);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_dot2c_f32_f16", "v_dot4c_i32_i8"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, true);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"flat_load_dword", "flat_load_dwordx2"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, true);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_cmpx_ge_i32_e64", "v_cmpx_le_u64_e64"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, true);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}

TEST_F(InstructionRefTest, AccMFMA)
{
    for(auto inst : {"v_accvgpr_read_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, true);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_accvgpr_write_b32"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, false);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, true);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, true);
    }

    for(auto inst :
        {"v_mfma_f32_16x16x16bf16_1k", "v_mfma_f32_16x16x1f32", "v_mfma_f32_32x32x8f16"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, true);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, false);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }

    for(auto inst : {"v_mfma_f64_16x16x4f64", "v_mfma_f64_4x4x4f64"})
    {
        EXPECT_CATEGORY_EQ(inst, isDLOP, false);
        EXPECT_CATEGORY_EQ(inst, isMFMA, true);
        EXPECT_CATEGORY_EQ(inst, isCMPX, false);

        EXPECT_CATEGORY_EQ(inst, isScalar, false);
        EXPECT_CATEGORY_EQ(inst, isSMEM, false);
        EXPECT_CATEGORY_EQ(inst, isSControl, false);
        EXPECT_CATEGORY_EQ(inst, isSALU, false);

        EXPECT_CATEGORY_EQ(inst, isVector, true);
        EXPECT_CATEGORY_EQ(inst, isVALU, false);
        EXPECT_CATEGORY_EQ(inst, isDGEMM, true);

        EXPECT_CATEGORY_EQ(inst, isVMEM, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMRead, false);
        EXPECT_CATEGORY_EQ(inst, isVMEMWrite, false);
        EXPECT_CATEGORY_EQ(inst, isFlat, false);

        EXPECT_CATEGORY_EQ(inst, isLDS, false);
        EXPECT_CATEGORY_EQ(inst, isLDSRead, false);
        EXPECT_CATEGORY_EQ(inst, isLDSWrite, false);

        EXPECT_CATEGORY_EQ(inst, isACCVGPRRead, false);
        EXPECT_CATEGORY_EQ(inst, isACCVGPRWrite, false);
    }
}
