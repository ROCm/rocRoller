#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using ::testing::HasSubstr;

namespace rocRollerTest
{
    class MFMA90aObserverTest : public GenericContextFixture
    {
    protected:
        std::string targetArchitecture()
        {
            return "gfx90a";
        }

        void peekAndSchedule(Instruction inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops);
            m_context->schedule(inst);
        }

        std::vector<Register::ValuePtr> createRegisters(Register::Type const regType,
                                                        DataType const       dataType,
                                                        size_t const         amount,
                                                        int const            regCount = 1)
        {
            std::vector<Register::ValuePtr> regs;
            for(size_t i = 0; i < amount; i++)
            {
                auto reg
                    = std::make_shared<Register::Value>(m_context, regType, dataType, regCount);
                reg->allocateNow();
                regs.push_back(reg);
            }
            return regs;
        }
    };

    TEST_F(MFMA90aObserverTest, NoWaitStates)
    {
        auto s = createRegisters(Register::Type::Scalar, DataType::Float, 3, 2);

        auto zero = Register::Value::Literal(0);

        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("s_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                   Instruction("s_load_dwordx2", {s[2]}, {s[0], zero}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            for(auto inst : insts)
            {
                peekAndSchedule(inst);
            }

            std::string expected = R"(
                                        s_load_dwordx2 s[2:3], s[0:1], 0
                                        s_load_dwordx2 s[4:5], s[0:1], 0
                                        s_endpgm
                                    )";

            EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
        }
    }

    TEST_F(MFMA90aObserverTest, CMPXWaitStates)
    {
        auto v_i64 = createRegisters(Register::Type::Vector, DataType::Int64, 1);
        auto v_f32 = createRegisters(Register::Type::Vector, DataType::Float, 2);
        auto v_f16 = createRegisters(Register::Type::Vector, DataType::Half, 2);
        auto a     = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts = {
                Instruction("v_cmpx_lt_f32", {m_context->getExec()}, {v_f32[0], v_f32[1]}, {}, ""),
                Instruction("v_mfma_f32_32x32x8f16", {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 4);

            EXPECT_THAT(output(), HasSubstr("s_nop 3"));
            clearOutput();
        }

        // No hazard if v_cmpx doesn't write to exec or a VGPR that is later read
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_cmpx_lt_f32", {v_i64[0]}, {v_f32[0], v_f32[1]}, {}, ""),
                   Instruction("v_mfma_f32_32x32x8f16", {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, VALUThenMFMA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }

        // No hazard if second instruction doesn't read same VGPRs
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[3], v[4], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 0);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop 1")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_SrcCExact)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // No hazard if SrcC is exactly overlapped
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[2], v[3], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_Different)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // No register overlap should not be a hazard
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts = {Instruction("v_mfma_f32_16x16x4f32",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f32_16x16x4f32",
                                                          {a[0]->subset({2, 3})},
                                                          {v[2], v[3], a[0]->subset({2, 3})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_SrcCOverlap)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // If SrcC is partially overlapped
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32",
                               {a[0]->subset({2, 3})},
                               {v[2], v[3], a[0]->subset({2, 3})},
                               {},
                               ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 8);

            EXPECT_THAT(output(), HasSubstr("s_nop 7"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_ReadA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        // If SrcC is partially overlapped
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[1]}, {v[2], v[3], a[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenFlat)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        // If SrcC is partially overlapped
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("flat_store_dword", {}, {v[3], v[2]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenVALU_Read)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        // If SrcC is partially overlapped
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[4]}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenVALU_Write)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        // If SrcC is partially overlapped
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[2]}, {v[3], v[4]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenVALU_WAR)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        // If SrcC is partially overlapped
        {
            Scheduling::InstructionStatus peeked;
            std::vector<Instruction>      insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], v[3]}, {}, ""),
                   Instruction("v_or_b32", {v[3]}, {v[4], v[5]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }
}
