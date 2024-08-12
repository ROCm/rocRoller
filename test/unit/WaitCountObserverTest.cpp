#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class WaitCountObserverTest : public GenericContextFixture
    {
    protected:
        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnkownInstructions, true);
            m_kernelOptions.assertWaitCntState = true;
            GenericContextFixture::SetUp();
        }

        std::string targetArchitecture() override
        {
            return "gfx90a";
        }
    };

    /**
     * This test checks that a wait will be inserted between s_loads that share a dst register,
     * but not between ones that only share src registers. It also ensures that a wait zero is
     * used with s_loads.
     **/
    TEST_F(WaitCountObserverTest, Basic)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto src2 = src1->subset({1});

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3 = dst1->subset({1});

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("s_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        auto inst3 = Instruction("s_load_dword", {dst3}, {src2, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0, ""));
        m_context->schedule(inst3);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   s_load_dwordx2 s[2:3], s[0:1], 0
                                   s_load_dwordx2 s[4:5], s[0:1], 0
                                   s_waitcnt lgkmcnt(0)
                                   s_load_dword s3, s1, 0
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    /**
     * This test just makes sure that a series of instructions that don't need a wait don't have one insterted.
     **/
    TEST_F(WaitCountObserverTest, NoWaits)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst3->allocateNow();

        auto dst4
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst4->allocateNow();

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("s_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        auto inst3 = Instruction("s_load_dwordx2", {dst3}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst3);

        auto inst4 = Instruction("s_load_dwordx2", {dst4}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst4);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   s_load_dwordx2 s[2:3], s[0:1], 0
                                   s_load_dwordx2 s[4:5], s[0:1], 0
                                   s_load_dwordx2 s[6:7], s[0:1], 0
                                   s_load_dwordx2 s[8:9], s[0:1], 0
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    /**
     * This test checks that the wait inserted only waits as long as is needed.
     **/
    TEST_F(WaitCountObserverTest, CorrectQueueBehavior)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst3->allocateNow();

        auto dst4 = dst1->subset({0});

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("buffer_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("buffer_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        auto inst3 = Instruction("buffer_load_dwordx2", {dst3}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst3);

        auto inst4 = Instruction("buffer_load_dword", {dst4}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::VMCnt(2));
        m_context->schedule(inst4);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   buffer_load_dwordx2 v[0:1], s[0:1], 0
                                   buffer_load_dwordx2 v[2:3], s[0:1], 0
                                   buffer_load_dwordx2 v[4:5], s[0:1], 0
                                   s_waitcnt vmcnt(2)
                                   buffer_load_dword v0, s[0:1], 0 //For this instruction we only have to wait for the first instruction to finish.
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_F(WaitCountObserverTest, BarrierLGKMWait)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        src1->allocateNow();

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("ds_write_b32", {}, {zero, src1}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst4 = Instruction("s_barrier", {}, {}, {}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0));
        m_context->schedule(inst4);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   ds_write_b32 0, v0
                                   s_waitcnt lgkmcnt(0)
                                   s_barrier
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    /**
     * This test makes sure that a wait zero is inserted if instruction types are mixed.
     **/
    TEST_F(WaitCountObserverTest, MixedInstructionTypes)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst3->allocateNow();

        auto dst4 = dst1->subset({0});

        auto dst5
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst5->allocateNow();

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("global_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("global_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        auto inst3 = Instruction("global_load_dwordx2", {dst3}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst3);

        auto inst4 = Instruction("s_sendmsg", {}, {}, {"sendmsg(MSG_INTERRUPT)"}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst4);

        auto inst5 = Instruction("global_load_dword", {dst4}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst5);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount(0, -1, -1, -1));
        m_context->schedule(inst5);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   global_load_dwordx2 v[0:1], s[0:1], 0
                                   global_load_dwordx2 v[2:3], s[0:1], 0
                                   global_load_dwordx2 v[4:5], s[0:1], 0
                                   s_sendmsg sendmsg(MSG_INTERRUPT)
                                   s_waitcnt vmcnt(0)
                                   global_load_dword v0, s[0:1], 0
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    /**
     * This test makes sure that wait queues are tracked independently.
     **/
    TEST_F(WaitCountObserverTest, QueuesIndependent)
    {
        rocRoller::Scheduling::InstructionStatus peeked;
        auto                                     expectedWaitLengths = peeked.waitLengths;

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst3->allocateNow();

        auto dst4
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst4->allocateNow();

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        expectedWaitLengths[GPUWaitQueueType::LGKMSmemQueue] = 1;
        EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        m_context->schedule(inst1);

        auto inst2 = Instruction("s_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        expectedWaitLengths[GPUWaitQueueType::LGKMSmemQueue] = 2;
        EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        m_context->schedule(inst2);

        auto inst3 = Instruction("buffer_load_dwordx2", {dst3}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        expectedWaitLengths[GPUWaitQueueType::VMQueue] = 1;
        EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        m_context->schedule(inst3);

        auto inst4 = Instruction("buffer_load_dwordx2", {dst4}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        expectedWaitLengths[GPUWaitQueueType::VMQueue] = 2;
        EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        m_context->schedule(inst4);

        auto inst5 = Instruction("buffer_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst5);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0));

        auto inst5Expected                             = expectedWaitLengths;
        inst5Expected[GPUWaitQueueType::LGKMSmemQueue] = 0;
        inst5Expected[GPUWaitQueueType::VMQueue]       = 3;
        EXPECT_EQ(inst5Expected, peeked.waitLengths);

        // Peek at an unrelated instruction, no wait should be needed
        {
            auto tmp_src0 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);
            auto tmp_src1 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);
            auto tmp_dst0 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto alloc0 = tmp_src0->allocate();
            auto alloc1 = tmp_src1->allocate();

            m_context->schedule(alloc0);
            m_context->schedule(alloc1);

            auto instPossible
                = Instruction("v_add_u32_e32", {tmp_dst0}, {tmp_src0, tmp_src1}, {}, "");
            peeked = m_context->observer()->peek(instPossible);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        }

        // Back to the original instruction, shouldn't change.
        peeked = m_context->observer()->peek(inst5);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0));
        EXPECT_EQ(inst5Expected, peeked.waitLengths);

        // Peek at an unrelated instruction, no wait should be needed
        {
            auto tmp_src0 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);
            auto tmp_src1 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);
            auto tmp_dst0 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            auto alloc0 = tmp_src0->allocate();
            auto alloc1 = tmp_src1->allocate();

            m_context->schedule(alloc0);
            m_context->schedule(alloc1);

            auto instPossible
                = Instruction("v_add_u32_e32", {tmp_dst0}, {tmp_src0, tmp_src1}, {}, "");
            peeked = m_context->observer()->peek(instPossible);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        }

        // Back to the original instruction, shouldn't change.
        peeked = m_context->observer()->peek(inst5);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0));
        EXPECT_EQ(inst5Expected, peeked.waitLengths);

        m_context->schedule(inst5);
        expectedWaitLengths = inst5Expected;

        auto inst6 = Instruction("buffer_load_dword", {dst3}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst6);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::VMCnt(2));
        expectedWaitLengths[GPUWaitQueueType::VMQueue] = 3; // We wait for 1 but then add 1.
        EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        m_context->schedule(inst6);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        EXPECT_EQ(expectedWaitLengths, peeked.waitLengths);
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   s_load_dwordx2 v[0:1], s[0:1], 0
                                   s_load_dwordx2 v[2:3], s[0:1], 0
                                   buffer_load_dwordx2 v[4:5], s[0:1], 0
                                   buffer_load_dwordx2 v[6:7], s[0:1], 0
                                   s_waitcnt lgkmcnt(0) //This wait for lgkmcnt shouldn't affect the vmcnt queue.
                                   buffer_load_dwordx2 v[0:1], s[0:1], 0
                                   s_waitcnt vmcnt(2) //Here we'll still need to wait on vmcnt.
                                   buffer_load_dword v[4:5], s[0:1], 0
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    /**
     * This test makes sure that manually inserted waitcnts affect the waitcnt queues.
     **/
    TEST_F(WaitCountObserverTest, ManualWaitCountsHandled)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst3->allocateNow();

        auto dst4 = dst1->subset({0});

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("buffer_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("buffer_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        auto inst_wait = Instruction::Wait(WaitCount::Zero(m_context->targetArchitecture()));
        peeked         = m_context->observer()->peek(inst_wait);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_wait);

        auto inst3 = Instruction("buffer_load_dwordx2", {dst3}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst3);

        auto inst4 = Instruction("buffer_load_dword", {dst4}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst4);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   buffer_load_dwordx2 v[0:1], s[0:1], 0
                                   buffer_load_dwordx2 v[2:3], s[0:1], 0
                                   s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0) // <-- This manual wait will keep us from waiting later.
                                   buffer_load_dwordx2 v[4:5], s[0:1], 0
                                   // s_waitcnt vmcnt(2) <-- This wait won't be inserted because we manually waited sooner.
                                   buffer_load_dword v0, s[0:1], 0
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    /**
     * This test makes sure that manually inserted waitcnts don't cause the observer to miss a required waitcntzero.
     **/
    TEST_F(WaitCountObserverTest, ManualWaitCountsIgnoredByWaitZero)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst3->allocateNow();

        auto dst4 = dst1->subset({0});

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("s_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        auto inst_wait = Instruction::Wait(WaitCount::LGKMCnt(1));
        peeked         = m_context->observer()->peek(inst_wait);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_wait);

        auto inst3 = Instruction("s_load_dwordx2", {dst3}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst3);

        auto inst4 = Instruction("s_load_dword", {dst4}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0));
        m_context->schedule(inst4);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   s_load_dwordx2 v[0:1], s[0:1], 0
                                   s_load_dwordx2 v[2:3], s[0:1], 0
                                   s_waitcnt lgkmcnt(1) // <-- This manual wait shouldn't keep us from waiting later.
                                   s_load_dwordx2 v[4:5], s[0:1], 0
                                   s_waitcnt lgkmcnt(0) // <-- We still have to wait zero becuse they could be out of order.
                                   s_load_dword v0, s[0:1], 0
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_F(WaitCountObserverTest, SaturatedWaitCounts)
    {
        rocRoller::Scheduling::InstructionStatus peeked;

        auto inst_wait_lgkm = Instruction::Wait(WaitCount::LGKMCnt(20));
        peeked              = m_context->observer()->peek(inst_wait_lgkm);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());

        m_context->schedule(inst_wait_lgkm);

        auto inst_wait_exp = Instruction::Wait(WaitCount::EXPCnt(20));
        peeked             = m_context->observer()->peek(inst_wait_exp);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_wait_exp);

        auto inst_wait_vm = Instruction::Wait(WaitCount::VMCnt(80));
        peeked            = m_context->observer()->peek(inst_wait_vm);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_wait_vm);

        auto inst_wait = Instruction::Wait(WaitCount(80, -1, 20, 20));
        peeked         = m_context->observer()->peek(inst_wait);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_wait);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst_end);

        std::string expected = R"(
                                   s_waitcnt lgkmcnt(15) // Max values
                                   s_waitcnt expcnt(7)
                                   s_waitcnt vmcnt(63)
                                   s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }
    class WaitCountObserverStateTest
        : public WaitCountObserverTest,
          public ::testing::WithParamInterface<std::tuple<bool, LogLevel>>
    {
        void SetUp() override
        {
            auto [waitBeforeBarrier, logLevel] = GetParam();
            Settings::getInstance()->set(Settings::LogLvl, logLevel);

            m_kernelOptions.alwaysWaitZeroBeforeBarrier = waitBeforeBarrier;
            WaitCountObserverTest::SetUp();
        }
    };

    TEST_P(WaitCountObserverStateTest, QueueStateTest)
    {
        auto [waitBeforeBarrier, logLevel] = GetParam();
        auto const& arch                   = m_context->targetArchitecture();

        auto src1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        src1->allocateNow();

        auto src2 = src1->subset({1});

        auto dst1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst1->allocateNow();

        auto dst2
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst2->allocateNow();

        auto dst3
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Float,
                                                2,
                                                Register::AllocationOptions::FullyContiguous());
        dst3->allocateNow();

        auto dst4 = dst1->subset({0});
        auto dst5 = dst1->subset({1});

        auto zero = Register::Value::Literal(0);

        auto prevOutput = output();

        auto getNewOutput = [&]() -> std::string {
            auto prevSize  = prevOutput.size();
            auto curOutput = output();
            EXPECT_EQ(prevOutput, curOutput.substr(0, prevSize));

            auto rv    = curOutput.substr(prevSize, -1);
            prevOutput = curOutput;

            return rv;
        };

        {
            auto inst   = Instruction("buffer_load_dwordx2", {dst1}, {src1, zero}, {}, "");
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            m_context->schedule(inst);

            auto        newOutput = getNewOutput();
            std::string expected  = R"(
                buffer_load_dwordx2 v[0:1], s[0:1], 0
            )";
            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst   = Instruction("buffer_load_dwordx2", {dst2}, {src1, zero}, {}, "");
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            m_context->schedule(inst);

            auto        newOutput = getNewOutput();
            std::string expected  = R"(
                buffer_load_dwordx2 v[2:3], s[0:1], 0 )";
            if(logLevel >= LogLevel::Debug)
            {
                expected += R"(//
                    // Wait Queue State:
                    // --Queue: VMQueue
                    // ----Needs Wait Zero: False
                    // ----Type In Queue : VMQueue
                    // ----Registers :
                    // ------Dst: {v[0:1], }
                )";
            }
            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst   = Instruction("buffer_load_dwordx2", {dst3}, {src1, zero}, {}, "");
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            m_context->schedule(inst);

            auto        newOutput = getNewOutput();
            std::string expected  = R"(
                buffer_load_dwordx2 v[4:5], s[0:1], 0 )";
            if(logLevel >= LogLevel::Debug)
            {
                expected += R"(//
                    // Wait Queue State:
                    // --Queue: VMQueue
                    // ----Needs Wait Zero: False
                    // ----Type In Queue : VMQueue
                    // ----Registers :
                    // ------Dst: {v[0:1], }
                    // ------Dst: {v[2:3], }
                )";
            }
            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst   = Instruction("buffer_load_dword", {dst4}, {src1, zero}, {}, "");
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::VMCnt(2));
            m_context->schedule(inst);

            auto        newOutput = getNewOutput();
            std::string expected  = R"(
                s_waitcnt vmcnt(2)
                buffer_load_dword v0, s[0:1], 0 )";

            if(logLevel >= LogLevel::Verbose)
            {
                expected += "// WaitCnt Needed: Intersects with registers in 'VMQueue', at 0 and "
                            "the queue size is 3, so a waitcnt of 2 is required.";
            }
            if(logLevel >= LogLevel::Debug)
            {
                expected += R"(
                    //
                    // Wait Queue State:
                    // --Queue: VMQueue
                    // ----Needs Wait Zero: False
                    // ----Type In Queue : VMQueue
                    // ----Registers :
                    // ------Dst: {v[0:1], }
                    // ------Dst: {v[2:3], }
                    // ------Dst: {v[4:5], }
                )";
            }
            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst_barrier = Instruction("s_barrier", {}, {}, {}, "");
            auto peeked       = m_context->observer()->peek(inst_barrier);
            if(waitBeforeBarrier)
            {
                EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::Zero(arch));
            }
            else
            {
                EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            }

            m_context->schedule(inst_barrier);

            auto        newOutput = getNewOutput();
            std::string expected;
            if(waitBeforeBarrier)
                expected += "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0) //\n";

            expected += "s_barrier ";

            if(waitBeforeBarrier && logLevel >= LogLevel::Verbose)
            {
                expected += "// WaitCnt Needed: alwaysWaitZeroBeforeBarrier is set.\n";
            }

            if(logLevel >= LogLevel::Debug)
            {
                expected += R"( //
                    // Wait Queue State:
                    // --Queue: VMQueue
                    // ----Needs Wait Zero: False
                    // ----Type In Queue : VMQueue
                    // ----Registers :
                    // ------Dst: {v[2:3], }
                    // ------Dst: {v[4:5], }
                    // ------Dst: {v0, }
                )";
            }
            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst   = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
            auto peeked = m_context->observer()->peek(inst);
            if(waitBeforeBarrier)
            {
                EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            }
            else
            {
                EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::VMCnt(0));
            }
            m_context->schedule(inst);

            auto        newOutput = getNewOutput();
            std::string expected;

            // Since the barrier didn't add a waitcnt, we now need one here.
            if(!waitBeforeBarrier)
            {
                expected += "s_waitcnt vmcnt(0)\n";
            }

            expected += "s_load_dwordx2 v[0:1], s[0:1], 0";

            if(!waitBeforeBarrier)
            {
                if(logLevel >= LogLevel::Verbose)
                {
                    expected += " // WaitCnt Needed: Intersects with registers in 'VMQueue', at 2 "
                                "and the queue size is 3, so a waitcnt of 0 is required.\n";
                }

                if(logLevel >= LogLevel::Debug)
                {
                    expected += R"( //
                    // Wait Queue State:
                    // --Queue: VMQueue
                    // ----Needs Wait Zero: False
                    // ----Type In Queue : VMQueue
                    // ----Registers :
                    // ------Dst: {v[2:3], }
                    // ------Dst: {v[4:5], }
                    // ------Dst: {v0, }
                )";
                }
            }
            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst = Instruction::Wait(WaitCount::Zero(arch));
            m_context->schedule(inst);

            std::string expected = "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0) //";

            if(logLevel >= LogLevel::Debug)
            {
                expected += R"(
                    //
                    // Wait Queue State:
                    // --Queue: LGKMQueue
                    // ----Needs Wait Zero: True
                    // ----Type In Queue : LGKMSmemQueue
                    // ----Registers :
                    // ------Dst: {v[0:1], }
                )";
            }
            auto newOutput = getNewOutput();

            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst   = Instruction("ds_read_b64", {dst2}, {src1, zero}, {}, "");
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            m_context->schedule(inst);

            std::string expected  = "ds_read_b64 v[2:3], s[0:1], 0";
            auto        newOutput = getNewOutput();

            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst_barrier = Instruction("s_barrier", {}, {}, {}, "");
            auto peeked       = m_context->observer()->peek(inst_barrier);
            if(waitBeforeBarrier)
            {
                EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::Zero(arch));
            }
            else
            {
                EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0));
            }

            m_context->schedule(inst_barrier);

            auto        newOutput = getNewOutput();
            std::string expected;

            if(waitBeforeBarrier)
            {
                expected += "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0) //\n";
            }
            else
            {
                expected += "s_waitcnt lgkmcnt(0)\n";
            }

            expected += "s_barrier ";

            if(logLevel >= LogLevel::Verbose)
            {
                if(waitBeforeBarrier)
                {
                    expected += "// WaitCnt Needed: alwaysWaitZeroBeforeBarrier is set.\n";
                }
                else
                {
                    expected += "// WaitCnt Needed: lgkmcnt(0) before an s_barrier since the lgkm "
                                "queue is not empty.\n";
                }
            }

            if(logLevel >= LogLevel::Debug)
            {
                expected += R"( //
                    // Wait Queue State:
                    // --Queue: LGKMQueue
                    // ----Needs Wait Zero: False
                    // ----Type In Queue : LGKMDSQueue
                    // ----Registers :
                    // ------Dst: {v[2:3], }
                )";
            }
            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst   = Instruction("s_load_dword", {dst5}, {src2, zero}, {}, "");
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            m_context->schedule(inst);

            auto        newOutput = getNewOutput();
            std::string expected  = "s_load_dword v1, s1, 0";
            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }

        {
            auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
            auto peeked   = m_context->observer()->peek(inst_end);
            EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
            m_context->schedule(inst_end);

            auto        newOutput = getNewOutput();
            std::string expected  = "s_endpgm";

            if(logLevel >= LogLevel::Debug)
            {
                expected += R"( //
                    // Wait Queue State:
                    // --Queue: LGKMQueue
                    // ----Needs Wait Zero: True
                    // ----Type In Queue : LGKMSmemQueue
                    // ----Registers :
                    // ------Dst: {v1, }
                )";
            }

            ASSERT_EQ(NormalizedSource(expected, true), NormalizedSource(newOutput, true));
        }
    }

    INSTANTIATE_TEST_SUITE_P(WaitCountObserverStateTest,
                             WaitCountObserverStateTest,
                             ::testing::ValuesIn({std::make_tuple(false, LogLevel::None),
                                                  std::make_tuple(false, LogLevel::Verbose),
                                                  std::make_tuple(false, LogLevel::Debug),
                                                  std::make_tuple(true, LogLevel::None),
                                                  std::make_tuple(true, LogLevel::Verbose),
                                                  std::make_tuple(true, LogLevel::Debug)}));

    TEST_F(WaitCountObserverTest, LoopWaitCntStateAssertFailCase)
    {
        EXPECT_TRUE(m_context->kernelOptions().assertWaitCntState);

        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src1->allocateNow();

        auto src2 = src1->subset({1});

        auto dst1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Float, 2);
        dst1->allocateNow();

        auto dst2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Float, 2);
        dst2->allocateNow();

        auto dst3 = dst1->subset({1});

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("s_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        //An wait LGKMCnt 0 is needed at this label.
        auto instlabel = Instruction::Label("test_label");
        peeked         = m_context->observer()->peek(instlabel);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(instlabel);

        //This instruction causes  wait, which clears the queue states.
        auto inst3 = Instruction("s_load_dword", {dst3}, {src2, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0, ""));
        m_context->schedule(inst3);

        //That means that here, the queue state is different from when the label was encountered.
        m_context->schedule(m_context->brancher()->branch(Register::Value::Label("test_label")));

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());

        EXPECT_THROW(m_context->schedule(inst_end), FatalError);
    }

    TEST_F(WaitCountObserverTest, LoopWaitCntStateAssertGoodCase)
    {
        EXPECT_TRUE(m_context->kernelOptions().assertWaitCntState);

        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src1->allocateNow();

        auto src2 = src1->subset({1});

        auto dst1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Float, 2);
        dst1->allocateNow();

        auto dst2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Float, 2);
        dst2->allocateNow();

        auto dst3 = dst1->subset({1});

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst1);

        auto inst2 = Instruction("s_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst2);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst2);

        //An wait LGKMCnt 0 is needed at this label.
        auto instlabel = Instruction::Label("test_label");
        peeked         = m_context->observer()->peek(instlabel);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(instlabel);

        //This instruction causes  wait, which clears the queue states.
        auto inst3 = Instruction("s_add", {dst3}, {src2, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst3);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount::LGKMCnt(0, ""));
        m_context->schedule(inst3);

        //These 2 instructions recreate the same queue state.
        auto inst4 = Instruction("s_load_dwordx2", {dst1}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst4);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst4);

        auto inst5 = Instruction("s_load_dwordx2", {dst2}, {src1, zero}, {}, "");
        peeked     = m_context->observer()->peek(inst5);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());
        m_context->schedule(inst5);

        //That means that here, the queue state is the same as when the label was encountered.
        auto instbranch
            = Instruction("s_branch", {}, {Register::Value::Label("test_label")}, {}, "");
        m_context->schedule(instbranch);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.waitCount, rocRoller::WaitCount());

        m_context->schedule(inst_end); //Now this shouldn't fail.
    }

    TEST_F(WaitCountObserverTest, LoopWaitCntStateAssertGoodCase2)
    {
        // This test sees that even though the wait count states
        // aren't equivalent between the branch and label,
        // there are no destination registers in the wait queues,
        // so it is still safe to do the branch.
        EXPECT_TRUE(m_context->kernelOptions().assertWaitCntState);

        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src1->allocateNow();

        auto src2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src2->allocateNow();

        auto src3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src3->allocateNow();

        auto dst1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        dst1->allocateNow();

        auto dst2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        dst2->allocateNow();

        auto dst3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        dst3->allocateNow();

        // Instructions

        auto inst1 = Instruction("ds_write_b64", {}, {dst1, src1}, {}, "");
        m_context->schedule(inst1);

        auto instbranch
            = Instruction("s_branch", {}, {Register::Value::Label("test_label")}, {}, "");
        m_context->schedule(instbranch);

        auto inst2 = Instruction("ds_write_b64", {}, {dst2, src2}, {}, "");
        m_context->schedule(inst2);

        auto inst3 = Instruction("ds_write_b64", {}, {dst3, src3}, {}, "");
        m_context->schedule(inst3);

        auto instlabel = Instruction::Label("test_label");
        m_context->schedule(instlabel);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        m_context->schedule(inst_end); //Now this shouldn't fail.
    }

    TEST_F(WaitCountObserverTest, LoopWaitCntStateAssertFailCase2)
    {
        // This test is similar to LoopWaitCntStateAssertGoodCase2, except that
        // it uses flat stores. These require the waitcount to always be set
        // to zero, so the mismatch at the branches causes an error.
        EXPECT_TRUE(m_context->kernelOptions().assertWaitCntState);

        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src1->allocateNow();

        auto src2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src2->allocateNow();

        auto src3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        src3->allocateNow();

        auto dst1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        dst1->allocateNow();

        auto dst2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        dst2->allocateNow();

        auto dst3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        dst3->allocateNow();

        // Instructions

        auto inst1 = Instruction("flat_store_dwordx2", {}, {dst1, src1}, {}, "");
        m_context->schedule(inst1);

        auto instbranch
            = Instruction("s_branch", {}, {Register::Value::Label("test_label")}, {}, "");
        m_context->schedule(instbranch);

        auto inst2 = Instruction("flat_store_dwordx2", {}, {dst2, src2}, {}, "");
        m_context->schedule(inst2);

        auto inst3 = Instruction("flat_store_dwordx2", {}, {dst3, src3}, {}, "");
        m_context->schedule(inst3);

        auto instlabel = Instruction::Label("test_label");
        m_context->schedule(instlabel);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        EXPECT_THROW(m_context->schedule(inst_end), FatalError);
    }
}
