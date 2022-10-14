
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

#define Inst(opcode) Instruction(opcode, {}, {}, {}, "")

using namespace rocRoller;

namespace rocRollerTest
{
    struct SchedulerTest : public GenericContextFixture
    {
        Generator<Instruction> testGeneratorWithComments();
    };

    Generator<Instruction> SchedulerTest::testGeneratorWithComments()
    {
        co_yield_(Instruction("s_sub_u32", {}, {}, {}, "Comment on an instruction"));
        co_yield Instruction::Comment("Pure comment 1");
        co_yield Instruction::Comment("Pure comment 2");

        co_yield_(Instruction("s_add_u32", {}, {}, {}, "Comment on an instruction"));
        co_yield Instruction::Comment("Pure comment 3");
        co_yield Instruction::Nop("Nop Comment");

        co_yield Instruction::Lock(Scheduling::Dependency::SCC, "Lock instruction");
        co_yield Instruction::Unlock("Unlock instruction");

        co_yield Instruction::Comment("Pure comment 4");
        co_yield Instruction::Comment("Pure comment 5");

        auto reg = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        co_yield reg->allocate();

        co_yield Instruction::Comment("Pure comment 6");
        co_yield Instruction::Directive(".set .amdgcn.next_free_vgpr, 0");

        co_yield Instruction::Label("FooLabel");

        co_yield Instruction::Comment("Pure comment 7");
        co_yield Instruction::Wait(WaitCount::Zero(m_context->targetArchitecture()));

        co_yield Instruction::Comment("Pure comment 8");
    }

    template <typename Begin, typename End>
    Instruction next(Begin& begin, End const& end)
    {
        AssertFatal(begin != end);
        auto inst = *begin;
        ++begin;
        return std::move(inst);
    }

    TEST_F(SchedulerTest, ConsumeCommentsTest)
    {
        using namespace rocRoller::Scheduling;

        auto gen   = testGeneratorWithComments();
        auto begin = gen.begin();
        auto end   = gen.end();

        auto inst = next(begin, end);
        EXPECT_EQ("s_sub_u32", inst.getOpCode());

        EXPECT_TRUE(begin->isCommentOnly());

        {
            m_context->schedule(consumeComments(begin, end));
            std::string expected = R"(// Pure comment 1
                                      // Pure comment 2)";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.getOpCode(), "s_add_u32");
        }

        EXPECT_TRUE(begin != end);

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 3\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.nopCount(), 1);
            EXPECT_EQ(inst.toString(Settings::LogLevel::Debug), "s_nop 0\n // Nop Comment\n");
        }

        {
            // Next is a lock, so there should be no comments consumed.
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(Settings::LogLevel::Debug), " // Lock instruction\n");
        }

        {
            // Next is an unlock, so there should be no comments consumed.
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(Settings::LogLevel::Debug), " // Unlock instruction\n");
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 4\n// Pure comment 5\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(Settings::LogLevel::Debug),
                      "// Allocated : 1 VGPR (Value: Float)\n");
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 6\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(Settings::LogLevel::Debug), ".set .amdgcn.next_free_vgpr, 0\n");
        }

        {
            // Next is a label, so there should be no comments consumed.
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(Settings::LogLevel::Debug), "FooLabel:\n\n");
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 7\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst       = next(begin, end);
            auto instString = R"(s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                                 s_waitcnt_vscnt 0)";
            EXPECT_EQ(NormalizedSource(inst.toString(Settings::LogLevel::Debug)),
                      NormalizedSource(instString));
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 8\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            EXPECT_EQ(begin, end);
        }
    }

    TEST_F(SchedulerTest, SequentialSchedulerTest)
    {
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 1"));
            co_yield_(Inst("Instruction 2, Generator 1"));
            co_yield_(Inst("Instruction 3, Generator 1"));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 2"));
            co_yield_(Inst("Instruction 2, Generator 2"));
            co_yield_(Inst("Instruction 3, Generator 2"));
            co_yield_(Inst("Instruction 4, Generator 2"));
        };

        std::vector<Generator<Instruction>> generators;
        generators.push_back(generator_one());
        generators.push_back(generator_two());

        std::string expected = R"( Instruction 1, Generator 1
                                    Instruction 2, Generator 1
                                    Instruction 3, Generator 1
                                    Instruction 1, Generator 2
                                    Instruction 2, Generator 2
                                    Instruction 3, Generator 2
                                    Instruction 4, Generator 2
                                )";

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::Sequential, m_context);
        m_context->schedule((*scheduler)(generators));
        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    TEST_F(SchedulerTest, RoundRobinSchedulerTest)
    {
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 1"));
            co_yield_(Inst("Instruction 2, Generator 1"));
            co_yield_(Inst("Instruction 3, Generator 1"));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 2"));
            co_yield_(Inst("Instruction 2, Generator 2"));
            co_yield_(Inst("Instruction 3, Generator 2"));
            co_yield_(Inst("Instruction 4, Generator 2"));
        };

        std::vector<Generator<Instruction>> generators;
        generators.push_back(generator_one());
        generators.push_back(generator_two());

        std::string expected = R"( Instruction 1, Generator 1
                                    Instruction 1, Generator 2
                                    Instruction 2, Generator 1
                                    Instruction 2, Generator 2
                                    Instruction 3, Generator 1
                                    Instruction 3, Generator 2
                                    Instruction 4, Generator 2
                                )";

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);
        m_context->schedule((*scheduler)(generators));
        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    // Can be run with the --gtest_also_run_disabled_tests option
    TEST_F(SchedulerTest, DISABLED_SchedulerWaitStressTest)
    {
        auto generator = []() -> Generator<Instruction> {
            for(size_t i = 0; i < 1000000; i++)
            {
                co_yield_(Instruction::Wait(WaitCount::VMCnt(1, "Comment")));
            }
        };

        m_context->schedule(generator());
    }

    // Can be run with the --gtest_also_run_disabled_tests option
    TEST_F(SchedulerTest, DISABLED_SchedulerCopyStressTest)
    {
        auto v_a
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_b
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);

        auto generator = [&]() -> Generator<Instruction> {
            co_yield v_a->allocate();
            co_yield v_b->allocate();

            for(size_t i = 0; i < 1000000; i++)
            {
                co_yield m_context->copier()->copy(v_a, v_b, "Comment");
            }
        };

        m_context->schedule(generator());
    }

    TEST_F(SchedulerTest, DoubleUnlocking)
    {
        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);

        std::vector<Generator<Instruction>> sequences;

        auto noUnlock = [&]() -> Generator<Instruction> {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC));
            co_yield(Instruction::Unlock());
            co_yield(Inst("Test"));
            co_yield(Instruction::Unlock());
        };

        sequences.push_back(noUnlock());

        EXPECT_THROW(m_context->schedule((*scheduler)(sequences));, FatalError);
    }

    TEST_F(SchedulerTest, noUnlocking)
    {
        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);

        std::vector<Generator<Instruction>> sequences;

        auto noUnlock = [&]() -> Generator<Instruction> {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC));
            co_yield(Inst("Test"));
        };

        sequences.push_back(noUnlock());

        EXPECT_THROW(m_context->schedule((*scheduler)(sequences));, FatalError);
    }

    TEST_F(SchedulerTest, SchedulerDepth)
    {
        auto schedulerA = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);
        auto schedulerB = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);
        auto schedulerC = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);

        std::vector<Generator<Instruction>> a_sequences;
        std::vector<Generator<Instruction>> b_sequences;
        std::vector<Generator<Instruction>> c_sequences;

        auto opB = [&]() -> Generator<Instruction> {
            co_yield(Inst("(C) Op B Begin"));
            co_yield(Inst("(C) Op B Instruction"));
            co_yield(Inst("(C) Op B End"));
        };

        auto ifBlock = [&]() -> Generator<Instruction> {
            co_yield(
                Inst("(C) If Begin").lock(Scheduling::Dependency::SCC, "(C) Scheduler C Lock"));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerC->getLockState().getDependency(), Scheduling::Dependency::SCC);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler C Lock Depth: "
                          + std::to_string(schedulerC->getLockState().getLockDepth())));
            co_yield(Inst("(C) If Instruction"));
            co_yield(Inst("(C) If End").unlock("(C) Scheduler C Unlock"));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerC->getLockState().getDependency(), Scheduling::Dependency::None);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler C Lock Depth: "
                          + std::to_string(schedulerC->getLockState().getLockDepth())));
        };

        c_sequences.push_back(opB());
        c_sequences.push_back(ifBlock());

        auto unroll0 = [&]() -> Generator<Instruction> {
            co_yield(Inst("(B) Unroll 0 Begin")
                         .lock(Scheduling::Dependency::VCC, "(B) Scheduler B Lock"));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::VCC);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
            co_yield((*schedulerC)(c_sequences));
            co_yield(Inst("(B) Unroll 0 End")).unlock("(B) Scheduler B Unlock");

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::None);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
        };

        auto unroll1 = [&]() -> Generator<Instruction> {
            co_yield(Inst("(B) Unroll 1 Begin"));
            co_yield(Inst("(B) Unroll 1 Instruction"));
            co_yield(Inst("(B) Unroll 1 End"));
        };

        b_sequences.push_back(unroll0());
        b_sequences.push_back(unroll1());

        auto opA = [&]() -> Generator<Instruction> {
            co_yield(Inst("(A) Op A Begin"));
            co_yield(Inst("(A) Op A Instruction"));
            co_yield(Inst("(A) Op A End"));
        };

        auto forloop = [&]() -> Generator<Instruction> {
            co_yield(Inst("(A) For Loop Begin")
                         .lock(Scheduling::Dependency::Branch, "(A) Scheduler A Lock"));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield((*schedulerB)(b_sequences));
            co_yield(Inst("(A) For Loop End").unlock("(A) Scheduler A Unlock)"));
            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::None);
        };

        a_sequences.push_back(opA());
        a_sequences.push_back(forloop());

        m_context->schedule((*schedulerA)(a_sequences));

        // TODO: Fix when m_comments.push_back() issue is fixed
        std::string expected = R"( (A) Op A Begin
                                    (A) For Loop Begin
                                    // (A) Scheduler A Lock
                                    // (A) Scheduler A Lock
                                    +++ Scheduler A Lock Depth: 1
                                    (B) Unroll 0 Begin
                                    // (B) Scheduler B Lock
                                    // (B) Scheduler B Lock
                                    +++ Scheduler A Lock Depth: 2
                                    +++ Scheduler B Lock Depth: 1
                                    (C) Op B Begin
                                    (C) If Begin
                                    // (C) Scheduler C Lock
                                    // (C) Scheduler C Lock
                                    +++ Scheduler A Lock Depth: 3
                                    +++ Scheduler B Lock Depth: 2
                                    +++ Scheduler C Lock Depth: 1
                                    (C) If Instruction
                                    (C) If End
                                    // (C) Scheduler C Unlock
                                    // (C) Scheduler C Unlock
                                    (C) Op B Instruction
                                    +++ Scheduler A Lock Depth: 2
                                    (C) Op B End
                                    +++ Scheduler B Lock Depth: 1
                                    +++ Scheduler C Lock Depth: 0
                                    (B) Unroll 0 End
                                    // (B) Scheduler B Unlock
                                    // (B) Scheduler B Unlock
                                    (B) Unroll 1 Begin
                                    +++ Scheduler A Lock Depth: 1
                                    (B) Unroll 1 Instruction
                                    +++ Scheduler B Lock Depth: 0
                                    (B) Unroll 1 End
                                    (A) For Loop End
                                    // (A) Scheduler A Unlock)
                                    // (A) Scheduler A Unlock)
                                    (A) Op A Instruction
                                    +++ Scheduler A Lock Depth: 0
                                    (A) Op A End
                                )";

        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }
}
