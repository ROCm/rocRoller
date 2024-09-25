#include <memory>

#include <rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;

struct LatencyAndOpCode
{
    std::string opCode;
};

class MFMAUnitObserverTest : public GPUContextFixtureParam<LatencyAndOpCode>
{
public:
    MFMAUnitObserverTest() {}

    std::string opCode()
    {
        return std::get<1>(GetParam()).opCode;
    }
};

TEST_P(MFMAUnitObserverTest, GPU_Direct)
{
    Scheduling::MFMAObserver observer(m_context);

    auto agpr
        = Register::Value::Placeholder(m_context, Register::Type::Accumulator, DataType::Float, 16);

    auto v0 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);
    auto v1 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);

    agpr->allocateNow();
    v0->allocateNow();
    v1->allocateNow();

    auto mfmaInst = Instruction(opCode(), {agpr}, {v0, v1, agpr}, {}, "");
    auto valuInst = Instruction("v_add_f32", {v0}, {v0, v1}, {}, "");

    EXPECT_EQ(0, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(mfmaInst);

    auto info    = m_context->targetArchitecture().GetInstructionInfo(mfmaInst.getOpCode());
    auto latency = info.getLatency();

    EXPECT_EQ(latency, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(valuInst);

    EXPECT_EQ(latency - 1, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);

    observer.observe(mfmaInst);

    EXPECT_EQ(latency, observer.peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, observer.peek(valuInst).stallCycles);
}

TEST_P(MFMAUnitObserverTest, GPU_InContext)
{
    auto agpr
        = Register::Value::Placeholder(m_context, Register::Type::Accumulator, DataType::Float, 16);

    auto v0 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);
    auto v1 = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 4);

    agpr->allocateNow();
    v0->allocateNow();
    v1->allocateNow();

    auto mfmaInst = Instruction(opCode(), {agpr}, {v0, v1, agpr}, {}, "");
    auto valuInst = Instruction("v_add_f32", {v0}, {v0, v1}, {}, "");

    EXPECT_EQ(0, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    auto info    = m_context->targetArchitecture().GetInstructionInfo(mfmaInst.getOpCode());
    auto latency = info.getLatency();

    m_context->schedule(mfmaInst);

    EXPECT_EQ(latency, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    m_context->schedule(valuInst);

    EXPECT_EQ(latency - 1, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);

    m_context->schedule(mfmaInst);

    EXPECT_EQ(latency, m_context->peek(mfmaInst).stallCycles);
    EXPECT_EQ(0, m_context->peek(valuInst).stallCycles);
}

INSTANTIATE_TEST_SUITE_P(
    MFMAUnitObserverTests,
    MFMAUnitObserverTest,
    ::testing::Combine(mfmaSupportedISAValues(),
                       ::testing::Values(LatencyAndOpCode{"v_mfma_f32_16x16x16f16"},
                                         LatencyAndOpCode{"v_mfma_f32_32x32x8f16"})));
