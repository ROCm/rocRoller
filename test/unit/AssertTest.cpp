#include "gtest/gtest.h"

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssertOpKinds.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using ::testing::HasSubstr;

namespace AssertTest
{
    class GPU_AssertTest
        : public CurrentGPUContextFixture,
          public ::testing::WithParamInterface<std::tuple<AssertOpKind, std::string>>
    {

    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            CurrentGPUContextFixture::SetUp();
            Settings::getInstance()->set(Settings::SaveAssembly, true);

            fastArith = Expression::FastArithmetic(m_context);
        }

        static std::string
            getTestSuffix(const testing::TestParamInfo<GPU_AssertTest::ParamType>& info)
        {
            const auto [assertOpKind, _] = info.param;
            return toString(assertOpKind);
        }
    };

    TEST_P(GPU_AssertTest, GPU_Assert)
    {
        if(!m_context->targetArchitecture().target().isCDNAGPU())
        {
            GTEST_SKIP() << "Skipping GPU assert tests for "
                         << m_context->targetArchitecture().target().toString();
        }

        AssertOpKind assertOpKind;
        std::string  outputMsg;
        std::tie(assertOpKind, outputMsg) = GetParam();

        ::testing::FLAGS_gtest_death_test_style = "threadsafe";

        if(assertOpKind == AssertOpKind::Count)
        {
            EXPECT_THAT(
                [&]() {
                    rocRoller::KernelGraph::KernelGraph kgraph;
                    auto                                k = m_context->kernel();
                    k->setKernelDimensions(1);
                    setKernelOptions({.assertOpKind = assertOpKind});
                    auto assertOp = kgraph.control.addElement(AssertOp{});
                    m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));
                },
                ::testing::ThrowsMessage<FatalError>(HasSubstr(outputMsg)));
        }
        else
        {
            auto one  = Expression::literal(1);
            auto zero = Expression::literal(0);

            rocRoller::KernelGraph::KernelGraph kgraph;

            auto k = m_context->kernel();
            k->setKernelDimensions(1);
            k->setWorkitemCount({one, one, one});
            k->setWorkgroupSize({1, 1, 1});
            setKernelOptions({.assertOpKind = assertOpKind});
            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());
            k->setDynamicSharedMemBytes(zero);

            auto                    testReg = kgraph.coordinates.addElement(Linear());
            Expression::DataFlowTag testRegTag{testReg, Register::Type::Scalar, DataType::UInt32};
            auto testRegExpr = std::make_shared<Expression::Expression>(testRegTag);

            int setToZero = kgraph.control.addElement(Assign{Register::Type::Scalar, zero});

            kgraph.mapper.connect(setToZero, testReg, NaryArgument::DEST);

            auto assertOp = kgraph.control.addElement(AssertOp{"Assert Test", testRegExpr == one});

            auto assignOne = kgraph.control.addElement(Assign{Register::Type::Scalar, one});

            auto kernelNode = kgraph.control.addElement(Kernel());
            kgraph.control.addElement(Body(), {kernelNode}, {setToZero});
            kgraph.control.addElement(Sequence(), {setToZero}, {assertOp});
            kgraph.control.addElement(Sequence(), {assertOp}, {assignOne});

            m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
            EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s3, 0"));
            if(assertOpKind != AssertOpKind::NoOp)
            {
                EXPECT_THAT(output(), testing::HasSubstr("s_cmp_eq_i32 s3, 1"));
                EXPECT_THAT(output(), testing::HasSubstr("s_cbranch_scc1"));
                EXPECT_THAT(output(), testing::HasSubstr("AssertFailed"));
                EXPECT_THAT(output(), testing::HasSubstr("// Lock for Assert Assert Test"));
                EXPECT_THAT(output(), testing::HasSubstr("// Unlock for Assert Assert Test"));
                if(assertOpKind == AssertOpKind::STrap)
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_trap 2"));
                }
                else
                { // MEMORY_VIOLATION
                    EXPECT_THAT(output(), testing::HasSubstr("s_mov_b64 s[4:5], 0"));
                    EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s6, 42"));
                    EXPECT_THAT(output(), testing::HasSubstr("s_store_dword s6, s[4:5], 0 glc"));
                }
                EXPECT_THAT(output(), testing::HasSubstr("AssertPassed"));
                EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s4, 1"));
            }
            else
            {
                EXPECT_THAT(output(), testing::HasSubstr("AssertOpKind == NoOp"));

                EXPECT_THAT(output(), Not(testing::HasSubstr("s_cmp_eq_i32 s3, 1")));
                EXPECT_THAT(output(), Not(testing::HasSubstr("s_cbranch_scc1")));
                EXPECT_THAT(output(), Not(testing::HasSubstr("AssertFailed")));
                EXPECT_THAT(output(), Not(testing::HasSubstr("// Lock for Assert Assert Test")));
                EXPECT_THAT(output(), Not(testing::HasSubstr("// Unlock for Assert Assert Test")));
            }

            if(isLocalDevice())
            {
                KernelArguments kargs;

                KernelInvocation kinv;
                kinv.workitemCount = {1, 1, 1};
                kinv.workgroupSize = {1, 1, 1};

                auto executableKernel = m_context->instructions()->getExecutableKernel();

                const auto runTest = [&]() {
                    executableKernel->executeKernel(kargs, kinv);
                    // Need to wait for signal, otherwise child process may terminate before signal is sent
                    (void)hipDeviceSynchronize();
                };
                if(assertOpKind != AssertOpKind::NoOp)
                {
                    EXPECT_EXIT({ runTest(); }, ::testing::KilledBySignal(SIGABRT), outputMsg);
                }
                else
                {
                    runTest();
                }
            }
        }
    }

    TEST_P(GPU_AssertTest, GPU_UnconditionalAssert)
    {
        if(!m_context->targetArchitecture().target().isCDNAGPU())
        {
            GTEST_SKIP() << "Skipping GPU assert tests for "
                         << m_context->targetArchitecture().target().toString();
        }

        AssertOpKind assertOpKind;
        std::string  outputMsg;
        std::tie(assertOpKind, outputMsg) = GetParam();

        ::testing::FLAGS_gtest_death_test_style = "threadsafe";

        if(assertOpKind == AssertOpKind::Count)
        {
            EXPECT_THAT(
                [&]() {
                    rocRoller::KernelGraph::KernelGraph kgraph;
                    auto                                k = m_context->kernel();
                    k->setKernelDimensions(1);
                    setKernelOptions({.assertOpKind = assertOpKind});
                    auto assertOp = kgraph.control.addElement(AssertOp{});
                    m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));
                },
                ::testing::ThrowsMessage<FatalError>(HasSubstr(outputMsg)));
        }
        else
        {
            auto one  = Expression::literal(1);
            auto zero = Expression::literal(0);

            rocRoller::KernelGraph::KernelGraph kgraph;

            auto k = m_context->kernel();
            k->setKernelDimensions(1);
            k->setWorkitemCount({one, one, one});
            k->setWorkgroupSize({1, 1, 1});
            setKernelOptions({.assertOpKind = assertOpKind});
            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());
            k->setDynamicSharedMemBytes(zero);

            int setToZero = kgraph.control.addElement(Assign{Register::Type::Scalar, zero});

            auto assertOp = kgraph.control.addElement(AssertOp{"Unconditional Assert"});

            auto assignOne = kgraph.control.addElement(Assign{Register::Type::Scalar, one});

            auto kernelNode = kgraph.control.addElement(Kernel());
            kgraph.control.addElement(Body(), {kernelNode}, {setToZero});
            kgraph.control.addElement(Sequence(), {setToZero}, {assertOp});
            kgraph.control.addElement(Sequence(), {assertOp}, {assignOne});

            m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
            if(assertOpKind != AssertOpKind::NoOp)
            {
                if(assertOpKind == AssertOpKind::STrap)
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_trap 2"));
                }
                else
                { // MEMORY_VIOLATION
                    EXPECT_THAT(output(), testing::HasSubstr("s_mov_b64 s[4:5], 0"));
                    EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s6, 42"));
                    EXPECT_THAT(output(), testing::HasSubstr("s_store_dword s6, s[4:5], 0 glc"));
                }
            }
            else
            {
                EXPECT_THAT(output(), testing::HasSubstr("AssertOpKind == NoOp"));
            }

            if(isLocalDevice())
            {
                KernelArguments kargs;

                KernelInvocation kinv;
                kinv.workitemCount = {1, 1, 1};
                kinv.workgroupSize = {1, 1, 1};

                auto executableKernel = m_context->instructions()->getExecutableKernel();

                const auto runTest = [&]() {
                    executableKernel->executeKernel(kargs, kinv);
                    // Need to wait for signal, otherwise child process may terminate before signal is sent
                    (void)hipDeviceSynchronize();
                };
                if(assertOpKind != AssertOpKind::NoOp)
                {
                    EXPECT_EXIT({ runTest(); }, ::testing::KilledBySignal(SIGABRT), outputMsg);
                }
                else
                {
                    runTest();
                }
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        AssertTest,
        GPU_AssertTest,
        ::testing::Values(std::tuple(AssertOpKind::MemoryViolation, "Memory access fault"),
                          std::tuple(AssertOpKind::STrap, "HSA_STATUS_ERROR_EXCEPTION"),
                          std::tuple(AssertOpKind::NoOp, "AssertOpKind == NoOp"),
                          std::tuple(AssertOpKind::Count, "Invalid AssertOpKind")),
        GPU_AssertTest::getTestSuffix);
}
