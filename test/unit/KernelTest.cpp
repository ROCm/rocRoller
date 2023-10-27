
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <iterator>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class KernelTest : public GenericContextFixture
    {
    };

    class ARCH_KernelTest : public GPUContextFixture
    {
    };

    TEST_F(KernelTest, Preamble)
    {
        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.preamble());

        std::string expected = R"(
            .amdgcn_target "amdgcn-amd-amdhsa--gfx1012:xnack+"
            .set .amdgcn.next_free_vgpr, 0
            .set .amdgcn.next_free_sgpr, 0
            .text
            .globl hello_world
            .p2align 8
            .type hello_world,@function
            hello_world:
        )";

        EXPECT_THAT(output(), MatchesSource(expected));
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(output()));
    }

    TEST_F(KernelTest, AddArgument)
    {
        auto kernel = m_context->kernel();

        kernel->addArgument({"foo", {DataType::Float}});
        EXPECT_EQ(0, kernel->arguments()[0].offset);
        EXPECT_EQ(4, kernel->arguments()[0].size);

        kernel->addArgument(
            {"bar", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly});
        EXPECT_EQ(8, kernel->arguments()[1].offset);
        EXPECT_EQ(8, kernel->arguments()[1].size);

        kernel->addArgument({"baz", {DataType::Double}});
        EXPECT_EQ(16, kernel->arguments()[2].offset);
        EXPECT_EQ(8, kernel->arguments()[2].size);
    }

    TEST_F(KernelTest, Postamble)
    {
        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.postamble());

        std::string expected = R"(
            s_endpgm
            .Lhello_world_end:
            .size hello_world, .Lhello_world_end-hello_world
            .rodata
            .p2align 6
            .amdhsa_kernel hello_world
            .amdhsa_next_free_vgpr 0
            .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
            .amdhsa_wavefront_size32 1
            //.amdhsa_accum_offset 4
            .amdhsa_user_sgpr_kernarg_segment_ptr 1
            .amdhsa_system_sgpr_workgroup_id_x 1
            .amdhsa_system_sgpr_workgroup_id_y 1
            .amdhsa_system_sgpr_workgroup_id_z 1
            .amdhsa_system_sgpr_workgroup_info 0
            .amdhsa_system_vgpr_workitem_id 2
            .end_amdhsa_kernel
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    class MaxRegisterKernelTest : public KernelTest
    {
    protected:
        std::string targetArchitecture() override
        {
            return "gfx90a";
        }

        void SetUp() override
        {
            m_kernelOptions.maxVGPRs             = 234;
            m_kernelOptions.setNextFreeVGPRToMax = true;

            GenericContextFixture::SetUp();
        }
    };

    TEST_F(MaxRegisterKernelTest, MaxVGPRsInPostamble)
    {
        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.postamble());

        EXPECT_EQ(countSubstring(output(), ".amdhsa_next_free_vgpr 234"), 1);
    }

    TEST_F(KernelTest, Metadata)
    {
        AssemblyKernel k(m_context, "hello_world");

        m_context->schedule(k.amdgpu_metadata());

        std::string expected = R"(
.amdgpu_metadata
---
amdhsa.version:
 - 1
 - 0
amdhsa.kernels:
 - .name: hello_world
   .symbol: hello_world.kd
   .kernarg_segment_size: 0
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .kernarg_segment_align: 8
   .wavefront_size: 32
   .sgpr_count: 0
   .vgpr_count: 0
   .agpr_count: 0
   .max_flat_workgroup_size: 1
   .args: []
...
.end_amdgpu_metadata
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_F(KernelTest, ArgumentsAndRegisters)
    {
        AssemblyKernel k(m_context, "hello_world");
        k.addArgument({"foo", {DataType::Float}});
        k.setWorkgroupSize({16, 8, 2});

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float);
        m_context->allocator(Register::Type::Scalar)->allocate(alloc0);

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Vector, DataType::Double);
        m_context->allocator(Register::Type::Vector)->allocate(alloc1);

        m_context->schedule(k.amdgpu_metadata());

        std::string expected = R"(
.amdgpu_metadata
---
amdhsa.version:
    - 1
    - 0
amdhsa.kernels:
    - .name: hello_world
      .symbol: hello_world.kd
      .kernarg_segment_size: 4
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .sgpr_count: 1
      .vgpr_count: 2
      .agpr_count: 0
      .max_flat_workgroup_size: 256
      .args:
          - .name: foo
            .size: 4
            .offset: 0
            .value_kind: by_value
...
.end_amdgpu_metadata)";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_P(ARCH_KernelTest, WholeKernel)
    {
        auto k = m_context->kernel();

        k->setKernelName("hello_world");
        k->setKernelDimensions(1);

        k->addArgument(
            {"ptr", {DataType::Float, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument({"val", {DataType::Float}});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            auto v_ptr   = Register::Value::Placeholder(m_context,
                                                      Register::Type::Vector,
                                                      {DataType::Float, PointerType::PointerGlobal},
                                                      1);
            auto v_value = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            co_yield v_ptr->allocate();

            co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

            co_yield v_value->allocate();

            co_yield m_context->copier()->copy(v_value, s_value, "Move value");

            co_yield m_context->mem()->storeFlat(v_ptr, v_value, 0, 4);
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto ptr = make_shared_device<float>();

            ASSERT_THAT(hipMemset(ptr.get(), 0, sizeof(float)), HasHipSuccess(0));

            KernelArguments kargs;
            kargs.append("ptr", ptr.get());
            kargs.append("val", 6.0f);
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            float resultValue = 0.0f;
            ASSERT_THAT(hipMemcpy(&resultValue, ptr.get(), sizeof(float), hipMemcpyDefault),
                        HasHipSuccess(0));

            EXPECT_EQ(resultValue, 6.0f);

            // Call the kernel a second time with different input.
            KernelArguments kargs2;
            kargs2.append("ptr", ptr.get());
            kargs2.append("val", 7.5f);

            executableKernel->executeKernel(kargs2, invocation);

            ASSERT_THAT(hipMemcpy(&resultValue, ptr.get(), sizeof(float), hipMemcpyDefault),
                        HasHipSuccess(0));

            EXPECT_EQ(resultValue, 7.5f);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }

        // kernel.s can be assembled with:
        // /opt/rocm/bin/amdclang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+ build/kernel.s -o build/kernel.o
    }

    INSTANTIATE_TEST_SUITE_P(ARCH_KernelTests, ARCH_KernelTest, supportedISATuples());

    class GPU_KernelTest : public CurrentGPUContextFixture
    {
    };

    TEST_F(GPU_KernelTest, WholeKernel)
    {
        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptr_arg  = command->allocateArgument(floatPtr);
        auto val_arg  = command->allocateArgument(floatVal);
        auto size_arg = command->allocateArgument(uintVal);

        auto ptr_exp  = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp  = std::make_shared<Expression::Expression>(val_arg);
        auto size_exp = std::make_shared<Expression::Expression>(size_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"ptr",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Float}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({size_exp, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            auto v_ptr   = Register::Value::Placeholder(m_context,
                                                      Register::Type::Vector,
                                                      {DataType::Float, PointerType::PointerGlobal},
                                                      1);
            auto v_value = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            co_yield v_ptr->allocate();

            co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

            co_yield v_value->allocate();

            co_yield m_context->copier()->copy(v_value, s_value, "Move value");

            co_yield m_context->mem()->storeFlat(v_ptr, v_value, 0, 4);
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        // Check for kernel's name
        ASSERT_STREQ(hipKernelNameRef(commandKernel.getHipFunction()),
                     "GPU_KernelTestWholeKernel_kernel");

        auto         ptr  = make_shared_device<float>();
        float        val  = 6.0f;
        unsigned int size = 1;

        ASSERT_THAT(hipMemset(ptr.get(), 0, sizeof(float)), HasHipSuccess(0));

        KernelArguments runtimeArgs;
        runtimeArgs.append("ptr", ptr.get());
        runtimeArgs.append("val", val);
        runtimeArgs.append("size", size);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        float resultValue = 0.0f;
        ASSERT_THAT(hipMemcpy(&resultValue, ptr.get(), sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        EXPECT_EQ(resultValue, 6.0f);
    }

}
