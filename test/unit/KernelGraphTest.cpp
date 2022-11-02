
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>
#include <variant>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordGraph/CoordinateHypergraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using ::testing::HasSubstr;

namespace KernelGraphTest
{
    class KernelGraphTestGPU : public CurrentGPUContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp()
        {
            CurrentGPUContextFixture::SetUp();
            Settings::getInstance()->set(Settings::SaveAssembly, true);

            fastArith = Expression::FastArithmetic(m_context);
        }

        static std::shared_ptr<Command> commonCommand()
        {
            auto command = std::make_shared<rocRoller::Command>();

            Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
            command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

            Operations::T_Load_Linear load_B(DataType::Int32, 1, 2);
            command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));

            Operations::T_Execute execute;
            execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(3, 2, 0)));
            execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Neg(4, 3)));
            execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(5, 3, 4)));

            command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

            Operations::T_Store_Linear store_C(1, 5);
            command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));
            return command;
        }

        void GPU_Translate04(bool reload);
    };

    class KernelGraphTest : public GenericContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp()
        {
            GenericContextFixture::SetUp();
            fastArith = Expression::FastArithmetic(m_context);
        }

        static std::shared_ptr<Command> commonCommand()
        {
            return KernelGraphTestGPU::commonCommand();
        }
    };

    class KernelGraphTestGPULoopSize : public KernelGraphTestGPU,
                                       public ::testing::WithParamInterface<int>
    {
    };

    TEST_P(KernelGraphTestGPULoopSize, MissingWorkitemCount)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        int  loopSize     = GetParam();
        auto loopSizeExpr = Expression::literal(loopSize);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / loopSizeExpr;

        ASSERT_THROW(
            {
                auto kgraph = KernelGraph::translate(command);

                kgraph = KernelGraph::lowerLinear(kgraph, m_context);

                kgraph = KernelGraph::lowerLinearLoop(kgraph, loopSizeExpr, m_context);

                kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

                m_context->kernel()->setWorkitemCount({numWorkitems, one, one});
            },
            FatalError);
    }

    TEST_P(KernelGraphTestGPULoopSize, TestForLoop)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        int  loopSize     = GetParam();
        auto loopSizeExpr = Expression::literal(loopSize);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / loopSizeExpr;

        m_context->kernel()->setWorkitemCount({numWorkitems, one, one});

        size_t origArgSize = m_context->kernel()->arguments().size();

        auto kgraph = KernelGraph::translate(command);

        kgraph = KernelGraph::lowerLinear(kgraph, m_context);

        kgraph = KernelGraph::lowerLinearLoop(kgraph, loopSizeExpr, m_context);

        kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        EXPECT_EQ(m_context->kernel()->arguments().size(), origArgSize + 1);
        ASSERT_NO_THROW(m_context->kernel()->findArgument("LAUNCH_WORKGROUPCOUNT_0"));

        CommandKernel commandKernel(command, m_context, kgraph);

        RandomGenerator random(1356);

        int              baseSize = workGroupSize * loopSize;
        std::vector<int> vecSizes = {baseSize, baseSize * 5, baseSize * 16, baseSize * 65};
        for(auto vecSize : vecSizes)
        {
            auto             a          = random.vector<int>(vecSize, -1000, 1000);
            auto             b          = random.vector<int>(vecSize, -1000, 1000);
            auto             c_expected = random.vector<int>(vecSize, -1000, 1000);
            auto             c_actual   = random.vector<int>(vecSize, -1000, 1000);
            std::vector<int> c(vecSize);
            for(int i = 0; i < vecSize; i++)
                c_expected[i] = -(a[i] + b[i]) * (a[i] + b[i]);

            auto a_d = make_shared_device<int>(vecSize);
            auto b_d = make_shared_device<int>(vecSize);
            auto c_d = make_shared_device<int>(vecSize);

            ASSERT_THAT(
                hipMemcpy(a_d.get(), a.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                HasHipSuccess(0));
            ASSERT_THAT(
                hipMemcpy(b_d.get(), b.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                HasHipSuccess(0));

            KernelArguments args;
            args.append("a", a_d.get());
            args.append<int64_t>("a_extent", vecSize);
            args.append<int64_t>("a_size", vecSize);
            args.append<int64_t>("a_stride", 1);

            args.append("b", b_d.get());
            args.append<int64_t>("b_extent", vecSize);
            args.append<int64_t>("b_size", vecSize);
            args.append<int64_t>("b_stride", 1);

            args.append("c", c_d.get());
            args.append<int64_t>("c_extent", vecSize);
            // args.append<int64_t>("c_size", vecSize);
            args.append<int64_t>("c_stride", 1);

            commandKernel.launchKernel(args.runtimeArguments());

            ASSERT_THAT(
                hipMemcpy(c_actual.data(), c_d.get(), vecSize * sizeof(int), hipMemcpyDeviceToHost),
                HasHipSuccess(0));

            EXPECT_THAT(output(), testing::HasSubstr("Lock For Loop"));
            EXPECT_THAT(output(), testing::HasSubstr("Unlock For Loop"));

            for(int i = 0; i < vecSize; i++)
                EXPECT_EQ(c_actual[i], c_expected[i]) << i << ", " << a[i] << ", " << b[i];
        }
    }

    INSTANTIATE_TEST_SUITE_P(KernelGraphTestGPULoopSize,
                             KernelGraphTestGPULoopSize,
                             ::testing::ValuesIn({1, 5, 16, 73}));

    TEST_F(KernelGraphTestGPU, TestKernelUnroll)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto unrollSize = Expression::literal(4);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / unrollSize;

        m_context->kernel()->setWorkitemCount({numWorkitems, one, one});

        auto kgraph = KernelGraph::translate(command);

        kgraph = KernelGraph::lowerLinear(kgraph, m_context);

        kgraph = KernelGraph::lowerLinearUnroll(kgraph, unrollSize, m_context);

        kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        m_context->kernel()->setKernelGraphMeta(std::make_shared<KernelGraph::KernelGraph>(kgraph));

        CommandKernel commandKernel(command, m_context, kgraph);

        RandomGenerator random(8379);

        int vecSize = 16384;

        auto             a          = random.vector<int>(vecSize, -1000, 1000);
        auto             b          = random.vector<int>(vecSize, -1000, 1000);
        auto             c_expected = random.vector<int>(vecSize, -1000, 1000);
        auto             c_actual   = random.vector<int>(vecSize, -1000, 1000);
        std::vector<int> c(vecSize);
        for(int i = 0; i < vecSize; i++)
            c_expected[i] = -(a[i] + b[i]) * (a[i] + b[i]);

        auto a_d = make_shared_device<int>(vecSize);
        auto b_d = make_shared_device<int>(vecSize);
        auto c_d = make_shared_device<int>(vecSize);

        ASSERT_THAT(hipMemcpy(a_d.get(), a.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));
        ASSERT_THAT(hipMemcpy(b_d.get(), b.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));

        KernelArguments args;
        args.append("a", a_d.get());
        args.append<int64_t>("a_extent", vecSize);
        args.append<int64_t>("a_size", vecSize);
        args.append<int64_t>("a_stride", 1);

        args.append("b", b_d.get());
        args.append<int64_t>("b_extent", vecSize);
        args.append<int64_t>("b_size", vecSize);
        args.append<int64_t>("b_stride", 1);

        args.append("c", c_d.get());
        args.append<int64_t>("c_extent", vecSize);
        // args.append<int64_t>("c_size", vecSize);
        args.append<int64_t>("c_stride", 1);

        commandKernel.launchKernel(args.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(c_actual.data(), c_d.get(), vecSize * sizeof(int), hipMemcpyDeviceToHost),
            HasHipSuccess(0));

        for(int i = 0; i < vecSize; i++)
            EXPECT_EQ(c_actual[i], c_expected[i]) << i << ", " << a[i] << ", " << b[i];
    }

    TEST_F(KernelGraphTestGPU, TestKernelUnrollAndLoop)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto unrollSize = Expression::literal(4);
        auto loopSize   = Expression::literal(16);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / (unrollSize * loopSize);

        m_context->kernel()->setWorkitemCount({numWorkitems, one, one});

        auto kgraph = KernelGraph::translate(command);

        kgraph = KernelGraph::lowerLinear(kgraph, m_context);

        kgraph = KernelGraph::lowerLinearLoop(kgraph, loopSize, m_context);
        kgraph = KernelGraph::lowerLinearUnroll(kgraph, unrollSize, m_context);

        kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        m_context->kernel()->setKernelGraphMeta(std::make_shared<KernelGraph::KernelGraph>(kgraph));

        CommandKernel commandKernel(command, m_context, kgraph);

        RandomGenerator random(68103);

        int vecSize = 16384;

        auto             a          = random.vector<int>(vecSize, -1000, 1000);
        auto             b          = random.vector<int>(vecSize, -1000, 1000);
        auto             c_expected = random.vector<int>(vecSize, -1000, 1000);
        auto             c_actual   = random.vector<int>(vecSize, -1000, 1000);
        std::vector<int> c(vecSize);
        for(int i = 0; i < vecSize; i++)
            c_expected[i] = -(a[i] + b[i]) * (a[i] + b[i]);

        auto a_d = make_shared_device<int>(vecSize);
        auto b_d = make_shared_device<int>(vecSize);
        auto c_d = make_shared_device<int>(vecSize);

        ASSERT_THAT(hipMemcpy(a_d.get(), a.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));
        ASSERT_THAT(hipMemcpy(b_d.get(), b.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));

        KernelArguments args;
        args.append("a", a_d.get());
        args.append<int64_t>("a_extent", vecSize);
        args.append<int64_t>("a_size", vecSize);
        args.append<int64_t>("a_stride", 1);

        args.append("b", b_d.get());
        args.append<int64_t>("b_extent", vecSize);
        args.append<int64_t>("b_size", vecSize);
        args.append<int64_t>("b_stride", 1);

        args.append("c", c_d.get());
        args.append<int64_t>("c_extent", vecSize);
        // args.append<int64_t>("c_size", vecSize);
        args.append<int64_t>("c_stride", 1);

        commandKernel.launchKernel(args.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(c_actual.data(), c_d.get(), vecSize * sizeof(int), hipMemcpyDeviceToHost),
            HasHipSuccess(0));

        for(int i = 0; i < vecSize; i++)
            EXPECT_EQ(c_actual[i], c_expected[i]) << i << ", " << a[i] << ", " << b[i];
    }

    TEST_F(KernelGraphTest, Translate01)
    {
        auto command = commonCommand();

        auto kgraph0 = KernelGraph::translate(command);

        auto bottom = kgraph0.coordinates.bottom();
        EXPECT_EQ(bottom.size(), 2);
        EXPECT_EQ(getTag(bottom[0]), getTag(KernelGraph::CoordinateTransform::User(0)));
        EXPECT_EQ(getTag(bottom[1]), getTag(KernelGraph::CoordinateTransform::User(2)));

        auto top = kgraph0.coordinates.top();
        EXPECT_EQ(top.size(), 1);
        EXPECT_EQ(getTag(top[0]), getTag(KernelGraph::CoordinateTransform::User(5, true)));

        std::string expected0 = R".(
          digraph {
           { "User{0, NA, i}" } -> { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } -> { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Flatten"]
           { "User{0, NA, i}" } -> { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } [color=red label="DataFlow"]
           { "User{2, NA, i}" } -> { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Flatten"]
           { "User{2, NA, i}" } -> { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } [color=red label="DataFlow"]
           { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}", "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Linear{3, NA, i}" } [color=red label="DataFlow"]
           { "Linear{3, NA, i}" } -> { "Linear{4, NA, i}" } [color=red label="DataFlow"]
           { "Linear{3, NA, i}", "Linear{4, NA, i}" } -> { "Linear{5, NA, i}" } [color=red label="DataFlow"]
           { "Linear{5, NA, i}" } -> { "Linear{5, NA, o}" } [color=blue label="MakeOutput"]
           { "Linear{5, NA, o}" } -> { "SubDimension{5, 0, NA, o}" } [color=blue label="Split"]
           { "SubDimension{5, 0, NA, o}" } -> { "User{5, NA, o}" } [color=blue label="Join"]
           { "Linear{5, NA, i}" } -> { "User{5, NA, o}" } [color=red label="DataFlow"]

          subgraph clusterCF {"krnKernel"[label="Kernel"];
          "krnLoadLinear(0)"[label="LoadLinear(0)"];
          "krnLoadLinear(2)"[label="LoadLinear(2)"];
          "krnElementOp(3)"[label="ElementOp(3)"];
          "krnElementOp(4)"[label="ElementOp(4)"];
          "krnElementOp(5)"[label="ElementOp(5)"];
          "krnStoreLinear(5)"[label="StoreLinear(5)"];
          "krnKernel" -> "krnLoadLinear(0)"[label="Body"];
          "krnKernel" -> "krnLoadLinear(2)"[label="Body"];
          "krnLoadLinear(0)" -> "krnElementOp(3)"[label="Sequence"];
          "krnLoadLinear(2)" -> "krnElementOp(3)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(4)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(4)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(5)" -> "krnStoreLinear(5)"[label="Sequence"];
          } }
        ).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT()));

        std::string expected1 = R".(
          digraph {
           { "User{0, NA, i}" } -> { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } -> { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Flatten"]
           { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } -> { "Workgroup{0, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{0, 0, 32j, i}" } [color=blue label="Tile"]
           { "Workgroup{0, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{0, 0, 32j, i}" } -> { "VGPR{0, NA, i}" } [color=blue label="Forget"]
           { "User{0, NA, i}" } -> { "VGPR{0, NA, i}" } [color=red label="DataFlow"]
           { "User{2, NA, i}" } -> { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Flatten"]
           { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Workgroup{2, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{2, 0, 32j, i}" } [color=blue label="Tile"]
           { "Workgroup{2, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{2, 0, 32j, i}" } -> { "VGPR{2, NA, i}" } [color=blue label="Forget"]
           { "User{2, NA, i}" } -> { "VGPR{2, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{0, NA, i}", "VGPR{2, NA, i}" } -> { "VGPR{3, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{3, NA, i}" } -> { "VGPR{4, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{3, NA, i}", "VGPR{4, NA, i}" } -> { "VGPR{5, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{5, NA, i}" } -> { "Workgroup{5, 0, LAUNCH_WORKGROUPCOUNT_0, o}", "Workitem{5, 0, 32j, o}" } [color=blue label="Inherit"]
           { "Workgroup{5, 0, LAUNCH_WORKGROUPCOUNT_0, o}", "Workitem{5, 0, 32j, o}" } -> { "Linear{5, NA, o}" } [color=blue label="Flatten"]
           { "Linear{5, NA, o}" } -> { "SubDimension{5, 0, NA, o}" } [color=blue label="Split"]
           { "SubDimension{5, 0, NA, o}" } -> { "User{5, NA, o}" } [color=blue label="Join"]
           { "VGPR{5, NA, i}" } -> { "User{5, NA, o}" } [color=red label="DataFlow"]

          subgraph clusterCF {"krnKernel"[label="Kernel"];
          "krnLoadVGPR(0)"[label="LoadVGPR(0)"];
          "krnLoadVGPR(2)"[label="LoadVGPR(2)"];
          "krnElementOp(3)"[label="ElementOp(3)"];
          "krnElementOp(4)"[label="ElementOp(4)"];
          "krnElementOp(5)"[label="ElementOp(5)"];
          "krnStoreVGPR(5)"[label="StoreVGPR(5)"];
          "krnKernel" -> "krnLoadVGPR(0)"[label="Body"];
          "krnKernel" -> "krnLoadVGPR(2)"[label="Body"];
          "krnLoadVGPR(0)" -> "krnElementOp(3)"[label="Sequence"];
          "krnLoadVGPR(2)" -> "krnElementOp(3)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(4)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(4)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(5)" -> "krnStoreVGPR(5)"[label="Sequence"];
          } }
        ).";

        auto one = Expression::literal(1u);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto kgraph1 = KernelGraph::lowerLinear(kgraph0, m_context);
        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT()));
    }

    TEST_F(KernelGraphTest, Translate01B)
    {
        auto command = commonCommand();
        auto kgraph0 = KernelGraph::translate2(command);

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(std::holds_alternative<KernelGraph::CoordGraph::User>(
                std::get<KernelGraph::CoordGraph::Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(std::holds_alternative<KernelGraph::CoordGraph::User>(
                std::get<KernelGraph::CoordGraph::Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto visitor = KernelGraph::BaseGraphVisitor2(m_context);
        auto kgraphC = rewrite(kgraph0, visitor);

        std::string expectedC = R".(
             digraph {
             "coord1"[label="User{NA}(1)"];
             "coord2"[label="User{NA}(2)"];
             "coord3"[label="SubDimension{0, CommandArgument(Load_Linear_0_size_0)}(3)"];
             "coord4"[label="Split(4)",shape=box];
             "coord5"[label="Linear{CommandArgument(Load_Linear_0_size_0)}(5)"];
             "coord6"[label="Flatten(6)",shape=box];
             "coord7"[label="DataFlow(7)",shape=box];
             "coord8"[label="SubDimension{0, CommandArgument(Load_Linear_2_size_0)}(8)"];
             "coord9"[label="Split(9)",shape=box];
             "coord10"[label="Linear{CommandArgument(Load_Linear_2_size_0)}(10)"];
             "coord11"[label="Flatten(11)",shape=box];
             "coord12"[label="DataFlow(12)",shape=box];
             "coord13"[label="Linear{NA}(13)"];
             "coord14"[label="DataFlow(14)",shape=box];
             "coord15"[label="Linear{NA}(15)"];
             "coord16"[label="DataFlow(16)",shape=box];
             "coord17"[label="Linear{NA}(17)"];
             "coord18"[label="DataFlow(18)",shape=box];
             "coord19"[label="SubDimension{0, NA}(19)"];
             "coord20"[label="Split(20)",shape=box];
             "coord21"[label="User{NA}(21)"];
             "coord22"[label="Join(22)",shape=box];
             "coord23"[label="DataFlow(23)",shape=box];
             "coord1" -> "coord4"
             "coord1" -> "coord7"
             "coord2" -> "coord9"
             "coord2" -> "coord12"
             "coord3" -> "coord6"
             "coord4" -> "coord3"
             "coord5" -> "coord14"
             "coord6" -> "coord5"
             "coord7" -> "coord5"
             "coord8" -> "coord11"
             "coord9" -> "coord8"
             "coord10" -> "coord14"
             "coord11" -> "coord10"
             "coord12" -> "coord10"
             "coord13" -> "coord16"
             "coord13" -> "coord18"
             "coord14" -> "coord13"
             "coord15" -> "coord18"
             "coord16" -> "coord15"
             "coord17" -> "coord20"
             "coord17" -> "coord23"
             "coord18" -> "coord17"
             "coord19" -> "coord22"
             "coord20" -> "coord19"
             "coord22" -> "coord21"
             "coord23" -> "coord21"
             {
             rank=same
             "coord5"->"coord10"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord15"->"coord13"[style=invis]
             rankdir=LR
             }
             subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
             "cntrl2"[label="LoadLinear(2)"];
             "cntrl3"[label="Body(3)",shape=box];
             "cntrl4"[label="LoadLinear(4)"];
             "cntrl5"[label="Body(5)",shape=box];
             "cntrl6"[label="ElementOp(4, 10)(6)"];
             "cntrl7"[label="Sequence(7)",shape=box];
             "cntrl8"[label="Sequence(8)",shape=box];
             "cntrl9"[label="ElementOp(13, -1)(9)"];
             "cntrl10"[label="Sequence(10)",shape=box];
             "cntrl11"[label="ElementOp(15, 13)(11)"];
             "cntrl12"[label="Sequence(12)",shape=box];
             "cntrl13"[label="Sequence(13)",shape=box];
             "cntrl14"[label="StoreLinear(14)"];
             "cntrl15"[label="Sequence(15)",shape=box];
             "cntrl1" -> "cntrl3"
             "cntrl1" -> "cntrl5"
             "cntrl2" -> "cntrl7"
             "cntrl3" -> "cntrl2"
             "cntrl4" -> "cntrl8"
             "cntrl5" -> "cntrl4"
             "cntrl6" -> "cntrl10"
             "cntrl6" -> "cntrl13"
             "cntrl7" -> "cntrl6"
             "cntrl8" -> "cntrl6"
             "cntrl9" -> "cntrl12"
             "cntrl10" -> "cntrl9"
             "cntrl11" -> "cntrl15"
             "cntrl12" -> "cntrl11"
             "cntrl13" -> "cntrl11"
             "cntrl15" -> "cntrl14"
             }
             "coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
             "coord5" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
             "coord2" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
             "coord10" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
             "coord13" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
             "coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
             "coord17" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
             "coord21" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
             "coord17" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
             }).";

        EXPECT_EQ(NormalizedSource(expectedC), NormalizedSource(kgraphC.toDOT(true)));

        std::string expected0 = R".(
             digraph {
             "coord1"[label="SubDimension{0, CommandArgument(Load_Linear_0_size_0)}(1)"];
             "coord2"[label="User{NA}(2)"];
             "coord3"[label="Split(3)",shape=box];
             "coord4"[label="Linear{CommandArgument(Load_Linear_0_size_0)}(4)"];
             "coord5"[label="Flatten(5)",shape=box];
             "coord6"[label="DataFlow(6)",shape=box];
             "coord7"[label="SubDimension{0, CommandArgument(Load_Linear_2_size_0)}(7)"];
             "coord8"[label="User{NA}(8)"];
             "coord9"[label="Split(9)",shape=box];
             "coord10"[label="Linear{CommandArgument(Load_Linear_2_size_0)}(10)"];
             "coord11"[label="Flatten(11)",shape=box];
             "coord12"[label="DataFlow(12)",shape=box];
             "coord13"[label="Linear{NA}(13)"];
             "coord14"[label="DataFlow(14)",shape=box];
             "coord15"[label="Linear{NA}(15)"];
             "coord16"[label="DataFlow(16)",shape=box];
             "coord17"[label="Linear{NA}(17)"];
             "coord18"[label="DataFlow(18)",shape=box];
             "coord19"[label="SubDimension{0, NA}(19)"];
             "coord20"[label="User{NA}(20)"];
             "coord21"[label="Split(21)",shape=box];
             "coord22"[label="Join(22)",shape=box];
             "coord23"[label="DataFlow(23)",shape=box];
             "coord1" -> "coord5"
             "coord2" -> "coord3"
             "coord2" -> "coord6"
             "coord3" -> "coord1"
             "coord4" -> "coord14"
             "coord5" -> "coord4"
             "coord6" -> "coord4"
             "coord7" -> "coord11"
             "coord8" -> "coord9"
             "coord8" -> "coord12"
             "coord9" -> "coord7"
             "coord10" -> "coord14"
             "coord11" -> "coord10"
             "coord12" -> "coord10"
             "coord13" -> "coord16"
             "coord13" -> "coord18"
             "coord14" -> "coord13"
             "coord15" -> "coord18"
             "coord16" -> "coord15"
             "coord17" -> "coord21"
             "coord17" -> "coord23"
             "coord18" -> "coord17"
             "coord19" -> "coord22"
             "coord21" -> "coord19"
             "coord22" -> "coord20"
             "coord23" -> "coord20"
             {
             rank=same
             "coord4"->"coord10"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord15"->"coord13"[style=invis]
             rankdir=LR
             }
             subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
             "cntrl2"[label="LoadLinear(2)"];
             "cntrl3"[label="Body(3)",shape=box];
             "cntrl4"[label="LoadLinear(4)"];
             "cntrl5"[label="Body(5)",shape=box];
             "cntrl6"[label="ElementOp(4, 10)(6)"];
             "cntrl7"[label="Sequence(7)",shape=box];
             "cntrl8"[label="Sequence(8)",shape=box];
             "cntrl9"[label="ElementOp(13, -1)(9)"];
             "cntrl10"[label="Sequence(10)",shape=box];
             "cntrl11"[label="ElementOp(15, 13)(11)"];
             "cntrl12"[label="Sequence(12)",shape=box];
             "cntrl13"[label="Sequence(13)",shape=box];
             "cntrl14"[label="StoreLinear(14)"];
             "cntrl15"[label="Sequence(15)",shape=box];
             "cntrl1" -> "cntrl3"
             "cntrl1" -> "cntrl5"
             "cntrl2" -> "cntrl7"
             "cntrl3" -> "cntrl2"
             "cntrl4" -> "cntrl8"
             "cntrl5" -> "cntrl4"
             "cntrl6" -> "cntrl10"
             "cntrl6" -> "cntrl13"
             "cntrl7" -> "cntrl6"
             "cntrl8" -> "cntrl6"
             "cntrl9" -> "cntrl12"
             "cntrl10" -> "cntrl9"
             "cntrl11" -> "cntrl15"
             "cntrl12" -> "cntrl11"
             "cntrl13" -> "cntrl11"
             "cntrl15" -> "cntrl14"
             }
             "coord2" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
             "coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
             "coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
             "coord10" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
             "coord13" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
             "coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
             "coord17" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
             "coord20" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
             "coord17" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
             }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        std::string expected1 = R".(
             digraph {
             "coord1"[label="User{NA}(1)"];
             "coord2"[label="User{NA}(2)"];
             "coord3"[label="SubDimension{0, CommandArgument(Load_Linear_0_size_0)}(3)"];
             "coord4"[label="Split(4)",shape=box];
             "coord5"[label="Linear{CommandArgument(Load_Linear_0_size_0)}(5)"];
             "coord6"[label="Flatten(6)",shape=box];
             "coord7"[label="SubDimension{0, CommandArgument(Load_Linear_2_size_0)}(7)"];
             "coord8"[label="Split(8)",shape=box];
             "coord9"[label="Linear{CommandArgument(Load_Linear_2_size_0)}(9)"];
             "coord10"[label="Flatten(10)",shape=box];
             "coord11"[label="Linear{NA}(11)"];
             "coord12"[label="SubDimension{0, NA}(12)"];
             "coord13"[label="Split(13)",shape=box];
             "coord14"[label="User{NA}(14)"];
             "coord15"[label="Join(15)",shape=box];
             "coord16"[label="Workgroup{0, NA}(16)"];
             "coord17"[label="Workitem{0, 32j}(17)"];
             "coord18"[label="VGPR{NA}(18)"];
             "coord19"[label="Tile(19)",shape=box];
             "coord20"[label="Forget(20)",shape=box];
             "coord21"[label="DataFlow(21)",shape=box];
             "coord22"[label="Workgroup{0, NA}(22)"];
             "coord23"[label="Workitem{0, 32j}(23)"];
             "coord24"[label="VGPR{NA}(24)"];
             "coord25"[label="Tile(25)",shape=box];
             "coord26"[label="Forget(26)",shape=box];
             "coord27"[label="DataFlow(27)",shape=box];
             "coord28"[label="VGPR{NA}(28)"];
             "coord29"[label="DataFlow(29)",shape=box];
             "coord30"[label="VGPR{NA}(30)"];
             "coord31"[label="DataFlow(31)",shape=box];
             "coord32"[label="VGPR{NA}(32)"];
             "coord33"[label="DataFlow(33)",shape=box];
             "coord34"[label="Workgroup{0, NA}(34)"];
             "coord35"[label="Workitem{0, 32j}(35)"];
             "coord36"[label="Inherit(36)",shape=box];
             "coord37"[label="Flatten(37)",shape=box];
             "coord38"[label="DataFlow(38)",shape=box];
             "coord1" -> "coord4"
             "coord1" -> "coord21"
             "coord2" -> "coord8"
             "coord2" -> "coord27"
             "coord3" -> "coord6"
             "coord4" -> "coord3"
             "coord5" -> "coord19"
             "coord6" -> "coord5"
             "coord7" -> "coord10"
             "coord8" -> "coord7"
             "coord9" -> "coord25"
             "coord10" -> "coord9"
             "coord11" -> "coord13"
             "coord12" -> "coord15"
             "coord13" -> "coord12"
             "coord15" -> "coord14"
             "coord16" -> "coord20"
             "coord17" -> "coord20"
             "coord18" -> "coord29"
             "coord19" -> "coord16"
             "coord19" -> "coord17"
             "coord20" -> "coord18"
             "coord21" -> "coord18"
             "coord22" -> "coord26"
             "coord23" -> "coord26"
             "coord24" -> "coord29"
             "coord25" -> "coord22"
             "coord25" -> "coord23"
             "coord26" -> "coord24"
             "coord27" -> "coord24"
             "coord28" -> "coord31"
             "coord28" -> "coord33"
             "coord29" -> "coord28"
             "coord30" -> "coord33"
             "coord31" -> "coord30"
             "coord32" -> "coord36"
             "coord32" -> "coord38"
             "coord33" -> "coord32"
             "coord34" -> "coord37"
             "coord35" -> "coord37"
             "coord36" -> "coord34"
             "coord36" -> "coord35"
             "coord37" -> "coord11"
             "coord38" -> "coord14"
             {
             rank=same
             "coord16"->"coord17"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord16"->"coord17"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord22"->"coord23"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord22"->"coord23"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord18"->"coord24"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord30"->"coord28"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord34"->"coord35"[style=invis]
             rankdir=LR
             }
             {
             rank=same
             "coord34"->"coord35"[style=invis]
             rankdir=LR
             }
             subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
             "cntrl2"[label="LoadVGPR(2)"];
             "cntrl3"[label="Body(3)",shape=box];
             "cntrl4"[label="LoadVGPR(4)"];
             "cntrl5"[label="Body(5)",shape=box];
             "cntrl6"[label="ElementOp(18, 24)(6)"];
             "cntrl7"[label="Sequence(7)",shape=box];
             "cntrl8"[label="Sequence(8)",shape=box];
             "cntrl9"[label="ElementOp(28, -1)(9)"];
             "cntrl10"[label="Sequence(10)",shape=box];
             "cntrl11"[label="ElementOp(30, 28)(11)"];
             "cntrl12"[label="Sequence(12)",shape=box];
             "cntrl13"[label="Sequence(13)",shape=box];
             "cntrl14"[label="StoreVGPR(14)"];
             "cntrl15"[label="Sequence(15)",shape=box];
             "cntrl1" -> "cntrl3"
             "cntrl1" -> "cntrl5"
             "cntrl2" -> "cntrl7"
             "cntrl3" -> "cntrl2"
             "cntrl4" -> "cntrl8"
             "cntrl5" -> "cntrl4"
             "cntrl6" -> "cntrl10"
             "cntrl6" -> "cntrl13"
             "cntrl7" -> "cntrl6"
             "cntrl8" -> "cntrl6"
             "cntrl9" -> "cntrl12"
             "cntrl10" -> "cntrl9"
             "cntrl11" -> "cntrl15"
             "cntrl12" -> "cntrl11"
             "cntrl13" -> "cntrl11"
             "cntrl15" -> "cntrl14"
             }
             "coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
             "coord18" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
             "coord2" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
             "coord24" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
             "coord28" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
             "coord30" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
             "coord32" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
             "coord14" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
             "coord32" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
             }).";

        auto one = Expression::literal(1u);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto kgraph1 = KernelGraph::lowerLinear(kgraph0, m_context);
        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TranslateTMul)
    {
        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 1))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1)));

        auto kgraph0 = KernelGraph::translate(command);

        std::string expected0 = R".(
          digraph {
           { "User{0, NA, i}" } -> { "SubDimension{0, 0, CommandArgument(Load_Tiled_0_size_0), i}", "SubDimension{0, 1, CommandArgument(Load_Tiled_0_size_1), i}" } [color=blue label="Split"]
           { "SubDimension{0, 0, CommandArgument(Load_Tiled_0_size_0), i}", "SubDimension{0, 1, CommandArgument(Load_Tiled_0_size_1), i}" } -> { "MacroTile{0, NA, i}" } [color=blue label="ConstructTensorTile"]
           { "User{0, NA, i}" } -> { "MacroTile{0, NA, i}" } [color=red label="DataFlow"]
           { "User{1, NA, i}" } -> { "SubDimension{1, 0, CommandArgument(Load_Tiled_1_size_0), i}", "SubDimension{1, 1, CommandArgument(Load_Tiled_1_size_1), i}" } [color=blue label="Split"]
           { "SubDimension{1, 0, CommandArgument(Load_Tiled_1_size_0), i}", "SubDimension{1, 1, CommandArgument(Load_Tiled_1_size_1), i}" } -> { "MacroTile{1, NA, i}" } [color=blue label="ConstructTensorTile"]
           { "User{1, NA, i}" } -> { "MacroTile{1, NA, i}" } [color=red label="DataFlow"]
           { "MacroTile{0, NA, i}", "MacroTile{1, NA, i}" } -> { "MacroTile{2, NA, i}" } [color=red label="DataFlow"]

          subgraph clusterCF {"krnKernel"[label="Kernel"];
          "krnLoadTiled(0)"[label="LoadTiled(0)"];
          "krnLoadTiled(1)"[label="LoadTiled(1)"];
          "krnTensorContraction(2, 0, 1)"[label="TensorContraction(2, 0, 1)"];
          "krnKernel" -> "krnLoadTiled(0)"[label="Body"];
          "krnKernel" -> "krnLoadTiled(1)"[label="Body"];
          "krnLoadTiled(0)" -> "krnTensorContraction(2, 0, 1)"[label="Sequence"];
          "krnLoadTiled(1)" -> "krnTensorContraction(2, 0, 1)"[label="Sequence"];
          } }
        ).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT()));
    }

    TEST_F(KernelGraphTest, Translate02)
    {
        auto command = commonCommand();

        auto one = Expression::literal(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto kgraph0 = KernelGraph::translate(command);
        auto kgraph1 = KernelGraph::lowerLinear(kgraph0, m_context);

        auto user0   = KernelGraph::CoordinateTransform::User(0);
        auto block0  = KernelGraph::CoordinateTransform::Workgroup(0);
        auto thread0 = KernelGraph::CoordinateTransform::Workitem(0);

        // given block id and thread id, compute regular (user) index for first (0) dataflow array
        auto block_id  = Expression::literal(2);
        auto thread_id = Expression::literal(33);

        auto exprs = kgraph1.coordinates.reverse(
            {block_id, thread_id}, {user0}, {block0, thread0}, nullptr);
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "Multiply(Add(Multiply(2i, 32j), 33i), CommandArgument(Load_Linear_0_stride_0))");

        exprs = kgraph1.coordinates.reverse(
            {block_id, thread_id}, {user0}, {block0, thread0}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "Multiply(97j, CommandArgument(Load_Linear_0_stride_0))");
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate03)
    {
        TIMER(t_total, "Translate03");
        TIMER(t_gpu, "Translate03::GPU");
        TIMER(t_hip, "Translate03::HIP");

        TIC(t_total);

        auto command = std::make_shared<rocRoller::Command>();

        Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

        Operations::T_Load_Linear load_B(DataType::Int32, 1, 2);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));

        Operations::T_Execute execute;
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(3, 2, 0)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(5, 3, 0)));

        command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

        Operations::T_Store_Linear store_C(1, 5);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));

        CommandKernel commandKernel(command, "Translate03");

        auto kgraph2 = commandKernel.getKernelGraph();

        auto kernelNode = kgraph2.control.getRootOperation();

        {
            auto expected = getTag(kernelNode);
            auto outputs
                = kgraph2.control.getOutputs<KernelGraph::ControlGraph::Body>(getTag(kernelNode));
            EXPECT_EQ(2, outputs.size());

            auto outputs2 = kgraph2.control.getOutputs<KernelGraph::ControlGraph::Sequence>(
                getTag(kernelNode));
            EXPECT_EQ(0, outputs2.size());

            auto outputs3
                = kgraph2.control.getOutputs(getTag(kernelNode), KernelGraph::ControlGraph::Body{});

            auto outputTags3 = kgraph2.control.getOutputTags(getTag(kernelNode),
                                                             KernelGraph::ControlGraph::Body{});

            EXPECT_EQ(outputs3.size(), outputTags3.size());
            for(size_t i = 0; i < outputs3.size(); i++)
            {
                EXPECT_EQ(getTag(outputs3[i]), outputTags3[i]);
            }

            EXPECT_EQ(getTag(outputs[0]), getTag(outputs3[0]));

            auto inputs1
                = kgraph2.control.getInputs<KernelGraph::ControlGraph::Body>(getTag(outputs.at(0)));
            ASSERT_EQ(1, inputs1.size());

            auto actual1 = getTag(inputs1.at(0));
            EXPECT_EQ(actual1, expected);

            auto inputs2 = kgraph2.control.getInputs(getTag(outputs.at(0)),
                                                     KernelGraph::ControlGraph::Body{});
            ASSERT_EQ(1, inputs2.size());

            auto inputTags2 = kgraph2.control.getInputTags(getTag(outputs.at(0)),
                                                           KernelGraph::ControlGraph::Body{});

            EXPECT_EQ(inputs2.size(), inputTags2.size());
            for(size_t i = 0; i < inputs2.size(); i++)
            {
                EXPECT_EQ(getTag(inputs2[i]), inputTags2[i]);
            }

            auto actual2 = getTag(inputs2.at(0));
            EXPECT_EQ(actual1, actual2);

            auto inputs3 = kgraph2.control.getInputs<KernelGraph::ControlGraph::Sequence>(
                getTag(outputs.at(0)));
            EXPECT_EQ(0, inputs3.size());

            auto inputs4 = kgraph2.control.getInputs<KernelGraph::ControlGraph::Initialize>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs4.size());

            auto inputs5 = kgraph2.control.getInputs<KernelGraph::ControlGraph::ForLoopIncrement>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs5.size());
        }

        {
            std::ostringstream msg;
            msg << kgraph2.control;

            std::ostringstream msg2;
            kgraph2.control.toDOT(msg2, "krn");

            EXPECT_EQ(msg.str(), msg2.str());
        }

        TIC(t_hip);
        size_t nx = 64;

        RandomGenerator random(17629u);
        auto            a = random.vector<int>(nx, -100, 100);
        auto            b = random.vector<int>(nx, -100, 100);

        auto user0 = make_shared_device(a);
        auto user2 = make_shared_device(b);
        auto user4 = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);
        TOC(t_hip);

        KernelArguments runtimeArgs;
        runtimeArgs.append("user0", user0.get());
        runtimeArgs.append("user1", nx);
        runtimeArgs.append("user2", nx);
        runtimeArgs.append("user3", (size_t)1);
        runtimeArgs.append("user4", user2.get());
        runtimeArgs.append("user5", nx);
        runtimeArgs.append("user6", nx);
        runtimeArgs.append("user7", (size_t)1);
        runtimeArgs.append("user8", user4.get());
        runtimeArgs.append("user9", nx);
        runtimeArgs.append("user10", (size_t)1);

        TIC(t_gpu);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        TOC(t_gpu);

        TIC(t_hip);
        ASSERT_THAT(hipMemcpy(r.data(), user4.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));
        TOC(t_hip);

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i] * (a[i] + b[i]);

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);

        TIC(t_hip);
        user0.reset();
        user2.reset();
        user4.reset();
        TOC(t_hip);

        TOC(t_total);

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    void KernelGraphTestGPU::GPU_Translate04(bool reload)
    {
        RandomGenerator random(1263u);

        size_t nx = 64;

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;
        int beta  = 33;

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 1))); // b
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar({DataType::Int32, PointerType::PointerGlobal},
                                                 2))); // alpha
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Int32, 3))); // beta

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(4, 0, 2))); // alpha * a
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(5, 1, 3))); // beta * b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(6, 4, 5))); // add above

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Linear(1, 6)));

        CommandKernel commandKernel(command, testKernelName());

        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", nx);
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        runtimeArgs.append("user2", d_alpha.get());

        runtimeArgs.append("user3", beta);

        runtimeArgs.append("user6", d_c.get());
        runtimeArgs.append("d_c_limit", nx);
        runtimeArgs.append("d_c_stride", (size_t)1);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        // launch again, using saved assembly
        auto assemblyFileName = m_context->assemblyFileName();

        if(reload)
        {
            commandKernel.loadKernelFromAssembly(assemblyFileName, testKernelName());
            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        }

        std::vector<int> r(nx), x(nx);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = alpha * a[i] + beta * b[i];

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);

        if(reload)
        {
            // load, using bad kernel name
            EXPECT_THROW(commandKernel.loadKernelFromAssembly(assemblyFileName, "Translate04_BAD"),
                         FatalError);

            // load, using non-existant file
            EXPECT_THROW(
                commandKernel.loadKernelFromAssembly(assemblyFileName + "_bad", testKernelName()),
                FatalError);

            std::filesystem::remove(assemblyFileName);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate04)
    {
        GPU_Translate04(false);
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate04LoadAssembly)
    {
        GPU_Translate04(true);
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate05)
    {
        auto command = std::make_shared<rocRoller::Command>();

        Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

        Operations::T_Store_Linear store_C(1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));

        CommandKernel commandKernel(command, "Translate05");

        size_t nx = 64;

        RandomGenerator random(135679u);
        auto            a = random.vector<int>(nx, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);

        KernelArguments runtimeArgs;
        runtimeArgs.append("d_a", d_a.get());
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);
        runtimeArgs.append("d_b", d_b.get());
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i];

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopy)
    {
        size_t nx  = 256; // tensor size x
        size_t ny  = 128; // tensor size y
        int    m   = 16; // macro tile size x
        int    n   = 4; // macro tile size y
        int    t_m = 4; // thread tile size x
        int    t_n = 2; // thread tile size y

        unsigned int workgroup_size_x = 4;
        unsigned int workgroup_size_y = 2;

        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0
                        && (size_t)m * n == t_m * t_n * workgroup_size_x * workgroup_size_y,
                    "MacroTile size mismatch");

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        RandomGenerator random(193674u);
        auto            a = random.vector<int>(nx * ny, -100, 100);
        auto            r = random.vector<int>(nx * ny, -100, 100);
        auto            x = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx * ny);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0)));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 0)));

        KernelArguments runtimeArgs;

        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("b", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_stride_0", (size_t)ny);
        runtimeArgs.append("d_b_stride_1", (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile_in
            = KernelGraph::CoordinateTransform::MacroTile(0, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_out = KernelGraph::CoordinateTransform::MacroTile(
            0, {m, n}, MemoryType::VGPR, {t_m, t_n}, true);

        params->setDimensionInfo(mac_tile_in);
        params->setDimensionInfo(mac_tile_out);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileCopy", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx * ny; ++i)
        {
            x[i] = a[i];
        }

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyLDS)
    {
        size_t nx  = 128; // tensor size x
        size_t ny  = 256; // tensor size y
        int    m   = 8; // macro tile size x
        int    n   = 16; // macro tile size y
        int    t_m = 2; // thread tile size x
        int    t_n = 8; // thread tile size y

        unsigned int workgroup_size_x = 4;
        unsigned int workgroup_size_y = 2;

        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0
                        && (size_t)m * n == t_m * t_n * workgroup_size_x * workgroup_size_y,
                    "MacroTile size mismatch");

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        RandomGenerator  random(193674u);
        auto             a = random.vector<int>(nx * ny, -100, 100);
        std::vector<int> r(nx * ny, 0);
        std::vector<int> x(nx * ny, 0);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx * ny);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0)));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 0)));

        KernelArguments runtimeArgs;

        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("b", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_stride_0", (size_t)ny);
        runtimeArgs.append("d_b_stride_1", (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile_in
            = KernelGraph::CoordinateTransform::MacroTile(0, {m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_out = KernelGraph::CoordinateTransform::MacroTile(
            0, {m, n}, MemoryType::VGPR, {t_m, t_n}, true);

        params->setDimensionInfo(mac_tile_in);
        params->setDimensionInfo(mac_tile_out);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileCopy", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx * ny; ++i)
        {
            x[i] = a[i];
        }

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileAdd)
    {
        size_t nx  = 256; // tensor size x
        size_t ny  = 512; // tensor size y
        int    m   = 8; // macro tile size x
        int    n   = 64; // macro tile size y
        int    t_m = 2; // thread tile size x
        int    t_n = 8; // thread tile size y

        uint workgroup_size_x = 4;
        uint workgroup_size_y = 8;

        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0
                        && (size_t)m * n == t_m * t_n * workgroup_size_x * workgroup_size_y,
                    "MacroTile size mismatch");

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        RandomGenerator random(129674u);
        auto            a = random.vector<int>(nx * ny, -100, 100);
        auto            b = random.vector<int>(nx * ny, -100, 100);
        auto            r = random.vector<int>(nx * ny, -100, 100);
        auto            x = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<int>(nx * ny);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 1))); // b

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(2, 0, 0))); // a + a
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(3, 1, 1))); // b + b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(4, 3, 2))); // 2a + 2b

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 4))); // c

        KernelArguments runtimeArgs;

        // tiled?
        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_size_0", (size_t)nx);
        runtimeArgs.append("d_b_size_1", (size_t)ny);
        runtimeArgs.append("d_b_stride_0", (size_t)ny);
        runtimeArgs.append("d_b_stride_1", (size_t)1);

        runtimeArgs.append("user2", d_c.get());
        runtimeArgs.append("d_c_limit", (size_t)nx * ny);
        runtimeArgs.append("d_c_stride_0", (size_t)ny);
        runtimeArgs.append("d_c_stride_1", (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        // TODO: Add a "fill" operation on the kernel graph to propagate tile sizes where appropriate
        auto mac_tile_0
            = KernelGraph::CoordinateTransform::MacroTile(0, {m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_1
            = KernelGraph::CoordinateTransform::MacroTile(1, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_2
            = KernelGraph::CoordinateTransform::MacroTile(2, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_3
            = KernelGraph::CoordinateTransform::MacroTile(3, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_4 = KernelGraph::CoordinateTransform::MacroTile(
            4, {m, n}, MemoryType::VGPR, {t_m, t_n}, true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_1);
        params->setDimensionInfo(mac_tile_2);
        params->setDimensionInfo(mac_tile_3);
        params->setDimensionInfo(mac_tile_4);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileAdd", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx * ny; ++i)
        {
            x[i] = a[i] + a[i] + b[i] + b[i];
        }

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileScale)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        // matrix size: A is MxK; B is KxN; D is MxN
        int M = 1024;
        int N = 1024;

        // output macro tile size
        int mac_m = 64;
        int mac_n = 64;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        // wave tile sizes
        int wave_m = 32;
        int wave_n = 32;
        int wave_k = 2;
        int wave_b = 1;

        uint workgroup_size_x = 256;
        uint workgroup_size_y = 1;

        // one macro tile per workgroup
        uint num_workgroup_x = M / mac_m;
        uint num_workgroup_y = N / mac_n;

        auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(61u);

        auto A = random.vector<float>(M * N, -1.f, 1.f);

        std::vector<float> B = {2.12f};

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_D = make_shared_device<float>(M * N);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar({DataType::Float, PointerType::PointerGlobal},
                                                 1))); // B

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(2, 0, 1))); // D = B * A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 2))); // D

        KernelArguments runtimeArgs;

        // tiled?
        runtimeArgs.append("A", d_A.get());
        runtimeArgs.append("d_a_limit", (size_t)M * N);
        runtimeArgs.append("d_a_size_0", (size_t)M);
        runtimeArgs.append("d_a_size_1", (size_t)N);
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)M);

        runtimeArgs.append("B", d_B.get());

        runtimeArgs.append("D", d_D.get());
        runtimeArgs.append("d_d_limit", (size_t)M * N);
        runtimeArgs.append("d_d_stride_0", (size_t)1);
        runtimeArgs.append("d_d_stride_1", (size_t)M);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile_0 = KernelGraph::CoordinateTransform::MacroTile(
            0, {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_2
            = KernelGraph::CoordinateTransform::MacroTile(2,
                                                          {mac_m, mac_n},
                                                          LayoutType::MATRIX_ACCUMULATOR,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_2);

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, -1, four, one, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 0, two, nullptr, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 1, two, nullptr, true));

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "BA", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        std::vector<float> D(M * N, 0.f);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<float> c_D(M * N, 0.f);
        for(size_t i = 0; i < c_D.size(); ++i)
            c_D[i] = B[0] * A[i];

        double rnorm = relativeNorm(D, c_D);
        ASSERT_LT(rnorm, 2.e-6);
    }

    TEST_F(KernelGraphTest, CleanExpression)
    {
        VariableType doubleVal{DataType::Double, PointerType::Value};
        auto         command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));
        auto b = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        m_context->kernel()->addCommandArguments(command->getArguments());

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        auto clean_expr = rocRoller::KernelGraph::cleanArguments(expr2, m_context->kernel());

        EXPECT_EQ(Expression::toString(clean_expr),
                  "Multiply(user_Int32_Value_1, Add(user_Int32_Value_0, user_Int32_Value_1))");
    }

    TEST_F(KernelGraphTest, CleanArguments)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto kgraph = KernelGraph::translate2(command);
        kgraph      = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        auto dot = kgraph.toDOT();
        EXPECT_THAT(dot, Not(HasSubstr("SubDimension{0, CommandArgument(Load_Linear_0_size_0)}")));
        EXPECT_THAT(dot, Not(HasSubstr("SubDimension{0, CommandArgument(Load_Linear_2_size_0)}")));
        EXPECT_THAT(
            dot, Not(HasSubstr("SubDimension{0, Linear{CommandArgument(Load_Linear_0_size_0)}")));
        EXPECT_THAT(
            dot, Not(HasSubstr("SubDimension{0, Linear{CommandArgument(Load_Linear_2_size_0)}")));

        EXPECT_THAT(dot, HasSubstr("SubDimension{0, Load_Linear_0_size_0}"));
        EXPECT_THAT(dot, HasSubstr("SubDimension{0, Load_Linear_2_size_0}"));
        EXPECT_THAT(dot, HasSubstr("Linear{Load_Linear_0_size_0}"));
        EXPECT_THAT(dot, HasSubstr("Linear{Load_Linear_2_size_0}"));
    }

}
