

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "SourceMatcher.hpp"

#include <rocRoller/Operations/Command.hpp>

using namespace rocRoller;

TEST(CommandTest, Basic)
{
    auto command = std::make_shared<rocRoller::Command>();

    EXPECT_EQ(0, command->operations().size());

    auto tagTensor = command->addOperation(Operations::Tensor(1, DataType::Int32));

    auto load_linear
        = std::make_shared<Operations::Operation>(Operations::T_Load_Linear(tagTensor));
    auto tagLoad = command->addOperation(load_linear);

    Operations::T_Store_Linear tsl(tagLoad, tagTensor);
    // tag value is assigned to an operation when it's added to the Command Graph
    EXPECT_EQ(-1, tsl.getTag());
    auto store_linear = std::make_shared<Operations::Operation>(std::move(tsl));
    auto execute
        = std::make_shared<Operations::Operation>(Operations::T_Execute(command->getNextTag()));

    command->addOperation(store_linear);
    command->addOperation(execute);

    EXPECT_EQ(load_linear, command->findTag(1));

    EXPECT_EQ(4, command->operations().size());
}

TEST(CommandTest, ToString)
{
    auto command = std::make_shared<rocRoller::Command>();

    EXPECT_EQ(0, command->operations().size());

    command->addOperation(std::make_shared<Operations::Operation>(Operations::T_Load_Linear(-1)));
    command->addOperation(
        std::make_shared<Operations::Operation>(Operations::T_Execute(command->getNextTag())));

    EXPECT_EQ(2, command->operations().size());

    std::ostringstream msg;
    msg << *command;

    EXPECT_THAT(msg.str(), ::testing::HasSubstr("T_LOAD_LINEAR"));
    EXPECT_THAT(msg.str(), ::testing::HasSubstr("T_EXECUTE"));

    EXPECT_EQ(msg.str(), command->toString());
}

TEST(CommandTest, VectorAdd)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto tagTensorA = command->addOperation(Operations::Tensor(1, DataType::Float));
    auto tagLoadA   = command->addOperation(Operations::T_Load_Linear(tagTensorA));

    auto tagTensorB = command->addOperation(Operations::Tensor(1, DataType::Float));
    auto tagLoadB   = command->addOperation(Operations::T_Load_Linear(tagTensorB));

    Operations::T_Execute execute(command->getNextTag());
    auto                  tagResult = execute.addXOp(Operations::E_Add(tagLoadA, tagLoadB));
    command->addOperation(std::move(execute));

    auto tagTensorResult = command->addOperation(Operations::Tensor(1, DataType::Float));
    command->addOperation(Operations::T_Store_Linear(tagResult, tagTensorResult));

    std::string result = R"(
        Tensor.Float.d1 0, (base=&0, lim=&8, sizes={&16 }, strides={&24 })
        T_LOAD_LINEAR 1 Tensor 0
        Tensor.Float.d1 2, (base=&32, lim=&40, sizes={&48 }, strides={&56 })
        T_LOAD_LINEAR 3 Tensor 2
        T_EXECUTE 1 3
        E_Add 4, 1, 3
        Tensor.Float.d1 6, (base=&64, lim=&72, sizes={&80 }, strides={&88 })
        T_STORE_LINEAR 7 Source 4 Tensor 6)";
    EXPECT_EQ(NormalizedSource(command->toString()), NormalizedSource(result));

    {
        std::string        expected = R"([
            Tensor_0_pointer: PointerGlobal: Float(offset: 0, size: 8, read_write),
            Tensor_0_extent: Value: Int64(offset: 8, size: 8, read_only),
            Tensor_0_size_0: Value: Int64(offset: 16, size: 8, read_only),
            Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only),
            Tensor_2_pointer: PointerGlobal: Float(offset: 32, size: 8, read_write),
            Tensor_2_extent: Value: Int64(offset: 40, size: 8, read_only),
            Tensor_2_size_0: Value: Int64(offset: 48, size: 8, read_only),
            Tensor_2_stride_0: Value: Int64(offset: 56, size: 8, read_only),
            Tensor_6_pointer: PointerGlobal: Float(offset: 64, size: 8, read_write),
            Tensor_6_extent: Value: Int64(offset: 72, size: 8, read_only),
            Tensor_6_size_0: Value: Int64(offset: 80, size: 8, read_only),
            Tensor_6_stride_0: Value: Int64(offset: 88, size: 8, read_only)
        ])";
        std::ostringstream msg;
        msg << command->getArguments();
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(msg.str()));
    }
}

TEST(CommandTest, DuplicateOp)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto execute
        = std::make_shared<Operations::Operation>(Operations::T_Execute(command->getNextTag()));

    command->addOperation(execute);
#ifdef NDEBUG
    GTEST_SKIP() << "Skipping assertion check in release mode.";
#endif

    EXPECT_THROW({ command->addOperation(execute); }, FatalError);
}

TEST(CommandTest, XopInputOutputs)
{
    auto command = std::make_shared<rocRoller::Command>();
    command->allocateTag();
    command->allocateTag();
    Operations::T_Execute execute(command->getNextTag());
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Sub(0, 1)));
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(2, 1)));

    EXPECT_EQ(execute.getInputs(), std::unordered_set<int>({0, 1}));
    EXPECT_EQ(execute.getOutputs(), std::unordered_set<int>({2, 3}));

    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Abs(3)));
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Neg(4)));

    EXPECT_EQ(execute.getInputs(), std::unordered_set<int>({0, 1}));
    EXPECT_EQ(execute.getOutputs(), std::unordered_set<int>({2, 3, 4, 5}));
}
