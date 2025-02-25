#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"

#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <chrono>
#include <thread>

using namespace rocRoller;

class TimerTest : public GenericContextFixture
{
};

class TimerTestGPU : public CurrentGPUContextFixture
{
    void SetUp() override
    {
        CurrentGPUContextFixture::SetUp();
    }
};

void timed_sleep()
{
    auto timer = Timer("rocRoller::TimerTest::timed_sleep");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void untimed_sleep()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

TEST_F(TimerTest, Base01)
{
    auto t = Timer("rocRoller::TimerTest.Base01");

    t.tic();
    timed_sleep();
    untimed_sleep();
    t.toc();

    auto t1 = TimerPool::milliseconds("rocRoller::TimerTest::timed_sleep");
    auto t2 = TimerPool::milliseconds("rocRoller::TimerTest.Base01");

    EXPECT_EQ(t.milliseconds(), t2);
    EXPECT_GT(t2, t1);
    EXPECT_GE(t1, 10);
    EXPECT_GE(t2, 20);
}

TEST_F(TimerTest, Destructor)
{
    TimerPool::clear();

    size_t t1, t2;
    {
        auto t = Timer("rocRoller::TimerTest.Destructor");

        t.tic();
        untimed_sleep();
        t.toc();

        t1 = t.milliseconds();
        t2 = TimerPool::milliseconds("rocRoller::TimerTest.Destructor");
        EXPECT_EQ(t1, t2);
    }
    std::cout << t1;
    EXPECT_EQ(t1, TimerPool::milliseconds("rocRoller::TimerTest.Destructor"));
}

TEST_F(TimerTestGPU, GPU_Destructor)
{
    TimerPool::clear();

    size_t t1, t2;
    {
        auto t = HIPTimer("rocRoller::TimerTestGPU.Destructor");

        t.tic();
        untimed_sleep();
        t.toc();
        t.sync();

        t1 = t.milliseconds();
        t2 = TimerPool::milliseconds("rocRoller::TimerTestGPU.Destructor");
        EXPECT_EQ(t1, t2);
    }
    std::cout << t1;
    EXPECT_EQ(t1, TimerPool::milliseconds("rocRoller::TimerTestGPU.Destructor"));
}
