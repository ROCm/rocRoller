
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "GenericContextFixture.hpp"
#include "Utilities.hpp"
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>

class RandomTest : public GenericContextFixture
{
};

using namespace rocRoller;

TEST_F(RandomTest, Seed)
{
    auto settings = Settings::getInstance();

    auto seed1 = 12867;
    auto size  = 64;
    auto min   = -10;
    auto max   = 10;

    settings->set(Settings::RandomSeed, seed1);

    // Deterministic seed
    auto random1 = RandomGenerator();
    auto x       = random1.vector<int>(size, min, max);

    EXPECT_EQ(x[0], 0);
    EXPECT_EQ(x[63], 6);

    // Generating again gives something different
    auto x2 = random1.vector<int>(size, min, max);

    EXPECT_NE(x, x2);

    // Re-seeding gives the same vector
    auto random2 = RandomGenerator();
    auto y       = random2.vector<int>(size, min, max);

    EXPECT_EQ(x, y);

    // Re-seeded generates same sequences
    auto y2 = random2.vector<int>(size, min, max);

    EXPECT_EQ(x2, y2);

    // Test explicit constructor that does not use Settings class
    auto seed2   = 1123;
    auto random3 = RandomGenerator(seed2);
    auto z       = random3.vector<int>(29, -10, 10);

    std::vector<int> Z{8, -7, -3, -3, -2, 3,  1, 6, 6, -2, 3,  -1, 1, 2, -1,
                       6, 2,  -9, -1, 1,  -5, 1, 2, 6, 8,  -2, -3, 7, 5};

    EXPECT_EQ(z, Z);

    // Re-seeding works again
    random3.seed(seed1);
    auto w = random3.vector<int>(size, min, max);

    EXPECT_EQ(x, w);

    // Re-seeded sequence is the same
    auto w2 = random3.vector<int>(size, min, max);

    EXPECT_EQ(x2, w2);

    // Different seeds produce different sequence of vectors
    random1.seed(1729u);
    random2.seed(1730u);
    for(int i = 0; i < 100; ++i)
    {
        auto a = random1.vector<int>(size, min, max);
        auto b = random2.vector<int>(size, min, max);
        EXPECT_NE(a, b);
    }
}
