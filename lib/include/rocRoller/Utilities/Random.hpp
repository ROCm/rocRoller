#pragma once

#include <cmath>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>

#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>

/*
 * Random vector generator.
 */

namespace rocRoller
{
    /**
     * Random vector generator.
     *
     * A seed must be passed to the constructor.  If the environment
     * variable specified by `ROCROLLER_RANDOM_SEED` is present, it
     * supercedes the seed passed to the constructor.
     *
     * A seed may be set programmatically (at any time) by calling
     * seed().
     */
    class RandomGenerator
    {
    public:
        RandomGenerator(int seedNumber);

        /**
         * Set a new seed.
         */
        void seed(int seedNumber);

        /**
         * Generate a random vector of length `nx`, with values
         * between `min` and `max`.
         */
        template <typename T, typename R>
        std::vector<T> vector(uint nx, R min, R max);

        template <std::integral T>
        T next(T min, T max);

    private:
        std::mt19937 m_gen;
    };
}

#include "Random_impl.hpp"
