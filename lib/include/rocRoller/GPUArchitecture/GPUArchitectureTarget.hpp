
#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{

    /**
     * @brief Represents GFX architecture version IDs
     *
     * NOTE: When adding new arches, be sure to
     * - Add the enum
     * - Update the `toString()` method
     * - Update the `name()` method
     * - Update relevant filters like `isGFX9X()`
     * - Double check Observers and other places the filters might be used
     *
     */
    enum class GPUArchitectureGFX : int32_t
    {
        UNKNOWN = 0,
        GFX908,
        GFX90A,
        GFX940,
        GFX941,
        GFX942,
        GFX950,
        GFX1010,
        GFX1011,
        GFX1012,
        GFX1030,
        GFX1200,
        GFX1201,

        Count,
    };
    std::string toString(GPUArchitectureGFX const& gfx);
    std::string name(GPUArchitectureGFX const& gfx);

    struct GPUArchitectureFeatures
    {
    public:
        bool sramecc = false;
        bool xnack   = false;

        std::string toString() const;

        // Return a string of features that can be provided as input to the LLVM Assembler.
        // These should have the ON/OFF symbol in front of each feature, and be comma
        // delimmited.
        std::string toLLVMString() const;

        auto operator<=>(const GPUArchitectureFeatures&) const = default;
    };

    struct GPUArchitectureTarget
    {
    public:
        GPUArchitectureGFX      gfx      = GPUArchitectureGFX::UNKNOWN;
        GPUArchitectureFeatures features = {};

        static GPUArchitectureTarget fromString(std::string const& archStr);
        std::string                  toString() const;

        std::string name() const;

        constexpr bool isCDNA1GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX908;
        }

        constexpr bool isCDNA2GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX90A;
        }

        constexpr bool isCDNA3GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX940 || gfx == GPUArchitectureGFX::GFX941
                   || gfx == GPUArchitectureGFX::GFX942;
        }

        constexpr bool isCDNA35GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX950;
        }

        constexpr bool isRDNA1GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX1012;
        }

        constexpr bool isRDNA2GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX1030;
        }

        constexpr bool isRDNA3GPU() const
        {
            return false;
        }

        constexpr bool isRDNA4GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX1200 || gfx == GPUArchitectureGFX::GFX1201;
        }

        constexpr bool isRDNAGPU() const
        {
            return isRDNA1GPU() || isRDNA2GPU() || isRDNA3GPU() || isRDNA4GPU();
        }

        constexpr bool isCDNAGPU() const
        {
            return isCDNA1GPU() || isCDNA2GPU() || isCDNA3GPU() || isCDNA35GPU();
        }

        auto operator<=>(const GPUArchitectureTarget&) const = default;

    private:
        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;
    };

    inline std::ostream& operator<<(std::ostream& os, GPUArchitectureTarget const& input)
    {
        os << input.toString();
        return os;
    }

    inline std::istream& operator>>(std::istream& is, GPUArchitectureTarget& input)
    {
        std::string recvd;
        is >> recvd;
        input = GPUArchitectureTarget::fromString(recvd);
        return is;
    }

    inline std::string toString(GPUArchitectureFeatures const& feat)
    {
        return feat.toString();
    }

    inline std::string toString(GPUArchitectureTarget const& target)
    {
        return target.toString();
    }

    inline std::string name(GPUArchitectureTarget const& target)
    {
        return target.name();
    }

    constexpr std::array<rocRoller::GPUArchitectureTarget, 18> SupportedArchitectures
        = {GPUArchitectureTarget{GPUArchitectureGFX::GFX908},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX908, {.xnack = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX908, {.sramecc = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX90A},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX90A, {.sramecc = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX940},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX941},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX941, {.sramecc = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX942},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX942, {.sramecc = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX1012},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX1012, {.xnack = true}},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX1030},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX1200},
           GPUArchitectureTarget{GPUArchitectureGFX::GFX1201}};
}

#include "GPUArchitectureTarget_impl.hpp"
