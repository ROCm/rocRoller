/**
 * Test suite utilites.
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>

#include <hip/amd_detail/amd_hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>

template <typename T>
std::shared_ptr<T> make_shared_device(std::size_t n = 1, T init = {})
{
    std::size_t size   = n * sizeof(T);
    T*          ptr    = nullptr;
    auto        result = hipMalloc(&ptr, size);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    result = hipMemset(ptr, init, size);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    return std::shared_ptr<T>(ptr, hipFree);
}

template <typename T>
std::shared_ptr<T> make_shared_device(const std::vector<T>& init, size_t padding = 0)
{
    std::size_t size   = init.size() * sizeof(T);
    T*          ptr    = nullptr;
    auto        result = hipMalloc(&ptr, size + padding);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    result = hipMemcpy(ptr, init.data(), size, hipMemcpyDefault);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    return std::shared_ptr<T>(ptr, hipFree);
}

template <typename T>
double normL2(std::vector<T> a)
{
    double r = 0.0;

#pragma omp parallel for reduction(+ : r)
    for(int i = 0; i < a.size(); ++i)
    {
        double t = double(a[i]);
        r        = r + t * t;
    }
    return std::sqrt(r);
}

template <typename T>
double relativeNormL2(std::vector<T> a, std::vector<T> b)
{
    double d = 0.0;
    double r = 0.0;

#pragma omp parallel for reduction(+ : d, r)
    for(size_t i = 0; i < a.size(); ++i)
    {
        double td = double(a[i] - b[i]);
        double t  = double(b[i]);
        d         = d + td * td;
        r         = r + t * t;
    }

    return std::sqrt(d / r);
}

template <typename T>
double normInf(std::vector<T> a)
{
    double r = 0.0;

#pragma omp parallel for reduction(max : r)
    for(int i = 0; i < a.size(); ++i)
    {
        double t = fabs(double(a[i]));
        if(t > r)
            r = t;
    }
    return r;
}

template <typename T>
double relativeNormInf(std::vector<T> a, std::vector<T> b)
{
    double d = 0.0;
    double r = 0.0;

#pragma omp parallel for reduction(max : d, r)
    for(size_t i = 0; i < a.size(); ++i)
    {
        double td = fabs(double(a[i] - b[i]));
        double t  = fabs(double(b[i]));

        if(td > d)
            d = td;
        if(t > r)
            r = t;
    }

    return d / r;
}

struct AcceptableError
{
    double      relativeL2Tolerance;
    std::string reasoning;
};

struct ComparisonResult
{
    bool ok;

    double relativeNormL2, relativeNormInf;
    double referenceNormL2, referenceNormInf;
    double xNormL2, xNormInf;

    AcceptableError acceptableError;

    std::string message() const
    {
        std::stringstream ss;
        ss << (ok ? "Comparison PASSED." : "Comparison FAILED.");
        ss << "  Relative norms:";
        ss << " L2 " << std::scientific << relativeNormL2;
        ss << " Inf " << std::scientific << relativeNormInf;
        ss << "  Norms (x/ref):";
        ss << " L2 " << std::scientific << xNormL2 << "/" << referenceNormL2;
        ss << " Inf " << std::scientific << xNormInf << "/" << referenceNormInf;
        ss << "  Tolerance: " << std::scientific << acceptableError.relativeL2Tolerance;
        ss << " (" << acceptableError.reasoning << ")";
        return ss.str();
    }
};

/**
 * Return expected machine epsilon for `T`.
 */
template <typename T>
double epsilon()
{
    using namespace rocRoller;

    double rv = 0.0;
    if constexpr(std::is_same_v<T, __half>)
        rv = std::pow(2.0, -10);
    else if constexpr(std::is_same_v<T, FP8>)
        rv = std::pow(2.0, -3);
    else if constexpr(std::is_same_v<T, BF8>)
        rv = std::pow(2.0, -2);
    else
        rv = std::numeric_limits<T>::epsilon();

    AssertFatal(rv != 0.0, "Unknown data type.");

    return rv;
}

/**
 * Return acceptable error for GEMM problems.
 *
 * Currently scales epsilon with the square-root of `K`.
 *
 * This assumes that the routines that compute various norms used for
 * comparison do not accumulate a significant error themselves (if
 * they did, we would want to include `M` and `N` in the scaling).
 */
template <typename TA, typename TB, typename TD>
AcceptableError
    gemmAcceptableError(int M, int N, int K, rocRoller::GPUArchitectureTarget const& arch)
{
    double eps       = epsilon<TD>();
    double tolerance = 0.0;

    std::stringstream ss;
    ss << "Output espilon: " << std::scientific << eps;

    if constexpr(std::is_same_v<TD, rocRoller::FP8> || std::is_same_v<TD, rocRoller::BF8>)
    {
        tolerance = eps;
        ss << " Error expected to be dominated by conversion.";
    }
    else
    {
        double scale = std::sqrt(K);
        double fudge = 5;

        ss << " K: " << K;
        ss << " Problem size scaling: " << scale;
        ss << " Fudge: " << fudge;

        tolerance = fudge * epsilon<TD>() * std::sqrt(K);
    }

    return {tolerance, ss.str()};
}

/**
 * @brief Compare `x` to a reference `r`.
 *
 * The boolean `ok` field of the return value is true if the relative
 * L2 norm between `x` and `r` is less than `scale` * `epsilon`.
 *
 * Various norms are computed and included in the return value.
 */
template <typename T>
ComparisonResult compare(std::vector<T> const&  x,
                         std::vector<T> const&  r,
                         AcceptableError const& acceptableError)
{
    ComparisonResult rv;

    rv.acceptableError  = acceptableError;
    rv.referenceNormL2  = normL2(r);
    rv.referenceNormInf = normInf(r);
    rv.xNormL2          = normL2(x);
    rv.xNormInf         = normInf(x);
    rv.relativeNormL2   = relativeNormL2(x, r);
    rv.relativeNormInf  = relativeNormInf(x, r);

    rv.ok = rv.relativeNormL2 < rv.acceptableError.relativeL2Tolerance;

    return rv;
}

int countSubstring(const std::string& str, const std::string& sub);

/*
 * Matrix multiply.
 */

namespace rocRoller
{
    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<float>& A,
               const std::vector<float>& B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA = false,
               bool                      transB = true);

    void CPUMM(std::vector<__half>&       D,
               const std::vector<__half>& C,
               const std::vector<__half>& A,
               const std::vector<__half>& B,
               int                        M,
               int                        N,
               int                        K,
               float                      alpha,
               float                      beta,
               bool                       transA = false,
               bool                       transB = true);

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<FP8>&   A,
               const std::vector<FP8>&   B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA = false,
               bool                      transB = true);

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<BF8>&   A,
               const std::vector<BF8>&   B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA = false,
               bool                      transB = true);
}

/*
 * Random vector generator.
 */

namespace rocRoller
{
    template <typename TA, typename TB, typename TC>
    void GenerateRandomInput(std::mt19937::result_type seed,
                             std::vector<TA>&          A,
                             size_t                    sizeA,
                             std::vector<TB>&          B,
                             size_t                    sizeB,
                             std::vector<TC>&          C,
                             size_t                    sizeC)
    {
        auto rngA = RandomGenerator(seed + 1);
        auto rngB = RandomGenerator(seed + 2);
        auto rngC = RandomGenerator(seed + 3);

#pragma omp parallel sections
        {
#pragma omp section
            {
                A = rngA.vector<TA>(sizeA, -1.f, 1.f);
            }

#pragma omp section
            {
                B = rngB.vector<TB>(sizeB, -1.f, 1.f);
            }

#pragma omp section
            {
                C = rngC.vector<TC>(sizeC, -1.f, 1.f);
            }
        }
    }

    template <typename T>
    void SetIdentityMatrix(std::vector<T>& mat, size_t cols, size_t rows)
    {
        for(size_t i = 0; i < cols; i++)
            for(size_t j = 0; j < rows; j++)
                mat[i + j * cols] = i == j ? static_cast<T>(1.0) : static_cast<T>(0.0);
    }

    template <>
    inline void SetIdentityMatrix(std::vector<FP4x8>& mat, size_t cols, size_t rows)
    {
        std::fill(mat.begin(), mat.end(), FP4x8()); // zero out the matrix

        // Notice `cols` and `rows` are NOT the actual dimensions of `mat`,
        // they are the dimensions before packed into FP4x8.
        size_t const row_bytes = 4 * cols / 8; // number of bytes in a row
        uint8_t      even      = 0b00100000;
        uint8_t      odd       = 0b00000010;

        //  Generate FP4 identity matrix with bit pattern like this:
        //    0010 0000 0000 0000
        //    0000 0010 0000 0000
        //    0000 0000 0010 0000
        //    0000 0000 0000 0010
        //    ...
        for(size_t i = 0; i < std::min(rows, cols); i += 2)
            std::memcpy(
                reinterpret_cast<uint8_t*>(mat.data()) + (row_bytes * i) + (4 * i / 8), &even, 1);
        for(size_t i = 1; i < std::min(rows, cols); i += 2)
            std::memcpy(
                reinterpret_cast<uint8_t*>(mat.data()) + (row_bytes * i) + (4 * i / 8), &odd, 1);
    }

    template <typename F6x16>
    requires(
        std::is_same_v<
            F6x16,
            FP6x16> || std::is_same_v<F6x16, BF6x16>) inline void SetIdentityMatrix(std::
                                                                                        vector<
                                                                                            F6x16>&
                                                                                            mat,
                                                                                    size_t  cols,
                                                                                    size_t  rows)
    {
        std::fill(mat.begin(), mat.end(), F6x16()); // zero out the matrix

        // Notice `cols` and `rows` are NOT the actual dimensions of `mat`,
        // they are the dimensions before packed into F6x16.
        size_t const row_bytes = 6 * cols / 8; // number of bytes in a row
        // clang-format off
        auto constexpr patterns = [] {
        if constexpr (std::is_same_v<F6x16, FP6x16>)
            return std::to_array<uint8_t> ({  //  Bit pattern of FP6 identity matrix:
                0b00100000,                   //    001000 000000 000000 000000 000000
                0b10000000,                   //    000000 001000 000000 000000 000000
                0b00000010,                   //    000000 000000 001000 000000 000000
                0b00001000,                   //    000000 000000 000000 001000 000000
            });                               //    000000 000000 000000 000000 001000 (repeat 1st row)
        else
            return std::to_array<uint8_t> ({  //  Bit pattern of BF6 identity matrix:
                0b00110000,                   //    001100 000000 000000 000000 000000
                0b11000000,                   //    000000 001100 000000 000000 000000
                0b00000011,                   //    000000 000000 001100 000000 000000
                0b00001100,                   //    000000 000000 000000 001100 000000
             });                              //    000000 000000 000000 000000 001100 (repeat 1st row)
        }();
        // clang-format on
        std::array constexpr shift = {0, 1, 0, 0};

        for(size_t i = 0; i < std::min(rows, cols); i++)
        {
            int byte_offset = (i * 6) / 8 + shift[i % 4];
            std::memcpy(reinterpret_cast<uint8_t*>(mat.data()) + (row_bytes * i) + byte_offset,
                        &patterns[i % 4],
                        1);
        }
    }
}
