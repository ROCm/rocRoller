#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ScaledMatrixMultiply.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/T_Execute.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "DataTypes/DataTypes.hpp"
#include "GPUContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TensorDescriptor.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace MatrixMultiplyTest
{
    template <typename T>
    concept isF8 = std::is_same_v<T, FP8> || std::is_same_v<T, BF8>;

    template <typename T>
    concept isF6F4 = std::is_same_v<T, FP6> || std::is_same_v<T, BF6> || std::is_same_v<T, FP4>;

    template <typename T>
    concept isF16 = std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>;

    /**
     * @brief Return a reasonable random value range for datatype T.
     *
     * The return value is usually passed to the random generator to
     * obtain values in (-range, range), and these will be used to
     * populate matrices for (small) GEMM problems.
     *
     * The value returned *may or may not* correspond to the maximum
     * representable value of T.
     */
    template <typename T>
    float range()
    {
        // Not the maximum range.
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, Half>)
            return 10.f;
        // Maximum range
        if constexpr(std::is_same_v<T, FP8>)
            return 448.f;
        // Maximum range; kinda extreme
        if constexpr(std::is_same_v<T, BF8>)
            return 57344.f;
        // Maximum range
        if constexpr(std::is_same_v<T, FP6>)
            return 7.5f;
        // Maximum range
        if constexpr(std::is_same_v<T, BF6>)
            return 28.f;
        // FP4, maximum range
        return 6.f;
    }

    template <typename... Ts>
    class BaseMatrixMultiplyContextFixture
        : public BaseGPUContextFixture,
          public ::testing::WithParamInterface<std::tuple<rocRoller::GPUArchitectureTarget, Ts...>>
    {
    protected:
        virtual rocRoller::ContextPtr createContext() override
        {
            GPUArchitectureTarget device = std::get<0>(this->GetParam());

            return this->createContextForArch(device);
        }

    public:
        CommandKernelPtr commandKernel;

        template <typename TA, typename TB, typename ACC>
        void matrixMultiplyMacroTile(int         wave_m,
                                     int         wave_n,
                                     int         wave_k,
                                     int         wave_b,
                                     double      acceptableError,
                                     bool        useLDSB = true,
                                     std::string transA  = "N",
                                     std::string transB  = "N",
                                     bool        scaleA  = false,
                                     bool        scaleB  = false)
        {
            commandKernel = nullptr;

            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
            if constexpr(isF8<TA> || isF8<TB>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
            }
            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            auto dataTypeA = TypeInfo<TA>::Var.dataType;
            auto dataTypeB = TypeInfo<TB>::Var.dataType;
            auto dataTypeD = TypeInfo<ACC>::Var.dataType;

            // matrix size: A is MxK; B is KxN; D is MxN
            int mac_m = wave_m;
            int mac_n = wave_n;
            int mac_k = 32;

            unsigned M = mac_m;
            unsigned N = mac_n;
            unsigned K = 32;

            if constexpr(isF8<TA> && isF8<TB>)
            {
                mac_k = 2 * wave_k;
                K     = 2 * mac_k;
            }
            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                mac_k = 2 * wave_k;
                K     = 4 * mac_k;
            }

            if constexpr(isF16<TA> || isF16<TB>)
            {
                mac_k = 4 * wave_k;
                K     = 8 * mac_k;
            }

            Log::debug("MatrixMultiplyMacroTile: Matrix {}x{}x{}", M, N, K);
            Log::debug("MatrixMultiplyMacroTile: WGTile {}x{}x{}", mac_m, mac_n, mac_k);
            Log::debug(
                "MatrixMultiplyMacroTile: MFMA   {}x{}x{}x{}", wave_m, wave_n, wave_k, wave_b);

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");
            AssertFatal(K % mac_k == 0, "MacroTile size mismatch (K)");

            AssertFatal(mac_m == wave_m, "Single output MacroTile.");
            AssertFatal(mac_n == wave_n, "Single output MacroTile.");

            uint workgroup_size_x = 64;
            uint workgroup_size_y = 1;

            auto bpe = CeilDivide(DataTypeInfo::Get(dataTypeA).elementBits, 8u);
            AssertFatal(mac_m * mac_k * bpe > wave_m * wave_k, "Not enough elements.");

            auto NX = std::make_shared<Expression::Expression>(workgroup_size_x);
            auto NY = std::make_shared<Expression::Expression>(workgroup_size_y);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            std::vector<size_t> unitStridesN = {1, 0};
            std::vector<size_t> unitStridesT = {0, 1};

            auto command    = std::make_shared<Command>();
            auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeA, transA == "N" ? unitStridesN : unitStridesT));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeB, transB == "N" ? unitStridesN : unitStridesT));
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            std::optional<rocRoller::Operations::OperationTag> tagTensorScaleA, tagLoadScaleA,
                tagTensorScaleB, tagLoadScaleB;

            if(scaleA)
            {
                tagTensorScaleA = command->addOperation(rocRoller::Operations::Tensor(
                    2, DataType::UInt8, transA == "N" ? unitStridesN : unitStridesT));
                tagLoadScaleA
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleA));
            }

            if(scaleB)
            {
                tagTensorScaleB = command->addOperation(rocRoller::Operations::Tensor(
                    2, DataType::UInt8, transB == "N" ? unitStridesN : unitStridesT));
                tagLoadScaleB
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleB));
            }

            rocRoller::Operations::OperationTag tagStoreD;

            if(!scaleA)
            {
                ASSERT_FALSE(scaleB);

                tagStoreD = command->addOperation(
                    rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B
            }
            else
            {
                ASSERT_TRUE(scaleB);

                auto scaledA = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadA, 2, tagLoadScaleA, {1, 32}));
                auto scaledB = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadB, 2, tagLoadScaleB, {32, 1}));

                tagStoreD = command->addOperation(
                    rocRoller::Operations::T_Mul(scaledA, scaledB)); // D = A * B
            }

            auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD));
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

            params->packMultipleElementsInto1VGPR = true;
            params->enableLongDwordInstructions   = true;

            params->transposeMemoryAccess[LayoutType::MATRIX_A] = transA == "T";
            params->transposeMemoryAccess[LayoutType::MATRIX_B] = transB == "T";

            // TODO: the translate step should figure out that there is a
            // T_Mul and do the right thing for the T_Load_Tiled commands
            auto macTileA
                = KernelGraph::CoordinateGraph::MacroTile({mac_m, mac_k},
                                                          LayoutType::MATRIX_A,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          MemoryType::WAVE_LDS);
            params->setDimensionInfo(tagLoadA, macTileA);

            if(scaleA)
            {
                auto macTileScaleA
                    = KernelGraph::CoordinateGraph::MacroTile({mac_m, mac_k / 32},
                                                              LayoutType::MATRIX_A,
                                                              {wave_m, wave_n, wave_k / 32, wave_b},
                                                              MemoryType::WAVE);
                params->setDimensionInfo(tagLoadScaleA.value(), macTileScaleA);
            }

            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {mac_k, mac_n},
                LayoutType::MATRIX_B,
                {wave_m, wave_n, wave_k, wave_b},
                useLDSB ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            params->setDimensionInfo(tagLoadB, macTileB);

            if(scaleB)
            {
                auto macTileScaleB
                    = KernelGraph::CoordinateGraph::MacroTile({mac_k / 32, mac_n},
                                                              LayoutType::MATRIX_B,
                                                              {wave_m, wave_n, wave_k / 32, wave_b},
                                                              MemoryType::WAVE);
                params->setDimensionInfo(tagLoadScaleB.value(), macTileScaleB);
            }

            params->setManualWavefrontCount({1u, 1u});

            commandKernel = std::make_shared<CommandKernel>(command, "MatrixMultiplyMacroTile");
            commandKernel->setContext(m_context);
            commandKernel->setCommandParameters(params);
            commandKernel->generateKernel();

            auto launch = std::make_shared<CommandLaunchParameters>();
            launch->setManualWorkitemCount({NX, NY, NZ});
            commandKernel->setLaunchParameters(launch);

            if(isLocalDevice())
            {
                RandomGenerator random(9861u);

                TensorDescriptor descA(dataTypeA, {M, K}, transA);
                TensorDescriptor descB(dataTypeB, {K, N}, transB);
                TensorDescriptor descD(dataTypeD, {M, N}, {1u, M});
                TensorDescriptor descScaleA(dataTypeA, {M, K / 32}, transA);
                TensorDescriptor descScaleB(dataTypeB, {K / 32, N}, transB);

                float rangeA = range<TA>();
                float rangeB = range<TB>();

                auto A = random.vector<TA>(descA.totalAllocatedElements(), -rangeA, rangeA);
                auto B = random.vector<TB>(descB.totalAllocatedElements(), -rangeB, rangeB);

                std::vector<uint8_t> hostScaleA, hostScaleB;

                auto d_A = make_shared_device(A);
                auto d_B = make_shared_device(B);
                auto d_D = make_shared_device<ACC>(descD.totalAllocatedElements());

                std::shared_ptr<uint8_t> d_scaleA, d_scaleB;

                if(scaleA)
                {
                    hostScaleA = random.vector<uint8_t>(
                        M * K / 32, floatToScale(0.03125f), floatToScale(1024.0f));
                    d_scaleA = make_shared_device(hostScaleA);
                }
                if(scaleB)
                {
                    hostScaleB = random.vector<uint8_t>(
                        K * N / 32, floatToScale(0.03125f), floatToScale(1024.0f));
                    d_scaleB = make_shared_device(hostScaleB);
                }

                CommandArguments commandArgs = command->createArguments();

                setCommandTensorArg(commandArgs, tagTensorA, descA, (TA*)d_A.get());
                setCommandTensorArg(commandArgs, tagTensorB, descB, (TB*)d_B.get());
                setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

                if(scaleA)
                {
                    setCommandTensorArg(
                        commandArgs, tagTensorScaleA.value(), descScaleA, d_scaleA.get());
                }
                if(scaleB)
                {
                    setCommandTensorArg(
                        commandArgs, tagTensorScaleB.value(), descScaleB, d_scaleB.get());
                }

                commandKernel->launchKernel(commandArgs.runtimeArguments());

                std::vector<ACC> D(descD.totalAllocatedElements());
                ASSERT_THAT(hipMemcpy(D.data(),
                                      d_D.get(),
                                      descD.totalAllocatedElements() * sizeof(ACC),
                                      hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<ACC> c_D(descD.totalAllocatedElements(), ACC{});
                std::vector<ACC> c_C(descD.totalAllocatedElements(), ACC{});

                float alpha = 1.0f;

                if(scaleA)
                {
                    ASSERT_TRUE(scaleB);

                    rocRoller::ScaledCPUMM(c_D,
                                           c_C,
                                           A,
                                           B,
                                           hostScaleA,
                                           hostScaleB,
                                           M,
                                           N,
                                           K,
                                           alpha,
                                           0.0,
                                           transA == "T",
                                           transB == "T");
                }
                else
                {
                    ASSERT_FALSE(scaleB);
                    CPUMM(c_D, c_C, A, B, M, N, K, alpha, 0.0, transA == "T", transB == "T");
                }

                auto tol = gemmAcceptableError<TA, TB, ACC>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(D, c_D, tol);

                Log::info("RNorm is {}", res.relativeNormL2);
                ASSERT_TRUE(res.ok) << res.message();
            }
        }

        template <typename TA>
        void matrixMultiplyMacroTileMixed(rocRoller::DataType typeB,
                                          int                 m,
                                          int                 n,
                                          int                 k,
                                          int                 b,
                                          double              err,
                                          bool                useLDSB = true,
                                          std::string         transA  = "N",
                                          std::string         transB  = "N",
                                          bool                scaleA  = false,
                                          bool                scaleB  = false)
        {
            if(typeB == rocRoller::DataType::FP8)
                matrixMultiplyMacroTile<TA, FP8, float>(
                    m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeB == rocRoller::DataType::BF8)
                matrixMultiplyMacroTile<TA, BF8, float>(
                    m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeB == rocRoller::DataType::FP6)
                matrixMultiplyMacroTile<TA, FP6, float>(
                    m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeB == rocRoller::DataType::BF6)
                matrixMultiplyMacroTile<TA, BF6, float>(
                    m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeB == rocRoller::DataType::FP4)
                matrixMultiplyMacroTile<TA, FP4, float>(
                    m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else
                Throw<FatalError>("Invalid type.");
        }

        void matrixMultiplyMacroTileMixed(rocRoller::DataType typeA,
                                          rocRoller::DataType typeB,
                                          int                 m,
                                          int                 n,
                                          int                 k,
                                          int                 b,
                                          double              err,
                                          bool                useLDSB = true,
                                          std::string         transA  = "N",
                                          std::string         transB  = "N",
                                          bool                scaleA  = false,
                                          bool                scaleB  = false)
        {
            if(typeA == rocRoller::DataType::FP8)
                matrixMultiplyMacroTileMixed<FP8>(
                    typeB, m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeA == rocRoller::DataType::BF8)
                matrixMultiplyMacroTileMixed<BF8>(
                    typeB, m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeA == rocRoller::DataType::FP6)
                matrixMultiplyMacroTileMixed<FP6>(
                    typeB, m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeA == rocRoller::DataType::BF6)
                matrixMultiplyMacroTileMixed<BF6>(
                    typeB, m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else if(typeA == rocRoller::DataType::FP4)
                matrixMultiplyMacroTileMixed<FP4>(
                    typeB, m, n, k, b, err, useLDSB, transA, transB, scaleA, scaleB);
            else
                Throw<FatalError>("Invalid type.");
        }

        template <typename T, typename ACC = T>
        void matrixMultiplyAB(int    wave_m,
                              int    wave_n,
                              int    wave_k,
                              int    wave_b,
                              double acceptableError,
                              bool   useLDS = false,
                              bool   transA = false,
                              bool   transB = false)
        {
            int const M = 1024;
            int const N = 1024;
            int const K = 512;
            // matrix size: A is MxK; B is KxN; D is MxN
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
            if constexpr(std::is_same_v<T, FP8> || std::is_same_v<T, BF8>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
            }

            auto dataTypeAB = TypeInfo<T>::Var.dataType;
            auto dataTypeD  = TypeInfo<ACC>::Var.dataType;

            // output macro tile size; we will launch 2x2 waves
            int mac_m = 2 * wave_m;
            int mac_n = 2 * wave_n;
            int mac_k = 2 * wave_k;

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

            uint workgroup_size_x = 256;
            uint workgroup_size_y = 1;

            auto bpe = CeilDivide(DataTypeInfo::Get(dataTypeAB).elementBits, 8u);
            AssertFatal(mac_m * mac_k * bpe > wave_m * wave_k, "Not enough elements.");

            uint num_workgroup_x = M / mac_m;
            uint num_workgroup_y = N / mac_n;

            auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
            auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            RandomGenerator random(61u);

            auto A = random.vector<T>(M * K, -1.f, 1.f);
            auto B = random.vector<T>(K * N, -1.f, 1.f);

            auto d_A = make_shared_device(A);
            auto d_B = make_shared_device(B);
            auto d_D = make_shared_device<ACC>(M * N);

            auto command  = std::make_shared<Command>();
            auto dataType = TypeInfo<T>::Var.dataType;

            std::vector<size_t> unitStridesN = {1, 0};
            std::vector<size_t> unitStridesT = {0, 1};

            auto tagTensorA = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeAB, transA ? unitStridesT : unitStridesN));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeAB, transB ? unitStridesT : unitStridesN)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagStoreD = command->addOperation(
                rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

            auto tagTensorD
                = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD)); // D
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            CommandArguments commandArgs = command->createArguments();

            TensorDescriptor descA(dataTypeAB, {M, K}, transA ? "T" : "N");
            TensorDescriptor descB(dataTypeAB, {K, N}, transB ? "T" : "N");
            TensorDescriptor descD(dataTypeD, {M, N}, {1u, M});

            setCommandTensorArg(commandArgs, tagTensorA, descA, (T*)d_A.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, (T*)d_B.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
            // TODO: the translate step should figure out that there is a
            // T_Mul and do the right thing for the T_Load_Tiled commands
            auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                {mac_m, mac_k},
                LayoutType::MATRIX_A,
                {wave_m, wave_n, wave_k, wave_b},
                useLDS ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {mac_k, mac_n},
                LayoutType::MATRIX_B,
                {wave_m, wave_n, wave_k, wave_b},
                useLDS ? MemoryType::WAVE_LDS : MemoryType::WAVE);

            params->setDimensionInfo(tagLoadA, macTileA);
            params->setDimensionInfo(tagLoadB, macTileB);
            params->setManualWavefrontCount({2u, 2u});
            params->transposeMemoryAccess[LayoutType::MATRIX_A] = transA;
            params->transposeMemoryAccess[LayoutType::MATRIX_B] = transB;

            auto launch = std::make_shared<CommandLaunchParameters>();
            launch->setManualWorkitemCount({NX, NY, NZ});

            CommandKernel commandKernel(command, "MatrixMultiplyAB");
            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            commandKernel.setLaunchParameters(launch);
            if(isLocalDevice())
            {
                commandKernel.launchKernel(commandArgs.runtimeArguments());

                std::vector<ACC> D(M * N);
                ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(ACC), hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<ACC> c_D(M * N, ACC{});
                std::vector<ACC> c_C(M * N, ACC{});

                CPUMM(c_D, c_C, A, B, M, N, K, 1.0f, 0.0, transA, transB);

                auto tol = gemmAcceptableError<T, T, ACC>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(D, c_D, tol);

                Log::info("RNorm is {}", res.relativeNormL2);
                ASSERT_TRUE(res.ok) << res.message();
            }
        }

        template <typename T>
        void matrixMultiplyABC(
            int wave_m, int wave_n, int wave_k, int wave_b, double acceptableError)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

            // matrix size: A is MxK; B is KxN; D is MxN
            unsigned M = 1024;
            unsigned N = 1024;
            unsigned K = 512;

            // output macro tile size
            int mac_m = 64;
            int mac_n = 64;
            int mac_k = 64;

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

            uint workgroup_size_x = 256;
            uint workgroup_size_y = 1;

            uint num_workgroup_x = M / mac_m;
            uint num_workgroup_y = N / mac_n;

            auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
            auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            RandomGenerator random(61u);

            auto A = random.vector<T>(M * K, -1.f, 1.f);
            auto B = random.vector<T>(K * N, -1.f, 1.f);
            auto C = random.vector<T>(M * N, -1.f, 1.f);

            auto d_A = make_shared_device(A);
            auto d_B = make_shared_device(B);
            auto d_C = make_shared_device(C);
            auto d_D = make_shared_device<T>(M * N);

            auto command  = std::make_shared<Command>();
            auto dataType = TypeInfo<T>::Var.dataType;

            auto tagTensorA
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagTensorC
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagAB
                = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // A * B

            auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
            auto tagStoreD
                = execute.addXOp(rocRoller::Operations::E_Add(tagAB, tagLoadC)); // D = A * B + C
            command->addOperation(std::move(execute));

            auto tagTensorD
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // D
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            CommandArguments commandArgs = command->createArguments();

            TensorDescriptor descA(dataType, {M, K}, {1u, M});
            TensorDescriptor descB(dataType, {K, N}, {1u, K});
            TensorDescriptor descC(dataType, {M, N}, {1u, M});
            TensorDescriptor descD(dataType, {M, N}, {1u, M});

            setCommandTensorArg(commandArgs, tagTensorA, descA, (T*)d_A.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, (T*)d_B.get());
            setCommandTensorArg(commandArgs, tagTensorC, descC, (T*)d_C.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);

            // TODO: the translate step should figure out that there is a
            // T_Mul and do the right thing for the T_Load_Tiled commands
            auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
            auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});

            params->setDimensionInfo(tagLoadA, macTileA);
            params->setDimensionInfo(tagLoadB, macTileB);
            params->setDimensionInfo(tagLoadC, macTileC);
            params->setManualWavefrontCount({2u, 2u});
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

            CommandKernel commandKernel(command, "ABC");
            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            auto launch = std::make_shared<CommandLaunchParameters>();
            launch->setManualWorkitemCount({NX, NY, NZ});
            commandKernel.setLaunchParameters(launch);

            if(isLocalDevice())
            {
                commandKernel.launchKernel(commandArgs.runtimeArguments());

                std::vector<T> D(M * N, 0.f);
                ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<T> c_D(M * N, 0.f);
                CPUMM(c_D, C, A, B, M, N, K, 1.0, 1.0, false, false);

                auto tol = gemmAcceptableError<T, T, T>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(D, c_D, tol);

                Log::info("RNorm is {}", res.relativeNormL2);
                ASSERT_TRUE(res.ok) << res.message();
            }
        }
    };

    class MatrixMultiplyTestGPU : public BaseMatrixMultiplyContextFixture<>
    {
    };

    // Params are: AB type, K tile size, (transA, transB)
    class MatrixMultiplyTestGPUF16
        : public BaseMatrixMultiplyContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>>>
    {
    };

    class MatrixMultiplyTestGPUF8 : public BaseMatrixMultiplyContextFixture<rocRoller::DataType>
    {
    };

    // Params are: AB type, K tile size, (transA, transB)
    class MatrixMultiplyF8F6F4TestGPU
        : public BaseMatrixMultiplyContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, K tile size, (transA, transB)
    class MatrixMultiplyMixedTestGPU
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    class MatrixMultiplyTestGPUBFloat16
        : public BaseMatrixMultiplyContextFixture<std::tuple<int, int, int>>
    {
    };

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTile)
    {
        matrixMultiplyMacroTile<float, float, float>(32, 32, 2, 1, 2.e-6);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP16)
    {
        matrixMultiplyMacroTile<Half, Half, Half>(32, 32, 8, 1, 2.e-6, false);

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int expectedLocalWriteOffset = 0;
        int numLocalRead             = 0;
        int expectedLocalReadOffset  = 0;
        for(auto const& instruction : instructions)
        {
            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 64;
            }

            if(instruction.starts_with("ds_read_u16"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 4 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 4 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 64;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 128);
        EXPECT_EQ(numLocalRead, 16);
    }

    TEST_P(MatrixMultiplyTestGPUBFloat16, GPU_MatrixMultiplyMacroTile_FP32_BF16)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x8);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);

        auto [mfma_m, mfma_n, mfma_k] = std::get<std::tuple<int, int, int>>(GetParam());

        matrixMultiplyMacroTile<BFloat16, BFloat16, float>(mfma_m, mfma_n, mfma_k, 1, 2.e-6, false);
    }

    TEST_P(MatrixMultiplyTestGPUBFloat16, GPU_MatrixMultiplyMacroTile_BF16_BF16)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x8);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);

        auto [mfma_m, mfma_n, mfma_k] = std::get<std::tuple<int, int, int>>(GetParam());

        matrixMultiplyMacroTile<BFloat16, BFloat16, BFloat16>(
            mfma_m, mfma_n, mfma_k, 1, 2.e-6, false);
    }

    TEST_P(MatrixMultiplyTestGPUF16, GPU_MatrixMultiplyMacroTileF16)
    {
        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 32) ? 16 : 32;
        uint const waveN = (MFMAK == 32) ? 16 : 32;
        uint const waveK = MFMAK;

        auto const transA = transOp.first;
        auto const transB = transOp.second;

        auto typeStr = "f16";
        switch(typeAB)
        {
        case DataType::Half:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_16x16x32_f16);
            }
            else
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_32x32x16_f16);
            }
            matrixMultiplyMacroTile<Half, Half, float>(
                waveM, waveN, waveK, 1, 2.e-6, true, transA, transB);
            break;
        case DataType::BFloat16:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_16x16x32_bf16);
            }
            else
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_32x32x16_bf16);
            }
            matrixMultiplyMacroTile<BFloat16, BFloat16, float>(
                waveM, waveN, waveK, 1, 2.e-6, true, transA, transB);
            typeStr = "bf16";
            break;
        default:
            Throw<FatalError>(std::format("Unexpected data type: {}. (Allowed: Half and Bfloat16)",
                                          toString(typeAB)));
        }

        std::string generatedCode = m_context->instructions()->toString();

        uint const elementBits = DataTypeInfo::Get(typeAB).elementBits;

        std::string const mfmaMnemonic{
            std::format("v_mfma_f32_{}x{}x{}_{}", waveM, waveN, waveK, typeStr)};
        std::string const trLoadMnemonic{transposeLoadMnemonic(elementBits)};

        uint const numMFMAs            = 4;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(transA == "N")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB == "T")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_16x16x32_NN)
    {
        if(m_context->targetArchitecture().target().isCDNA3GPU())
        {
            GTEST_SKIP() << "FIXME: Skipping test for gfx94X";
        }

        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, FP8, float>(16, 16, 32, 1, 7.5e-6, false, "N", "N");
        else
            matrixMultiplyMacroTile<BF8, BF8, float>(16, 16, 32, 1, 7.5e-6, false, "N", "N");

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int               expectedLocalWriteOffset = 0;
        int               numLocalRead             = 0;
        int               expectedLocalReadOffset  = 0;
        int               numMFMA                  = 0;
        std::string const mfma_pattern
            = isFP8 ? "v_mfma_f32_16x16x32_fp8_fp8" : "v_mfma_f32_16x16x32_bf8_bf8";
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with(mfma_pattern))
                numMFMA++;

            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 1024;
            }

            if(instruction.starts_with("ds_read_u8"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 8 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 8 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 16;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 1024);
        EXPECT_EQ(numLocalRead, 16);
        EXPECT_EQ(numMFMA, 2);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_32x32x16_NN)
    {
        if(m_context->targetArchitecture().target().isCDNA3GPU())
        {
            GTEST_SKIP() << "FIXME: Skipping test for gfx94X";
        }

        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, FP8, float>(32, 32, 16, 1, 7.5e-6, false, "N", "N");
        else
            matrixMultiplyMacroTile<BF8, BF8, float>(32, 32, 16, 1, 7.5e-6, false, "N", "N");

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int               expectedLocalWriteOffset = 0;
        int               numLocalRead             = 0;
        int               expectedLocalReadOffset  = 0;
        int               numMFMA                  = 0;
        std::string const mfma_pattern
            = isFP8 ? "v_mfma_f32_32x32x16_fp8_fp8" : "v_mfma_f32_32x32x16_bf8_bf8";
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with(mfma_pattern))
                numMFMA++;

            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 1024;
            }

            if(instruction.starts_with("ds_read_u8"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 8 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 8 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 32;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 1024);
        EXPECT_EQ(numLocalRead, 16);
        EXPECT_EQ(numMFMA, 2);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_16x16x32_TN)
    {
        if(m_context->targetArchitecture().target().isCDNA3GPU())
        {
            GTEST_SKIP() << "FIXME: Skipping test for gfx94X";
        }

        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, FP8, float>(16, 16, 32, 1, 2.e-5, true, "T", "N");
        else
            matrixMultiplyMacroTile<BF8, BF8, float>(16, 16, 32, 1, 2.e-5, true, "T", "N");
    }

    TEST_P(MatrixMultiplyF8F6F4TestGPU, GPU_MatrixMultiplyMacroTileF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 128) ? 16 : 32;
        uint const waveN = (MFMAK == 128) ? 16 : 32;
        uint const waveK = MFMAK;

        std::string const mfmaMnemonic{
            std::format("v_mfma_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        auto const [transA, transB] = transOp;

        uint const        elementBits = DataTypeInfo::Get(typeAB).elementBits;
        std::string const trLoadMnemonic{transposeLoadMnemonic(elementBits)};

        // TODO: enable non-TN F6 tests
        if(elementBits == 6 && (transA != "T" || transB != "N"))
        {
            GTEST_SKIP();
        }

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            matrixMultiplyMacroTile<FP8, FP8, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB);
            break;
        case DataType::BF8:
            matrixMultiplyMacroTile<BF8, BF8, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            matrixMultiplyMacroTile<FP6, FP6, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            matrixMultiplyMacroTile<BF6, BF6, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            matrixMultiplyMacroTile<FP4, FP4, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                std::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const numMFMAs            = 2;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(transA == "N")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB == "T")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), 2);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
        EXPECT_EQ(countSubstring(generatedCode, modifiers), 2);
    }

    TEST_P(MatrixMultiplyF8F6F4TestGPU, GPU_ScaledMatrixMultiplyMacroTileF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 128) ? 16 : 32;
        uint const waveN = (MFMAK == 128) ? 16 : 32;
        uint const waveK = MFMAK;

        std::string const mfmaMnemonic{
            std::format("v_mfma_scale_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        auto const [transA, transB] = transOp;

        uint        elementBits = DataTypeInfo::Get(typeAB).elementBits;
        std::string trLoadMnemonic{transposeLoadMnemonic(elementBits)};

        // TODO: enable non-TN F6 tests
        if(elementBits == 6 && (transA != "T" || transB != "N"))
        {
            GTEST_SKIP();
        }

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            matrixMultiplyMacroTile<FP8, FP8, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB, true, true);
            break;
        case DataType::BF8:
            matrixMultiplyMacroTile<BF8, BF8, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB, true, true);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            matrixMultiplyMacroTile<FP6, FP6, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB, true, true);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            matrixMultiplyMacroTile<BF6, BF6, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB, true, true);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            matrixMultiplyMacroTile<FP4, FP4, float>(
                waveM, waveN, waveK, 1, 7.5e-6, true, transA, transB, true, true);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                std::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const numMFMAs            = 2;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(transA == "N")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB == "T")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), 2);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
        EXPECT_EQ(countSubstring(generatedCode, modifiers), 2);
    }

    TEST_P(MatrixMultiplyMixedTestGPU, GPU_MatrixMultiplyMacroTileMixed)
    {
        auto [typeA, typeB, MFMAK, transOp] = std::get<1>(GetParam());

        int wave_m = (MFMAK == 128) ? 16 : 32;
        int wave_n = (MFMAK == 128) ? 16 : 32;
        int wave_k = MFMAK;

        uint const elementBitsA = DataTypeInfo::Get(typeA).elementBits;
        uint const elementBitsB = DataTypeInfo::Get(typeB).elementBits;
        auto [transA, transB]   = transOp;
        // TODO: enable non-TN F6 tests
        if((elementBitsA == 6 || elementBitsB == 6) && (transA != "T" || transB != "N"))
        {
            GTEST_SKIP();
        }

        matrixMultiplyMacroTileMixed(
            typeA, typeB, wave_m, wave_n, wave_k, 1, 1.e-5, true, "T", "N");
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyAB)
    {
        matrixMultiplyAB<float>(32, 32, 2, 1, 2.e-6);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABFP16)
    {
        matrixMultiplyAB<Half>(32, 32, 8, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_16x16x32)
    {
        if(m_context->targetArchitecture().target().isCDNA3GPU())
        {
            GTEST_SKIP() << "FIXME: Skipping test for gfx94X";
        }

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(16, 16, 32, 1, 2.e-5);
        else
            matrixMultiplyAB<BF8, float>(16, 16, 32, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_32x32x16)
    {
        if(m_context->targetArchitecture().target().isCDNA3GPU())
        {
            GTEST_SKIP() << "FIXME: Skipping test for gfx94X";
        }

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(32, 32, 16, 1, 2.e-5);
        else
            matrixMultiplyAB<BF8, float>(32, 32, 16, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyF8F6F4TestGPU, GPU_MatrixMultiplyABF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 128) ? 16 : 32;
        uint const waveN = (MFMAK == 128) ? 16 : 32;
        uint const waveK = MFMAK;

        std::string const mfmaMnemonic{
            std::format("v_mfma_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        auto const transA = transOp.first == "T";
        auto const transB = transOp.second == "T";

        uint const        elementBits = DataTypeInfo::Get(typeAB).elementBits;
        std::string const trLoadMnemonic{transposeLoadMnemonic(elementBits)};

        // TODO: enable non-TN F6 tests
        if(elementBits == 6 && (!transA || transB))
        {
            GTEST_SKIP();
        }

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            matrixMultiplyAB<FP8, float>(waveM, waveN, waveK, 1, 2.e-5, true, transA, transB);
            break;
        case DataType::BF8:
            matrixMultiplyAB<BF8, float>(waveM, waveN, waveK, 1, 2.e-5, true, transA, transB);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            matrixMultiplyAB<FP6, float>(waveM, waveN, waveK, 1, 2.e-5, true, transA, transB);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            matrixMultiplyAB<BF6, float>(waveM, waveN, waveK, 1, 2.e-5, true, transA, transB);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            matrixMultiplyAB<FP4, float>(waveM, waveN, waveK, 1, 2.e-5, true, transA, transB);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                std::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const numMFMAs            = 2;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(!transA)
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB)
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, modifiers), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABC)
    {
        matrixMultiplyABC<float>(32, 32, 2, 1, 2.e-6);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABCFP16)
    {
        matrixMultiplyABC<Half>(32, 32, 8, 1, 2.e-5);
    }

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyTest, MatrixMultiplyTestGPU, mfmaSupportedISATuples());

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyTest,
                             MatrixMultiplyTestGPUF8,
                             ::testing::Combine(mfmaSupportedISAValues(),
                                                ::testing::Values(rocRoller::DataType::FP8,
                                                                  rocRoller::DataType::BF8)));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        MatrixMultiplyTestGPUF16,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Combine(::testing::Values(rocRoller::DataType::Half,
                                                 rocRoller::DataType::BFloat16),
                               ::testing::Values(16, 32),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        MatrixMultiplyF8F6F4TestGPU,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        MatrixMultiplyMixedTestGPU,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    // Params are: A type, B type, scale pair, K tile size
    class ScaledMatrixMultiplyMixedTestGPU
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    TEST_P(ScaledMatrixMultiplyMixedTestGPU, GPU_ScaledMatrixMultiplyMacroTileMixed)
    {
        auto [typeA, typeB, MFMAK, transOp] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        uint const elementBitsA = DataTypeInfo::Get(typeA).elementBits;
        uint const elementBitsB = DataTypeInfo::Get(typeB).elementBits;
        auto [transA, transB]   = transOp;
        // TODO: enable non-TN F6 tests
        if((elementBitsA == 6 || elementBitsB == 6) && (transA != "T" || transB != "N"))
        {
            GTEST_SKIP();
        }

        matrixMultiplyMacroTileMixed(
            typeA, typeB, waveM, waveN, waveK, 1, 1.e-5, true, transA, transB, true, true);
    }

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        ScaledMatrixMultiplyMixedTestGPU,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    class ScaledMMTest
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             std::pair<uint8_t, uint8_t>,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    template <typename TA, typename TB>
    void exeScaledCPUMM(const int   M,
                        const int   N,
                        const int   K,
                        const float scaleA,
                        const float scaleB,
                        float       alpha,
                        double      err,
                        bool        transA,
                        bool        transB)
    {
        RandomGenerator random(9861u);

        auto A     = random.vector<TA>(M * K, -1.f, 1.f);
        auto B     = random.vector<TB>(K * N, -1.f, 1.f);
        auto C     = std::vector<float>(M * N);
        auto D     = std::vector<float>(M * N);
        auto ref_D = std::vector<float>(M * N);

        auto AX = std::vector<uint8_t>(M * K / 32);
        auto BX = std::vector<uint8_t>(K * N / 32);
        std::fill(AX.begin(), AX.end(), scaleA);
        std::fill(BX.begin(), BX.end(), scaleB);

        // TODO: now only works for _TN for A and B, need to enable other data layout
        ScaledCPUMM(D, C, A, B, AX, BX, M, N, K, alpha, 0.0, transA, transB);

        alpha *= std::pow(2.0f, int(scaleA) - 127) * std::pow(2.0f, int(scaleB) - 127);

        CPUMM(ref_D, C, A, B, M, N, K, alpha, 0.0, transA, transB);

        double rnorm = relativeNormL2(D, ref_D);
        Log::info("RNorm is {}", rnorm);
        ASSERT_LT(rnorm, err);
    }

    template <typename TA>
    void scaledCPUMMMixed(rocRoller::DataType typeB,
                          const int           m,
                          const int           n,
                          const int           k,
                          const float         scaleA,
                          const float         scaleB,
                          float               alpha,
                          double              err,
                          bool                transA,
                          bool                transB)
    {
        if(typeB == rocRoller::DataType::FP8)
            exeScaledCPUMM<TA, FP8>(m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeB == rocRoller::DataType::BF8)
            exeScaledCPUMM<TA, BF8>(m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeB == rocRoller::DataType::FP6)
            exeScaledCPUMM<TA, FP6>(m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeB == rocRoller::DataType::BF6)
            exeScaledCPUMM<TA, BF6>(m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeB == rocRoller::DataType::FP4)
            exeScaledCPUMM<TA, FP4>(m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else
            Throw<FatalError>("Invalid type.");
    }

    void scaledCPUMMMixed(rocRoller::DataType typeA,
                          rocRoller::DataType typeB,
                          const int           m,
                          const int           n,
                          const int           k,
                          const float         scaleA,
                          const float         scaleB,
                          float               alpha,
                          double              err,
                          bool                transA,
                          bool                transB)
    {
        if(typeA == rocRoller::DataType::FP8)
            scaledCPUMMMixed<FP8>(typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeA == rocRoller::DataType::BF8)
            scaledCPUMMMixed<BF8>(typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeA == rocRoller::DataType::FP6)
            scaledCPUMMMixed<FP6>(typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeA == rocRoller::DataType::BF6)
            scaledCPUMMMixed<BF6>(typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else if(typeA == rocRoller::DataType::FP4)
            scaledCPUMMMixed<FP4>(typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB);
        else
            Throw<FatalError>("Invalid type.");
    }

    TEST_P(ScaledMMTest, ScaledMMTestCPU)
    {
        auto [typeA, typeB, scales, MFMAK, transOp] = std::get<1>(GetParam());

        auto [scaleA, scaleB] = scales;
        auto [transA, transB] = transOp;

        int M = (MFMAK == 128) ? 16 : 32;
        int N = (MFMAK == 128) ? 16 : 32;
        int K = MFMAK;

        float alpha = 1.0f;

        scaledCPUMMMixed(
            typeA, typeB, M, N, K, scaleA, scaleB, alpha, 1.e-5, transA == "T", transB == "T");
    }

    INSTANTIATE_TEST_SUITE_P(
        ScaledMMCPU,
        ScaledMMTest,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(std::pair<uint8_t, uint8_t>{125u, 125u},
                                                 std::pair<uint8_t, uint8_t>{125u, 128u},
                                                 std::pair<uint8_t, uint8_t>{128u, 125u},
                                                 std::pair<uint8_t, uint8_t>{128u, 128u}),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTestGPUBFloat16,
        MatrixMultiplyTestGPUBFloat16,
        ::testing::Combine(mfmaSupportedISAValues(),
                           ::testing::Values(std::tuple<int, int, int>{32, 32, 4},
                                             std::tuple<int, int, int>{16, 16, 8})));
}
