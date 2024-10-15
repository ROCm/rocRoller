
#include <filesystem>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/GPUArchitecture.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Version.hpp>

#include <common/Utilities.hpp>
#include <common/mxDataGen.hpp>

#include "include/DataParallelGEMMSolution.hpp"
#include "include/GEMMParameters.hpp"
#include "include/StreamKGEMMSolution.hpp"
#include "include/TensileGEMMSolution.hpp"

#include <CLI/CLI.hpp>

using namespace rocRoller;

const std::string clientName = "GEMMv00";

namespace rocRoller::Client::GEMMClient
{
    struct GEMMSolutionBlob
    {
        GPUArchitectureTarget architecture;
        SolutionParameters    solution;
        std::string           assembly;
    };
}

template <typename IO>
struct rocRoller::Serialization::
    MappingTraits<Client::GEMMClient::Result, IO, rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, Client::GEMMClient::Result& result)
    {
        iot::mapRequired(io, "client", clientName);
        iot::mapRequired(io, "device", result.benchmarkResults.runParams.device);
        iot::mapRequired(io, "M", result.problemParams.m);
        iot::mapRequired(io, "N", result.problemParams.n);
        iot::mapRequired(io, "K", result.problemParams.k);
        iot::mapRequired(io, "alpha", result.problemParams.alpha);
        iot::mapRequired(io, "beta", result.problemParams.beta);
        iot::mapRequired(io, "trans_A", Client::GEMMClient::toString(result.solutionParams.transA));
        iot::mapRequired(io, "trans_B", Client::GEMMClient::toString(result.solutionParams.transB));

        iot::mapRequired(io, "type_A", result.solutionParams.typeA);
        iot::mapRequired(io, "type_B", result.solutionParams.typeB);
        iot::mapRequired(io, "type_C", result.solutionParams.typeC);
        iot::mapRequired(io, "type_D", result.solutionParams.typeD);
        iot::mapRequired(io, "type_acc", result.solutionParams.typeAcc);

        iot::mapRequired(io, "mac_m", result.solutionParams.macM);
        iot::mapRequired(io, "mac_n", result.solutionParams.macN);
        iot::mapRequired(io, "mac_k", result.solutionParams.macK);
        iot::mapRequired(io, "wave_m", result.solutionParams.waveM);
        iot::mapRequired(io, "wave_n", result.solutionParams.waveN);
        iot::mapRequired(io, "wave_k", result.solutionParams.waveK);
        iot::mapRequired(io, "wave_b", result.solutionParams.waveB);
        iot::mapRequired(io, "workgroup_size_x", result.solutionParams.workgroupSizeX);
        iot::mapRequired(io, "workgroup_size_y", result.solutionParams.workgroupSizeY);
        iot::mapRequired(io, "unroll_x", result.solutionParams.unrollX);
        iot::mapRequired(io, "unroll_y", result.solutionParams.unrollY);
        iot::mapRequired(io, "loadLDS_A", result.solutionParams.loadLDSA);
        iot::mapRequired(io, "loadLDS_B", result.solutionParams.loadLDSB);
        iot::mapRequired(io, "storeLDS_D", result.solutionParams.storeLDSD);
        iot::mapRequired(io, "prefetch", result.solutionParams.prefetch);
        iot::mapRequired(io, "prefetchInFlight", result.solutionParams.prefetchInFlight);
        iot::mapRequired(io, "prefetchLDSFactor", result.solutionParams.prefetchLDSFactor);
        iot::mapRequired(io, "betaInFma", result.solutionParams.betaInFma);
        iot::mapRequired(io, "scheduler", result.solutionParams.scheduler);

        iot::mapRequired(io, "streamK", result.solutionParams.streamK);
        iot::mapRequired(io, "numWGs", result.benchmarkResults.runParams.numWGs);
        iot::mapRequired(io, "streamKTwoTile", result.solutionParams.streamKTwoTile);

        iot::mapRequired(io, "numWarmUp", result.benchmarkResults.runParams.numWarmUp);
        iot::mapRequired(io, "numOuter", result.benchmarkResults.runParams.numOuter);
        iot::mapRequired(io, "numInner", result.benchmarkResults.runParams.numInner);

        iot::mapRequired(io, "kernelGenerate", result.benchmarkResults.kernelGenerate);
        iot::mapRequired(io, "kernelAssemble", result.benchmarkResults.kernelAssemble);
        iot::mapRequired(io, "kernelExecute", result.benchmarkResults.kernelExecute);

        iot::mapRequired(io, "checked", result.benchmarkResults.checked);
        iot::mapRequired(io, "correct", result.benchmarkResults.correct);
        iot::mapRequired(io, "rnorm", result.benchmarkResults.rnorm);
    }

    static void mapping(IO& io, Client::GEMMClient::Result& result, EmptyContext& ctx)
    {
        mapping(io, result);
    }
};

template <typename IO>
struct rocRoller::Serialization::MappingTraits<Client::GEMMClient::SolutionParameters,
                                               IO,
                                               rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, Client::GEMMClient::SolutionParameters& params)
    {
        iot::mapRequired(io, "mac_m", params.macM);
        iot::mapRequired(io, "mac_n", params.macN);
        iot::mapRequired(io, "mac_k", params.macK);
        iot::mapRequired(io, "wave_m", params.waveM);
        iot::mapRequired(io, "wave_n", params.waveN);
        iot::mapRequired(io, "wave_k", params.waveK);
        iot::mapRequired(io, "wave_b", params.waveB);
        iot::mapRequired(io, "workgroup_size_x", params.workgroupSizeX);
        iot::mapRequired(io, "workgroup_size_y", params.workgroupSizeY);
        iot::mapRequired(io, "unroll_x", params.unrollX);
        iot::mapRequired(io, "unroll_y", params.unrollY);
        iot::mapRequired(io, "loadLDS_A", params.loadLDSA);
        iot::mapRequired(io, "loadLDS_B", params.loadLDSB);
        iot::mapRequired(io, "storeLDS_D", params.storeLDSD);
        iot::mapRequired(io, "prefetch", params.prefetch);
        iot::mapRequired(io, "prefetchInFlight", params.prefetchInFlight);
        iot::mapRequired(io, "prefetchLDSFactor", params.prefetchLDSFactor);
        iot::mapRequired(io, "betaInFma", params.betaInFma);
        iot::mapRequired(io, "scheduler", params.scheduler);

        iot::mapRequired(io, "trans_A", params.transA);
        iot::mapRequired(io, "trans_B", params.transB);

        iot::mapRequired(io, "type_A", params.typeA);
        iot::mapRequired(io, "type_B", params.typeB);
        iot::mapRequired(io, "type_C", params.typeC);
        iot::mapRequired(io, "type_D", params.typeD);
        iot::mapRequired(io, "type_acc", params.typeAcc);

        iot::mapRequired(io, "streamK", params.streamK);
        iot::mapRequired(io, "streamKTwoTile", params.streamKTwoTile);

        iot::mapRequired(io, "version", params.version);
    }

    static void mapping(IO& io, Client::GEMMClient::SolutionParameters& params, EmptyContext& ctx)
    {
        mapping(io, params);
    }
};

template <typename IO>
struct rocRoller::Serialization::
    MappingTraits<Client::GEMMClient::GEMMSolutionBlob, IO, rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, Client::GEMMClient::GEMMSolutionBlob& soln)
    {
        iot::mapRequired(io, "architecture", soln.architecture);
        iot::mapRequired(io, "solution", soln.solution);
        iot::mapRequired(io, "assembly", soln.assembly);
    }

    static void mapping(IO& io, Client::GEMMClient::GEMMSolutionBlob& soln, EmptyContext& ctx)
    {
        mapping(io, soln);
    }
};

namespace rocRoller::Client::GEMMClient
{
    template <typename A, typename B, typename C, typename D>
    std::pair<bool, double>
        validate(std::vector<A> const&                                   h_A,
                 std::vector<B> const&                                   h_B,
                 std::vector<C> const&                                   h_C,
                 std::vector<D> const&                                   h_D,
                 rocRoller::Client::GEMMClient::ProblemParameters const& problemParams,
                 GPUArchitecture const&                                  arch)
    {
        using namespace rocRoller::Client::GEMMClient;

        // Host result
        std::vector<D> h_result(problemParams.m * problemParams.n, static_cast<D>(0.0));
        CPUMM(h_result,
              h_C,
              h_A,
              h_B,
              problemParams.m,
              problemParams.n,
              problemParams.k,
              problemParams.alpha,
              problemParams.beta,
              problemParams.transA == TransposeType::T,
              problemParams.transB == TransposeType::T);

        auto tol = gemmAcceptableError<A, B, D>(
            problemParams.m, problemParams.n, problemParams.k, arch.target());
        auto res = compare(h_D, h_result, tol);

        Log::debug(res.message());

        std::cout << "Result: " << (res.ok ? "Correct" : "Incorrect") << std::endl;
        std::cout << "RNorm: " << res.relativeNormL2 << std::endl;
        if(!res.ok)
        {
            std::cerr << "WARNING: Result incorrect.  " << res.message() << std::endl;
        }
        return {res.ok, res.relativeNormL2};
    }

    // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
    template <typename A, typename B, typename C, typename D>
    Client::BenchmarkResults GEMM(std::shared_ptr<Client::GEMMClient::GEMMSolution> gemm,
                                  Client::GEMMClient::SolutionParameters const&     solutionParams,
                                  Client::GEMMClient::ProblemParameters const&      problemParams,
                                  Client::RunParameters const&                      runParams,
                                  std::string            loadAssemblyFile,
                                  GPUArchitecture const& arch)
    {
        using namespace rocRoller::Client;
        using namespace rocRoller::Client::GEMMClient;

        // Host Data
        std::cout << "Generating input data..." << std::endl;
        auto           seed = 31415u;
        std::vector<A> h_A  = DGenVector<A>(problemParams.m, problemParams.k, -1.0, 1.0, seed + 1);
        std::vector<B> h_B  = DGenVector<B>(problemParams.k, problemParams.n, -1.0, 1.0, seed + 2);
        std::vector<C> h_C  = DGenVector<C>(problemParams.m, problemParams.n, -1.0, 1.0, seed + 3);
        std::vector<D> h_D(problemParams.m * problemParams.n, static_cast<D>(0.0));

        auto d_A = make_shared_device(h_A);
        auto d_B = make_shared_device(h_B);
        auto d_C = make_shared_device(h_C);
        auto d_D = make_shared_device(h_D);

        std::cout << "Generating lauch parameters and runtime arguments..." << std::endl;
        auto commandKernel = gemm->commandKernel();
        auto launchParams  = gemm->makeLaunchParameters(problemParams, solutionParams, runParams);

        commandKernel->loadKernel();
        commandKernel->setLaunchParameters(launchParams);
        auto commandArgs = gemm->commandArguments(problemParams, runParams);

        auto [aTag, bTag, cTag, dTag] = gemm->getABCDTags();
        commandArgs.setArgument(aTag, ArgumentType::Value, (A*)d_A.get());
        commandArgs.setArgument(bTag, ArgumentType::Value, (B*)d_B.get());
        commandArgs.setArgument(cTag, ArgumentType::Value, (C*)d_C.get());
        commandArgs.setArgument(dTag, ArgumentType::Value, (D*)d_D.get());

        if(!loadAssemblyFile.empty())
        {
            std::cout << "Loading kernel from: " << loadAssemblyFile << std::endl;
            commandKernel->loadKernelFromAssembly(loadAssemblyFile,
                                                  solutionParams.generateKernelName());
        }

        gemm->validateRunParameters(solutionParams, problemParams, runParams, commandKernel);

        auto runtimeArgs = commandArgs.runtimeArguments();

        auto scratchSpaceRequired = commandKernel->scratchSpaceRequired(runtimeArgs);
        if(scratchSpaceRequired > 0)
        {
            auto deviceScratch = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
            commandArgs.setArgument(
                gemm->getScratchTag(), ArgumentType::Value, deviceScratch.get());
        }

        if(runParams.visualize)
        {
            Client::visualize(gemm->command(), *commandKernel, commandArgs);
        }

        std::cout << std::endl;
        std::cout << "Solution:" << std::endl;
        std::cout << solutionParams << std::endl;

        std::cout << "Problem:" << std::endl;
        std::cout << problemParams << std::endl;

        std::cout << "Launching GPU kernel(s)..." << std::endl;

        BenchmarkResults result;
        result.runParams = runParams;

        // Benchmark runs
        for(int outer = 0; outer < runParams.numOuter; ++outer)
        {
            // Warmup runs
            for(int i = 0; i < runParams.numWarmUp; ++i)
            {
                commandKernel->launchKernel(runtimeArgs);
            }

            HIP_TIMER(t_kernel, "GEMM", runParams.numInner);
            for(int inner = 0; inner < runParams.numInner; ++inner)
            {
                commandKernel->launchKernel(runtimeArgs, t_kernel, inner);
            }
            HIP_SYNC(t_kernel);
            t_kernel->sleep(50);
            auto nanoseconds = t_kernel->allNanoseconds();
            result.kernelExecute.insert(
                result.kernelExecute.end(), nanoseconds.begin(), nanoseconds.end());
        }

        double totalTime = 0;
        for(auto ke : result.kernelExecute)
            totalTime += static_cast<double>(ke) / 1.e9;
        double averageTime = totalTime / (runParams.numInner * runParams.numOuter);

        std::cout << "Average runtime (s): " << averageTime << std::endl;
        std::cout << "Average GFLOPS:      "
                  << (double)problemParams.m * problemParams.n * problemParams.k * 2.0 / averageTime
                         * 1.e-9
                  << std::endl;

        result.kernelAssemble = TimerPool::nanoseconds("CommandKernel::assembleKernel");
        result.kernelGenerate = TimerPool::nanoseconds("CommandKernel::generateKernel");

        if(runParams.check)
        {
            AssertFatal(hipMemcpy(h_D.data(),
                                  d_D.get(),
                                  problemParams.m * problemParams.n * sizeof(D),
                                  hipMemcpyDeviceToHost)
                        == (hipError_t)HIP_SUCCESS);

            auto [correct, rnorm] = validate<A, B, C, D>(h_A, h_B, h_C, h_D, problemParams, arch);

            result.checked = true;
            result.correct = correct;
            result.rnorm   = rnorm;
        }

        return result;
    }

    /*
     * Generate an instance of GEMMSolution and call generateSolution.
     *
     * The kind of GEMMSolution returned is based on the parameters in
     * solutionParams.
     */
    std::shared_ptr<Client::GEMMClient::GEMMSolution>
        generateGEMMSolution(rocRoller::ContextPtr                         context,
                             Client::GEMMClient::SolutionParameters const& solutionParams)
    {
        std::shared_ptr<Client::GEMMClient::GEMMSolution> gemmSolution;

        auto arch = context->targetArchitecture().target();

        if(solutionParams.scheduler == "TENSILE_ASM")
        {
            if(arch.isCDNA2GPU())
            {
                gemmSolution = std::make_shared<Client::GEMMClient::TensileGEMMSolution>(context);
            }
            else
            {
                std::cout << "Not running TENSILE_ASM for " << arch.toString() << std::endl;
                return nullptr;
            }
        }
        else if(solutionParams.streamK)
        {
            if(context->targetArchitecture().HasCapability(GPUCapability::ArchAccUnifiedRegs))
            {
                gemmSolution = std::make_shared<Client::GEMMClient::StreamKGEMMSolution>(context);
            }
            else
            {
                std::cout << "Not running StreamK for " << arch.toString() << std::endl;
                return nullptr;
            }
        }
        else
        {
            gemmSolution = std::make_shared<Client::GEMMClient::DataParallelGEMMSolution>(context);
        }

        AssertFatal(gemmSolution, "No solution!");

        gemmSolution->generateSolution(solutionParams);

        return gemmSolution;
    }

    /*
     * Dispatch
     */
    struct ArchitectureParameters
    {
        GPUArchitectureTarget target;
    };

    struct TypeParameters
    {
        std::string typeA   = "float";
        std::string typeB   = "float";
        std::string typeC   = "float";
        std::string typeD   = "float";
        std::string typeAcc = "float";

        Client::GEMMClient::TransposeType transA = Client::GEMMClient::TransposeType::N;
        Client::GEMMClient::TransposeType transB = Client::GEMMClient::TransposeType::N;
    };

    struct IOParameters
    {
        bool        doSave;
        std::string savePath, loadPath, resultsPath;
    };

    int RunGEMMCLI(bool                   doGenerate,
                   bool                   doValidate,
                   bool                   doBenchmark,
                   ArchitectureParameters architecture,
                   SolutionParameters     solution,
                   ProblemParameters      problem,
                   TypeParameters         types,
                   RunParameters          run,
                   IOParameters           io)
    {
        std::shared_ptr<Client::GEMMClient::GEMMSolution> gemm;

        // Changing settings has to go before creating the context :(
        if(io.doSave)
        {
            if(io.savePath.empty())
                io.savePath = solution.generateKernelName() + ".s";

            std::filesystem::path assemblyPath{io.savePath};
            assemblyPath.replace_extension(".s");

            Settings::getInstance()->set(Settings::SaveAssembly, true);
            Settings::getInstance()->set(Settings::AssemblyFile, std::string(assemblyPath));
        }

        // Changing settings has to go before creating the context :(
        std::string loadAssemblyFile;
        if(!io.loadPath.empty())
        {
            auto soln
                = Serialization::readYAMLFile<rocRoller::Client::GEMMClient::GEMMSolutionBlob>(
                    io.loadPath);
            solution            = soln.solution;
            architecture.target = soln.architecture;
            loadAssemblyFile    = soln.assembly;
            if(solution.version != rocRoller::Version::Git())
            {
                std::cout << "Warning: this version of rocRoller (" << rocRoller::Version::Git()
                          << ") differs from the one that generated " << io.loadPath << std::endl;
            }
        }

        auto arch = GPUArchitectureLibrary::getInstance()->GetArch(architecture.target);

        if(solution.scheduler != "" && solution.scheduler != "TENSILE_ASM")
        {
            auto schedulerValue = fromString<Scheduling::SchedulerProcedure>(solution.scheduler);
            Settings::getInstance()->set(Settings::Scheduler, schedulerValue);
        }

        auto context = Context::ForTarget(arch, solution.generateKernelName());

        // If no subcommands were given, default behaviour is: generate
        // and benchmark.
        bool willRunOnGPU = doValidate || doBenchmark;
        if(willRunOnGPU)
        {
            std::cout << "Setting HIP device to " << run.device << std::endl;
            HIP_CHECK(hipSetDevice(run.device));
        }

        if(willRunOnGPU && solution.streamK)
        {
            if(run.numWGs == 0)
            {
                hipDeviceProp_t deviceProperties;
                AssertFatal(hipGetDeviceProperties(&deviceProperties, 0)
                            == (hipError_t)HIP_SUCCESS);
                run.numWGs = deviceProperties.multiProcessorCount;
            }
            AssertFatal(!solution.streamKTwoTile || solution.streamK);
        }

        if(doGenerate)
        {
            std::cout << "Generating for architecture: "
                      << context->targetArchitecture().target().toString() << std::endl;

            std::cout << std::endl;
            std::cout << "Solution:" << std::endl;
            std::cout << solution << std::endl;

            std::cout << "Generating: " << solution.generateKernelName() << "..." << std::endl;

            gemm = generateGEMMSolution(context, solution);

            if(gemm && !io.savePath.empty())
            {
                std::filesystem::path assemblyPath{io.savePath};
                assemblyPath.replace_extension(".s");

                std::filesystem::path yamlPath{io.savePath};
                yamlPath.replace_extension(".yaml");

                std::ofstream file(yamlPath);
                Serialization::writeYAML(file,
                                         rocRoller::Client::GEMMClient::GEMMSolutionBlob{
                                             architecture.target, solution, assemblyPath});

                std::cout << "Wrote: " << assemblyPath.string() << std::endl;
                std::cout << "Wrote: " << yamlPath.string() << std::endl;
            }
        }
        else
        {
            gemm = generateGEMMSolution(context, solution);
        }

        if(!gemm)
            return 0;

        if(doValidate || doBenchmark)
        {
            std::cout << "Running..." << std::endl;

            if(doValidate)
            {
                run.check     = true;
                run.numWarmUp = 0;
                run.numOuter  = 1;
                run.numInner  = 1;
            }

            Client::GEMMClient::Result result;

            result.problemParams              = problem;
            result.solutionParams             = solution;
            result.benchmarkResults.runParams = run;

            if(types.typeA == "float" && types.typeB == "float" && types.typeC == "float"
               && types.typeD == "float")
            {
                result.benchmarkResults = GEMM<float, float, float, float>(
                    gemm, solution, problem, run, loadAssemblyFile, arch);
            }
            else if(types.typeA == "half" && types.typeB == "half" && types.typeC == "half"
                    && types.typeD == "half")
            {
                result.benchmarkResults = GEMM<Half, Half, Half, Half>(
                    gemm, solution, problem, run, loadAssemblyFile, arch);
            }
            else if(types.typeA == "bf16" && types.typeB == "bf16" && types.typeC == "float"
                    && types.typeD == "float")
            {
                result.benchmarkResults = GEMM<BFloat16, BFloat16, float, float>(
                    gemm, solution, problem, run, loadAssemblyFile, arch);
            }
            else if(types.typeA == "bf16" && types.typeB == "bf16" && types.typeC == "bf16"
                    && types.typeD == "bf16")
            {
                result.benchmarkResults = GEMM<BFloat16, BFloat16, BFloat16, BFloat16>(
                    gemm, solution, problem, run, loadAssemblyFile, arch);
            }
            else if(types.typeA == "fp8" && types.typeB == "fp8" && types.typeC == "float"
                    && types.typeD == "float")
            {
                result.benchmarkResults = GEMM<FP8, FP8, float, float>(
                    gemm, solution, problem, run, loadAssemblyFile, arch);
            }
            else if(types.typeA == "bf8" && types.typeB == "bf8" && types.typeC == "float"
                    && types.typeD == "float")
            {
                result.benchmarkResults = GEMM<BF8, BF8, float, float>(
                    gemm, solution, problem, run, loadAssemblyFile, arch);
            }
            else
            {
                Throw<FatalError>("Unsupported combination of datatypes for GEMM");
            }

            if(!io.resultsPath.empty())
            {
                std::ofstream file(io.resultsPath);
                Serialization::writeYAML(file, result);
            }

            if(!result.benchmarkResults.correct)
                return 1;
        }

        return 0;
    }
}

/*
 * Parse the command line and dispatch.
 */
int main(int argc, const char* argv[])
{
    CLI::App app{"GEMM Driver: D (MxN) = alpha * A (MxK) * B (KxN) + beta * C (MxN)"};
    app.footer(Settings::getInstance()->help());

    //
    // Parameters
    //
    std::string                                           architectureName;
    rocRoller::Client::GEMMClient::ArchitectureParameters architecture;

    rocRoller::Client::GEMMClient::SolutionParameters solution{
        .macM = 64,
        .macN = 64,
        .macK = 64,

        .waveM = -1,
        .waveN = -1,
        .waveK = -1,
        .waveB = -1,

        .workgroupSizeX = 128,
        .workgroupSizeY = 2,

        .loadLDSA  = true,
        .loadLDSB  = true,
        .storeLDSD = true,

        .prefetch          = false,
        .prefetchInFlight  = 0,
        .prefetchLDSFactor = 0,

        .betaInFma = true,

        .unrollX = 0,
        .unrollY = 0,

        .scheduler         = "Priority",
        .matchMemoryAccess = true,

        .streamK        = false,
        .streamKTwoTile = false,

        .version = rocRoller::Version::Git(),
    };

    rocRoller::Client::GEMMClient::ProblemParameters problem{
        .m = 3072,
        .n = 4096,
        .k = 4096,

        .alpha = 2.0f,
        .beta  = 0.5f,
    };

    rocRoller::Client::GEMMClient::TypeParameters types;

    rocRoller::Client::RunParameters run{
        .device    = 0,
        .numWarmUp = 3,
        .numOuter  = 5,
        .numInner  = 2,
        .check     = true,
        .visualize = false,
        .numWGs    = 0,
    };

    rocRoller::Client::GEMMClient::IOParameters io;

    //
    // Architecture
    //
    app.option_defaults()->ignore_case()->group("Architecture parameters");
    app.add_option("--arch", architectureName, "GPU architecture name (eg, gfx90a).");

    //
    // Problem definition
    //
    app.option_defaults()->ignore_case()->group("Problem parameters");
    app.add_option("-M,--M", problem.m, "Tensor size M.");
    app.add_option("-N,--N", problem.n, "Tensor size N.");
    app.add_option("-K,--K", problem.k, "Tensor size K.");
    app.add_option("--alpha", problem.alpha, "Alpha scalar.");
    app.add_option("--beta", problem.beta, "Beta scalar.");

    //
    // Problem types
    //
    app.option_defaults()->ignore_case()->group("Type parameters");
    app.add_option("--type_A",
                   types.typeA,
                   "Datatype of A matrix [float | half | bf16 | fp8 | bf8].  Default: float.");
    app.add_option("--type_B",
                   types.typeB,
                   "Datatype of B matrix [float | half | bf16 | fp8 | bf8].  Default: float.");
    app.add_option(
        "--type_C", types.typeC, "Datatype of C matrix [float | half | bf16].  Default: float.");
    app.add_option(
        "--type_D", types.typeD, "Datatype of D matrix [float | half | bf16].  Default: float.");
    app.add_option("--type_acc", types.typeAcc, "Datatype of accumulation [float]");
    app.add_option(
        "--trans_A",
        [&types](auto res) -> bool {
            types.transA = fromString<Client::GEMMClient::TransposeType>(res[0]);
            return true;
        },
        "N: A is not to be transposed.  T: A is to be transposed.",
        "N");
    app.add_option(
        "--trans_B",
        [&types](auto res) -> bool {
            types.transB = fromString<Client::GEMMClient::TransposeType>(res[0]);
            return true;
        },
        "N: B is not to be transposed.  T: B is to be transposed.",
        "N");

    //
    // Kernel options
    //
    app.option_defaults()->ignore_case()->group("Solution parameters");
    app.add_option("--mac_m", solution.macM, "(Macro) Tile size M.");
    app.add_option("--mac_n", solution.macN, "(Macro) Tile size N.");
    app.add_option("--mac_k", solution.macK, "(Macro) Tile size K.");
    app.add_option("--wave_m", solution.waveM, "(MFMA) Tile size M.");
    app.add_option("--wave_n", solution.waveN, "(MFMA) Tile size N.");
    app.add_option("--wave_k", solution.waveK, "(MFMA) Tile size K.");
    app.add_option("--wave_b", solution.waveB, "(MFMA) Tile size K.");

    app.add_option(
        "--mfma",
        [&solution](auto res) -> bool {
            auto mfma = res[0];
            if(!mfma.empty())
            {
                bool fail = false;
                try
                {
                    std::istringstream iss(mfma);
                    std::string        token;

                    iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit
                                   | std::ifstream::badbit);
                    std::getline(iss, token, 'x');
                    solution.waveM = std::stoi(token);
                    std::getline(iss, token, 'x');
                    solution.waveN = std::stoi(token);
                    std::getline(iss, token, 'x');
                    solution.waveK = std::stoi(token);
                    iss.exceptions(std::ifstream::failbit | std::ifstream::badbit);
                    std::getline(iss, token, 'x');
                    solution.waveB = std::stoi(token);
                }
                catch(const std::invalid_argument&)
                {
                    fail = true;
                }
                catch(const std::ios_base::failure&)
                {
                    fail = true;
                }
                fail |= (solution.waveM < 1) || (solution.waveN < 1) || (solution.waveK < 1)
                        || (solution.waveB < 1);
                if(fail)
                {
                    std::cout << "Invalid format for MFMA instruction." << std::endl;
                    std::cout << std::endl;
                    std::cout << "The MFMA argument should be formatted like:" << std::endl;
                    std::cout << std::endl;
                    std::cout << "    --mfma=MxNxKxB" << std::endl;
                    std::cout << std::endl;
                    std::cout << "For example: --mfma=32x32x2x1" << std::endl;
                }
                return !fail;
            }
            return false;
        },
        "MFMA instruction to use.  Default 32x32x2x1 for floats, 32x32x8x1 for halfs.");

    app.add_option(
        "--workgroup_size_x", solution.workgroupSizeX, "Workgroup size in the x dimension.");
    app.add_option(
        "--workgroup_size_y", solution.workgroupSizeY, "Workgroup size in the y dimension.");
    app.add_option("--unroll_x", solution.unrollX, "Unroll size in X.");
    app.add_option("--unroll_y", solution.unrollY, "Unroll size in Y.");
    app.add_flag("--loadLDS_A", solution.loadLDSA, "Use LDS when loading A.");
    app.add_flag("--loadLDS_B", solution.loadLDSB, "Use LDS when loading B.");
    app.add_flag("--storeLDS_D", solution.storeLDSD, "Use LDS when storing D.");
    app.add_flag(
        "--betaInFma", solution.betaInFma, "Use beta in FMA instruction instead of alpha.");
    app.add_option("--scheduler", solution.scheduler, "Which scheduler to use.");
    app.add_flag("--match_memory_access",
                 solution.matchMemoryAccess,
                 "Match memory access to transpose.  Currently decreases performance.");
    app.add_flag("--prefetch", solution.prefetch, "Enable prefetching (UnrollK=2 implied).");
    app.add_option("--prefetchInFlight",
                   solution.prefetchInFlight,
                   "Number of prefetches in flight at the same time");
    app.add_option("--prefetchLDSFactor",
                   solution.prefetchLDSFactor,
                   "Prefetch 1/prefetchLDSFactor of MacroTile from LDS");
    app.add_flag("--streamK", solution.streamK, "Enable StreamK algorithm.");
    app.add_flag("--streamKTwoTile", solution.streamKTwoTile, "Enable two-tile StreamK algorithm.");

    //
    // Benchmarking options
    //
    app.option_defaults()->ignore_case()->group("Benchmarking parameters");
    app.add_option("--num_warmup", run.numWarmUp, "Number of warm-up runs.");
    app.add_option("--num_outer", run.numOuter, "Number of outer runs.");
    app.add_option("--num_inner", run.numInner, "Number of inner runs.");
    app.add_option("--device", run.device, "GPU device ordinal");
    app.add_option("--numWGs",
                   run.numWGs,
                   "Number of workgroups to use with StreamK algorithm.  Defaults to number of WGs "
                   "present on local device.");

    //
    // Client params and shortcuts
    //

    app.option_defaults()->ignore_case()->group("Client options and shortcuts");

    app.add_flag(
        "--hgemm",
        [&types](auto res) -> bool {
            types.typeA   = "half";
            types.typeB   = "half";
            types.typeC   = "half";
            types.typeD   = "half";
            types.typeAcc = "float";
            return true;
        },
        "Overwrite types to: --type_A=half --type_B=half --type_C=half --type_D=half "
        "--type_acc=float.");

    app.add_flag(
        "--visualize", run.visualize, "Dump out volumes describing memory access patterns.");

    bool noCheckResult = false;
    app.add_flag("--no-check", noCheckResult, "Do not verify GEMM results against OpenBLAS.");

    app.add_option("--yaml", io.resultsPath, "Save results to file.");

    auto generate = app.add_subcommand("generate", "Generate a GEMM solution.")->fallthrough();
    auto validate
        = app.add_subcommand("validate",
                             "Run and validate a GEMM solution (only runs the solution once).")
              ->fallthrough();
    auto benchmark
        = app.add_subcommand("benchmark",
                             "Benchmark a GEMM solution (may run the solution many times).")
              ->fallthrough();

    auto saveOption
        = generate->add_option("--save", io.savePath, "Save assembly to file.")->expected(0, 1);
    validate->add_option("--load", io.loadPath, "Load solution from assembly file.");
    benchmark->add_option("--load", io.loadPath, "Load solution from assembly file.");

    CLI11_PARSE(app, argc, argv);

    //
    // Update/validate problem definition
    //

    if(architectureName.empty())
        architecture.target
            = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch(run.device).target();
    else
        architecture.target = GPUArchitectureTarget::fromString(architectureName);

    if(!io.loadPath.empty())
    {
        auto soln = Serialization::readYAMLFile<rocRoller::Client::GEMMClient::GEMMSolutionBlob>(
            io.loadPath);

        if((types.typeA != soln.solution.typeA) || (types.typeB != soln.solution.typeB)
           || (types.typeC != soln.solution.typeC) || (types.typeD != soln.solution.typeD)
           || (types.typeAcc != soln.solution.typeAcc))
        {
            std::cout << "NOTE: Types from command line have been superceded by types from "
                         "solution."
                      << std::endl;
        }
        if((types.transA != soln.solution.transA) || (types.transB != soln.solution.transB))
        {
            std::cout << "NOTE: Transposes from command line have been superceded by transposes "
                         "from solution."
                      << std::endl;
        }

        types.typeA   = soln.solution.typeA;
        types.typeB   = soln.solution.typeB;
        types.typeC   = soln.solution.typeC;
        types.typeD   = soln.solution.typeD;
        types.typeAcc = soln.solution.typeAcc;
        types.transA  = soln.solution.transA;
        types.transB  = soln.solution.transB;
    }

    // Currently, we only support F32 accumulation
    AssertFatal(types.typeAcc == "float");

    problem.typeA   = types.typeA;
    problem.typeB   = types.typeB;
    problem.typeC   = types.typeC;
    problem.typeD   = types.typeD;
    problem.typeAcc = types.typeAcc;
    problem.transA  = types.transA;
    problem.transB  = types.transB;

    solution.typeA   = types.typeA;
    solution.typeB   = types.typeB;
    solution.typeC   = types.typeC;
    solution.typeD   = types.typeD;
    solution.typeAcc = types.typeAcc;
    solution.transA  = types.transA;
    solution.transB  = types.transB;

    run.check = !noCheckResult;

    io.doSave = saveOption->count() > 0;

    // Set default MFMA sizes
    if(solution.typeA == "float" && solution.typeB == "float")
    {
        if(solution.waveM == -1)
            solution.waveM = 32;
        if(solution.waveN == -1)
            solution.waveN = 32;
        if(solution.waveK == -1)
            solution.waveK = 2;
        if(solution.waveB == -1)
            solution.waveB = 1;
    }
    else if(solution.typeA == "half" && solution.typeB == "half")
    {
        if(solution.waveM == -1)
            solution.waveM = 32;
        if(solution.waveN == -1)
            solution.waveN = 32;
        if(solution.waveK == -1)
            solution.waveK = 8;
        if(solution.waveB == -1)
            solution.waveB = 1;
    }
    else if(solution.typeA == "bf16" && solution.typeB == "bf16")
    {
        if(solution.waveM == -1)
            solution.waveM = 16;
        if(solution.waveN == -1)
            solution.waveN = 16;
        if(solution.waveK == -1)
            solution.waveK = 8;
        if(solution.waveB == -1)
            solution.waveB = 1;
    }
    else if((solution.typeA == "fp8" && solution.typeB == "fp8")
            || (solution.typeA == "bf8" && solution.typeB == "bf8"))
    {
        if(solution.waveM == -1)
            solution.waveM = 16;
        if(solution.waveN == -1)
            solution.waveN = 16;
        if(solution.waveK == -1)
            solution.waveK = 32;
        if(solution.waveB == -1)
            solution.waveB = 1;
    }

    //
    // Run!
    //

    // If no subcommands were given, default behaviour is: generate
    // and benchmark.
    bool generateAndBenchmark = !generate->parsed() && !validate->parsed() && !benchmark->parsed();

    bool doGenerate  = generateAndBenchmark || generate->parsed();
    bool doValidate  = validate->parsed();
    bool doBenchmark = generateAndBenchmark || benchmark->parsed();

    return rocRoller::Client::GEMMClient::RunGEMMCLI(
        doGenerate, doValidate, doBenchmark, architecture, solution, problem, types, run, io);
}
