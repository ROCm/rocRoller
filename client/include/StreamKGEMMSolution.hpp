#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>

#include "DataParallelGEMMSolution.hpp"
#include "GEMMParameters.hpp"
#include "GEMMSolution.hpp"
#include "visualize.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            template <typename A, typename B, typename C, typename D>
            class StreamKGEMMSolution : public DataParallelGEMMSolution<A, B, C, D>
            {
            public:
                StreamKGEMMSolution(SolutionParameters const& solutionParams)
                {
                    this->m_problemParams  = solutionParams.problemParams;
                    this->m_solutionParams = solutionParams;

                    this->m_command = makeCommand();
                    this->m_kernel  = std::make_shared<CommandKernel>(
                        DataParallelGEMMSolution<A, B, C, D>::makeKernel(
                            makeKernelOptions(), this->m_solutionParams.numWGs, 1));
                }

                Result benchmark(Client::RunParameters const& runParams,
                                 bool                         checkResult,
                                 bool                         doVisualize,
                                 std::vector<A> const&        h_A,
                                 std::vector<B> const&        h_B,
                                 std::vector<C> const&        h_C,
                                 std::vector<D>&              h_D)
                {
                    Result result;
                    result.solutionParams = this->m_solutionParams;

                    auto d_A = make_shared_device(h_A);
                    auto d_B = make_shared_device(h_B);
                    auto d_C = make_shared_device(h_C);
                    auto d_D = make_shared_device(h_D);

                    // Create scratch space
                    auto scratchSpaceRequired = this->m_kernel->scratchSpaceRequired();
                    auto deviceScratch = make_shared_device<uint8_t>(scratchSpaceRequired, 0);

                    auto runtimeArgs = makeArgs(d_A, d_B, d_C, d_D, deviceScratch);

                    if(doVisualize)
                    {
                        Client::visualize(this->m_command, *(this->m_kernel), runtimeArgs);
                    }

                    result.benchmarkResults
                        = GEMMSolution<A, B, C, D>::benchmark(runParams, runtimeArgs);

                    if(checkResult)
                    {
                        AssertFatal(
                            hipMemcpy(h_D.data(),
                                      d_D.get(),
                                      this->m_problemParams.m * this->m_problemParams.n * sizeof(D),
                                      hipMemcpyDeviceToHost)
                            == (hipError_t)HIP_SUCCESS);

                        result.benchmarkResults.checked = true;
                        result.benchmarkResults.correct = this->validate(h_A, h_B, h_C, h_D);
                    }

                    return result;
                }

            private:
                CommandPtr makeCommand()
                {
                    auto command = DataParallelGEMMSolution<A, B, C, D>::makeCommand();

                    command->allocateArgument(
                        VariableType(DataType::UInt32, PointerType::PointerGlobal),
                        DataDirection::ReadWrite,
                        rocRoller::SCRATCH);

                    return command;
                }

                std::shared_ptr<KernelOptions> makeKernelOptions()
                {
                    auto kernelOptions = DataParallelGEMMSolution<A, B, C, D>::makeKernelOptions();

                    uint num_workgroup_x = this->m_solutionParams.numWGs;
                    uint num_workgroup_y = 1;

                    AssertFatal(num_workgroup_y == 1,
                                "Current scratch space implementation assumes that the kernel "
                                "is launched "
                                "with workgroup_size_y == 1");

                    kernelOptions->numScratchTiles = std::min((uint)(this->m_solutionParams.numWGs),
                                                              num_workgroup_x * num_workgroup_y);

                    kernelOptions->loopOverOutputTilesDimensions = {0, 1};
                    kernelOptions->loopOverOutputTilesCoordSizes
                        = {static_cast<uint>(this->m_solutionParams.problemParams.m
                                             / this->m_solutionParams.macM),
                           static_cast<uint>(this->m_solutionParams.problemParams.n
                                             / this->m_solutionParams.macN)};
                    kernelOptions->streamK = true;
                    return kernelOptions;
                }

                KernelArguments makeArgs(std::shared_ptr<A>       m_dA,
                                         std::shared_ptr<B>       m_dB,
                                         std::shared_ptr<C>       m_dC,
                                         std::shared_ptr<D>       m_dD,
                                         std::shared_ptr<uint8_t> m_dScratch)
                {
                    auto runtimeArgs
                        = DataParallelGEMMSolution<A, B, C, D>::makeArgs(m_dA, m_dB, m_dC, m_dD);

                    runtimeArgs.append(rocRoller::SCRATCH, static_cast<void*>(m_dScratch.get()));

                    // Determine the number of CUs on the device
                    hipDeviceProp_t deviceProperties;
                    AssertFatal(hipGetDeviceProperties(&deviceProperties, 0)
                                == (hipError_t)HIP_SUCCESS);
                    auto numCUs = deviceProperties.multiProcessorCount;

                    // Determine the occupancy for the kernel
                    int occupancy;
                    AssertFatal(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                                    &occupancy,
                                    this->m_kernel->getHipFunction(),
                                    this->m_solutionParams.workgroupSizeX
                                        * this->m_solutionParams.workgroupSizeY,
                                    0)
                                == (hipError_t)HIP_SUCCESS);

                    AssertFatal(this->m_solutionParams.numWGs <= numCUs * occupancy,
                                "StreamK kernel requires that the number of workgroups is not "
                                "greater than the number of compute units * occupancy.");

                    runtimeArgs.append("numWGs", this->m_solutionParams.numWGs);

                    bool logArgs = Log::getLogger()->should_log(spdlog::level::debug);

                    if(logArgs)
                        Log::getLogger()->debug(runtimeArgs.toString());

                    return runtimeArgs;
                }
            };
        }
    }
}
