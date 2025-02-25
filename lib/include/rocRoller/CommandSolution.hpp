#pragma once

#include <memory>

#include "AssemblyKernel.hpp"
#include "ExecutableKernel_fwd.hpp"
#include "KernelArguments.hpp"
#include "Operations/Command_fwd.hpp"
#include "Utilities/HIPTimer.hpp"

namespace rocRoller
{
    KernelArguments  getKernelArguments(AssemblyKernelPtr kernel, RuntimeArguments const& args);
    KernelInvocation getKernelInvocation(AssemblyKernelPtr kernel, RuntimeArguments const& args);

    /**
     * CommandParameters - tunable command parameters.
     */
    class CommandParameters
    {
    public:
        CommandParameters();

        /**
         * Set (reset) a dimensions properties.
         */
        void setDimensionInfo(Operations::OperationTag                tag,
                              KernelGraph::CoordinateGraph::Dimension dim);
        std::map<Operations::OperationTag, KernelGraph::CoordinateGraph::Dimension>
            getDimensionInfo() const;

        /**
         * Manually override kernel launch dimensions.
         *
         * TODO remove this.
         */
        void setManualKernelDimension(int dim);
        int  getManualKernelDimension() const;

        /**
         * Manually overrite workgroup sizes.
         *
         * TODO remove this.
         */
        void setManualWorkgroupSize(std::array<unsigned int, 3> const&);
        std::optional<std::array<unsigned int, 3>> getManualWorkgroupSize() const;

        void setManualWavefrontCount(std::pair<uint, uint> wavefrontCounts);
        std::optional<std::pair<uint, uint>> getManualWavefrontCounts() const;

        /**
         * @brief Set the number of wave tiles to execute within a workgroup
         *
         * @param x Number of wave tiles to execute in the x dimension
         * @param y Number of wave tiles to execute in the y dimension
         */
        void setWaveTilesPerWavefront(unsigned int x = 1, unsigned int y = 1);

        /**
         * @brief Get the number of wave tiles to execute within a workgroup
         *
         * @return std::vector<unsigned int>
         */
        std::vector<unsigned int> getWaveTilesPerWavefront() const;

        /**
         * @brief Enable/disable wave-storage strategy.
         *
         * When storing through LDS; the LDS traffic is done following
         * the MFMA accumulator layout intead of straight threads.
         */
        void setSplitStoreTileIntoWaveBlocks(bool);
        bool getSplitStoreTileIntoWaveBlocks() const;

        /**
         * Lowering parameters.
         */
        bool allowAmbiguousMemoryNodes   = false;
        bool enableLongDwordInstructions = true;

        uint numScratchTiles = 0;

        EnumBitset<LayoutType> transposeMemoryAccess = {};

        bool packMultipleElementsInto1VGPR = true;

        unsigned int unrollX   = 0;
        unsigned int unrollY   = 0;
        unsigned int unrollK   = 0;
        bool         fuseLoops = true;

        bool prefetch          = false;
        int  prefetchInFlight  = 1;
        int  prefetchLDSFactor = 0;
        bool prefetchMixMemOps = false;

        bool streamK        = false;
        bool streamKTwoTile = false;

        std::vector<int>  loopOverOutputTilesDimensions    = {};
        std::string       loopOverOutputTilesTopLoop       = XLOOP;
        std::vector<uint> loopOverOutputTilesCoordSizes    = {};
        uint              loopOverOutputTilesIteratedTiles = 0;

    private:
        std::map<Operations::OperationTag, KernelGraph::CoordinateGraph::Dimension> m_dimInfo;
        std::optional<std::array<unsigned int, 3>>                                  m_workgroupSize;
        std::optional<std::pair<uint, uint>> m_wavefrontCounts;

        int m_kernelDimension = 0;

        std::vector<unsigned int> m_waveTilesPerWavefront;

        bool m_splitStoreTileIntoWaveBlocks = true;
    };

    /**
     * CommandLaunchParameters - manual kernel launch parameters.
     *
     * TODO: Remove this!
     */
    class CommandLaunchParameters
    {
    public:
        CommandLaunchParameters() = default;

        /**
         * Manually override work item counts.
         */
        void setManualWorkitemCount(std::array<Expression::ExpressionPtr, 3> const&);
        std::optional<std::array<Expression::ExpressionPtr, 3>> getManualWorkitemCount() const;

    private:
        std::optional<std::array<Expression::ExpressionPtr, 3>> m_workitemCount;
    };

    class CommandKernel
    {
    public:
        CommandKernel() = default;

        /**
         * @brief Create a CommandKernel based on a Command object.
         *
         * Callers should set the context, parameters, and then call
         * `generateKernel`.
         */
        CommandKernel(CommandPtr command, std::string kernelName);

        /**
         * @brief Set context.
	 *
	 * This must be called before `generateKernel`.
         */
        void setContext(ContextPtr);

        /**
         * @brief CommandParameters applied to the kernel graph after
         * translation from a command to a graph but before all
         * lowering stages.
         */
        void                 setCommandParameters(CommandParametersPtr commandParams);
        CommandParametersPtr getCommandParameters() const;

        /**
         * @brief Generates the kernel, assembles it, and records the
         * info needed to launch the kernel, given command arguments.
         */
        void generateKernel();

        void loadKernel();

        void addPredicate(Expression::ExpressionPtr expression);
        bool matchesPredicates(/* args */) const;

        /**
	 * @brief Set (manual) launch parameters.
	 */
        void setLaunchParameters(CommandLaunchParametersPtr);

        /**
         * Load (and compile) a kernel from the assembly source file `fileName`.
         */
        void loadKernelFromAssembly(std::string const& fileName, std::string const& kernelName);

        /**
         * @brief Determines launch bounds and arguments, and launches the kernel.
         *
         * @param args The runtime arguments being passed to the kernel
        */
        void launchKernel(RuntimeArguments const& args);

        /**
         * @brief Determines launch bounds and arguments, and launches the kernel.
         *
         * @param args The runtime arguments being passed to the kernel
         * @param timer HIPTimer that will record how long the kernel took to execute
         * @param iteration Iteration number within the timer
        */
        void launchKernel(RuntimeArguments const&   args,
                          std::shared_ptr<HIPTimer> timer,
                          int                       iteration);

        KernelGraph::KernelGraph getKernelGraph() const;

        std::string getInstructions() const;

        ContextPtr getContext();

        /**
         * Determines kernel arguments for a particular invocation.
         */
        KernelArguments getKernelArguments(RuntimeArguments const& args);
        /**
         * Determines launch bounds for a particular invocation.
         */
        KernelInvocation getKernelInvocation(RuntimeArguments const& args);

        /**
         * @brief Returns the total number of bytes required for scratch space
         *
         * If this value is greather than 0, the user is required to allocate this
         * amount of device memory and pass it into the kernel.
         *
         * @return size_t
         */
        size_t scratchSpaceRequired() const;

        /**
         * @brief Returns the hipFunction for the kernel
        */
        hipFunction_t getHipFunction() const;

    private:
        CommandPtr  m_command;
        std::string m_name;

        std::vector<Expression::ExpressionPtr> m_predicates;

        KernelGraph::KernelGraph          m_kernelGraph;
        ContextPtr                        m_context;
        std::shared_ptr<ExecutableKernel> m_executableKernel;
        CommandParametersPtr              m_commandParameters;
        CommandLaunchParametersPtr        m_launchParameters;

        Generator<Instruction> commandComments();

        void generateKernelGraph(std::string name);
        void generateKernelSource();

        Generator<Instruction> kernelInstructions();

        void assembleKernel();
    };

    class CommandSolution
    {
    public:
        CommandSolution(CommandPtr command);

        void appendKernel(std::shared_ptr<CommandKernel> kernel);
        std::vector<std::shared_ptr<CommandKernel>> const& kernels() const;

        void generateKernels();

        void launchKernels();
    };
}
