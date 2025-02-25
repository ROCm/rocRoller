
#include <algorithm>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    CommandParameters::CommandParameters()
        : m_waveTilesPerWavefront({1, 1})
    {
    }

    void CommandParameters::setDimensionInfo(Operations::OperationTag                tag,
                                             KernelGraph::CoordinateGraph::Dimension dim)
    {
        m_dimInfo[tag] = dim;
    }

    std::map<Operations::OperationTag, KernelGraph::CoordinateGraph::Dimension>
        CommandParameters::getDimensionInfo() const
    {
        return m_dimInfo;
    }

    int CommandParameters::getManualKernelDimension() const
    {
        return m_kernelDimension;
    }

    void CommandParameters::setManualKernelDimension(int dim)
    {
        m_kernelDimension = dim;
    }

    void CommandParameters::setManualWorkgroupSize(std::array<unsigned int, 3> const& v)
    {
        m_workgroupSize = v;
    }

    std::optional<std::array<unsigned int, 3>> CommandParameters::getManualWorkgroupSize() const
    {
        return m_workgroupSize;
    }

    void CommandParameters::setManualWavefrontCount(std::pair<uint, uint> wavefrontCounts)
    {
        m_wavefrontCounts = wavefrontCounts;
    }

    std::optional<std::pair<uint, uint>> CommandParameters::getManualWavefrontCounts() const
    {
        return m_wavefrontCounts;
    }

    void CommandLaunchParameters::setManualWorkitemCount(
        std::array<Expression::ExpressionPtr, 3> const& v)
    {
        m_workitemCount = v;
    }

    std::optional<std::array<Expression::ExpressionPtr, 3>>
        CommandLaunchParameters::getManualWorkitemCount() const
    {
        return m_workitemCount;
    }

    void CommandParameters::setWaveTilesPerWavefront(unsigned int x, unsigned int y)
    {
        m_waveTilesPerWavefront[0] = x;
        m_waveTilesPerWavefront[1] = y;
    }

    std::vector<unsigned int> CommandParameters::getWaveTilesPerWavefront() const
    {
        return m_waveTilesPerWavefront;
    }

    void CommandParameters::setSplitStoreTileIntoWaveBlocks(bool x)
    {
        m_splitStoreTileIntoWaveBlocks = x;
    }

    bool CommandParameters::getSplitStoreTileIntoWaveBlocks() const
    {
        return m_splitStoreTileIntoWaveBlocks;
    }

    KernelArguments CommandKernel::getKernelArguments(RuntimeArguments const& args)
    {
        TIMER(t, "CommandKernel::getKernelArguments");

        bool            log = Log::getLogger()->should_log(spdlog::level::debug);
        KernelArguments rv(log);

        auto const& argStructs = m_context->kernel()->arguments();

        rv.reserve(m_context->kernel()->argumentSize(), argStructs.size());

        for(auto& arg : argStructs)
        {
            auto value = Expression::evaluate(arg.expression, args);

            if(variableType(value) != arg.variableType)
            {
                throw std::runtime_error(concatenate("Evaluated argument type ",
                                                     variableType(value),
                                                     " doesn't match expected type ",
                                                     arg.variableType,
                                                     ", Expression: ",
                                                     toString(arg.expression),
                                                     ", name: ",
                                                     arg.name));
            }

            rv.append(arg.name, value);
        }

        return rv;
    }

    KernelInvocation CommandKernel::getKernelInvocation(RuntimeArguments const& args)
    {
        TIMER(t, "CommandKernel::getKernelInvocation");

        KernelInvocation rv;

        rv.workgroupSize = m_context->kernel()->workgroupSize();

        auto const& workitems = m_context->kernel()->workitemCount();
        if(workitems[0])
            rv.workitemCount[0] = getUnsignedInt(evaluate(workitems[0], args));
        if(workitems[1])
            rv.workitemCount[1] = getUnsignedInt(evaluate(workitems[1], args));
        if(workitems[2])
            rv.workitemCount[2] = getUnsignedInt(evaluate(workitems[2], args));

        auto const& sharedMem = m_context->kernel()->dynamicSharedMemBytes();
        if(sharedMem)
            rv.sharedMemBytes = getUnsignedInt(evaluate(sharedMem, args));

        return rv;
    }

    void CommandKernel::setContext(ContextPtr context)
    {
        m_context = context;
    }

    CommandKernel::CommandKernel(CommandPtr command, std::string kernelName)
        : m_command(command)
        , m_name(kernelName)
    {
    }

    KernelGraph::KernelGraph CommandKernel::getKernelGraph() const
    {
        return m_kernelGraph;
    }

    hipFunction_t CommandKernel::getHipFunction() const
    {
        return m_executableKernel->getHipFunction();
    }

    std::string CommandKernel::getInstructions() const
    {
        return m_context->instructions()->toString();
    }

    Generator<Instruction> CommandKernel::commandComments()
    {
        co_yield Instruction::Comment(m_command->toString());
        co_yield Instruction::Comment(m_command->argInfo());
    }

    void CommandKernel::generateKernelGraph(std::string name)
    {
        TIMER(t, "CommandKernel::generateKernelGraph");

        AssertFatal(m_context);

        KernelGraph::ConstraintStatus check;

        if(!m_commandParameters)
            m_commandParameters = std::make_shared<CommandParameters>();

        // TODO: Determine the correct kernel dimensions
        if(m_commandParameters->getManualKernelDimension() > 0)
            m_context->kernel()->setKernelDimensions(
                m_commandParameters->getManualKernelDimension());
        else
            m_context->kernel()->setKernelDimensions(1);

        // TODO: Determine the correct work group size
        if(m_commandParameters->getManualWorkgroupSize())
            m_context->kernel()->setWorkgroupSize(*m_commandParameters->getManualWorkgroupSize());
        else
            m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto zero = std::make_shared<Expression::Expression>(0u);
        m_context->kernel()->setDynamicSharedMemBytes(zero);

        if(!m_context->kernelOptions().lazyAddArguments)
            m_context->kernel()->addCommandArguments(m_command->getArguments());

        m_kernelGraph = KernelGraph::translate(m_command);

        if(Settings::getInstance()->get(Settings::LogGraphs))
            Log::debug("CommandKernel::generateKernel: post translate: {}",
                       m_kernelGraph.toDOT(false, "CommandKernel::generateKernel: post translate"));

        if(Settings::getInstance()->get(Settings::EnforceGraphConstraints))
        {
            check = m_kernelGraph.checkConstraints();
            AssertFatal(
                check.satisfied,
                concatenate("CommandKernel::generateKernel: post translate:\n", check.explanation));
        }

        std::vector<KernelGraph::GraphTransformPtr> transforms;

        transforms.push_back(std::make_shared<KernelGraph::OrderMemory>(
            !m_commandParameters->allowAmbiguousMemoryNodes));
        transforms.push_back(std::make_shared<KernelGraph::UpdateParameters>(m_commandParameters));
        transforms.push_back(std::make_shared<KernelGraph::LowerLinear>(m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::LowerTile>(m_commandParameters, m_context));
        transforms.push_back(
            std::make_shared<KernelGraph::LowerTensorContraction>(m_commandParameters, m_context));

        // TODO: remove the condition by making ConstantPropagation and Streamk work simultaneously
        if(!m_commandParameters->streamK)
        {
            transforms.push_back(std::make_shared<KernelGraph::ConstantPropagation>());
        }

        transforms.push_back(std::make_shared<KernelGraph::FuseExpressions>());
        if(m_commandParameters->streamK)
        {
            Expression::ExpressionPtr numWGsExpr;
            {
                auto arguments = m_command->getArguments();
                auto it        = std::find_if(arguments.cbegin(), arguments.cend(), [](auto x) {
                    return x->name() == rocRoller::NUMWGS;
                });
                AssertFatal(it != arguments.cend(),
                            "Can not find numWGs Command argument required for StreamK kernels.");
                numWGsExpr = std::make_shared<Expression::Expression>(*it);
            }

            transforms.push_back(std::make_shared<KernelGraph::AddStreamK>(
                m_commandParameters->loopOverOutputTilesDimensions,
                rocRoller::XLOOP,
                rocRoller::KLOOP,
                m_commandParameters->streamKTwoTile,
                numWGsExpr,
                m_commandParameters,
                m_context));
        }
        else if(!m_commandParameters->loopOverOutputTilesDimensions.empty())
        {
            transforms.push_back(std::make_shared<KernelGraph::LoopOverTileNumbers>(
                m_commandParameters->loopOverOutputTilesDimensions,
                m_commandParameters->loopOverOutputTilesCoordSizes,
                m_commandParameters->loopOverOutputTilesIteratedTiles,
                m_commandParameters->loopOverOutputTilesTopLoop,
                m_context));
        }
        transforms.push_back(std::make_shared<KernelGraph::ConnectWorkgroups>());
        transforms.push_back(
            std::make_shared<KernelGraph::UnrollLoops>(m_commandParameters, m_context));
        if(m_commandParameters->fuseLoops)
        {
            transforms.push_back(std::make_shared<KernelGraph::FuseLoops>());
        }
        transforms.push_back(std::make_shared<KernelGraph::OrderEpilogueBlocks>());
        transforms.push_back(std::make_shared<KernelGraph::AddLDS>(m_commandParameters, m_context));
        transforms.push_back(std::make_shared<KernelGraph::CleanLoops>());
        transforms.push_back(std::make_shared<KernelGraph::AddF6LDSPadding>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::AddComputeIndex>());
        transforms.push_back(std::make_shared<KernelGraph::AddConvert>());
        transforms.push_back(std::make_shared<KernelGraph::AddPRNG>(m_context));
        transforms.push_back(std::make_shared<KernelGraph::AddDeallocate>());
        transforms.push_back(std::make_shared<KernelGraph::InlineIncrements>());
        transforms.push_back(std::make_shared<KernelGraph::Simplify>());
        transforms.push_back(std::make_shared<KernelGraph::CleanArguments>(m_context, m_command));
        transforms.push_back(
            std::make_shared<KernelGraph::UpdateWavefrontParameters>(m_commandParameters));
        transforms.push_back(std::make_shared<KernelGraph::SetWorkitemCount>(m_context));

        for(auto& t : transforms)
        {
            m_kernelGraph = m_kernelGraph.transform(t);
        }
    }

    Generator<Instruction> CommandKernel::kernelInstructions()
    {
        co_yield commandComments();
        co_yield m_context->kernel()->preamble();
        co_yield m_context->kernel()->prolog();

        co_yield KernelGraph::generate(m_kernelGraph, m_context->kernel());

        co_yield m_context->kernel()->postamble();
        co_yield m_context->kernel()->amdgpu_metadata();
    }

    void CommandKernel::generateKernelSource()
    {
        TIMER(t, "CommandKernel::generateKernelSource");
        m_context->kernel()->setKernelGraphMeta(
            std::make_shared<KernelGraph::KernelGraph>(m_kernelGraph));
        m_context->kernel()->setCommandMeta(m_command);

        m_context->schedule(kernelInstructions());
    }

    std::vector<char> CommandKernel::assembleKernel()
    {
        TIMER(t, "CommandKernel::assembleKernel");

        return m_context->instructions()->assemble();
    }

    void CommandKernel::loadKernel()
    {
        TIMER(t, "CommandKernel::loadKernel");

        if(!m_executableKernel)
            m_executableKernel = m_context->instructions()->getExecutableKernel();
    }

    void CommandKernel::generateKernel()
    {
        TIMER(t, "CommandKernel::generateKernel");

        if(m_command)
        {
            generateKernelGraph(m_name);
            generateKernelSource();
        }
        else
        {
            // Probably from a unit test.  The context should contain
            // scheduled instructions already.
        }
    }

    void CommandKernel::addPredicate(Expression::ExpressionPtr expression)
    {
        m_predicates.push_back(expression);
    }

    bool CommandKernel::matchesPredicates(RuntimeArguments const&   args,
                                          spdlog::level::level_enum level) const
    {
        bool retVal = true;
        for(auto predicate : m_predicates)
        {
            try
            {
                if(!std::get<bool>(Expression::evaluate(predicate, args)))
                {
                    retVal = false;
                    std::string comment;
                    if(!(comment = Expression::getComment(predicate)).empty())
                    {
                        Log::log(level, "Predicate mismatch for {}: {}", m_name, comment);
                    }
                    else
                    {
                        Log::log(level,
                                 "Predicate {} for {} is false.",
                                 Expression::toString(predicate),
                                 m_name);
                    }
                }
            }
            catch(std::bad_variant_access err)
            {
                Throw<FatalError>(
                    "Predicate ", Expression::toString(predicate), " does not evaluate to a bool");
            }
        }

        return retVal;
    }

    void CommandKernel::setCommandParameters(CommandParametersPtr commandParams)
    {
        m_commandParameters = commandParams;
    }

    CommandParametersPtr CommandKernel::getCommandParameters() const
    {
        return m_commandParameters;
    }

    //
    // 2024-11-05: This is only used in a few tests.  Please see the
    // note in Command::createWorkItemCount.
    //
    void CommandKernel::setLaunchParameters(CommandLaunchParametersPtr launch)
    {
        m_launchParameters = launch;
    }

    void CommandKernel::launchKernel(RuntimeArguments const& args)
    {
        launchKernel(args, nullptr, 0);
    }

    void CommandKernel::launchKernel(RuntimeArguments const&   args,
                                     std::shared_ptr<HIPTimer> timer,
                                     int                       iteration)
    {
        TIMER(t, "CommandKernel::launchKernel");

        AssertFatal(m_context, "Unable to launch kernel: CommandKernel must have a Context.");
        AssertFatal(m_context->kernel(),
                    "Unable to launch kernel: Context must have an AssemblyKernel.");
        AssertFatal(matchesPredicates(args, spdlog::level::err),
                    "Unable to launch kernel: all CommandKernel predicates must match.");

        if(m_launchParameters)
        {
            if(m_launchParameters->getManualWorkitemCount())
                m_context->kernel()->setWorkitemCount(
                    *m_launchParameters->getManualWorkitemCount());
        }

        auto kargs = getKernelArguments(args);
        auto inv   = getKernelInvocation(args);

        loadKernel();

        m_executableKernel->executeKernel(kargs, inv, timer, iteration);
    }

    void CommandKernel::loadKernelFromAssembly(const std::string& fileName,
                                               const std::string& kernelName)
    {
        AssertFatal(m_context);
        AssertFatal(m_context->kernel());

        m_executableKernel = std::make_shared<ExecutableKernel>();
        m_executableKernel->loadKernelFromFile(
            fileName, kernelName, m_context->targetArchitecture().target());
    }

    AssemblyKernelPtr CommandKernel::loadKernelFromCodeObject(const std::string& fileName,
                                                              const std::string& kernelName)
    {
        AssertFatal(m_context);

        m_executableKernel = std::make_shared<ExecutableKernel>();
        m_executableKernel->loadKernelFromCodeObjectFile(
            fileName, kernelName, m_context->targetArchitecture().target());

        auto yaml   = readMetaDataFromCodeObject(fileName);
        auto kernel = AssemblyKernels::fromYAML(yaml).kernels[0];

        // XXX Instead of adding `setKernel`, should the context load from a code object?
        auto kernelPtr = std::make_shared<AssemblyKernel>(kernel);
        m_context->setKernel(kernelPtr);
        return kernelPtr;
    }

    ContextPtr CommandKernel::getContext()
    {
        return m_context;
    }

    size_t CommandKernel::scratchSpaceRequired(RuntimeArguments const& args) const
    {
        auto amount = m_context->getScratchAmount();

        auto times = evaluationTimes(amount);
        AssertFatal(times[Expression::EvaluationTime::Translate]
                        || times[Expression::EvaluationTime::KernelLaunch],
                    "Unable to evaluate the scratch space required",
                    ShowValue(toString(amount)));

        return getUnsignedInt(evaluate(amount, args));
    }
}
