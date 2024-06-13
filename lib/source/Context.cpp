#include <rocRoller/Context.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/CrashKernelGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    Context::Context()
        : m_scratchAllocator(Expression::literal(0u))
    {
    }

    ContextPtr Context::ForDefaultHipDevice(std::string const&   kernelName,
                                            KernelOptions const& kernelOpts)
    {
        int  idx  = -1;
        auto arch = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch(idx);

        return Create(idx, arch, kernelName, kernelOpts);
    }

    ContextPtr Context::ForHipDevice(int                  deviceIdx,
                                     std::string const&   kernelName,
                                     KernelOptions const& kernelOpts)
    {
        auto arch = GPUArchitectureLibrary::getInstance()->GetHipDeviceArch(deviceIdx);

        return Create(deviceIdx, arch, kernelName, kernelOpts);
    }

    ContextPtr Context::ForTarget(GPUArchitectureTarget const& target,
                                  std::string const&           kernelName,
                                  KernelOptions const&         kernelOpts)
    {
        auto arch = GPUArchitectureLibrary::getInstance()->GetArch(target);
        return ForTarget(arch, kernelName, kernelOpts);
    }

    ContextPtr Context::ForTarget(GPUArchitecture const& arch,
                                  std::string const&     kernelName,
                                  KernelOptions const&   kernelOpts)
    {
        return Create(-1, arch, kernelName, kernelOpts);
    }

    void Context::setRandomSeed(int seed)
    {
        m_random = std::make_shared<RandomGenerator>(seed);
    }

    ContextPtr Context::Create(int                    deviceIdx,
                               GPUArchitecture const& arch,
                               std::string const&     kernelName,
                               KernelOptions const&   kernelOpts)
    {
        auto rv = std::make_shared<Context>();

        rv->m_kernelOptions = kernelOpts;

        rv->m_hipDeviceIdx = deviceIdx;
        rv->m_targetArch   = arch;

        for(size_t i = 0; i < rv->m_allocators.size(); i++)
        {
            auto regType = static_cast<Register::Type>(i);
            switch(regType)
            {
            case Register::Type::Accumulator:
                rv->m_allocators[i]
                    = std::make_shared<Register::Allocator>(regType, kernelOpts.maxACCVGPRs);
                break;
            case Register::Type::Vector:
                rv->m_allocators[i]
                    = std::make_shared<Register::Allocator>(regType, kernelOpts.maxVGPRs);
                break;
            case Register::Type::Scalar:
                rv->m_allocators[i]
                    = std::make_shared<Register::Allocator>(regType, kernelOpts.maxSGPRs);
                break;
            default:
                // We do not allocate other types of register representation
                rv->m_allocators[i] = std::make_shared<Register::Allocator>(regType, 0);
                break;
            }
        }

        rv->m_kernel       = std::make_shared<AssemblyKernel>(rv, kernelName);
        rv->m_argLoader    = std::make_shared<ArgumentLoader>(rv->m_kernel);
        rv->m_instructions = std::make_shared<ScheduledInstructions>(rv);
        rv->m_mem          = std::make_shared<MemoryInstructions>(rv);
        rv->m_copier       = std::make_shared<CopyGenerator>(rv);
        rv->m_brancher     = std::make_shared<BranchGenerator>(rv);
        rv->m_crasher      = std::make_shared<CrashKernelGenerator>(rv);
        // Using a seed of zero by default ensures consistent results.
        // Set ROCROLLER_RANDOM_SEED to specify a different seed.
        rv->m_random = std::make_shared<RandomGenerator>(0);

        rv->m_labelAllocator = std::make_shared<LabelAllocator>(kernelName);

        rv->m_assemblyFileName = Settings::getInstance()->get(Settings::AssemblyFile);
        if(rv->m_assemblyFileName.empty())
        {
            rv->m_assemblyFileName = kernelName.size() ? kernelName : "a";
            rv->m_assemblyFileName += "_" + arch.target().toString();
            std::replace(rv->m_assemblyFileName.begin(), rv->m_assemblyFileName.end(), ':', '-');
            rv->m_assemblyFileName += ".s";
        }
        rv->m_ldsAllocator = std::make_shared<LDSAllocator>(
            rv->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize));

        rv->m_observer = Scheduling::createObserver(rv);

        rv->m_registerTagMan = std::make_shared<RegisterTagManager>(rv);
        return rv;
    }

    Context::~Context() = default;

    Register::ValuePtr Context::getVCC()
    {
        if(kernel()->wavefront_size() == 32)
            return std::make_shared<Register::Value>(
                shared_from_this(), Register::Type::VCC_LO, DataType::Bool32, 1);
        else
            return std::make_shared<Register::Value>(
                shared_from_this(), Register::Type::VCC, DataType::Bool64, 1);
    }

    Register::ValuePtr Context::getVCC_LO()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::VCC_LO, DataType::Bool32, 1);
    }

    Register::ValuePtr Context::getVCC_HI()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::VCC_HI, DataType::Bool32, 1);
    }

    Register::ValuePtr Context::getSCC()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::SCC, DataType::Bool, 1);
    }

    Register::ValuePtr Context::getExec()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::EXEC, DataType::Bool32, 1);
    }

    std::ostream& operator<<(std::ostream& stream, ContextPtr const& ctx)
    {
        return stream << "Context " << ctx.get();
    }

    std::shared_ptr<KernelGraph::ScopeManager> Context::getScopeManager() const
    {
        return m_scope;
    }

    void Context::setScopeManager(std::shared_ptr<KernelGraph::ScopeManager> scope)
    {
        m_scope = scope;
    }

    Expression::ExpressionPtr Context::getScratchAmount() const
    {
        return m_scratchAllocator;
    }

    void Context::allocateScratch(Expression::ExpressionPtr size)
    {
        m_scratchAllocator = simplify(m_scratchAllocator + size);
    }

}
