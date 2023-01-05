#include <rocRoller/Context.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
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
    Context::Context() {}

    ContextPtr Context::ForDefaultHipDevice(std::string const& kernelName)
    {
        int  idx  = -1;
        auto arch = GPUArchitectureLibrary::GetDefaultHipDeviceArch(idx);

        return Create(idx, arch, kernelName);
    }

    ContextPtr Context::ForHipDevice(int deviceIdx, std::string const& kernelName)
    {
        auto arch = GPUArchitectureLibrary::GetHipDeviceArch(deviceIdx);

        return Create(deviceIdx, arch, kernelName);
    }

    ContextPtr Context::ForTarget(GPUArchitectureTarget const& target,
                                  std::string const&           kernelName)
    {
        auto arch = GPUArchitectureLibrary::GetArch(target);
        return ForTarget(arch, kernelName);
    }

    ContextPtr Context::ForTarget(GPUArchitecture const& arch, std::string const& kernelName)
    {
        return Create(-1, arch, kernelName);
    }

    void Context::setRandomSeed(int seed)
    {
        m_random = std::make_shared<RandomGenerator>(seed);
    }

    ContextPtr
        Context::Create(int deviceIdx, GPUArchitecture const& arch, std::string const& kernelName)
    {
        auto rv = std::make_shared<Context>();

        rv->m_hipDeviceIdx = deviceIdx;
        rv->m_targetArch   = arch;

        for(size_t i = 0; i < rv->m_allocators.size(); i++)
        {
            auto regType        = static_cast<Register::Type>(i);
            rv->m_allocators[i] = std::make_shared<Register::Allocator>(regType, 256);
        }

        rv->m_kernel       = std::make_shared<AssemblyKernel>(rv, kernelName);
        rv->m_argLoader    = std::make_shared<ArgumentLoader>(rv->m_kernel);
        rv->m_instructions = std::make_shared<ScheduledInstructions>(rv);
        rv->m_mem          = std::make_shared<MemoryInstructions>(rv);
        rv->m_copier       = std::make_shared<CopyGenerator>(rv);
        rv->m_brancher     = std::make_shared<BranchGenerator>(rv);
        rv->m_random       = std::make_shared<RandomGenerator>(0);

        rv->m_regMap = std::make_shared<RegisterHazardMap>();

        rv->m_labelAllocator = std::make_shared<LabelAllocator>(kernelName);

        rv->m_assemblyFileName = Settings::getInstance()->get(Settings::AssemblyFile);
        if(rv->m_assemblyFileName.empty())
        {
            rv->m_assemblyFileName = kernelName.size() ? kernelName : "a";
            rv->m_assemblyFileName += "_" + arch.target().ToString();
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

    std::shared_ptr<Register::Value> Context::getVCC()
    {
        auto name = kernel()->wavefront_size() == 32 ? Register::SpecialType::VCC_LO
                                                     : Register::SpecialType::VCC;
        return Register::Value::Special(name, shared_from_this());
    }

    std::shared_ptr<Register::Value> Context::getVCC_LO()
    {
        return Register::Value::Special(Register::SpecialType::VCC_LO, shared_from_this());
    }

    std::shared_ptr<Register::Value> Context::getVCC_HI()
    {
        return Register::Value::Special(Register::SpecialType::VCC_HI, shared_from_this());
    }

    std::shared_ptr<Register::Value> Context::getSCC()
    {
        return Register::Value::Special(Register::SpecialType::SCC, shared_from_this());
    }

    std::shared_ptr<Register::Value> Context::getExec()
    {
        return Register::Value::Special(Register::SpecialType::EXEC, shared_from_this());
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

}
