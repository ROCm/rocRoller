
#pragma once

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

#include "Base.hpp"
#include "Containers.hpp"
#include "Enum.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        const std::string KeyValue            = "Value";
        const std::string KeyMajorVersion     = "MajorVersion";
        const std::string KeyMinorVersion     = "MinorVersion";
        const std::string KeyPointVersion     = "PointVersion";
        const std::string KeySramecc          = "Sramecc";
        const std::string KeyStringRep        = "StringRep";
        const std::string KeyVersionRep       = "VersionRep";
        const std::string KeyLLVMFeaturesRep  = "LLVMFeaturesRep";
        const std::string KeyInstruction      = "Instruction";
        const std::string KeyWaitCount        = "WaitCount";
        const std::string KeyWaitQueues       = "WaitQueues";
        const std::string KeyLatency          = "Latency";
        const std::string KeyImplicitAccess   = "ImplicitAccess";
        const std::string KeyIsBranch         = "IsBranch";
        const std::string KeyISAVersion       = "ISAVersion";
        const std::string KeyInstructionInfos = "InstructionInfos";
        const std::string KeyCapabilities     = "Capabilities";
        const std::string KeyArchitectures    = "Architectures";

        /**
         * GPUWaitQueueType is actually a class that look like an enum, so it is not handled by the
         * generic enum serialization.
         */
        template <>
        struct ScalarTraits<GPUWaitQueueType>
        {
            static std::string output(const GPUWaitQueueType& value)
            {
                return toString(value);
            }

            static void input(std::string const& scalar, GPUWaitQueueType& value)
            {
                value = GPUWaitQueueType(scalar);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUCapability, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUCapability& cap)
            {
                iot::mapRequired(io, KeyValue.c_str(), cap.m_value);
            }

            static void mapping(IO& io, GPUCapability& info, EmptyContext& ctx)
            {
                mapping(io, info);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUArchitectureTarget, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUArchitectureTarget& arch)
            {
                iot::mapRequired(io, KeyMajorVersion.c_str(), arch.m_majorVersion);
                iot::mapRequired(io, KeyMinorVersion.c_str(), arch.m_minorVersion);
                iot::mapRequired(io, KeyPointVersion.c_str(), arch.m_pointVersion);
                iot::mapRequired(io, KeySramecc.c_str(), arch.m_sramecc);
                iot::mapRequired(io, KeyStringRep.c_str(), arch.m_stringRep);
                iot::mapRequired(io, KeyVersionRep.c_str(), arch.m_versionRep);
                iot::mapRequired(io, KeyLLVMFeaturesRep.c_str(), arch.m_llvmFeaturesRep);
            }

            static void mapping(IO& io, GPUArchitectureTarget& arch, EmptyContext& ctx)
            {
                mapping(io, arch);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUInstructionInfo, IO, EmptyContext>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUInstructionInfo& info)
            {
                iot::mapRequired(io, KeyInstruction.c_str(), info.m_instruction);
                iot::mapRequired(io, KeyWaitCount.c_str(), info.m_waitCount);
                iot::mapRequired(io, KeyWaitQueues.c_str(), info.m_waitQueues);
                iot::mapRequired(io, KeyLatency.c_str(), info.m_latency);
                iot::mapRequired(io, KeyImplicitAccess.c_str(), info.m_implicitAccess);
                iot::mapRequired(io, KeyIsBranch.c_str(), info.m_isBranch);
            }

            static void mapping(IO& io, GPUInstructionInfo& info, EmptyContext& ctx)
            {
                mapping(io, info);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUArchitecture, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUArchitecture& arch)
            {
                iot::mapRequired(io, KeyISAVersion.c_str(), arch.m_isaVersion);
                iot::mapRequired(io, KeyInstructionInfos.c_str(), arch.m_instructionInfos);
                iot::mapRequired(io, KeyCapabilities.c_str(), arch.m_capabilities);
            }

            static void mapping(IO& io, GPUArchitecture& arch, EmptyContext& ctx)
            {
                mapping(io, arch);
            }
        };

        template <typename IO>
        struct MappingTraits<GPUArchitecturesStruct, IO, EmptyContext>
        {
            static const bool flow = false;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, GPUArchitecturesStruct& arch)
            {
                iot::mapRequired(io, KeyArchitectures.c_str(), arch.architectures);
            }

            static void mapping(IO& io, GPUArchitecturesStruct& arch, EmptyContext& ctx)
            {
                mapping(io, arch);
            }
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<std::string, GPUInstructionInfo>, IO>
            : public DefaultCustomMappingTraits<std::map<std::string, GPUInstructionInfo>,
                                                IO,
                                                false,
                                                true>
        {
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<GPUCapability, int>, IO>
            : public DefaultCustomMappingTraits<std::map<GPUCapability, int>, IO, false, true>
        {
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<GPUArchitectureTarget, GPUArchitecture>, IO>
            : public DefaultCustomMappingTraits<std::map<GPUArchitectureTarget, GPUArchitecture>,
                                                IO,
                                                false,
                                                true>
        {
        };

        ROCROLLER_SERIALIZE_VECTOR(true, GPUWaitQueueType);
    }
}
