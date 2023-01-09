
#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
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
    class GPUWaitQueueType
    {
    public:
        enum Value : uint8_t
        {
            VMQueue = 0,
            LGKMSendMsgQueue,
            LGKMDSQueue,
            LGKMSmemQueue,
            EXPQueue,
            VSQueue,
            FinalInstruction,
            Count,
            None = Count,
        };

        GPUWaitQueueType() = default;
        // cppcheck-suppress noExplicitConstructor
        constexpr GPUWaitQueueType(Value input)
            : m_value(input)
        {
        }
        explicit GPUWaitQueueType(std::string const& input)
            : m_value(m_stringMap.at(input))
        {
        }
        explicit constexpr GPUWaitQueueType(uint8_t input)
            : m_value(static_cast<Value>(input))
        {
        }

        operator uint8_t() const
        {
            return static_cast<uint8_t>(m_value);
        }

        std::string ToString() const;

        static std::string ToString(Value);

        struct Hash
        {
            std::size_t operator()(const GPUWaitQueueType& input) const
            {
                return std::hash<uint8_t>()((uint8_t)input.m_value);
            };
        };

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        template <typename T1, typename T2>
        friend struct rocRoller::Serialization::EnumTraits;

    private:
        Value                                               m_value = Value::Count;
        static const std::unordered_map<std::string, Value> m_stringMap;
    };

    class GPUWaitQueue
    {
    public:
        enum Value : uint8_t
        {
            VMQueue = 0,
            LGKMQueue,
            EXPQueue,
            VSQueue,
            Count,
            None = Count,
        };

        GPUWaitQueue() = default;
        // cppcheck-suppress noExplicitConstructor
        constexpr GPUWaitQueue(Value input)
            : m_value(input)
        {
        }

        explicit GPUWaitQueue(std::string const& input)
            : m_value(GPUWaitQueue::m_stringMap[input])
        {
        }
        explicit constexpr GPUWaitQueue(uint8_t input)
            : m_value(static_cast<Value>(input))
        {
        }
        // cppcheck-suppress noExplicitConstructor
        constexpr GPUWaitQueue(GPUWaitQueueType input)
        {
            switch(input)
            {
            case GPUWaitQueueType::VMQueue:
                m_value = Value::VMQueue;
                break;
            case GPUWaitQueueType::LGKMSendMsgQueue:
            case GPUWaitQueueType::LGKMDSQueue:
            case GPUWaitQueueType::LGKMSmemQueue:
                m_value = Value::LGKMQueue;
                break;
            case GPUWaitQueueType::EXPQueue:
                m_value = Value::EXPQueue;
                break;
            case GPUWaitQueueType::VSQueue:
                m_value = Value::VSQueue;
                break;
            default:
                m_value = Value::None;
            }
        }

        operator uint8_t() const
        {
            return static_cast<uint8_t>(m_value);
        }

        std::string ToString() const;

        static std::string ToString(Value);

        struct Hash
        {
            std::size_t operator()(const GPUWaitQueue& input) const
            {
                return std::hash<uint8_t>()((uint8_t)input.m_value);
            };
        };

    private:
        Value                                         m_value = Value::Count;
        static std::unordered_map<std::string, Value> m_stringMap;
    };

    class GPUInstructionInfo
    {
    public:
        GPUInstructionInfo() = default;
        GPUInstructionInfo(std::string const& instruction,
                           int,
                           std::vector<GPUWaitQueueType> const&,
                           int  = 0,
                           bool = false,
                           bool = false);

        std::string                   getInstruction() const;
        int                           getWaitCount() const;
        std::vector<GPUWaitQueueType> getWaitQueues() const;
        int                           getLatency() const;
        bool                          hasImplicitAccess() const;
        bool                          isBranch() const;

        friend std::ostream& operator<<(std::ostream& os, const GPUInstructionInfo& d);

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

    private:
        std::string                   m_instruction = "";
        int                           m_waitCount   = -1;
        std::vector<GPUWaitQueueType> m_waitQueues;
        int                           m_latency        = -1;
        bool                          m_implicitAccess = false;
        bool                          m_isBranch       = false;
    };

    std::string ToString(GPUWaitQueueType);
}

#include "GPUInstructionInfo_impl.hpp"
