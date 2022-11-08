/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "MetaObserver.hpp"
#include "Scheduling.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        template <CObserver... Types>
        MetaObserver<Types...>::MetaObserver() = default;

        template <CObserver... Types>
        MetaObserver<Types...>::MetaObserver(Tup const& tup)
            : m_tuple(tup)
        {
        }

        template <CObserver... Types>
        MetaObserver<Types...>::~MetaObserver() = default;

        namespace Detail
        {
            template <CObserver T, CObserver... Rest>
            InstructionStatus Peek(Instruction const& inst, T const& obs, Rest const&... rest)
            {
                auto rv = obs.peek(inst);
                if constexpr(sizeof...(rest) > 0)
                {
                    rv.combine(Peek(inst, rest...));
                }
                return rv;
            }

        }

        template <>
        inline InstructionStatus MetaObserver<>::peek(Instruction const& inst) const
        {
            return {};
        }

        template <CObserver... Types>
        InstructionStatus MetaObserver<Types...>::peek(Instruction const& inst) const
        {
            return std::apply([&inst](auto&&... args) { return Detail::Peek(inst, args...); },
                              m_tuple);
        }

        namespace Detail
        {
            template <CObserver T, CObserver... Rest>
            void Modify(Instruction& inst, T const& obs, Rest const&... rest)
            {
                obs.modify(inst);
                if constexpr(sizeof...(rest) > 0)
                {
                    Modify(inst, rest...);
                }
            }
        }

        template <>
        inline void MetaObserver<>::modify(Instruction& inst) const
        {
            return;
        }

        template <CObserver... Types>
        void MetaObserver<Types...>::modify(Instruction& inst) const
        {
            std::apply([&inst](auto&&... args) { return Detail::Modify(inst, args...); }, m_tuple);
        }

        namespace Detail
        {
            template <CObserver T, CObserver... Rest>
            InstructionStatus Observe(Instruction const& inst, T& obs, Rest&... rest)
            {
                auto rv = obs.observe(inst);
                if constexpr(sizeof...(rest) > 0)
                {
                    rv.combine(Observe(inst, rest...));
                }
                return rv;
            }

        }

        template <>
        inline InstructionStatus MetaObserver<>::observe(Instruction const& inst)
        {
            return {};
        }

        template <CObserver... Types>
        InstructionStatus MetaObserver<Types...>::observe(Instruction const& inst)
        {
            return std::apply([&inst](auto&&... args) { return Detail::Observe(inst, args...); },
                              m_tuple);
        }
    }
}
