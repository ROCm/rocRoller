#pragma once

#include <functional>

#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    template <typename T>
    inline void ControlToCoordinateMapper::connect(int control, int coordinate, int subDimension)
    {
        connect(control, coordinate, Connections::TypeAndSubDimension{name<T>(), subDimension});
    }

    template <typename T>
    inline void ControlToCoordinateMapper::disconnect(int control, int coordinate, int subDimension)
    {
        disconnect(control, coordinate, Connections::TypeAndSubDimension{name<T>(), subDimension});
    }

    template <typename T>
    inline int ControlToCoordinateMapper::get(int control, int subDimension) const
    {
        return get(control, Connections::TypeAndSubDimension{name<T>(), subDimension});
    }

    inline int ControlToCoordinateMapper::get(int control, Connections::ConnectionSpec conn) const
    {
        auto key = key_type{control, conn};
        auto it  = m_map.find(key);
        if(it == m_map.end())
            return -1;
        return it->second;
    }

    inline int ControlToCoordinateMapper::get(int control, NaryArgument arg) const
    {
        return get(control, Connections::JustNaryArgument{arg});
    }
}
