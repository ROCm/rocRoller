#pragma once

#include <string>

#include "CoordinateEdge_fwd.hpp"

#include <rocRoller/KernelGraph/StructUtils.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /**
         * EdgeType - type of edge in the coordinate transfrom graph.
         * 
         * NOTE: The order of nodes matter!
         * 
         * Traversal routines can be limited to traversing only a
         * particular type of edge.
         */
        enum class EdgeType : int
        {
            CoordinateTransform,
            DataFlow,
            Any,
            Count,
            None = Count
        };

        /*
         * Coordinate transform edges
         *
         * Represents the arithmetic from going to one set of coordinates
         * to another.
         */

        /**
         * DataFlow - used to denote data flow through storage locations.
         */
        RR_EMPTY_STRUCT_WITH_NAME(DataFlow);

        /**
         * Buffer - denotes SRD for MUBUF instructions
         */
        RR_EMPTY_STRUCT_WITH_NAME(Buffer);

        /**
         * Offset - denotes offset between target/increment
         * dimensions.
         *
         * See ComputeIndex.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Offset);

        /**
         * Stride - denotes stride between target/increment
         * dimensions.
         *
         * See ComputeIndex.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Stride);

        /**
         * View - denotes a virtual node with is a view into another
         * node.
         */
        RR_EMPTY_STRUCT_WITH_NAME(View);

        /**
         * Construct MacroTile.
         *
         * Joins SubDimensions to MacroTile during translation and
         * lowering.  Should not persist.
         */
        RR_EMPTY_STRUCT_WITH_NAME(ConstructMacroTile);

        /**
         * Destruct MacroTile.
         *
         * Splits MacroTile into SubDimensions during translation and
         * lowering.  Should not persist.
         */
        RR_EMPTY_STRUCT_WITH_NAME(DestructMacroTile);

        /**
         * Flatten dimensions (row-major, contiguous storage).
         *
         * For example, with input dimensions
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *   J = Dimension(size=n_j, stride=s_j)
         *
         * and output dimension
         *
         *   O = Dimension()
         *
         * the coordinate transform
         *
         *   Flatten(I, J; O)(i, j) = i * n_j + j
         *
         * and the inverse coordinate transform is
         *
         *   Flatten'(O; I, J)(o) = { o / n_j, o % n_j }.
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Flatten);

        /**
         * Forget (drop) dimensions.
         *
         * Used to express a transform where coordinates are "lost"
         * and can't be reconstructed or transformed further.  For
         * example, a scalar VGPR doesn't have a coordinate.
         *
         * Forward and reverse transforms aren't defined.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Forget);

        /**
         * Inherit dimension(s).
         *
         * Used to express a transform where coordinates are inherited
         * from other coordinates.  For example, a scalar VGPR doesn't
         * have a coordinate, but may conceptually inherit coordinates
         * from other dimensions (eg, workgroup and wavefront).
         *
         * The forward coordinate transform returns the destination
         * indexes.  The reverse coordinate transform returns the
         * source indexes.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Inherit);

        /**
         * Join dimensions (forward version of Split).
         *
         * For example, with input dimensions
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *   J = Dimension(size=n_j, stride=s_j)
         *
         * and output dimensions
         *
         *   F = Dimension()
         *
         * the coordinate transform is
         *
         *   Join(I, J; F)(i, j) = i * s_i + j * s_j
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Join);

        /**
         * Make output (subsequent dims should be tagged as output).
         */
        RR_EMPTY_STRUCT_WITH_NAME(MakeOutput);

        /**
         * PassThrough (identity).
         *
         * Forward and reverse transforms are the identity.
         */
        RR_EMPTY_STRUCT_WITH_NAME(PassThrough);

        /**
         * Split a tensor into subdimensions.
         *
         * For example, with input dimensions
         *
         *   F = Dimension()
         *
         * and output dimensions
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *   J = Dimension(size=n_j, stride=s_j)
         *
         * the inverse coordinate transform is
         *
         *   Split'(I, J; F)(i, j) = i * s_i + j * s_j
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Split);

        /**
         * Tile a dimension.
         *
         * For example, with an input dimension of
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *
         * and output dimensions of
         *
         *   B = Dimension(size=null)
         *   T = Dimension(size=64)
         *
         * the coordinate transform of Tile(I; B, T) is
         *
         *   Tile(I; B, T)(i) = { i / 64, i % 64 }
         *
         * and the reverse coordinate transform Tile'(B, T; I) is
         *
         *   Tile'(B, T; I)(b, t) = b * 64 + t
         */
        RR_EMPTY_STRUCT_WITH_NAME(Tile);

        /*
         * Helpers
         */

        inline std::string toString(const CoordinateTransformEdge& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        template <CConcreteCoordinateTransformEdge T>
        inline std::string name(T const& x)
        {
            return x.toString();
        }

        inline std::string name(const CoordinateTransformEdge& x)
        {
            return toString(x);
        }

        inline std::string toString(const DataFlowEdge& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        template <CConcreteDataFlowEdge T>
        inline std::string name(T const& x)
        {
            return x.toString();
        }

        inline std::string name(const DataFlowEdge& x)
        {
            return toString(x);
        }

        inline std::string toString(const Edge& x)
        {
            return std::visit([](const auto& a) { return toString(a); }, x);
        }

        inline std::string name(Edge const& x)
        {
            return std::visit(
                rocRoller::overloaded{[](const CoordinateTransformEdge&) { return "Transform"; },
                                      [](const DataFlowEdge&) { return "DataFlow"; }},
                x);
        }

        template <typename T>
        requires(
            std::constructible_from<
                CoordinateTransformEdge,
                T> || std::constructible_from<DataFlowEdge, T>) inline bool isEdge(const Edge& x)
        {
            if constexpr(std::is_same_v<DataFlowEdge, T>)
            {
                if(std::holds_alternative<DataFlowEdge>(x))
                {
                    return true;
                }
            }
            else if constexpr(std::constructible_from<DataFlowEdge, T>)
            {
                if(std::holds_alternative<DataFlowEdge>(x))
                {
                    if(std::holds_alternative<T>(std::get<DataFlowEdge>(x)))
                        return true;
                }
            }
            else if constexpr(std::is_same_v<CoordinateTransformEdge, T>)
            {
                if(std::holds_alternative<CoordinateTransformEdge>(x))
                {
                    return true;
                }
            }
            else if constexpr(std::constructible_from<CoordinateTransformEdge, T>)
            {
                if(std::holds_alternative<CoordinateTransformEdge>(x))
                {
                    if(std::holds_alternative<T>(std::get<CoordinateTransformEdge>(x)))
                        return true;
                }
            }
            return false;
        }
    }
}
