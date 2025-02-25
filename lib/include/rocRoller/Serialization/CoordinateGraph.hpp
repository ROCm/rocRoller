#pragma once

#include "../DataTypes/DataTypes.hpp"

#include "AssemblyKernel.hpp"
#include "Base.hpp"
#include "Containers.hpp"
#include "Enum.hpp"
#include "HasTraits.hpp"
#include "Variant.hpp"

#include "Hypergraph.hpp"

#include "../Graph/Hypergraph.hpp"
#include "../KernelGraph/CoordinateGraph/CoordinateGraph.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void
                mapping(IO& io, KernelGraph::CoordinateGraph::BaseDimension& dim, Context& ctx)
            {
                iot::mapRequired(io, "size", dim.size);
                iot::mapRequired(io, "stride", dim.stride);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::BaseDimension& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::LDS, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::LDS& lds, Context& ctx)
            {
                iot::mapRequired(io, "size", lds.size);
                iot::mapRequired(io, "stride", lds.stride);
                iot::mapRequired(io, "holdsTransposedTile", lds.holdsTransposedTile);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::LDS& lds)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, lds, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::Adhoc, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::Adhoc& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                // Outputting of `name` should be handled by the variant.

                if(!iot::outputting(io))
                {
                    if constexpr(std::same_as<Context, std::string>)
                    {
                        dim.m_name = ctx;
                    }
                }
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::Adhoc& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::SubDimension, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void
                mapping(IO& io, KernelGraph::CoordinateGraph::SubDimension& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "dim", dim.dim);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::SubDimension& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::User, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::User& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "argumentName", dim.argumentName);
                iot::mapRequired(io, "needsPadding", dim.needsPadding);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::User& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::MacroTile, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::MacroTile& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "rank", dim.rank);
                iot::mapRequired(io, "memoryType", dim.memoryType);
                iot::mapRequired(io, "layoutType", dim.layoutType);

                iot::mapRequired(io, "sizes", dim.sizes);
                iot::mapRequired(io, "subTileSizes", dim.subTileSizes);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::MacroTile& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::ThreadTile, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::ThreadTile& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "rank", dim.rank);

                iot::mapRequired(io, "sizes", dim.sizes);
                iot::mapRequired(io, "wsizes", dim.wsizes);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::ThreadTile& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::WaveTile, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, KernelGraph::CoordinateGraph::WaveTile& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);

                iot::mapRequired(io, "rank", dim.rank);

                iot::mapRequired(io, "sizes", dim.sizes);
                iot::mapRequired(io, "layout", dim.layout);
                iot::mapRequired(io, "vgpr", dim.vgpr);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::WaveTile& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename T, typename IO, typename Context>
        requires(std::constructible_from<KernelGraph::CoordinateGraph::Edge,
                                         T>) struct MappingTraits<T, IO, Context>
            : public EmptyMappingTraits<T, IO, Context>
        {
        };

        template <typename T, typename IO, typename Context>
        requires(std::constructible_from<KernelGraph::CoordinateGraph::Dimension, T>&&
                     std::derived_from<T, KernelGraph::CoordinateGraph::SubDimension>&& T::HasValue
                 == false) struct MappingTraits<T, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, T& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::SubDimension, IO, Context>::mapping(
                    io, dim, ctx);
            }

            static void mapping(IO& io, T& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        template <typename T, typename IO, typename Context>
        requires(
            std::constructible_from<KernelGraph::CoordinateGraph::Dimension, T>&& std::derived_from<
                T,
                KernelGraph::CoordinateGraph::
                    BaseDimension> && !std::derived_from<T, KernelGraph::CoordinateGraph::SubDimension> && T::HasValue == false) struct
            MappingTraits<T, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, T& dim, Context& ctx)
            {
                MappingTraits<KernelGraph::CoordinateGraph::BaseDimension, IO, Context>::mapping(
                    io, dim, ctx);
            }

            static void mapping(IO& io, T& dim)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, dim, ctx);
            }
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::CoordinateTransformEdge>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::CoordinateTransformEdge, IO, Context>
            : public DefaultVariantMappingTraits<
                  KernelGraph::CoordinateGraph::CoordinateTransformEdge,
                  IO,
                  Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::DataFlowEdge>);

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::DataFlowEdge, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::CoordinateGraph::DataFlowEdge,
                                                 IO,
                                                 Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::Edge>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::Edge, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::CoordinateGraph::Edge, IO, Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::Dimension>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::Dimension, IO, Context>
            : public DefaultVariantMappingTraits<KernelGraph::CoordinateGraph::Dimension,
                                                 IO,
                                                 Context>
        {
        };

        static_assert(CNamedVariant<KernelGraph::CoordinateGraph::CoordinateGraph::Element>);
        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::CoordinateGraph::Element, IO, Context>
            : public DefaultVariantMappingTraits<
                  KernelGraph::CoordinateGraph::CoordinateGraph::Element,
                  IO,
                  Context>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<KernelGraph::CoordinateGraph::CoordinateGraph, IO, Context>
        {
            using iot = IOTraits<IO>;
            using HG  = typename KernelGraph::CoordinateGraph::CoordinateGraph::Base;

            static void
                mapping(IO& io, KernelGraph::CoordinateGraph::CoordinateGraph& graph, Context& ctx)
            {
                MappingTraits<HG, IO, Context>::mapping(io, graph, ctx);
            }

            static void mapping(IO& io, KernelGraph::CoordinateGraph::CoordinateGraph& graph)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, graph, ctx);
            }
        };
    }
}
