#include <string>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {

        BaseDimension::BaseDimension() noexcept = default;

        BaseDimension::BaseDimension(Expression::ExpressionPtr size,
                                     Expression::ExpressionPtr stride)
            : size(size)
            , stride(stride)
        {
        }

        std::string BaseDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto stag  = "{" + _size + "}";
            return name() + stag;
        }

        Adhoc::Adhoc() = default;

        Adhoc::Adhoc(std::string const&        name,
                     Expression::ExpressionPtr size,
                     Expression::ExpressionPtr stride)
            : BaseDimension(size, stride)
            , m_name(name)
        {
        }

        Adhoc::Adhoc(std::string const& name)
            : Adhoc(name, nullptr, nullptr)
        {
        }

        SubDimension::SubDimension(int const                 dim,
                                   Expression::ExpressionPtr size,
                                   Expression::ExpressionPtr stride)
            : BaseDimension(size, stride)
            , dim(dim)
        {
        }

        SubDimension::SubDimension(int const dim)
            : BaseDimension()
            , dim(dim)
        {
        }

        std::string SubDimension::name() const
        {
            return "SubDimension";
        }

        std::string SubDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto _sdim = std::to_string(dim);
            auto stag  = "{" + _sdim + ", " + _size + "}";
            return name() + stag;
        }

        User::User(std::string const& name)
            : BaseDimension()
            , argumentName(name)
        {
        }

        User::User(std::string const& name, Expression::ExpressionPtr size)
            : BaseDimension(size, Expression::literal(1u))
            , argumentName(name)
        {
        }
        Workgroup::Workgroup(int const dim)
            : SubDimension(dim)
        {
        }

        Workitem::Workitem() = default;

        Workitem::Workitem(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        Unroll::Unroll() = default;

        Unroll::Unroll(uint const usize)
            : BaseDimension(Expression::literal(usize), Expression::literal(1))
        {
        }

        Unroll::Unroll(Expression::ExpressionPtr usize)
            : BaseDimension(usize, Expression::literal(1))
        {
        }

        MacroTile::MacroTile() = default;

        MacroTile::MacroTile(int const rank)
            : BaseDimension()
            , rank(rank)
        {
        }

        MacroTile::MacroTile(std::vector<int> const& sizes,
                             MemoryType              memoryType,
                             std::vector<int> const& subTileSizes)
            : BaseDimension()
            , rank(sizes.size())
            , sizes(sizes)
            , memoryType(memoryType)
            , layoutType(LayoutType::None)
            , subTileSizes(subTileSizes)
        {
        }

        MacroTile::MacroTile(std::vector<int> const& sizes,
                             LayoutType              layoutType,
                             std::vector<int> const& subTileSizes,
                             MemoryType              memoryType)
            : BaseDimension()
            , rank(sizes.size())
            , sizes(sizes)
            , memoryType(memoryType)
            , layoutType(layoutType)
            , subTileSizes(subTileSizes)
        {
            if(this->memoryType == MemoryType::LDS)
                this->memoryType = MemoryType::WAVE_LDS;
            AssertFatal(layoutType != LayoutType::None, "Invalid layout type.");
        }

        std::string MacroTile::toString() const
        {
            if(!sizes.empty())
            {
                std::string _sizes = "{";
                for(int s : sizes)
                    _sizes += std::to_string(s) + ",";
                _sizes[_sizes.length() - 1] = '}';
                return name() + _sizes;
            }

            return BaseDimension::toString();
        }

        MacroTileNumber MacroTile::tileNumber(int sdim) const
        {
            return MacroTileNumber(sdim, Expression::literal(1u), Expression::literal(1u));
        }

        MacroTileIndex MacroTile::tileIndex(int sdim) const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return MacroTileIndex(sdim,
                                  Expression::literal(static_cast<uint>(sizes.at(sdim))),
                                  Expression::literal(stride));
        }

        int MacroTile::elements() const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            return product(sizes);
        }

        ThreadTileIndex::ThreadTileIndex() = default;
        ThreadTileIndex::ThreadTileIndex(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        ThreadTileNumber::ThreadTileNumber() = default;
        ThreadTileNumber::ThreadTileNumber(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        ThreadTile::ThreadTile() = default;

        ThreadTile::ThreadTile(MacroTile const& mac_tile)
            : BaseDimension()
            , rank(mac_tile.rank)
            , sizes(mac_tile.subTileSizes)
        {
            wsizes.resize(rank);
            for(int i = 0; i < rank; ++i)
            {
                wsizes[i] = mac_tile.sizes[i] / sizes[i];
            }
        }

        WaveTile::WaveTile() = default;

        WaveTile::WaveTile(int rank)
            : BaseDimension()
            , rank(rank)
            , layout(LayoutType::None)
        {
        }

        /**
             * Construct WaveTile dimension with fully specified sizes.
             */
        WaveTile::WaveTile(std::vector<int> const& sizes, LayoutType layout)
            : BaseDimension(Expression::literal(product(sizes)), Expression::literal(1u))
            , rank(sizes.size())
            , sizes(sizes)
            , layout(layout)
        {
        }

        WaveTileNumber WaveTile::tileNumber(int sdim) const
        {
            return WaveTileNumber(sdim, Expression::literal(1u), Expression::literal(1u));
        }

        WaveTileIndex WaveTile::tileIndex(int sdim) const
        {
            AssertFatal(!sizes.empty(), "WaveTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return WaveTileIndex(sdim,
                                 Expression::literal(static_cast<uint>(sizes.at(sdim))),
                                 Expression::literal(stride));
        }

        ElementNumber::ElementNumber() = default;
        ElementNumber::ElementNumber(int const dim, Expression::ExpressionPtr size)
            : SubDimension(dim, size, Expression::literal(1u))
        {
        }

        std::string Adhoc::name() const
        {
            if(m_name.empty())
                return "Adhoc";
            else
                return "Adhoc." + m_name;
        }

#define DEFAULT_DIM_NAME(cls)     \
    std::string cls::name() const \
    {                             \
        return #cls;              \
    }

        DEFAULT_DIM_NAME(User);
        DEFAULT_DIM_NAME(Linear);
        DEFAULT_DIM_NAME(Wavefront);
        DEFAULT_DIM_NAME(Lane);
        DEFAULT_DIM_NAME(Workgroup);
        DEFAULT_DIM_NAME(Workitem);
        DEFAULT_DIM_NAME(VGPR);
        DEFAULT_DIM_NAME(VGPRBlockNumber);
        DEFAULT_DIM_NAME(VGPRBlockIndex);
        DEFAULT_DIM_NAME(LDS);
        DEFAULT_DIM_NAME(ForLoop);
        DEFAULT_DIM_NAME(Unroll);
        DEFAULT_DIM_NAME(MacroTileIndex);
        DEFAULT_DIM_NAME(MacroTileNumber);
        DEFAULT_DIM_NAME(MacroTile);
        DEFAULT_DIM_NAME(ThreadTileIndex);
        DEFAULT_DIM_NAME(ThreadTileNumber);
        DEFAULT_DIM_NAME(ThreadTile);
        DEFAULT_DIM_NAME(WaveTileIndex);
        DEFAULT_DIM_NAME(WaveTileNumber);
        DEFAULT_DIM_NAME(WaveTile);
        DEFAULT_DIM_NAME(JammedWaveTileNumber);
        DEFAULT_DIM_NAME(ElementNumber);
    }
}
