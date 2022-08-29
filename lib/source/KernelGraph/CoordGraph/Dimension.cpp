#include <string>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordGraph/Dimension.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordGraph
    {
        /*
         * Dimension methods
         */
        std::string BaseDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto stag  = "{" + _size + ", " + (output ? "o" : "i") + "}";
            return name() + stag;
        }

        std::string SubDimension::toString() const
        {
            auto _size = size ? rocRoller::Expression::toString(size) : "NA";
            auto _sdim = std::to_string(dim);
            auto stag  = "{" + _sdim + ", " + _size + ", " + (output ? "o" : "i") + "}";
            return name() + stag;
        }

        MacroTileNumber MacroTile::tileNumber(int sdim, bool output) const
        {
            return CoordGraph::MacroTileNumber(
                sdim, Expression::literal(1u), Expression::literal(1u), output);
        }

        MacroTileIndex MacroTile::tileIndex(int sdim, bool output) const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return CoordGraph::MacroTileIndex(
                sdim,
                Expression::literal(static_cast<uint>(sizes.at(sdim))),
                Expression::literal(stride),
                output);
        }

        ThreadTileNumber ThreadTile::tileNumber(int sdim, bool output) const
        {
            return CoordGraph::ThreadTileNumber(
                sdim, Expression::literal(1u), Expression::literal(1u), output);
        }

        ThreadTileIndex ThreadTile::tileIndex(int sdim, bool output) const
        {
            AssertFatal(!sizes.empty(), "ThreadTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return CoordGraph::ThreadTileIndex(
                sdim,
                Expression::literal(static_cast<uint>(sizes.at(sdim))),
                Expression::literal(stride),
                output);
        }

        WaveTileNumber WaveTile::tileNumber(int sdim, bool output) const
        {
            return CoordGraph::WaveTileNumber(
                sdim, Expression::literal(1u), Expression::literal(1u), output);
        }

        WaveTileIndex WaveTile::tileIndex(int sdim, bool output) const
        {
            AssertFatal(!sizes.empty(), "WaveTile doesn't have sizes set.");
            int stride = 1;
            for(int d = sizes.size() - 1; d > sdim; --d)
            {
                AssertFatal(sizes[d] > 0, "Invalid tile size: ", ShowValue(sizes[d]));
                stride = stride * sizes[d];
            }
            return CoordGraph::WaveTileIndex(sdim,
                                             Expression::literal(static_cast<uint>(sizes.at(sdim))),
                                             Expression::literal(stride),
                                             output);
        }

        int MacroTile::elements() const
        {
            AssertFatal(!sizes.empty(), "MacroTile doesn't have sizes set.");
            return product(sizes);
        }
    }
}
