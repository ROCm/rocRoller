/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <rocRoller/rocRoller.hpp>

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /*
         * Nodes (Dimensions)
         */

        struct ROCROLLER_DECLSPEC ForLoop;
        struct ROCROLLER_DECLSPEC Adhoc;
        struct ROCROLLER_DECLSPEC ElementNumber;
        struct ROCROLLER_DECLSPEC Lane;
        struct ROCROLLER_DECLSPEC Linear;
        struct ROCROLLER_DECLSPEC LDS;
        struct ROCROLLER_DECLSPEC MacroTile;
        struct ROCROLLER_DECLSPEC MacroTileIndex;
        struct ROCROLLER_DECLSPEC MacroTileNumber;
        struct ROCROLLER_DECLSPEC SubDimension;
        struct ROCROLLER_DECLSPEC ThreadTile;
        struct ROCROLLER_DECLSPEC ThreadTileIndex;
        struct ROCROLLER_DECLSPEC ThreadTileNumber;
        struct ROCROLLER_DECLSPEC Unroll;
        struct ROCROLLER_DECLSPEC User;
        struct ROCROLLER_DECLSPEC VGPR;
        struct ROCROLLER_DECLSPEC VGPRBlockNumber;
        struct ROCROLLER_DECLSPEC VGPRBlockIndex;
        struct ROCROLLER_DECLSPEC WaveTile;
        struct ROCROLLER_DECLSPEC WaveTileIndex;
        struct ROCROLLER_DECLSPEC WaveTileNumber;
        struct ROCROLLER_DECLSPEC JammedWaveTileNumber;
        struct ROCROLLER_DECLSPEC Wavefront;
        struct ROCROLLER_DECLSPEC Workgroup;
        struct ROCROLLER_DECLSPEC Workitem;

        using Dimension = std::variant<ForLoop,
                                       Adhoc,
                                       ElementNumber,
                                       Lane,
                                       LDS,
                                       Linear,
                                       MacroTile,
                                       MacroTileIndex,
                                       MacroTileNumber,
                                       SubDimension,
                                       ThreadTile,
                                       ThreadTileIndex,
                                       ThreadTileNumber,
                                       Unroll,
                                       User,
                                       VGPR,
                                       VGPRBlockNumber,
                                       VGPRBlockIndex,
                                       WaveTile,
                                       WaveTileIndex,
                                       WaveTileNumber,
                                       JammedWaveTileNumber,
                                       Wavefront,
                                       Workgroup,
                                       Workitem>;

        template <typename T>
        concept CDimension = std::constructible_from<Dimension, T>;

        template <typename T>
        concept CConcreteDimension = (CDimension<T> && !std::same_as<Dimension, T>);
    }
}
