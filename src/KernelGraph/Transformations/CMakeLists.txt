################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the Software), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/AddComputeIndex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddConvert.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddDeallocate.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddDirect2LDS.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddF6LDSPadding.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddLDS.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddPRNG.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddPrefetch.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddStreamK.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddTransposeLoadCT.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AliasDataFlowTags.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CleanArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CleanLoops.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ConnectWorkgroups.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ConstantPropagation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FuseExpressions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FuseLoops.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/IdentifyParallelDimensions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/InlineIncrements.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/InlineInits.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraph_loop.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LoadPacked.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LoopOverTileNumbers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerLinear.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerTensorContraction.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerTile.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/NopExtraScopes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/OrderEpilogueBlocks.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/OrderMemory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/OrderMultiplyNodes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PrefetchScale.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RemoveDuplicates.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Simplify.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SwizzleScale.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UnrollLoops.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UpdateParameters.cpp"
)
