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
        "${CMAKE_CURRENT_SOURCE_DIR}/CombineShifts.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataFlowTagPropagation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FastArithmetic.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FastMultiplication.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FuseAssociative.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FuseTernary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Identity.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LaunchTimeSubExpressions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerBitfieldValues.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerExponential.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerPRNG.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PositionalArgumentPropagation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RestoreCommandArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Simplify.cpp"
)
