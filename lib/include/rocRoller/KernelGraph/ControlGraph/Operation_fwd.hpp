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
    namespace KernelGraph::ControlGraph
    {
        struct ROCROLLER_DECLSPEC Assign;
        struct ROCROLLER_DECLSPEC Barrier;
        struct ROCROLLER_DECLSPEC ComputeIndex;
        struct ROCROLLER_DECLSPEC ConditionalOp;
        struct ROCROLLER_DECLSPEC AssertOp;
        struct ROCROLLER_DECLSPEC Deallocate;
        struct ROCROLLER_DECLSPEC DoWhileOp;
        struct ROCROLLER_DECLSPEC Exchange;
        struct ROCROLLER_DECLSPEC ForLoopOp;
        struct ROCROLLER_DECLSPEC Kernel;
        struct ROCROLLER_DECLSPEC LoadLDSTile;
        struct ROCROLLER_DECLSPEC LoadLinear;
        struct ROCROLLER_DECLSPEC LoadVGPR;
        struct ROCROLLER_DECLSPEC LoadSGPR;
        struct ROCROLLER_DECLSPEC LoadTiled;
        struct ROCROLLER_DECLSPEC Multiply;
        struct ROCROLLER_DECLSPEC NOP;
        struct ROCROLLER_DECLSPEC Block;
        struct ROCROLLER_DECLSPEC Scope;
        struct ROCROLLER_DECLSPEC SetCoordinate;
        struct ROCROLLER_DECLSPEC StoreLDSTile;
        struct ROCROLLER_DECLSPEC LoadTileDirect2LDS;
        struct ROCROLLER_DECLSPEC StoreLinear;
        struct ROCROLLER_DECLSPEC StoreTiled;
        struct ROCROLLER_DECLSPEC StoreVGPR;
        struct ROCROLLER_DECLSPEC StoreSGPR;
        struct ROCROLLER_DECLSPEC TensorContraction;
        struct ROCROLLER_DECLSPEC UnrollOp;
        struct ROCROLLER_DECLSPEC WaitZero;
        struct ROCROLLER_DECLSPEC SeedPRNG;

        using Operation = std::variant<Assign,
                                       Barrier,
                                       ComputeIndex,
                                       ConditionalOp,
                                       AssertOp,
                                       Deallocate,
                                       DoWhileOp,
                                       Exchange,
                                       ForLoopOp,
                                       Kernel,
                                       LoadLDSTile,
                                       LoadLinear,
                                       LoadTiled,
                                       LoadVGPR,
                                       LoadSGPR,
                                       LoadTileDirect2LDS,
                                       Multiply,
                                       NOP,
                                       Block,
                                       Scope,
                                       SetCoordinate,
                                       StoreLDSTile,
                                       StoreLinear,
                                       StoreTiled,
                                       StoreVGPR,
                                       StoreSGPR,
                                       TensorContraction,
                                       UnrollOp,
                                       WaitZero,
                                       SeedPRNG>;

        template <typename T>
        concept COperation = std::constructible_from<Operation, T>;

        template <typename T>
        concept CConcreteOperation = (COperation<T> && !std::same_as<Operation, T>);
    }
}
