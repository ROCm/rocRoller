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

/**
 *
 */

#pragma once

#include <rocRoller/rocRoller.hpp>
#include <string>
#include <variant>

namespace rocRoller
{
    namespace Operations
    {
        struct ROCROLLER_DECLSPEC Tensor;
        struct ROCROLLER_DECLSPEC Scalar;
        struct ROCROLLER_DECLSPEC Literal;
        struct ROCROLLER_DECLSPEC BlockScale;
        struct ROCROLLER_DECLSPEC T_Load_Linear;
        struct ROCROLLER_DECLSPEC T_Load_Scalar;
        struct ROCROLLER_DECLSPEC T_Load_Tiled;
        struct ROCROLLER_DECLSPEC T_Mul;
        struct ROCROLLER_DECLSPEC T_Store_Linear;
        struct ROCROLLER_DECLSPEC T_Store_Tiled;
        struct ROCROLLER_DECLSPEC T_Execute;
        struct ROCROLLER_DECLSPEC Nop;
        struct ROCROLLER_DECLSPEC RandomNumberGenerator;
        using Operation = std::variant<Tensor,
                                       Scalar,
                                       Literal,
                                       BlockScale,
                                       T_Load_Linear,
                                       T_Load_Scalar,
                                       T_Load_Tiled,
                                       T_Mul,
                                       T_Store_Linear,
                                       T_Store_Tiled,
                                       T_Execute,
                                       Nop,
                                       RandomNumberGenerator>;

        template <typename T>
        concept COperation = std::constructible_from<Operation, T>;

        template <typename T>
        concept CConcreteOperation = (COperation<T> && !std::same_as<Operation, T>);

        struct ROCROLLER_DECLSPEC Inputs;
        struct ROCROLLER_DECLSPEC Outputs;
        struct ROCROLLER_DECLSPEC TagVisitor;

        std::string name(Operation const&);

        template <CConcreteOperation T>
        std::string name();

    }
}
