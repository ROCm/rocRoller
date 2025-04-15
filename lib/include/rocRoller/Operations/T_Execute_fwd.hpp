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

#include <string>
#include <variant>

namespace rocRoller
{
    namespace Operations
    {
        struct ROCROLLER_DECLSPEC T_Execute;
        struct ROCROLLER_DECLSPEC E_Unary;
        struct ROCROLLER_DECLSPEC E_Binary;
        struct ROCROLLER_DECLSPEC E_Ternary;
        struct ROCROLLER_DECLSPEC E_Neg;
        struct ROCROLLER_DECLSPEC E_Abs;
        struct ROCROLLER_DECLSPEC E_RandomNumber;
        struct ROCROLLER_DECLSPEC E_Not;
        struct ROCROLLER_DECLSPEC E_Cvt;
        struct ROCROLLER_DECLSPEC E_StochasticRoundingCvt;
        struct ROCROLLER_DECLSPEC E_Add;
        struct ROCROLLER_DECLSPEC E_Sub;
        struct ROCROLLER_DECLSPEC E_Mul;
        struct ROCROLLER_DECLSPEC E_Div;
        struct ROCROLLER_DECLSPEC E_And;
        struct ROCROLLER_DECLSPEC E_Or;
        struct ROCROLLER_DECLSPEC E_GreaterThan;
        struct ROCROLLER_DECLSPEC E_Conditional;

        using XOp = std::variant<E_Neg,
                                 E_Abs,
                                 E_Not,
                                 E_RandomNumber,
                                 E_Cvt,
                                 E_StochasticRoundingCvt,
                                 E_Add,
                                 E_Sub,
                                 E_Mul,
                                 E_Div,
                                 E_And,
                                 E_Or,
                                 E_GreaterThan,
                                 E_Conditional>;

        template <typename T>
        concept CXOp = requires()
        {
            requires std::constructible_from<XOp, T>;
            requires !std::same_as<XOp, T>;
        };

        std::string name(XOp const&);

        template <CXOp T>
        std::string name();
    }
}
