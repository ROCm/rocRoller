/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024 Advanced Micro Devices, Inc.
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

#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
    namespace Operations
    {
        struct OperationTag final : public DistinctType<int32_t, OperationTag>
        {
            explicit OperationTag(int value)
                : DistinctType<int32_t, OperationTag>(value)
            {
            }

            OperationTag()
                : OperationTag(-1)
            {
            }

            // Prefix increment operator
            OperationTag& operator++();

            // Postfix increment operator
            OperationTag operator++(int);

            auto operator<=>(OperationTag const&) const = default;

            bool uninitialized() const;
        };
    }
}

#include <rocRoller/Operations/OperationTag_impl.hpp>
