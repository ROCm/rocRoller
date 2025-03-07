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

#include <rocRoller/Expression.hpp>

#include <rocRoller/Serialization/Expression.hpp>
#include <rocRoller/Serialization/YAML.hpp>

#ifdef ROCROLLER_USE_LLVM

static_assert(rocRoller::Serialization::
                  CMappedType<rocRoller::Expression::Expression, llvm::yaml::IO, std::string>);
static_assert(
    rocRoller::Serialization::CMappedType<rocRoller::Expression::Expression, llvm::yaml::IO>);
static_assert(rocRoller::Serialization::
                  CMappedType<rocRoller::Expression::ExpressionPtr, llvm::yaml::IO, std::string>);

static_assert(
    rocRoller::Serialization::CMappedType<rocRoller::Expression::ExpressionPtr, llvm::yaml::IO>);
static_assert(
    rocRoller::Serialization::has_SerializationTraits<rocRoller::Expression::ExpressionPtr,
                                                      llvm::yaml::IO>::value);
static_assert(!llvm::yaml::has_FlowTraits<rocRoller::Expression::ExpressionPtr>::value);
#endif

namespace rocRoller
{
    namespace Expression
    {
        std::string toYAML(ExpressionPtr const& expr)
        {
            return Serialization::toYAML(expr);
        }

        ExpressionPtr fromYAML(std::string const& str)
        {
            return Serialization::fromYAML<ExpressionPtr>(str);
        }
    }
}
