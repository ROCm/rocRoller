/**
 * Copyright 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "InstructionValues/Register_fwd.hpp"
#include "Operations/CommandArgument_fwd.hpp"

#include "DataTypes/DataTypes.hpp"

namespace rocRoller
{
    class KernelArguments
    {
    public:
        KernelArguments(bool log = true);
        virtual ~KernelArguments();

        void reserve(size_t bytes, size_t count);

        template <typename T>
        void append(std::string const& name, T value);

        template <typename T>
        void appendUnbound(std::string const& name);

        template <typename T>
        void bind(std::string const& name, T value);

        bool isFullyBound() const;

        bool                        log() const;
        void const*                 data() const;
        size_t                      size() const;
        std::vector<uint8_t> const& dataVector() const;
        RuntimeArguments            runtimeArguments() const;

        friend std::ostream& operator<<(std::ostream& stream, const KernelArguments& t);
        friend class const_iterator;

        using ArgPair = std::pair<void const*, size_t>;

        class const_iterator
        {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type        = ArgPair const;
            using difference_type   = std::ptrdiff_t;
            using pointer           = ArgPair const*;
            using reference         = ArgPair const&;

            const_iterator(KernelArguments const& args);
            const_iterator(KernelArguments const& args, std::string const& name);
            const_iterator(const const_iterator& other) = default;
            const_iterator& operator++();
            const_iterator  operator++(int);
            bool            operator==(const const_iterator& rhs) const;
            bool            operator!=(const const_iterator& rhs) const;
            ArgPair const&  operator*() const;
            ArgPair const*  operator->() const;
            void            reset();
            template <typename T>
            operator T() const;

        private:
            void assignCurrentArg();

            std::vector<std::string>::const_iterator m_currentArg;
            KernelArguments const&                   m_args;
            ArgPair                                  m_value;
        };

        const_iterator begin() const;
        const_iterator end() const;

    private:
        enum
        {
            ArgOffset,
            ArgSize,
            ArgBound,
            ArgString,
            NumArgFields
        };
        using Arg = std::tuple<size_t, size_t, bool, std::string>;
        static_assert(std::tuple_size<Arg>::value == NumArgFields,
                      "Enum for fields of Arg tuple doesn't match size of tuple.");

        void alignTo(size_t alignment);

        template <typename T>
        void append(std::string const& name, T value, bool bound);

        template <typename T>
        std::string stringForValue(T value, bool bound);

        void appendRecord(std::string const& name, Arg info);

        template <typename T>
        void writeValue(size_t offset, T value);

        std::vector<uint8_t> m_data;

        std::vector<std::string>             m_names;
        std::unordered_map<std::string, Arg> m_argRecords;

        bool m_log;
    };

    KernelArguments::const_iterator begin(KernelArguments const&);
    KernelArguments::const_iterator end(KernelArguments const&);

} // namespace rocRoller

#include "KernelArguments_impl.hpp"
