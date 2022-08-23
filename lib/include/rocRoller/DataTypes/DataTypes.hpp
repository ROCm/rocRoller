/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
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

#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "DataTypes_BFloat16.hpp"
#include "DataTypes_Half.hpp"
#include "DataTypes_Int8.hpp"
#include "DataTypes_Int8x4.hpp"

#include "../Utilities/Comparison.hpp"

namespace rocRoller
{
    /**
     * \ingroup rocRoller
     * \defgroup DataTypes Data Type Info
     *
     * @brief Definitions and metadata on supported data types.
     */

    enum class DataDirection
    {
        ReadOnly = 0,
        WriteOnly,
        ReadWrite,
        Count
    };

    std::string   ToString(DataDirection dir);
    std::ostream& operator<<(std::ostream& stream, DataDirection dir);

    /**
     * \ingroup DataTypes
     * @{
     */

    /**
     * Data Type
     */
    enum class DataType : int
    {
        Float = 0,
        Double,
        ComplexFloat,
        ComplexDouble,
        Half,
        Halfx2,
        Int8x4,
        Int32,
        Int64,
        BFloat16,
        Int8,
        Raw32,
        UInt32,
        UInt64,
        Bool,
        Bool32,
        Count,
        None = Count
    };

    std::string   ToString(DataType d);
    std::string   TypeAbbrev(DataType d);
    std::ostream& operator<<(std::ostream& stream, DataType const& t);
    std::istream& operator>>(std::istream& stream, DataType& t);

    /**
     * Pointer Type
     */
    enum class PointerType : int
    {
        Value,
        PointerLocal,
        PointerGlobal,
        Buffer,

        Count,
        None = Count
    };

    /**
     * Memory location type; where a buffer is stored.
     */
    enum class MemoryType : int
    {
        Global,
        LDS,
        AGPR,
        VGPR,
        WAVE,

        Count,
        None = Count
    };

    /**
     * Layout of wavetile for MFMA instructions.
     */
    enum class LayoutType : int
    {
        MATRIX_A,
        MATRIX_B,
        MATRIX_ACCUMULATOR,

        Count,
        None = Count
    };

    /**
     * VariableType
     */
    struct VariableType
    {
        constexpr VariableType()
            : dataType()
        {
        }
        constexpr VariableType(VariableType const& v)
            : dataType(v.dataType)
            , pointerType(v.pointerType)
        {
        }
        constexpr VariableType(DataType d)
            : dataType(d)
            , pointerType(PointerType::Value)
        {
        }
        constexpr VariableType(DataType d, PointerType p)
            : dataType(d)
            , pointerType(p)
        {
        }

        DataType    dataType;
        PointerType pointerType = PointerType::Value;

        size_t getElementSize() const;

        inline bool isPointer() const
        {
            return pointerType != PointerType::Value;
        }
        inline bool isGlobalPointer() const
        {
            return pointerType == PointerType::PointerGlobal;
        }

        inline VariableType getDereferencedType() const
        {
            return VariableType(dataType);
        }

        inline VariableType getPointer() const
        {
            assert(pointerType == PointerType::Value);
            return VariableType(dataType, PointerType::PointerGlobal);
        }

        inline constexpr bool operator==(const VariableType& rhs) const
        {
            return (dataType == rhs.dataType) && (pointerType == rhs.pointerType);
        }
        inline bool operator<(const VariableType& rhs) const
        {
            return pointerType < rhs.pointerType
                   || (pointerType == rhs.pointerType && pointerType == PointerType::Value
                       && dataType < rhs.dataType);
        }

        /**
         * Returns a VariableType representing the result of an arithmetic operation
         * between lhs and rhs. Will throw an exception for an invalid combination of inputs.
         *
         * Accepts up to one pointer value and one integral type, or two values of compatible
         * types.
         *
         * Does no conversion between different categories (float, integral, bool).
         */
        static VariableType Promote(VariableType lhs, VariableType rhs);
    };

    std::string   ToString(PointerType const& v);
    std::ostream& operator<<(std::ostream& stream, PointerType const& p);

    std::string   ToString(VariableType const& v);
    std::string   ValueString(VariableType const& v);
    std::string   TypeAbbrev(VariableType const& v);
    std::ostream& operator<<(std::ostream& stream, VariableType const& v);

    /**
     * \ingroup DataTypes
     * \brief Runtime accessible data type metadata
     */
    struct DataTypeInfo
    {
        static DataTypeInfo const& Get(int index);
        static DataTypeInfo const& Get(DataType t);
        static DataTypeInfo const& Get(VariableType const& v);
        static DataTypeInfo const& Get(std::string const& str);

        VariableType variableType;
        std::string  name;
        std::string  abbrev;

        unsigned int elementSize;
        size_t       packing;
        size_t       segmentSize;
        size_t       alignment;

        bool isComplex;
        bool isIntegral;
        bool isSigned;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <typename T>
        static void registerTypeInfo();

        static void addInfoObject(DataTypeInfo const& info);

        static std::map<VariableType, DataTypeInfo> data;
        static std::map<std::string, VariableType>  typeNames;
    };

    /**
     * \ingroup DataTypes
     * \brief Compile-time accessible data type metadata.
     */
    template <typename T>
    struct TypeInfo
    {
    };

    template <typename T,
              DataType    T_DEnum,
              PointerType T_PEnum,
              int         T_Packing,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    struct BaseTypeInfo
    {
        constexpr static VariableType Var = VariableType(T_DEnum, T_PEnum);

        /// Bytes of one element.  May contain multiple segments.
        constexpr static size_t ElementSize = sizeof(T);
        /// Segments per element.
        constexpr static size_t Packing = T_Packing;
        /// Bytes per segment.
        constexpr static size_t SegmentSize = ElementSize / Packing;
        constexpr static size_t Alignment   = alignof(T);

        constexpr static bool IsComplex  = T_IsComplex;
        constexpr static bool IsIntegral = T_IsIntegral;
        constexpr static bool IsSigned   = T_IsSigned;

        static inline std::string Name()
        {
            if(Var.pointerType == PointerType::Value)
                return ToString(Var.dataType);
            else
                return ToString(Var.pointerType);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Var);
        }
    };

    template <typename T,
              DataType    T_DEnum,
              PointerType T_PEnum,
              int         T_Packing,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr VariableType
        BaseTypeInfo<T, T_DEnum, T_PEnum, T_Packing, T_IsComplex, T_IsIntegral, T_IsSigned>::Var;
    template <typename T,
              DataType    T_DEnum,
              PointerType T_PEnum,
              int         T_Packing,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t
        BaseTypeInfo<T, T_DEnum, T_PEnum, T_Packing, T_IsComplex, T_IsIntegral, T_IsSigned>::
            ElementSize;
    template <typename T,
              DataType    T_DEnum,
              PointerType T_PEnum,
              int         T_Packing,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t
        BaseTypeInfo<T, T_DEnum, T_PEnum, T_Packing, T_IsComplex, T_IsIntegral, T_IsSigned>::
            Packing;
    template <typename T,
              DataType    T_DEnum,
              PointerType T_PEnum,
              int         T_Packing,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t
        BaseTypeInfo<T, T_DEnum, T_PEnum, T_Packing, T_IsComplex, T_IsIntegral, T_IsSigned>::
            SegmentSize;

    template <typename T,
              DataType    T_DEnum,
              PointerType T_PEnum,
              int         T_Packing,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr bool
        BaseTypeInfo<T, T_DEnum, T_PEnum, T_Packing, T_IsComplex, T_IsIntegral, T_IsSigned>::
            IsComplex;
    template <typename T,
              DataType    T_DEnum,
              PointerType T_PEnum,
              int         T_Packing,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr bool
        BaseTypeInfo<T, T_DEnum, T_PEnum, T_Packing, T_IsComplex, T_IsIntegral, T_IsSigned>::
            IsIntegral;

    template <>
    struct TypeInfo<float>
        : public BaseTypeInfo<float, DataType::Float, PointerType::Value, 1, false, false, true>
    {
    };

    template <>
    struct TypeInfo<double>
        : public BaseTypeInfo<double, DataType::Double, PointerType::Value, 1, false, false, true>
    {
    };

    template <>
    struct TypeInfo<std::complex<float>> : public BaseTypeInfo<std::complex<float>,
                                                               DataType::ComplexFloat,
                                                               PointerType::Value,
                                                               1,
                                                               true,
                                                               false,
                                                               true>
    {
    };

    template <>
    struct TypeInfo<std::complex<double>> : public BaseTypeInfo<std::complex<double>,
                                                                DataType::ComplexDouble,
                                                                PointerType::Value,
                                                                1,
                                                                true,
                                                                false,
                                                                true>
    {
    };

    template <>
    struct TypeInfo<Int8x4>
        : public BaseTypeInfo<Int8x4, DataType::Int8x4, PointerType::Value, 4, false, true, true>
    {
    };

    template <>
    struct TypeInfo<int32_t>
        : public BaseTypeInfo<int32_t, DataType::Int32, PointerType::Value, 1, false, true, true>
    {
    };

    template <>
    struct TypeInfo<int64_t>
        : public BaseTypeInfo<int64_t, DataType::Int64, PointerType::Value, 1, false, true, true>
    {
    };

    template <>
    struct TypeInfo<Half>
        : public BaseTypeInfo<Half, DataType::Half, PointerType::Value, 1, false, false, true>
    {
    };

    struct Halfx2 : public DistinctType<uint32_t, Halfx2>
    {
    };

    template <>
    struct TypeInfo<Halfx2>
        : public BaseTypeInfo<Halfx2, DataType::Halfx2, PointerType::Value, 2, false, false, true>
    {
    };

    template <>
    struct TypeInfo<BFloat16> : public BaseTypeInfo<BFloat16,
                                                    DataType::BFloat16,
                                                    PointerType::Value,
                                                    1,
                                                    false,
                                                    false,
                                                    true>
    {
    };

    // Enum DataType::Int8 maps to int8_t, struct rocRoller::Int8 is only used for LogTensor now
    template <>
    struct TypeInfo<int8_t>
        : public BaseTypeInfo<int8_t, DataType::Int8, PointerType::Value, 1, false, true, true>
    {
    };

    template <>
    struct TypeInfo<uint32_t>
        : public BaseTypeInfo<uint32_t, DataType::UInt32, PointerType::Value, 1, false, true, false>
    {
    };

    struct Raw32 : public DistinctType<uint32_t, Raw32>
    {
    };

    template <>
    struct TypeInfo<Raw32>
        : public BaseTypeInfo<Raw32, DataType::Raw32, PointerType::Value, 1, false, true, false>
    {
    };

    template <>
    struct TypeInfo<uint64_t>
        : public BaseTypeInfo<uint64_t, DataType::UInt64, PointerType::Value, 1, false, true, false>
    {
    };

    template <>
    struct TypeInfo<bool>
        : public BaseTypeInfo<bool, DataType::Bool, PointerType::Value, 1, false, true, false>
    {
    };

    struct Bool32 : public DistinctType<uint32_t, Bool32>
    {
    };

    template <>
    struct TypeInfo<Bool32>
        : public BaseTypeInfo<Bool32, DataType::Bool32, PointerType::Value, 1, false, false, false>
    {
    };

    struct PointerLocal : public DistinctType<uint32_t, PointerLocal>
    {
    };

    template <>
    struct TypeInfo<PointerLocal> : public BaseTypeInfo<PointerLocal,
                                                        DataType::None,
                                                        PointerType::PointerLocal,
                                                        1,
                                                        false,
                                                        true,
                                                        false>
    {
    };

    struct PointerGlobal : public DistinctType<uint64_t, PointerGlobal>
    {
    };

    template <>
    struct TypeInfo<PointerGlobal> : public BaseTypeInfo<PointerGlobal,
                                                         DataType::None,
                                                         PointerType::PointerGlobal,
                                                         1,
                                                         false,
                                                         true,
                                                         false>
    {
    };

    struct Buffer
    {
        uint32_t desc0;
        uint32_t desc1;
        uint32_t desc2;
        uint32_t desc3;
    };

    template <>
    struct TypeInfo<Buffer>
        : public BaseTypeInfo<Buffer, DataType::None, PointerType::Buffer, 1, false, true, false>
    {
    };

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    template <>
    struct hash<rocRoller::VariableType>
    {
        inline size_t operator()(rocRoller::VariableType const& varType) const
        {
            return rocRoller::hash_combine(static_cast<size_t>(varType.dataType),
                                           static_cast<size_t>(varType.pointerType));
        }
    };

}
