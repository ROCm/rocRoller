/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2023 Advanced Micro Devices, Inc.
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
#include "../Utilities/Error.hpp"

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

    std::string   toString(DataDirection dir);
    std::ostream& operator<<(std::ostream& stream, DataDirection dir);

    /**
     * \ingroup DataTypes
     * @{
     */

    /**
     * Data Type
     *
     * A value of `None` generically means any/unspecified/deferred.
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
        Int8,
        Int16,
        Int32,
        Int64,
        BFloat16,
        Raw32,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Bool,
        Bool32,
        Count,
        None = Count /**< Represents: any, unknown/unspecified, or a deferred type. */
    };

    std::string   toString(DataType d);
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
        WAVE_LDS,

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

    std::string   toString(LayoutType l);
    std::ostream& operator<<(std::ostream& stream, LayoutType l);

    enum class NaryArgument : int
    {
        DEST = 0,
        LHS,
        RHS,

        Count,
        None = Count
    };

    std::string   toString(NaryArgument n);
    std::ostream& operator<<(std::ostream& stream, NaryArgument n);

    inline constexpr DataType getIntegerType(bool isSigned, int sizeBytes)
    {
        if(isSigned)
        {
            switch(sizeBytes)
            {
            case 1:
                return DataType::Int8;
            case 2:
                return DataType::Int16;
            case 4:
                return DataType::Int32;
            case 8:
                return DataType::Int64;
            }
        }
        else
        {
            switch(sizeBytes)
            {
            case 1:
                return DataType::UInt8;
            case 2:
                return DataType::UInt16;
            case 4:
                return DataType::UInt32;
            case 8:
                return DataType::UInt64;
            }
        }

        auto name = isSigned ? "signed" : "unsigned";

        Throw<FatalError>("No enumeration for ", name, " integer with size ", sizeBytes, " bytes.");

        // cppcheck doesn't seem to notice that Throw<>() is marked [[noreturn]] so it will
        // complain if this isn't here.
        return DataType::None;
    }

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
        // cppcheck-suppress noExplicitConstructor
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

        inline DataType getArithmeticType() const
        {
            if(pointerType == PointerType::Value)
                return dataType;

            return getIntegerType(false, getElementSize());
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

    std::string   toString(PointerType const& p);
    std::ostream& operator<<(std::ostream& stream, PointerType const& p);

    std::string   toString(VariableType const& v);
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
        VariableType segmentVariableType;
        std::string  name;
        std::string  abbrev;

        unsigned int elementSize;
        unsigned int registerCount;
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
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    struct BaseTypeInfo
    {
        using Type = T;

        constexpr static VariableType Var                 = VariableType(T_DEnum, T_PEnum);
        constexpr static VariableType SegmentVariableType = VariableType(T_SegmentType, T_PEnum);

        /// Bytes of one element.  May contain multiple segments.
        constexpr static size_t ElementSize = sizeof(T);
        /// Segments per element.
        constexpr static size_t Packing = T_Packing;
        /// Bytes per segment.
        constexpr static size_t SegmentSize   = ElementSize / Packing;
        constexpr static size_t Alignment     = alignof(T);
        constexpr static size_t RegisterCount = T_RegCount;

        constexpr static bool IsComplex  = T_IsComplex;
        constexpr static bool IsIntegral = T_IsIntegral;
        constexpr static bool IsSigned   = T_IsSigned;

        static inline std::string Name()
        {
            if(Var.pointerType == PointerType::Value)
                return toString(Var.dataType);
            else
                return toString(Var.pointerType);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Var);
        }
    };

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr VariableType BaseTypeInfo<T,
                                        T_DEnum,
                                        T_SegmentType,
                                        T_PEnum,
                                        T_Packing,
                                        T_RegCount,
                                        T_IsComplex,
                                        T_IsIntegral,
                                        T_IsSigned>::Var;
    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr VariableType BaseTypeInfo<T,
                                        T_DEnum,
                                        T_SegmentType,
                                        T_PEnum,
                                        T_Packing,
                                        T_RegCount,
                                        T_IsComplex,
                                        T_IsIntegral,
                                        T_IsSigned>::SegmentVariableType;
    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::ElementSize;
    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::Packing;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::RegisterCount;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::SegmentSize;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr bool BaseTypeInfo<T,
                                T_DEnum,
                                T_SegmentType,
                                T_PEnum,
                                T_Packing,
                                T_RegCount,
                                T_IsComplex,
                                T_IsIntegral,
                                T_IsSigned>::IsComplex;
    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr bool BaseTypeInfo<T,
                                T_DEnum,
                                T_SegmentType,
                                T_PEnum,
                                T_Packing,
                                T_RegCount,
                                T_IsComplex,
                                T_IsIntegral,
                                T_IsSigned>::IsIntegral;

#define DeclareDefaultValueTypeInfo(dtype, enumVal)                         \
    template <>                                                             \
    struct TypeInfo<dtype> : public BaseTypeInfo<dtype,                     \
                                                 DataType::enumVal,         \
                                                 DataType::enumVal,         \
                                                 PointerType::Value,        \
                                                 1,                         \
                                                 1,                         \
                                                 false,                     \
                                                 std::is_integral_v<dtype>, \
                                                 std::is_signed_v<dtype>>   \
    {                                                                       \
    }

    DeclareDefaultValueTypeInfo(float, Float);

    DeclareDefaultValueTypeInfo(int8_t, Int8);
    DeclareDefaultValueTypeInfo(int16_t, Int16);
    DeclareDefaultValueTypeInfo(int32_t, Int32);

    DeclareDefaultValueTypeInfo(uint8_t, UInt8);
    DeclareDefaultValueTypeInfo(uint16_t, UInt16);
    DeclareDefaultValueTypeInfo(uint32_t, UInt32);

#undef DeclareDefaultValueTypeInfo

    template <>
    struct TypeInfo<uint64_t> : public BaseTypeInfo<uint64_t,
                                                    DataType::UInt64,
                                                    DataType::UInt64,
                                                    PointerType::Value,
                                                    1,
                                                    2,
                                                    false,
                                                    true,
                                                    false>
    {
    };

    template <>
    struct TypeInfo<int64_t> : public BaseTypeInfo<int64_t,
                                                   DataType::Int64,
                                                   DataType::Int64,
                                                   PointerType::Value,
                                                   1,
                                                   2,
                                                   false,
                                                   true,
                                                   true>
    {
    };

    template <>
    struct TypeInfo<double> : public BaseTypeInfo<double,
                                                  DataType::Double,
                                                  DataType::Double,
                                                  PointerType::Value,
                                                  1,
                                                  2,
                                                  false,
                                                  false,
                                                  true>
    {
    };

    template <>
    struct TypeInfo<std::complex<float>> : public BaseTypeInfo<std::complex<float>,
                                                               DataType::ComplexFloat,
                                                               DataType::ComplexFloat,
                                                               PointerType::Value,
                                                               1,
                                                               2,
                                                               true,
                                                               false,
                                                               true>
    {
    };

    template <>
    struct TypeInfo<std::complex<double>> : public BaseTypeInfo<std::complex<double>,
                                                                DataType::ComplexDouble,
                                                                DataType::ComplexDouble,
                                                                PointerType::Value,
                                                                1,
                                                                4,
                                                                true,
                                                                false,
                                                                true>
    {
    };

    template <>
    struct TypeInfo<Int8x4> : public BaseTypeInfo<Int8x4,
                                                  DataType::Int8x4,
                                                  DataType::Int8,
                                                  PointerType::Value,
                                                  4,
                                                  1,
                                                  false,
                                                  true,
                                                  true>
    {
    };

    template <>
    struct TypeInfo<Half> : public BaseTypeInfo<Half,
                                                DataType::Half,
                                                DataType::Half,
                                                PointerType::Value,
                                                1,
                                                1,
                                                false,
                                                false,
                                                true>
    {
    };

    struct Halfx2 : public DistinctType<uint32_t, Halfx2>
    {
    };

    template <>
    struct TypeInfo<Halfx2> : public BaseTypeInfo<Halfx2,
                                                  DataType::Halfx2,
                                                  DataType::Half,
                                                  PointerType::Value,
                                                  2,
                                                  1,
                                                  false,
                                                  false,
                                                  true>
    {
    };

    template <>
    struct TypeInfo<BFloat16> : public BaseTypeInfo<BFloat16,
                                                    DataType::BFloat16,
                                                    DataType::BFloat16,
                                                    PointerType::Value,
                                                    1,
                                                    1,
                                                    false,
                                                    false,
                                                    true>
    {
    };

    struct Raw32 : public DistinctType<uint32_t, Raw32>
    {
    };

    template <>
    struct TypeInfo<Raw32> : public BaseTypeInfo<Raw32,
                                                 DataType::Raw32,
                                                 DataType::Raw32,
                                                 PointerType::Value,
                                                 1,
                                                 1,
                                                 false,
                                                 true,
                                                 false>
    {
    };

    template <>
    struct TypeInfo<bool> : public BaseTypeInfo<bool,
                                                DataType::Bool,
                                                DataType::Bool,
                                                PointerType::Value,
                                                1,
                                                1,
                                                false,
                                                true,
                                                false>
    {
    };

    struct Bool32 : public DistinctType<uint32_t, Bool32>
    {
    };

    template <>
    struct TypeInfo<Bool32> : public BaseTypeInfo<Bool32,
                                                  DataType::Bool32,
                                                  DataType::Bool32,
                                                  PointerType::Value,
                                                  1,
                                                  1,
                                                  false,
                                                  false,
                                                  false>
    {
    };

    struct PointerLocal : public DistinctType<uint32_t, PointerLocal>
    {
    };

    template <>
    struct TypeInfo<PointerLocal> : public BaseTypeInfo<PointerLocal,
                                                        DataType::None,
                                                        DataType::None,
                                                        PointerType::PointerLocal,
                                                        1,
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
                                                         DataType::None,
                                                         PointerType::PointerGlobal,
                                                         1,
                                                         2,
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
    struct TypeInfo<Buffer> : public BaseTypeInfo<Buffer,
                                                  DataType::None,
                                                  DataType::None,
                                                  PointerType::Buffer,
                                                  1,
                                                  4,
                                                  false,
                                                  true,
                                                  false>
    {
    };

    template <DataType T_DataType>
    struct EnumTypeInfo
    {
    };

#define DeclareEnumTypeInfo(typeEnum, dtype)                         \
    template <>                                                      \
    struct EnumTypeInfo<DataType::typeEnum> : public TypeInfo<dtype> \
    {                                                                \
    }

    DeclareEnumTypeInfo(Float, float);
    DeclareEnumTypeInfo(Double, double);
    DeclareEnumTypeInfo(ComplexFloat, std::complex<float>);
    DeclareEnumTypeInfo(ComplexDouble, std::complex<double>);
    DeclareEnumTypeInfo(Half, Half);
    DeclareEnumTypeInfo(Halfx2, Halfx2);
    DeclareEnumTypeInfo(Int8x4, Int8x4);
    DeclareEnumTypeInfo(Int32, int32_t);
    DeclareEnumTypeInfo(Int64, int64_t);
    DeclareEnumTypeInfo(BFloat16, BFloat16);
    DeclareEnumTypeInfo(Int8, int8_t);
    DeclareEnumTypeInfo(Raw32, Raw32);
    DeclareEnumTypeInfo(UInt32, uint32_t);
    DeclareEnumTypeInfo(UInt64, uint64_t);
    DeclareEnumTypeInfo(Bool, bool);
    DeclareEnumTypeInfo(Bool32, Bool32);

#undef DeclareEnumTypeInfo

    template <typename T>
    concept CArithmeticType = std::integral<T> || std::floating_point<T>;

    template <typename Result, typename T>
    concept CCanStaticCastTo = requires(T val) //
    {
        {
            static_cast<Result>(val)
            } -> std::same_as<Result>;
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
