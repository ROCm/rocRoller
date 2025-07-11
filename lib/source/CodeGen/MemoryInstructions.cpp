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

#include <rocRoller/CodeGen/MemoryInstructions.hpp>

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    std::string toString(MemoryInstructions::MemoryDirection const& d)
    {
        switch(d)
        {
        case MemoryInstructions::MemoryDirection::Load:
            return "Load";
        case MemoryInstructions::MemoryDirection::Store:
            return "Store";
        default:
            break;
        }

        Throw<FatalError>("Invalid MemoryDirection");
    }

    std::ostream& operator<<(std::ostream& stream, MemoryInstructions::MemoryDirection d)
    {
        return stream << toString(d);
    }

    std::string toString(MemoryInstructions::MemoryKind const& k)
    {
        switch(k)
        {
        case MemoryInstructions::MemoryKind::Global:
            return "Global";
        case MemoryInstructions::MemoryKind::Scalar:
            return "Scalar";
        case MemoryInstructions::MemoryKind::Local:
            return "Local";
        case MemoryInstructions::MemoryKind::Buffer:
            return "Buffer";
        case MemoryInstructions::MemoryKind::Buffer2LDS:
            return "Buffer2LDS";
        default:
            break;
        }

        Throw<FatalError>("Invalid MemoryKind");
    }

    std::ostream& operator<<(std::ostream& stream, MemoryInstructions::MemoryKind k)
    {
        return stream << toString(k);
    }

    uint bitsPerTransposeLoad(GPUArchitecture const& arch, uint elementBits)
    {
        switch(elementBits)
        {
        case 16:
            if(arch.HasCapability(GPUCapability::ds_read_b64_tr_b16))
            {
                return 64;
            }
            else
            {
                Throw<FatalError>(
                    "No 16-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        case 8:
            if(arch.HasCapability(GPUCapability::ds_read_b64_tr_b8))
            {
                return 64;
            }
            else
            {
                Throw<FatalError>(
                    "No 8-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        case 6:
            if(arch.HasCapability(GPUCapability::ds_read_b96_tr_b6))
            {
                return 96;
            }
            else
            {
                Throw<FatalError>(
                    "No 6-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        case 4:
            if(arch.HasCapability(GPUCapability::ds_read_b64_tr_b4))
            {
                return 64;
            }
            else
            {
                Throw<FatalError>(
                    "No 4-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        default:
            Throw<FatalError>(
                "Transpose load from LDS only available for 16, 8, 6, and 4-bit datatypes.");
        }

        // unsupported number of bits
        Throw<FatalError>("unreachable");
        return 0;
    }

    uint extraLDSBytesPerElementBlock(GPUArchitecture const& arch, uint elementBits)
    {
        if(arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes) && elementBits == 6)
        {
            return arch.GetCapability(GPUCapability::DSReadTransposeB6PaddingBytes);
        }
        return 0;
    }

    std::string transposeLoadMnemonic(GPUArchitecture const& arch, uint elementBits)
    {
        switch(elementBits)
        {
        case 16:
            if(arch.HasCapability(GPUCapability::ds_read_b64_tr_b16))
            {
                return "ds_read_b64_tr_b16";
            }
            else
            {
                Throw<FatalError>(
                    "No 16-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        case 8:
            if(arch.HasCapability(GPUCapability::ds_read_b64_tr_b8))
            {
                return "ds_read_b64_tr_b8";
            }
            else
            {
                Throw<FatalError>(
                    "No 8-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        case 6:
            if(arch.HasCapability(GPUCapability::ds_read_b96_tr_b6))
            {
                return "ds_read_b96_tr_b6";
            }
            else
            {
                Throw<FatalError>(
                    "No 6-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        case 4:
            if(arch.HasCapability(GPUCapability::ds_read_b64_tr_b4))
            {
                return "ds_read_b64_tr_b4";
            }
            else
            {
                Throw<FatalError>(
                    "No 4-bit transpose load from LDS instruction is not supported on ",
                    arch.target().toString());
            }
        default:
            Throw<FatalError>(
                "Transpose load from LDS only available for 16, 8, 6, and 4-bit datatypes.");
        }

        // unsupported number of bits
        Throw<FatalError>("unreachable");
        return "";
    }

    Generator<Instruction> MemoryInstructions::transposeLoadLocal(Register::ValuePtr dest,
                                                                  Register::ValuePtr addr,
                                                                  int                offset,
                                                                  int                numBytes,
                                                                  uint               elementBits,
                                                                  std::string const  comment)
    {
        AssertFatal(dest != nullptr);
        AssertFatal(addr != nullptr);

        auto const& arch = m_context.lock()->targetArchitecture();

        AssertFatal((elementBits == 16 || elementBits == 8 || elementBits == 6 || elementBits == 4),
                    "Transpose load from LDS only available for 16, 8, 6, and 4-bit datatypes.",
                    ShowValue(elementBits));

        AssertFatal(numBytes > 0 && (numBytes < m_wordSize || numBytes % m_wordSize == 0),
                    "Invalid number of bytes");
        // 6-bit transposes are special as they require 128b alignment even though they only load 96 bits.
        const uint extraLDSBytes  = extraLDSBytesPerElementBlock(arch, elementBits);
        const uint bytesPerTrLoad = bitsPerTransposeLoad(arch, elementBits) / 8 + extraLDSBytes;
        const std::string dsReadTrMnemonic{transposeLoadMnemonic(arch, elementBits)};

        AssertFatal(
            numBytes % bytesPerTrLoad == 0,
            "Number of bytes must be a multiple of bytes loaded per lane by each transpose load.");

        auto newAddr = addr;
        co_yield genLocalAddr(newAddr);
        auto ctx = m_context.lock();

        // TODO: consider multiple load case
        AssertFatal(numBytes == bytesPerTrLoad, "TODO: consider multiple transpose loads!");

        auto offsetModifier = genOffsetModifier(offset);
        co_yield_(Instruction(dsReadTrMnemonic,
                              {dest},
                              {newAddr},
                              {offsetModifier},
                              concatenate("Transpose load local data ", comment)));

        if(ctx->kernelOptions()->alwaysWaitAfterLoad)
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait after load"));
    }

    Generator<Instruction>
        MemoryInstructions::loadAndPack(MemoryKind                        kind,
                                        Register::ValuePtr                dest,
                                        Register::ValuePtr                addr1,
                                        Register::ValuePtr                offset1,
                                        Register::ValuePtr                addr2,
                                        Register::ValuePtr                offset2,
                                        std::string const                 comment,
                                        std::shared_ptr<BufferDescriptor> buffDesc,
                                        BufferInstructionOptions          buffOpts)
    {
        AssertFatal(dest && dest->regType() == Register::Type::Vector
                        && dest->variableType() == DataType::Halfx2,
                    "loadAndPack destination must be a vector register of type Halfx2");

        // Use the same register for the destination and the temporary val1
        auto val1 = std::make_shared<Register::Value>(
            dest->allocation(), Register::Type::Vector, DataType::Half, dest->allocationCoord());
        auto val2 = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Half, 1);

        co_yield load(kind, val1, addr1, offset1, 2, comment, false, buffDesc, buffOpts);
        co_yield load(kind, val2, addr2, offset2, 2, comment, true, buffDesc, buffOpts);

        // Zero-out upper 16-bits
        co_yield generateOp<Expression::BitwiseAnd>(
            val1, val1, Register::Value::Literal(0x0000FFFF));
        // Zero-out lower 16-bits
        co_yield generateOp<Expression::BitwiseAnd>(
            val2, val2, Register::Value::Literal(0xFFFF0000));
        co_yield generateOp<Expression::BitwiseOr>(dest, val1, val2);
    }

    Generator<Instruction> MemoryInstructions::packAndStore(MemoryKind         kind,
                                                            Register::ValuePtr addr,
                                                            Register::ValuePtr data1,
                                                            Register::ValuePtr data2,
                                                            Register::ValuePtr offset,
                                                            std::string const  comment)
    {
        auto val = Register::Value::Placeholder(
            m_context.lock(), Register::Type::Vector, DataType::Halfx2, 1);

        co_yield m_context.lock()->copier()->packHalf(val, data1, data2);

        co_yield store(kind, addr, val, offset, 4, comment);
    }

    /**
     * @brief Pack values from toPack into result. There may be multiple values within toPack.
     *
     * Currently only works for going from Half to Halfx2
     *
     * @param result
     * @param toPack
     * @return Generator<Instruction>
     */
    Generator<Instruction> MemoryInstructions::packForStore(Register::ValuePtr& result,
                                                            Register::ValuePtr  toPack) const
    {
        auto valuesPerWord = m_wordSize / toPack->variableType().getElementSize();
        auto packed        = DataTypeInfo::Get(toPack->variableType()).packedVariableType();
        if(!packed)
        {
            Throw<FatalError>("Packed variable type not found for ",
                              ShowValue(toPack->variableType()));
        }

        result = Register::Value::Placeholder(toPack->context(),
                                              toPack->regType(),
                                              *packed,
                                              toPack->valueCount() / valuesPerWord,
                                              Register::AllocationOptions::FullyContiguous());
        for(int i = 0; i < result->registerCount(); i++)
        {
            std::vector<Register::ValuePtr> values;
            for(int j = 0; j < valuesPerWord; j++)
            {
                values.push_back(toPack->element({i * valuesPerWord + j}));
            }
            co_yield m_context.lock()->copier()->pack(result->element({i}), values);
        }
    }
}
