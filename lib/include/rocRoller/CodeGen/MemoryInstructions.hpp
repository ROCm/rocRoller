/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#pragma once

#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{

    class MemoryInstructions
    {
    public:
        MemoryInstructions(ContextPtr context);

        MemoryInstructions(MemoryInstructions const& rhs) = default;
        MemoryInstructions(MemoryInstructions&& rhs)      = default;
        ~MemoryInstructions()                             = default;

        enum MemoryKind
        {
            Global,
            Scalar,
            Local,
            Buffer
        };

        enum MemoryDirection : int
        {
            Load = 0,
            Store
        };

        /**
         * @brief Generate the instructions required to perform a load.
         *
         *
         * @param kind The kind of memory operation to perform.
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Register containing an offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        Generator<Instruction> load(MemoryKind                        kind,
                                    Register::ValuePtr                dest,
                                    Register::ValuePtr                addr,
                                    Register::ValuePtr                offset,
                                    int                               numBytes,
                                    std::string                       comment  = "",
                                    bool                              high     = false,
                                    std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                    BufferInstructionOptions buffOpts = BufferInstructionOptions());

        /**
         * @brief Generate the instructions required to perform a store.
         *
         *
         * @param kind The kind of memory operation to perform.
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Register containing an offset to be added to addr.
         * @param numBytes The number of bytes to store.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        Generator<Instruction> store(MemoryKind                        kind,
                                     Register::ValuePtr                addr,
                                     Register::ValuePtr                data,
                                     Register::ValuePtr                offset,
                                     int                               numBytes,
                                     std::string const                 comment  = "",
                                     bool                              high     = false,
                                     std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                     BufferInstructionOptions          buffOpts
                                     = BufferInstructionOptions());

        /**
         * @brief Generate the instructions required to perform a load or store.
         *
         * @tparam Dir Whether to load or store data
         * @param kind The kind of memory operation to perform.
         * @param addr The register containing the address.
         * @param data  The register containing the data.
         * @param offset Register containing an offset to be added to addr.
         * @param numBytes The number of bytes.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded or stored into the high bits of the register. (Default=false)
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        template <MemoryDirection Dir>
        Generator<Instruction> moveData(MemoryKind                        kind,
                                        Register::ValuePtr                addr,
                                        Register::ValuePtr                data,
                                        Register::ValuePtr                offset,
                                        int                               numBytes,
                                        std::string const                 comment  = "",
                                        bool                              high     = false,
                                        std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                        BufferInstructionOptions          buffOpts
                                        = BufferInstructionOptions());

        /**
         * @brief Generate instructions that will load two 16bit values and pack them into
         *        a single register.
         *
         * @param kind The kind of memory operation to perform.
         * @param dest The register to store the loaded data in.
         * @param addr1 The register containing the address of the first 16bit value to load the data from.
         * @param offset1 Register containing an offset to be added to addr1.
         * @param addr2 The register containing the address of the second 16bit value to load the data from.
         * @param offset2 Register containing an offset to be added to addr2.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param buffDesc Buffer descriptor to use when `kind` is `Buffer`. (Default = nullptr)
         * @param buffOpts Buffer options. (Default = BufferInstructionOptions())
         */
        Generator<Instruction> loadAndPack(MemoryKind                        kind,
                                           Register::ValuePtr                dest,
                                           Register::ValuePtr                addr1,
                                           Register::ValuePtr                offset1,
                                           Register::ValuePtr                addr2,
                                           Register::ValuePtr                offset2,
                                           std::string const                 comment  = "",
                                           std::shared_ptr<BufferDescriptor> buffDesc = nullptr,
                                           BufferInstructionOptions          buffOpts
                                           = BufferInstructionOptions());

        /**
         * @brief Generate instructions that will pack 2 16bit values into a single 32bit register and store the value
         *        to memmory.
         *
         * @param kind The kind of memory operation to perform.
         * @param addr The register containing the address to store the data.
         * @param data1 The register containing the first 16bit value to store.
         * @param data2 The register containing the second 16bit value to store.
         * @param offset Register containing an offset to be added to addr.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         */
        Generator<Instruction> packAndStore(MemoryKind         kind,
                                            Register::ValuePtr addr,
                                            Register::ValuePtr data1,
                                            Register::ValuePtr data2,
                                            Register::ValuePtr offset,
                                            std::string const  comment = "");

        /**
         * @brief Generate the instructions required to perform a global load.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> loadGlobal(Register::ValuePtr dest,
                                          Register::ValuePtr addr,
                                          int                offset,
                                          int                numBytes,
                                          bool               high = false);

        /**
         * @brief Generate the instructions required to perform a global store.
         *
         *
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> storeGlobal(Register::ValuePtr addr,
                                           Register::ValuePtr data,
                                           int                offset,
                                           int                numBytes,
                                           bool               high = false);

        /**
         * @brief Generate the instructions required to load scalar data into a register from memory.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param base  The register containing the address to load the data from.
         * @param offset The value containing an offset to be added to the address in base.
         * @param numBytes The total number of bytes to load.
         * @param glc Globally Coherent modifier, controls L1 cache policy: false=hit_lru, true=miss_evict. (Default=false)
         */
        Generator<Instruction> loadScalar(Register::ValuePtr dest,
                                          Register::ValuePtr base,
                                          int                offset,
                                          int                numBytes,
                                          bool               glc = false);

        /**
         * @brief Generate the instructions required to perform a scalar store.
         *
         *
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param glc Globally Coherent modifier, controls L1 cache bypass: false=write-combine, true=write-thru. (Default=true)
         */
        Generator<Instruction> storeScalar(Register::ValuePtr addr,
                                           Register::ValuePtr data,
                                           int                offset,
                                           int                numBytes,
                                           bool               glc = true);

        /**
         * @brief Generate the instructions required to perform an LDS load.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> loadLocal(Register::ValuePtr dest,
                                         Register::ValuePtr addr,
                                         int                offset,
                                         int                numBytes,
                                         std::string const  comment = "",
                                         bool               high    = false);

        /**
         * @brief Generate the instructions required to perform a transpose load from LDS.
         *
         *
         * @param dest The register to store the transpose-loaded data in.
         * @param addr  The register containing the address to transpose load the data from.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param elementBits number of bits of variable type to load.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         */
        Generator<Instruction> transposeLoadLocal(Register::ValuePtr dest,
                                                  Register::ValuePtr addr,
                                                  int                offset,
                                                  int                numBytes,
                                                  uint               elementBits,
                                                  std::string const  comment = "");

        /**
         * @brief Generate the instructions required to perform an LDS store.
         *
         *
         * @param addr The register containing the address to store the data.
         * @param data  The register containing the data to store.
         * @param offset Offset to be added to addr.
         * @param numBytes The number of bytes to load.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> storeLocal(Register::ValuePtr addr,
                                          Register::ValuePtr data,
                                          int                offset,
                                          int                numBytes,
                                          std::string const  comment = "",
                                          bool               high    = false);

        /**
         * @brief Generate the instructions required to perform a buffer load.
         *
         *
         * @param dest The register to store the loaded data in.
         * @param addr  The register containing the address to load the data from.
         * @param offset Offset to be added to addr.
         * @param buffDesc Buffer descriptor to use.
         * @param buffOpts Buffer options
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> loadBuffer(Register::ValuePtr                dest,
                                          Register::ValuePtr                addr,
                                          int                               offset,
                                          std::shared_ptr<BufferDescriptor> buffDesc,
                                          BufferInstructionOptions          buffOpts,
                                          int                               numBytes,
                                          bool                              high = false);

        /**
         * @brief Generate the instructions required to perform a buffer store.
         *
         *
         * @param data  The register containing the data to store.
         * @param addr The register containing the address to store the data.
         * @param offset Offset to be added to addr.
         * @param buffDesc Buffer descriptor to use
         * @param buffOpts Buffer options
         * @param numBytes The number of bytes to load.
         * @param high Whether the value will be loaded into the high bits of the register. (Default=false)
         */
        Generator<Instruction> storeBuffer(Register::ValuePtr                data,
                                           Register::ValuePtr                addr,
                                           int                               offset,
                                           std::shared_ptr<BufferDescriptor> buffDesc,
                                           BufferInstructionOptions          buffOpts,
                                           int                               numBytes,
                                           bool                              high = false);

        /**
         * @brief Generate the instructions required to perform a direct global to lds buffer load.
         *
         *
         * @param addr  The register containing the LDS address to write data.
         * @param data The register containing the data to store.
         * @param buffDesc Buffer descriptor to use.
         * @param buffOpts Buffer options
         * @param numBytes The number of bytes to load.
         */
        Generator<Instruction> bufferLoad2LDS(Register::ValuePtr                addr,
                                              Register::ValuePtr                data,
                                              std::shared_ptr<BufferDescriptor> buffDesc,
                                              BufferInstructionOptions          buffOpts,
                                              int                               numBytes);

        /**
         * @brief Generate the instructions required to add a memory barrier.
         *
         * @return Generator<Instruction>
         */
        Generator<Instruction> barrier();

        /**
         * @brief Add the offset to a new register if the offset is greater than maxOffset allowed by the
         *        instruction.
         *
         *
         * @param offset Offset to be added to addr.
         * @param addr The register containing the address to store the data.
         * @param inst The instruction to be queried for maxOffset
         */
        Generator<Instruction>
            addLargerOffset2Addr(int& offset, Register::ValuePtr& addr, std::string inst);

    private:
        const int m_wordSize = 4; // in bytes

        std::weak_ptr<Context> m_context;

        int chooseWidth(int numWords, const std::vector<int>& potentialWidths, int maxWidth) const;
        std::string            genOffsetModifier(int) const;
        Generator<Instruction> genLocalAddr(Register::ValuePtr& addr) const;
        Generator<Instruction> packForStore(Register::ValuePtr& result,
                                            Register::ValuePtr  toPack) const;
    };

    std::string   toString(MemoryInstructions::MemoryDirection const& d);
    std::ostream& operator<<(std::ostream& stream, MemoryInstructions::MemoryDirection n);
}

#include <rocRoller/CodeGen/MemoryInstructions_impl.hpp>
