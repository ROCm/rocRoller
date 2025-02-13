/**
 * @copyright Copyright 2023 Advanced Micro Devices, Inc.
 */

#pragma once

#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

#include <rocRoller/Expression_fwd.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Class for generating instructions related to loading and storing tiles
         *        to and from memory.
         *
         */
        class LoadStoreTileGenerator
        {
        public:
            LoadStoreTileGenerator(KernelGraphPtr, ContextPtr, unsigned int);

            /**
             * @brief Generate instructions needed to load a tile from global memory
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction> genLoadTile(int                            tag,
                                               ControlGraph::LoadTiled const& load,
                                               CoordinateGraph::Transformer   coords);

            /**
             * @brief Generate instructions needed to load a tile from LDS
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction> genLoadLDSTile(int                              tag,
                                                  ControlGraph::LoadLDSTile const& load,
                                                  CoordinateGraph::Transformer     coords);

            /**
             * @brief Generate instructions needed to load a tile from global memory direct to lds
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction>
                genLoadTileDirect2LDS(int                                     tag,
                                      ControlGraph::LoadTileDirect2LDS const& load,
                                      CoordinateGraph::Transformer            coords);

            /**
             * @brief Generate instructions needed to store a tile to global memory
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction> genStoreTile(int                             tag,
                                                ControlGraph::StoreTiled const& store,
                                                CoordinateGraph::Transformer    coords);

            /**
             * @brief Generate instructions needed to store a tile to LDS
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction> genStoreLDSTile(int                               tag,
                                                   ControlGraph::StoreLDSTile const& store,
                                                   CoordinateGraph::Transformer      coords);

            /**
             * @brief Generate instructions needed to calculate offset and stride information
             *
             * @param tag The tag of the node in the control graph
             * @param load The node in the control graph
             * @param coords Known coordinates
             * @return Generator<Instruction>
             */
            Generator<Instruction> genComputeIndex(int                               tag,
                                                   ControlGraph::ComputeIndex const& ci,
                                                   CoordinateGraph::Transformer      coords);

        private:
            std::map<int, int>               m_baseOffsets;
            ContextPtr                       m_context;
            KernelGraphPtr                   m_graph;
            Expression::ExpressionTransducer m_fastArith;
            unsigned int                     m_workgroupSizeTotal;

            // Information needed in order to load or store a tile.
            struct LoadStoreTileInfo
            {
                MemoryInstructions::MemoryKind    kind;
                uint64_t                          m;
                uint64_t                          n;
                uint32_t                          elementBits;
                bool                              unitStride;
                uint32_t                          packedAmount;
                uint32_t                          ldsWriteStride;
                Register::ValuePtr                data;
                Register::ValuePtr                rowOffsetReg;
                Register::ValuePtr                rowStrideReg;
                Register::ValuePtr                colStrideReg;
                Register::ValuePtr                offset;
                std::shared_ptr<BufferDescriptor> bufDesc;
                BufferInstructionOptions          bufOpts;
            };

            inline Generator<Instruction> generate(auto&                     dest,
                                                   Expression::ExpressionPtr expr) const;

            // Index calculation Helpers
            std::shared_ptr<BufferDescriptor> getBufferDesc(int tag);
            Expression::ExpressionPtr         getOffsetExpr(int                                 offsetTag,
                                                            CoordinateGraph::Transformer const& coords);
            Generator<Instruction>            getOffset(LoadStoreTileInfo&           info,
                                                        CoordinateGraph::Transformer coords,
                                                        int                          tag,
                                                        bool                         preserveOffset,
                                                        bool                         direct2LDS = false);

            /**
             * @brief Generate stride (in bytes).
             *
             * The `unitStride` flag is set if the generated
             * byte-stride corresponds to a unit element-stride.  A
             * unit element-stride is a unitary (=1) stride with
             * respect to the element of the underlying data type.
             *
             * The generated stride is in bytes.  This facilitates,
             * eg, advancing offset registers to the next macro tile
             * by simply adding the stride in the increment of a for
             * loop.
             *
             * However, determining whether a byte-stride in a
             * stride-expression is a unit-stride is tricky for
             * sub-byte datatypes.  To make this more robust,
             * stride-expressions have meta-data attached to the
             * expression to make this explicit.
             *
             * For example, if we only knew the byte-stride:
             *
             * | data type | byte-stride | unit element-stride |
             * |-----------|-------------|---------------------|
             * | FP64      | 8           | true                |
             * | FP32      | 4           | true                |
             * | FP32      | 8           | false               |
             * | FP16      | 2           | true                |
             * | FP8       | 1           | true                |
             * | Sub-byte  | 1           | maybe!              |
             */
            Generator<Instruction>
                generateStride(Register::ValuePtr& stride, bool& isUnit, int tag, int dimension);

            // Move Tile Helpers
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTile(MemoryInstructions::MemoryKind kind,
                                            uint64_t                       m,
                                            uint64_t                       n,
                                            VariableType                   dataType,
                                            int                            tag,
                                            Register::ValuePtr             vgpr,
                                            Register::ValuePtr             offset,
                                            CoordinateGraph::Transformer&  coords,
                                            BufferInstructionOptions       bufOpts = {});
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileLiteralStrides(LoadStoreTileInfo& info);
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileColStrideOne(LoadStoreTileInfo& info);
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileRuntimeStrides(LoadStoreTileInfo& info);
            template <MemoryInstructions::MemoryDirection Dir>
            Generator<Instruction> moveTileDirect2LDS(LoadStoreTileInfo& info,
                                                      int                numBytes,
                                                      bool               setM0,
                                                      Register::ValuePtr readOffset,
                                                      Register::ValuePtr readAddr);
            Generator<Instruction> loadTileLiteralStridesPack(LoadStoreTileInfo& info);
            Generator<Instruction> loadTileRuntimeStridesPack(LoadStoreTileInfo& info);

            // Load Tile Helpers
            Generator<Instruction> loadMacroTileVGPR(int                            tag,
                                                     ControlGraph::LoadTiled const& load,
                                                     CoordinateGraph::Transformer   coords);
            Generator<Instruction> loadMacroTileLDS(int                              tag,
                                                    ControlGraph::LoadLDSTile const& load,
                                                    CoordinateGraph::Transformer     coords);
            Generator<Instruction> loadMacroTileWAVELDS(int                              tag,
                                                        ControlGraph::LoadLDSTile const& load,
                                                        CoordinateGraph::Transformer     coords);
            Generator<Instruction> loadMacroTileWAVE(int                            tag,
                                                     ControlGraph::LoadTiled const& load,
                                                     CoordinateGraph::Transformer   coords);
            Generator<Instruction> loadMacroTileWAVECIACCUM(int                            tag,
                                                            ControlGraph::LoadTiled const& load,
                                                            CoordinateGraph::Transformer   coords);
            Generator<Instruction>
                loadMacroTileDirect2LDS(int                                     tag,
                                        ControlGraph::LoadTileDirect2LDS const& load,
                                        CoordinateGraph::Transformer            coords);

            // Store Tile Helpers
            Generator<Instruction> storeMacroTileLDS(int                               tag,
                                                     ControlGraph::StoreLDSTile const& store,
                                                     CoordinateGraph::Transformer      coords);
            Generator<Instruction> storeMacroTileVGPR(int                             tag,
                                                      ControlGraph::StoreTiled const& store,
                                                      CoordinateGraph::Transformer    coords);
            Generator<Instruction> storeMacroTileWAVELDS(int                               tag,
                                                         ControlGraph::StoreLDSTile const& store,
                                                         CoordinateGraph::Transformer      coords);
            Generator<Instruction> storeMacroTileWAVE(int                             tag,
                                                      ControlGraph::StoreTiled const& store,
                                                      CoordinateGraph::Transformer    coords);
        };
    }
}
