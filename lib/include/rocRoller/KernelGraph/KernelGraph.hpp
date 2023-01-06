#pragma once

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Context.hpp>

#include "KernelHypergraph.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /*
         * Kernel graph containers
         */

        class KernelHypergraph
        {
        public:
            ControlGraph::ControlGraph       control;
            CoordGraph::CoordinateHypergraph coordinates;
            ControlToCoordinateMapper        mapper;

            std::string toDOT(bool drawMappings = false) const;

            template <typename T>
            std::pair<int, T> getDimension(int controlIndex, int subDimension = 0) const
            {
                int  tag     = mapper.get<T>(controlIndex, subDimension);
                auto element = coordinates.getElement(tag);
                AssertFatal(std::holds_alternative<CoordGraph::Dimension>(element),
                            "Invalid connection: element isn't a Dimension.",
                            ShowValue(controlIndex));
                auto dim = std::get<CoordGraph::Dimension>(element);
                AssertFatal(std::holds_alternative<T>(dim),
                            "Invalid connection: Dimension type mismatch.",
                            ShowValue(controlIndex),
                            ShowValue(typeid(T).name()),
                            ShowValue(dim));
                return {tag, std::get<T>(dim)};
            }
        };

        /**
         * Translate from Command to (initial) KernelGraph.
         *
         * Resulting KernelGraph matches the Command operations
         * closely.
         */
        KernelHypergraph translate(std::shared_ptr<Command>);

        /**
         * Rewrite KernelGraph to distribute linear packets onto GPU.
         *
         * Linear dimensions are packed/flattened, tiled onto
         * workgroups and wavefronts, and then operated on.
         */
        KernelHypergraph lowerLinear(KernelHypergraph, ContextPtr);

        /**
         * Rewrite KernelGraph to additionally distribute linear dimensions onto a For loop.
         */
        KernelHypergraph lowerLinearLoop(KernelHypergraph const&,
                                         Expression::ExpressionPtr loopSize,
                                         ContextPtr);

        /**
         * Rewrite KernelGraph to additionally unroll linear dimensions.
         */

        /**
         * Rewrite KernelGraph to distribute tiled packets onto GPU.
         *
         * When loading tiles, the tile size, storage location (eg,
         * VGPRs vs LDS), and affinity (eg, owned by a thread vs
         * workgroup) of each tile is specified by the destination
         * tile.  These attributes do not need to be known at
         * translation time.  To specify these attributes, call
         * `setDimension`.
         */
        KernelHypergraph
            lowerTile(KernelHypergraph, std::shared_ptr<CommandParameters>, ContextPtr);

        /**
         * @brief Rewrite KernelGraph to add ComputeIndex operations.
         *
         * The ComputeIndex operations are used so that indices do not need to be
         * completely recalculated everytime when iterating through a tile of data.
         *
         * ComputeIndex operations are added before For Loops and calculate the
         * first index in the loop.
         *
         * A new ForLoopIncrement is added to the loop as well to increment the index
         * by the stride amount.
         *
         * Offset, Stride and Buffer edges are added to the DataFlow portion of the
         * Coordinate graph to keep track of the data needed to perform the operations.
         *
         * @param original
         * @return KernelHypergraph
         */
        KernelHypergraph addComputeIndexOperations(KernelHypergraph const& original);

        /**
         * @brief Rewrite KernelGraph to add Deallocate operations.
         *
         * The control graph is analysed to determine register
         * lifetimes.  Deallocate operations are added when registers
         * are no longer needed.
         */
        KernelHypergraph addDeallocate(KernelHypergraph const&);

        /**
         * Rewrite KernelGraphs to make sure no more CommandArgument
         * values are present within the graph.
         */
        KernelHypergraph cleanArguments(KernelHypergraph, std::shared_ptr<AssemblyKernel>);

        /**
         * Removes all CommandArgruments found within an expression
         * with the appropriate AssemblyKernel Argument.
         */
        Expression::ExpressionPtr cleanArguments(Expression::ExpressionPtr,
                                                 std::shared_ptr<AssemblyKernel>);

        KernelHypergraph unrollLoops(KernelHypergraph const&, ContextPtr);

        /**
         * Rewrite KernelGraphs to set dimension/operation perameters.
         */
        KernelHypergraph updateParameters(KernelHypergraph, std::shared_ptr<CommandParameters>);

        /*
         * Code generation
         */
        Generator<Instruction> generate(KernelHypergraph, std::shared_ptr<AssemblyKernel>);

    }
}
