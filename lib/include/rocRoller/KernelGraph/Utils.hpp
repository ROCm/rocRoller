
#pragma once

#include <optional>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Create a range-based for loop.
         */
        std::pair<int, int>
            rangeFor(KernelGraph& graph, Expression::ExpressionPtr size, const std::string& name);

        /**
         * @brief Remove a range-based for loop created by rangeFor.
         */
        void purgeFor(KernelGraph& graph, int tag);

        /**
         * @brief Remove a node and all of its children from the control graph
         *
         * Also purges the mapper of references to the deleted nodes.
         *
         * @param kgraph
         * @param node
         */
        void purgeNodeAndChildren(KernelGraph& kgraph, int node);

        bool isHardwareCoordinate(int tag, KernelGraph const& kgraph);
        bool isLoopishCoordinate(int tag, KernelGraph const& kgraph);
        bool isStorageCoordinate(int tag, KernelGraph const& kgraph);

        /**
         * @brief Filter coordinates by type.
         */
        template <typename T>
        std::unordered_set<int> filterCoordinates(std::vector<int>   candidates,
                                                  KernelGraph const& kgraph);

        /**
         * @brief Find storage neighbour in either direction.
         *
         * Looks upstream and downstream for a neighbour that
         * satisfies isStorageCoordinate.
         *
         * If found, returns the neighbour tag, and the direction to
         * search for required coordinates.
         *
         * Tries upstream first.
         */
        std::optional<std::pair<int, Graph::Direction>>
            findStorageNeighbour(int tag, KernelGraph const& kgraph);

        /**
         * @brief Return target coordinate for load/store operation.
         *
         * For loads, the target is the source (User or LDS) of the
         * load.
         *
         * For stores, the target is the destination (User or LDS) of
         * the store.
         */
        std::pair<int, Graph::Direction> getOperationTarget(int tag, KernelGraph const& kgraph);

        /**
         * @brief Find all required coordintes needed to compute
         * indexes for the target dimension.
         *
         * @return Pair of: vector required coordinates; set of
         * coordinates in the connecting path.
         */
        std::pair<std::vector<int>, std::unordered_set<int>> findRequiredCoordinates(
            int target, Graph::Direction direction, KernelGraph const& kgraph);

        /**
         * @brief Find the operation of type T that contains the
         * candidate load/store operation.
         */
        template <typename T>
        std::optional<int> findContainingOperation(int candidate, KernelGraph const& kgraph);

        /**
         * Replace operation with a new operation.  Does not delete the original operation.
         */
        int replaceWith(KernelGraph& graph, int op, int newOp, bool includeBody = true);

        /**
         * @brief Find load/store operations that need their indexes
         * precomputed by ComputeIndex.
         */
        std::vector<int> findComputeIndexCandidates(KernelGraph const& kgraph, int start);

        void loadMacroTile(KernelGraph&                       graph,
                           int                                load_tag,
                           int                                user_tag,
                           int                                mac_tile_tag,
                           std::vector<int>&                  sdim,
                           std::array<unsigned int, 3> const& workgroupSizes,
                           int                                wavefrontSize,
                           std::vector<unsigned int> const&   wavetilesPerWorkgroup);

        void loadMacroTileForLDS(KernelGraph&                       graph,
                                 int                                load_tag,
                                 int                                user_tag,
                                 int                                mac_tile_tag,
                                 std::vector<int>&                  sdim,
                                 int                                K,
                                 std::array<unsigned int, 3> const& workgroupSizes,
                                 bool                               useSwappedAccess);

        void updateLoadLDSMacroTile(KernelGraph&                      graph,
                                    CoordinateGraph::MacroTile const& mac_tile,
                                    int                               load_tag,
                                    std::vector<int>&                 sdims,
                                    int                               K,
                                    int                               lds,
                                    bool                              useSwappedAccess);

        void loadWaveMacroTile(KernelGraph&                      graph,
                               CoordinateGraph::MacroTile const& mac_tile,
                               int                               load_tag,
                               int                               i_mac_x,
                               int                               i_mac_y,
                               int                               user_tag,
                               int                               wavefrontSize,
                               std::vector<unsigned int> const&  wavetilesPerWorkgroup);

        void storeMacroTile(KernelGraph&                       graph,
                            int                                store_tag,
                            int                                user_tag,
                            int                                mac_tile_tag,
                            std::vector<int>&                  sdims,
                            std::array<unsigned int, 3> const& workgroupSizes,
                            int                                wavefrontSize,
                            std::vector<unsigned int> const&   wavetilesPerWorkgroup);

        void storeWaveMacroTile(KernelGraph&                      graph,
                                CoordinateGraph::MacroTile const& mac_tile,
                                int                               store_tag,
                                int                               i_mac_x,
                                int                               i_mac_y,
                                int                               workitem,
                                int                               user_tag,
                                int                               wavefrontSize,
                                std::vector<unsigned int> const&  wavetilesPerWorkgroup);

        void storeMacroTileForLDS(KernelGraph&                       graph,
                                  int                                store_tag,
                                  int                                user_tag,
                                  int                                mac_tile_tag,
                                  std::vector<int>&                  sdims,
                                  std::array<unsigned int, 3> const& workgroupSizes);

        void updateStoreLDSMacroTile(KernelGraph&                      graph,
                                     CoordinateGraph::MacroTile const& mac_tile,
                                     int                               store_tag,
                                     std::vector<int>&                 sdims,
                                     int                               lds);

        void storeMacroTileIntoLDS(KernelGraph&                       graph,
                                   int                                store_tag,
                                   int                                lds_tag,
                                   int                                mac_tile_tag,
                                   std::array<unsigned int, 3> const& workgroupSizes,
                                   bool                               useSwappedAccess);

        void loadMacroTileFromLDS(KernelGraph&                       graph,
                                  int                                load_tag,
                                  int                                lds_tag,
                                  int                                mac_tile_tag,
                                  std::array<unsigned int, 3> const& workgroupSizes);

        void addConnectionsMultiply(KernelGraph& graph, int waveMult, int loadA, int loadB);

        /**
         * @brief Get a pair of expressions representing a for loop increment
         *
         * This assumes that there is only a single for loop increment for a given loop.
         *
         * This also assumes that the increment is of the form: Add(DataFlowTag(N), Val),
         * where N is the data tag associated with the for loop.
         *
         * The first item in the pair is the data flow tag associated with the for loop.
         *
         * The second item is the amount that it is being incremented by.
         *
         * @param graph
         * @param forLoop
         * @return std::pair<ExpressionPtr, ExpressionPtr>
         */
        std::pair<Expression::ExpressionPtr, Expression::ExpressionPtr>
            getForLoopIncrement(KernelGraph const& graph, int forLoop);

        /**
         * @brief Return first entry of vector.
         *
         * If vector does not contain a single result, return empty.
         */
        template <typename T>
        inline std::optional<T> only(std::vector<T> v);

        /**
         * @brief Return first result of generator.
         *
         * If generator does not return a single result, return empty.
         */
        inline std::optional<int> only(Generator<int> g);

        int duplicateControlNode(KernelGraph& graph, int tag);

        /**
         * Updates the threadtile size for enabling the use of long dword instructions
         */
        void updateThreadTileForLongDwords(int& t_m,
                                           int& t_n,
                                           int  maxWidth,
                                           int  numDwordsPerElement);
    }
}

#include "Utils_impl.hpp"
