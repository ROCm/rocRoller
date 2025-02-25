
#pragma once

#include <functional>
#include <optional>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
        * @brief Return DataFlowTag of LHS of binary expression in Assign node.
        */
        template <Expression::CBinary T>
        std::tuple<int, Expression::ExpressionPtr> getBinaryLHS(KernelGraph const& kgraph,
                                                                int                assign);

        /**
        * @brief Return DataFlowTag of RHS of binary expression in Assign node.
        */
        template <Expression::CBinary T>
        std::tuple<int, Expression::ExpressionPtr> getBinaryRHS(KernelGraph const& kgraph,
                                                                int                assign);

        /**
         * @brief Create a range-based for loop.
         *
         * returns {dimension, operation}
         */
        std::pair<int, int> rangeFor(KernelGraph&              graph,
                                     Expression::ExpressionPtr size,
                                     const std::string&        name,
                                     VariableType              vtype        = DataType::None,
                                     int                       forLoopCoord = -1);

        /**
         * @brief Remove a range-based for loop created by rangeFor.
         */
        void purgeFor(KernelGraph& graph, int tag);

        /**
         * @brief Create a clone of a ForLoopOp. This new ForLoopOp
         * will use the same ForLoop Dimension as the original
         * ForLoopOp.
        */
        int cloneForLoop(KernelGraph& graph, int tag);

        /**
         * @brief Remove a node and all of its children from the control graph
         *
         * Also purges the mapper of references to the deleted nodes.
         *
         * @param kgraph
         * @param node
         */
        void purgeNodeAndChildren(KernelGraph& kgraph, int node);

        template <std::ranges::forward_range Range = std::initializer_list<int>>
        void purgeNodes(KernelGraph& kgraph, Range nodes);

        bool isHardwareCoordinate(int tag, KernelGraph const& kgraph);
        bool isLoopishCoordinate(int tag, KernelGraph const& kgraph);
        bool isStorageCoordinate(int tag, KernelGraph const& kgraph);

        /**
         * @brief Filter coordinates by type.
         */
        template <typename T>
        std::unordered_set<int> filterCoordinates(auto const&        candidates,
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
        * @brief Return DataFlowTag of DEST of Assign node.
        */
        int getDEST(KernelGraph const& kgraph, int assign);

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

        std::pair<std::vector<int>, std::unordered_set<int>>
            findRequiredCoordinates(int                      target,
                                    Graph::Direction         direction,
                                    std::function<bool(int)> fullStop,
                                    KernelGraph const&       kgraph);

        std::pair<std::unordered_set<int>, std::unordered_set<int>>
            findAllRequiredCoordinates(int op, KernelGraph const& graph);

        /**
         * @brief Find the operation of type T that contains the
         * candidate load/store operation.
         */
        template <typename T>
        std::optional<int> findContainingOperation(int candidate, KernelGraph const& kgraph);

        /**
     * @brief Reconnect incoming/outgoing edges from op to newop.
     */
        template <Graph::Direction direction>
        void reconnect(KernelGraph& graph, int newop, int op);

        /**
         * @brief Find the operation of type T that contains the
         * candidate load/store operation. Then return the top element of the
         * body of that operation.
         */
        template <typename T>
        std::optional<int> findTopOfContainingOperation(int candidate, KernelGraph const& kgraph);

        /**
         * @brief Create a new coordinate representing data within the scratch space. This will return a
         * coordinate that can be added to a coordinate graph. It also allocates the required scratch space
         * within the context.
         *
         * @param size
         * @param varType
         * @param context
         * @return User
         */
        rocRoller::KernelGraph::CoordinateGraph::User newScratchCoordinate(
            Expression::ExpressionPtr size, VariableType varType, ContextPtr context);

        /**
         * Replace operation with a new operation.  Does not delete the original operation.
         */
        int replaceWith(KernelGraph& graph, int op, int newOp, bool includeBody = true);

        /**
         * @brief Insert chain (from top to bottom) above operation.
         *
         * Bottom is attached to op via a Sequence edge.
         */
        void insertBefore(KernelGraph& graph, int op, int top, int bottom);

        /**
         * @brief Replace operation with a new operation.
         */
        void insertWithBody(KernelGraph& graph, int op, int newOp);

        /**
         * @brief Find load/store operations that need their indexes
         * precomputed by ComputeIndex.
         */
        std::vector<int> findComputeIndexCandidates(KernelGraph const& kgraph, int start);

        /**
         * Removes all CommandArgruments found within an expression
         * with the appropriate AssemblyKernel Argument.
         */
        Expression::ExpressionPtr cleanArguments(Expression::ExpressionPtr, AssemblyKernelPtr);

        /**
         * @brief Get ForLoop and increment (Linear) dimensions
         * assciated with ForLoopOp.
         */
        std::pair<int, int> getForLoopCoords(int forLoopOp, KernelGraph const& kgraph);

        template <CForwardRangeOf<int> Range>
        std::optional<int>
            getForLoopCoord(std::optional<int> forLoopOp, KernelGraph const& kgraph, Range within);

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

        int duplicateControlNode(KernelGraph& graph, int tag);

        /**
         * Updates the threadtile size for enabling the use of long dword instructions
         */
        void updateThreadTileForLongDwords(int& t_m,
                                           int& t_n,
                                           int  maxWidth,
                                           uint macTileFastMovingDimSize,
                                           int  numDwordsPerElement);

        /**
         * @brief Get the tag of the highest SetCoordinate directly upstream from load.
         *
         * @param graph
         * @param load
         * @return int
         */
        int getTopSetCoordinate(KernelGraph const& graph, int load);

        /**
         * @brief Get the unique tags of the highest SetCoordinate nodes directly upstream from each load.
         *
         * @param graph
         * @param loads
         * @return std::set<int>
         */
        std::set<int> getTopSetCoordinates(KernelGraph& graph, std::vector<int> loads);

        /**
         * @brief Get the SetCoordinate object upstream from load that sets the
         * coordinate for the dimension dim.
         *
         * @param graph
         * @param dim
         * @param load
         * @return int
         */
        int getSetCoordinateForDim(KernelGraph const& graph, int dim, int load);

        /**
         * Gets the unroll coordinate value that is set by a SetCoordinate node upstream
         * from the operation op, for the dimension unrollDim.
         */
        unsigned int getUnrollValueForOp(KernelGraph const& graph, int unrollDim, int op);

        /**
         * @brief Create duplicates of all of the nodes downstream of the provided
         *        start nodes.
         *        Add the duplicates to the provided graph.
         *        Return the location of the new start nodes.
         *
         * @param graph KernelGraph that nodes are duplicated from and into.
         * @param startNodes Starting nodes of sub-graph to duplicate.
         * @param reindexer Graph reindexer.
         * @param dontDuplicate Predicate to determine if a coordinate node is duplicated.
         *
         * @return New start nodes for the duplicated sub-graph.
         */
        template <std::predicate<int> Predicate>
        std::vector<int> duplicateControlNodes(KernelGraph&                    graph,
                                               std::shared_ptr<GraphReindexer> reindexer,
                                               std::vector<int> const&         startNodes,
                                               Predicate                       dontDuplicate);

        /**
         * @brief Return VariableType of load/store operation.
         */
        VariableType getVariableType(KernelGraph const& graph, int opTag);

        /**
         * @brief Add coordinate-transforms for storing a MacroTile
         * from a ThreadTile into global.
         *
         * Implemented in LowerTile.cpp.
         */
        void storeMacroTile_VGPR(KernelGraph&                     graph,
                                 std::vector<DeferredConnection>& connections,
                                 int                              userTag,
                                 int                              macTileTag,
                                 std::vector<int> const&          sdim,
                                 std::vector<unsigned int> const& jammedTiles,
                                 ContextPtr                       context);

        /**
         * @brief Add coordinate-transforms for loading a MacroTile
         * from global into a ThreadTile.
         */
        void loadMacroTile_VGPR(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              userTag,
                                int                              macTileTag,
                                std::vector<int> const&          sdim,
                                std::vector<unsigned int> const& jammedTiles,
                                ContextPtr                       context);

        /**
         * @brief Store version of addLoadThreadTileCT.
         */
        void addStoreThreadTileCT(KernelGraph&                       graph,
                                  std::vector<DeferredConnection>&   connections,
                                  int                                macTileTag,
                                  int                                iMacX,
                                  int                                iMacY,
                                  std::array<unsigned int, 3> const& workgroupSizes,
                                  std::vector<unsigned int> const&   jammedTiles,
                                  bool                               useSwappedAccess);

        /**
         * @brief Store version of addLoadMacroTileCT.
         */
        std::tuple<int, int, int, int>
            addStoreMacroTileCT(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              macTileTag,
                                std::vector<int> const&          sdim);

        /**
         * @brief Add coordinate-transforms for tiling two
         * SubDimension coordinates into macro number/index
         * coordinates.
         *
         * The geometry of the tiling is taken from the MacroTile
         * associated with `macTileTag`.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         *
         * @return Tuple of: row MacroTileNumber, row MacroTileIndex,
         * column MacroTileNumber, column MacroTileIndex.
         */
        std::tuple<int, int, int, int>
            addLoadMacroTileCT(KernelGraph&                     graph,
                               std::vector<DeferredConnection>& connections,
                               int                              macTileTag,
                               std::vector<int> const&          sdim);

        /**
         * @brief Add coordinate-transforms for loading a ThreadTile
         * from row/column coordinates iMacX and iMacY.
         *
         * The geometry of the ThreadTile is taken from the MacroTile
         * associated with `macTileTag`.
         *
         * By default:
         *
         *   - For A/B matrix layouts, the Y thread tile number is
         *     fast wrt the workitem/lane index and the X thread tile
         *     number is slow.  For other layous, the X/Y thread tile
         *     numbers are taken from the X/Y workitem index.
         *
         *   - The row index of a thread tile is fast wrt the VGPR
         *     index.
         *
         * When `useSwappedAccess` is true, both of these orders are
         * reversed.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         */
        void addLoadThreadTileCT(KernelGraph&                       graph,
                                 std::vector<DeferredConnection>&   connections,
                                 int                                macTileTag,
                                 int                                iMacX,
                                 int                                iMacY,
                                 std::array<unsigned int, 3> const& workgroupSizes,
                                 std::vector<unsigned int> const&   jammedTiles,
                                 bool                               useSwappedAccess);

        /**
         * @brief Create an internal tile backed by a ThreadTile.
         *
         * Implemented in LowerTile.cpp.
         */
        int createInternalTile(KernelGraph&         graph,
                               VariableType         varType,
                               int                  macTileTag,
                               CommandParametersPtr params,
                               ContextPtr           context);

        /**
         * @brief Create an internal tile backed by a ThreadTile.  The
         * internal tile is reduced in size according to numWaveTiles.
         *
         * Implemented in LowerTile.cpp.
         */
        int createInternalTile(KernelGraph&                     graph,
                               VariableType                     varType,
                               int                              macTileTag,
                               std::vector<unsigned int> const& numWaveTiles,
                               CommandParametersPtr             params,
                               ContextPtr                       context);

        /**
         * @brief Order all input pairs of memory nodes in graph.
         *
         * @param graph
         * @param pairs Pairs of memory nodes to be ordered.
         * @param ordered If true, the pairs are passed in order.
         */
        void orderMemoryNodes(KernelGraph&                         graph,
                              std::set<std::pair<int, int>> const& pairs,
                              bool                                 ordered);

        /**
         * @brief Order all memory nodes in srcs with respect to all memory nodes in dests.
         *
         * @param graph
         * @param srcs
         * @param dests
         * @param ordered If true, all orderings will be src -> dest.
         */
        void orderMemoryNodes(KernelGraph&         graph,
                              std::set<int> const& srcs,
                              std::set<int> const& dests,
                              bool                 ordered);

        /**
         * @brief Order all input nodes with respect to each other.
         *
         * @param graph
         * @param nodes
         * @param ordered If true, all orderings will be nodes[i-1] -> nodes[i].
         */
        void orderMemoryNodes(KernelGraph& graph, std::vector<int> const& nodes, bool ordered);

        /**
         * Replace the use of an old macrotile in the given control
         * nodes with a new macrotile.
         */
        void replaceMacroTile(KernelGraph&                   graph,
                              std::unordered_set<int> const& ops,
                              int                            oldMacTileTag,
                              int                            newMacTileTag);
    }
}

#include "Utils_impl.hpp"
