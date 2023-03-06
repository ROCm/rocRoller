
#include <msgpack.hpp>

#include <algorithm>

#include "include/GraphInspector.hpp"

namespace rocRoller
{
    namespace Client
    {

        template <typename T>
        void writeMatrix(std::string const&    filename,
                         size_t                size0,
                         size_t                size1,
                         std::vector<T> const& matrix)
        {
            {
                std::ofstream                  of(filename);
                msgpack::packer<std::ofstream> packer(of);

                packer.pack_map(2);

                packer.pack("sizes");
                packer.pack_array(2);
                packer.pack(size0);
                packer.pack(size1);

                packer.pack("data");
                packer.pack(matrix);
            }

            std::cout << "Wrote " << filename << std::endl;
        }

        using ElementNumbers = std::vector<std::tuple<int, size_t>>;

        ElementNumbers findElementNumberNodes(auto coords, int startingNode, Graph::Direction dir)
        {

            auto getElementNumberAndSize = [coords](int elem) -> std::tuple<int, size_t> {
                auto node = coords->template get<KernelGraph::CoordinateGraph::ElementNumber>(elem);
                if(!node)
                    return {-1, 0};

                auto size = std::visit(to_size_t, Expression::evaluate(node->size));
                return {elem, size};
            };

            ElementNumbers rv;

            for(int elem : coords->depthFirstVisit(startingNode, dir))
            {
                auto tup = getElementNumberAndSize(elem);
                if(std::get<0>(tup) > 0)
                    rv.push_back(std::move(tup));
            }

            return rv;
        }

        int findLDSNode(auto coords, int startingNode, Graph::Direction dir)
        {
            for(int elem : coords->depthFirstVisit(startingNode, dir))
            {
                if(isNode<KernelGraph::CoordinateGraph::LDS>(coords)(elem))
                    return elem;
            }

            Throw<FatalError>("Could not find an LDS node starting at ", ShowValue(startingNode));
        }

        void setIndexCoords(GraphInspector&       gi,
                            std::vector<size_t>&  elementCoords,
                            size_t                value,
                            ElementNumbers const& fixedElements)
        {
            elementCoords.resize(fixedElements.size());
            for(int coordIdx = 0; coordIdx < fixedElements.size(); coordIdx++)
            {
                auto coord              = std::get<0>(fixedElements[coordIdx]);
                auto thisSize           = std::get<1>(fixedElements[coordIdx]);
                elementCoords[coordIdx] = value % thisSize;
                value                   = value / thisSize;

                gi.setCoordinate(coord, elementCoords[coordIdx]);
            }
        }

        void getMatrixDims(size_t&            size0,
                           size_t&            size1,
                           GraphInspector&    gi,
                           std::string const& argPrefix)
        {
            size0 = std::visit(to_size_t, gi.argValues().at(argPrefix + "_size_0"));
            size1 = std::visit(to_size_t, gi.argValues().at(argPrefix + "_size_1"));

            auto stride0 = std::visit(to_size_t, gi.argValues().at(argPrefix + "_stride_0"));
            auto stride1 = std::visit(to_size_t, gi.argValues().at(argPrefix + "_stride_1"));

            // TODO: Once we do transposing in the graph, remove this.
            bool transposed = stride0 > stride1;
            if(transposed)
                std::swap(size0, size1);
        }

        /**
         * Write a volume where the value is the workitem index, for all memory locations read
         * by one MacroTile of the specified loadDim (workgroup + K for loop iteration).

         * \param vfile Log file
         * \param gi
         * \param loadDim Dimension of the coordinate graph associated with the matrix
         * \param argPrefix The prefix of the argument names associated with this matrix. "Load_Tiled_X" where X is a number
         * \param matName A name for the matrix, to be used in logging and naming the output file.
         */
        void writeMacrotileByWorkitem(std::ostream&      vfile,
                                      GraphInspector&    gi,
                                      int                loadDim,
                                      std::string const& argPrefix,
                                      std::string const& matName)
        {
            auto invocation = gi.kernelInvocation();

            size_t size0, size1;
            getMatrixDims(size0, size1, gi, argPrefix);

            std::vector<int> matrix(size0 * size1, -10);

            auto workitemIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::Workitem>(gi.coords()))
                      .to<std::vector>();

            AssertFatal(gi.kernelInvocation().workgroupSize[1] == 1);

            auto totalWorkgroupSize = product(invocation.workgroupSize);

            auto fixedElements
                = findElementNumberNodes(gi.coords(), loadDim, Graph::Direction::Downstream);

            vfile << "FEs:";
            for(auto const& tup : fixedElements)
                vfile << " {" << std::get<0>(tup) << ", " << std::get<1>(tup) << "}";
            vfile << std::endl;

            size_t totalElements = 1;
            for(auto const& tup : fixedElements)
                totalElements *= std::get<1>(tup);
            std::vector<size_t> elementCoords(fixedElements.size(), 0);

            for(int i = 0; i < totalWorkgroupSize; i++)
            {
                gi.setCoordinate(workitemIndices, i);

                for(size_t j = 0; j < totalElements; j++)
                {
                    setIndexCoords(gi, elementCoords, j, fixedElements);

                    auto matIdx = gi.getLoadIndex(loadDim);
                    AssertFatal(
                        matrix.at(matIdx) == -10, ShowValue(matIdx), ShowValue(i), ShowValue(j));
                    matrix.at(matIdx) = (i * totalElements) + j;
                    vfile << "Index of " << matName << " for thread " << i << " Element " << j
                          << " (";
                    streamJoin(vfile, elementCoords, ", ");
                    vfile << "): " << matIdx << " (" << (matIdx % 512) << ", " << (matIdx / 512)
                          << ")" << std::endl;
                    vfile.flush();
                }
            }

            vfile << std::endl << std::endl << std::endl;

            writeMatrix("workitem_" + matName + ".dat", size0, size1, matrix);
        }

        /**
         * Writes the LDS memory access pattern to a data file. Currently,
         * the dimensions of LDS are not inferred from the graph so they
         * must be passed in.
         */
        void writeLDSByWorkitem(std::ostream&      vfile,
                                GraphInspector&    gi,
                                int                ldsDim,
                                size_t             size0,
                                size_t             size1,
                                std::string const& matName)
        {
            using namespace KernelGraph::CoordinateGraph;
            auto invocation = gi.kernelInvocation();
            auto coords     = gi.coords();

            auto fixedElements = findElementNumberNodes(coords, ldsDim, Graph::Direction::Upstream);
            size_t totalElements = 1;
            for(auto const& tup : fixedElements)
                totalElements *= std::get<1>(tup);

            auto workitemIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::Workitem>(gi.coords()))
                      .to<std::vector>();

            std::vector<size_t> elementCoords;

            auto lds = coords->getNode<LDS>(ldsDim);

            std::vector<int> matrix(0, -10);

            auto totalWorkgroupSize = product(invocation.workgroupSize);
            for(int i = 0; i < totalWorkgroupSize; i++)
            {
                gi.setCoordinate(workitemIndices, i);

                for(int j = 0; j < totalElements; j++)
                {
                    setIndexCoords(gi, elementCoords, j, fixedElements);

                    auto matIdx = gi.getStoreIndex(ldsDim);
                    if(matIdx >= matrix.size())
                        matrix.resize(matIdx + 1, -1);

                    matrix.at(matIdx) = i;
                }
            }

            writeMatrix("LDS_" + matName + ".dat", size0, size1, matrix);
        }

        /**
         * Write a volume where the value is the linear workgroup ID which reads the specified matrix,
         * for all workgroups, for one iteration of the K loop.
         *
         * \param vfile Log file
         * \param gi
         * \param loadDim Dimension of the coordinate graph associated with the matrix
         * \param hipWorkgroupIndex 0, 1, or 2, are we iterating over workgroup x, y, or z?
         * \param argPrefix The prefix of the argument names associated with this matrix. "Load_Tiled_X" where X is a number
         * \param matName A name for the matrix, to be used in logging and naming the output file.
         */
        void writeKIterByWorkitem(std::ostream&      vfile, // Log file
                                  GraphInspector&    gi,
                                  int                loadDim,
                                  int                hipWorkgroupIndex,
                                  std::string const& argPrefix,
                                  std::string const& matName)
        {
            auto invocation     = gi.kernelInvocation();
            auto workgroupCount = invocation.workitemCount;
            for(int i = 0; i < 3; i++)
                workgroupCount[i] = invocation.workitemCount[i] / invocation.workgroupSize[i];
            vfile << "Workgroups: " << workgroupCount[0] << "x" << workgroupCount[1] << "x"
                  << workgroupCount[2] << std::endl;

            size_t size0, size1;
            getMatrixDims(size0, size1, gi, argPrefix);

            std::vector<int> matrix(size0 * size1, -10);

            auto fixedElements
                = findElementNumberNodes(gi.coords(), loadDim, Graph::Direction::Downstream);

            vfile << "FEs:";
            for(auto const& tup : fixedElements)
                vfile << " {" << std::get<0>(tup) << ", " << std::get<1>(tup) << "}";
            vfile << std::endl;

            size_t totalElements = 1;
            for(auto const& tup : fixedElements)
                totalElements *= std::get<1>(tup);
            std::vector<size_t> elementCoords(fixedElements.size(), 0);

            auto workitemIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::Workitem>(gi.coords()))
                      .to<std::vector>();
            std::array<std::vector<int>, 3> workgroupIndices;
            for(int i = 0; i < 3; i++)
            {
                workgroupIndices[i]
                    = gi.coords()
                          ->findElements(
                              isSubDim<KernelGraph::CoordinateGraph::Workgroup>(gi.coords(), i))
                          .to<std::vector>();
            }

            // The main loop will only iterate over the one workgroup
            // dimension that is associated with this matrix.
            // Set all of them to 0 beforehand.
            for(int i = 0; i < 3; i++)
                gi.setCoordinate(workgroupIndices[i], 0);

            auto totalWorkgroupSize = product(invocation.workgroupSize);

            for(size_t j = 0; j < totalElements; j++)
            {
                setIndexCoords(gi, elementCoords, j, fixedElements);

                for(int threadIdx = 0; threadIdx < totalWorkgroupSize; threadIdx++)
                {
                    gi.setCoordinate(workitemIndices, threadIdx);

                    for(int workgroupIndex = 0; workgroupIndex < workgroupCount[hipWorkgroupIndex];
                        workgroupIndex++)
                    {
                        gi.setCoordinate(workgroupIndices[hipWorkgroupIndex], workgroupIndex);

                        auto matIdx    = gi.getLoadIndex(loadDim);
                        matrix[matIdx] = workgroupIndex;
                    }
                }
            }

            writeMatrix("workgroups_" + matName + ".dat", size0, size1, matrix);
        }

        /**
         * Write a volume where the value is the K for loop index which reads the
         * A matrix, for the entire A matrix.
         *
         * \param vfile Log file
         * \param gi
         * \param loadDim Dimension of the coordinate graph associated with the matrix
         * \param tileSizeK Size of the kernel's macrotile in the K dimension.
         * \param hipWorkgroupIndex 0, 1, or 2, are we iterating over workgroup x, y, or z?
         * \param argPrefix The prefix of the argument names associated with this matrix. "Load_Tiled_X" where X is a number
         * \param matName A name for the matrix, to be used in logging and naming the output file.
         */
        void writeMatByKIter(std::ostream&      vfile,
                             GraphInspector&    gi,
                             int                loadDim,
                             int                hipWorkgroupIndex,
                             int                tileSizeK,
                             std::string const& argPrefix,
                             std::string const& matName)
        {
            AssertFatal(tileSizeK > 0, ShowValue(tileSizeK));

            auto invocation     = gi.kernelInvocation();
            auto workgroupCount = invocation.workitemCount;
            for(int i = 0; i < 3; i++)
                workgroupCount[i] = invocation.workitemCount[i] / invocation.workgroupSize[i];
            vfile << "Workgroups: " << workgroupCount[0] << "x" << workgroupCount[1] << "x"
                  << workgroupCount[2] << std::endl;

            size_t size0, size1;
            getMatrixDims(size0, size1, gi, argPrefix);

            std::vector<int> matrix(size0 * size1, -10);

            auto workitemIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::Workitem>(gi.coords()))
                      .to<std::vector>();
            std::array<std::vector<int>, 3> workgroupIndices;
            for(int i = 0; i < 3; i++)
            {
                workgroupIndices[i]
                    = gi.coords()
                          ->findElements(
                              isSubDim<KernelGraph::CoordinateGraph::Workgroup>(gi.coords(), i))
                          .to<std::vector>();
            }

            // The main loop will only iterate over the one workgroup
            // dimension that is associated with this matrix.
            // Set all of them to 0 beforehand.
            for(int i = 0; i < 3; i++)
                gi.setCoordinate(workgroupIndices[i], 0);

            auto forLoopIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::ForLoop>(gi.coords()))
                      .to<std::vector>();

            auto totalWorkgroupSize = product(invocation.workgroupSize);

            auto fixedElements
                = findElementNumberNodes(gi.coords(), loadDim, Graph::Direction::Downstream);

            size_t totalElements = 1;
            for(auto const& tup : fixedElements)
                totalElements *= std::get<1>(tup);
            std::vector<size_t> elementCoords(fixedElements.size(), 0);

            for(size_t j = 0; j < totalElements; j++)
            {
                setIndexCoords(gi, elementCoords, j, fixedElements);
                for(int threadIdx = 0; threadIdx < totalWorkgroupSize; threadIdx++)
                {
                    gi.setCoordinate(workitemIndices, threadIdx);

                    for(int blockIdx_x = 0; blockIdx_x < workgroupCount[hipWorkgroupIndex];
                        blockIdx_x++)
                    {
                        gi.setCoordinate(workgroupIndices[hipWorkgroupIndex], blockIdx_x);

                        for(int i = 0; i < size1 / tileSizeK; i++)
                        {
                            gi.setCoordinate(forLoopIndices, i);
                            auto matIdx    = gi.getLoadIndex(loadDim);
                            matrix[matIdx] = i;
                        }
                    }
                }
            }

            writeMatrix("loop_idx_" + matName + ".dat", size0, size1, matrix);
        }

        auto threadTileSize(
            std::shared_ptr<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph> coords,
            size_t                                                                    size)
        {
            return [coords, size](int idx) -> bool {
                if(coords->getElementType(idx) != Graph::ElementType::Node)
                    return false;

                auto const& node
                    = std::get<KernelGraph::CoordinateGraph::Dimension>(coords->getElement(idx));
                if(!std::holds_alternative<KernelGraph::CoordinateGraph::ElementNumber>(node))
                    return false;

                auto const& tt = std::get<KernelGraph::CoordinateGraph::ElementNumber>(node);

                return std::visit(to_size_t, Expression::evaluate(tt.size)) == size;
            };
        }

        void visualize(std::shared_ptr<Command> command,
                       CommandKernel&           kc,
                       KernelArguments const&   commandArgs)
        {
            auto filename = Settings::getInstance()->get(Settings::LogFile) + "gemm.vis";
            std::cout << "Visualizing to " << filename << std::endl;
            std::ofstream vfile(filename);

            GraphInspector gi(command, kc, commandArgs);
            gi.inventExecutionCoordinates();

            auto coords = gi.coords();

            int loadA  = gi.findLoadTile(0);
            int loadB  = gi.findLoadTile(1);
            int loadC  = gi.findLoadTile(2);
            int storeD = gi.findStoreTile(8);

            int ldsA = findLDSNode(coords, loadA, Graph::Direction::Downstream);
            int ldsB = findLDSNode(coords, loadB, Graph::Direction::Downstream);

            writeMacrotileByWorkitem(vfile, gi, loadA, "Load_Tiled_0", "A");
            writeMacrotileByWorkitem(vfile, gi, loadB, "Load_Tiled_1", "B");

            writeLDSByWorkitem(vfile, gi, ldsA, 128, 16, "A");
            writeLDSByWorkitem(vfile, gi, ldsB, 256, 16, "B");

            writeKIterByWorkitem(vfile, gi, loadA, 0, "Load_Tiled_0", "A");
            writeKIterByWorkitem(vfile, gi, loadB, 1, "Load_Tiled_1", "B");

            // NOTE: The last argument here needs to be changed to match the MacroTile K dimension.
            // It's currently set for the Guidepost value of 16.
            writeMatByKIter(vfile, gi, loadA, 0, 16, "Load_Tiled_0", "A");
            writeMatByKIter(vfile, gi, loadB, 1, 16, "Load_Tiled_1", "B");

            auto cElemNodes = findElementNumberNodes(coords, loadC, Graph::Direction::Downstream);
            auto dElemNodes = findElementNumberNodes(coords, storeD, Graph::Direction::Upstream);

            vfile << "C Element Numbers:";
            for(auto const& tup : cElemNodes)
                vfile << " {" << std::get<0>(tup) << ", " << std::get<1>(tup) << "}";
            vfile << std::endl;

            vfile << "D Element Numbers:";
            for(auto const& tup : dElemNodes)
                vfile << " {" << std::get<0>(tup) << ", " << std::get<1>(tup) << "}";
            vfile << std::endl;
        }

    }
}
