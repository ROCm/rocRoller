/**
 * @brief Common graphs used for unit tests.
 */

#pragma once

#include <vector>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>

#include <common/GEMMProblem.hpp>

namespace rocRollerTest
{
    namespace Graphs
    {
        using CommandLaunchParameters    = rocRoller::CommandLaunchParameters;
        using CommandLaunchParametersPtr = rocRoller::CommandLaunchParametersPtr;
        using CommandParameters          = rocRoller::CommandParameters;
        using CommandParametersPtr       = rocRoller::CommandParametersPtr;
        using CommandPtr                 = rocRoller::CommandPtr;
        using ContextPtr                 = rocRoller::ContextPtr;
        using KernelArguments            = rocRoller::KernelArguments;
        using KernelGraph                = rocRoller::KernelGraph::KernelGraph;
        using DataType                   = rocRoller::DataType;

        /**
         * @brief Graph for linear: alpha x + beta y.
         *
         * Creates a command graph that is essentially:
         * - LoadScalar(alpha)
         * - LoadScalar(beta)
         * - LoadLinear(x)
         * - LoadLinear(y)
         * - Assign(z = alpha x + beta y)
         * - StoreLinear(z)
         */
        template <typename T>
        class VectorAdd
        {
        public:
            VectorAdd();
            VectorAdd(bool useBeta);

            CommandPtr      getCommand();
            KernelGraph     getKernelGraph();
            KernelArguments getRuntimeArguments(size_t nx, T* alpha, T beta, T* x, T* y, T* rv);
            KernelArguments getRuntimeArguments(size_t nx, T* alpha, T* x, T* y, T* rv);
            std::vector<T>
                referenceSolution(T alpha, std::vector<T> const& x, std::vector<T> const& y);
            std::vector<T> referenceSolution(T                     alpha,
                                             T                     beta,
                                             std::vector<T> const& x,
                                             std::vector<T> const& y);

        private:
            void createCommand();

            bool       m_useBeta;
            CommandPtr m_command;
        };

        /**
         * @brief Graph for linear: - (x + y) * (x + y).
         *
         * Creates a command graph that is essentially:
         * - LoadLinear(x)
         * - LoadLinear(y)
         * - Assign(w = x + y)
         * - Assign(z = - w * w)
         * - StoreLinear(z)
         */
        template <typename T>
        class VectorAddNegSquare
        {
        public:
            VectorAddNegSquare();
            VectorAddNegSquare(bool useScalarLoads);

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

        private:
            void createCommand();

            bool       m_useScalarLoads;
            CommandPtr m_command;
        };

        /**
         * @brief Graph for tiled: A B.
         *
         * Creates a command graph that is essentially:
         * - LoadTiled(A)
         * - LoadTiled(B)
         * - Assign(D = TensorContraction(A, B))
         * - StoreTiled(D)
         */
        class MatrixMultiply
        {
        public:
            MatrixMultiply(rocRoller::DataType              aType,
                           rocRoller::DataType              bType  = rocRoller::DataType::None,
                           rocRoller::DataType              cdType = rocRoller::DataType::None,
                           rocRoller::Operations::ScaleMode aMode
                           = rocRoller::Operations::ScaleMode::None,
                           rocRoller::Operations::ScaleMode bMode
                           = rocRoller::Operations::ScaleMode::None);

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n, int k);
            void setMFMA(int m, int n, int k, int b);
            void setUseLDS(bool a, bool b, bool d);

            std::shared_ptr<CommandParameters> getCommandParameters() const;

        private:
            void createCommand();

            rocRoller::DataType m_aType;
            rocRoller::DataType m_bType;
            rocRoller::DataType m_cdType;

            rocRoller::Operations::ScaleMode m_aMode;
            rocRoller::Operations::ScaleMode m_bMode;

            int  m_macM, m_macN, m_macK;
            int  m_waveM, m_waveN, m_waveK, m_waveB;
            bool m_useLDSA = false, m_useLDSB = false, m_useLDSD = false;

            rocRoller::Operations::OperationTag m_tagA, m_tagB, m_tagD;
            rocRoller::Operations::OperationTag m_tagScaleA, m_tagScaleB;

            CommandPtr m_command;
        };

        /**
         * @brief Graph for GEMM: alpha A B + beta C
         *
         * Creates a command graph that is essentially:
         * - LoadScalar(alpha)
         * - LoadScalar(beta)
         * - LoadTiled(A)
         * - LoadTiled(B)
         * - LoadTiled(C)
         * - Assign(AB = TensorContraction(A, B))
         * - Assign(D = alpha * AB + beta * C)
         * - StoreTiled(D)
         */
        class GEMM
        {
        public:
            GEMM(DataType ta);
            GEMM(DataType ta, DataType tb);
            GEMM(DataType ta, DataType tb, DataType tc);
            GEMM(DataType ta, DataType tb, DataType tc, DataType td);

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n, int k);
            void setMFMA(int m, int n, int k, int b);
            void setUseLDS(bool a, bool b, bool d);
            void setPrefetch(bool prefetch,
                             int  prefetchInFlight,
                             int  prefetchLDSFactor,
                             bool prefetchMixMemOps);
            void setProblem(GEMMProblem const& problem);

            GEMMProblem const&   getProblem() const;
            CommandParametersPtr getCommandParameters() const;

            rocRoller::Operations::OperationTag m_tagTensorA, m_tagTensorB, m_tagTensorC,
                m_tagTensorD, m_tagScalarAlpha, m_tagScalarBeta, m_tagScalarSeed, m_tagScratch;
            rocRoller::Operations::OperationTag m_tagNumWGs;

            DataType m_ta, m_tb, m_tc, m_td;

        private:
            void createCommand();

            int  m_macM, m_macN, m_macK;
            int  m_waveM, m_waveN, m_waveK, m_waveB;
            bool m_useLDSA = false, m_useLDSB = false, m_useLDSD = false;

            rocRoller::Operations::OperationTag m_tagA, m_tagB, m_tagC, m_tagD;

            CommandPtr  m_command;
            GEMMProblem m_problem;
        };

        /**
         * @brief Graph for tiled: (x + x) + (y + y).
         *
         * Creates a command graph that is essentially:
         * - LoadTiled(x)
         * - LoadTiled(y)
         * - Assign(z = (x + x) + (y + y))
         * - StoreTiled(z)
         */
        template <typename T>
        class TileDoubleAdd
        {
        public:
            TileDoubleAdd();

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n);
            void setSubTileSize(int m, int n);

            CommandParametersPtr       getCommandParameters(size_t nx, size_t ny) const;
            CommandLaunchParametersPtr getCommandLaunchParameters(size_t nx, size_t ny) const;

            KernelArguments getRuntimeArguments(size_t nx, size_t ny, T* x, T* y, T* rv);

            std::vector<T> referenceSolution(std::vector<T> const& x, std::vector<T> const& y);

        private:
            void createCommand();

            int m_macM, m_macN;
            int m_thrM, m_thrN;

            rocRoller::Operations::OperationTag m_tagA, m_tagB, m_tagD;

            CommandPtr m_command;
        };

        /**
         * @brief Graph for tiled: x.
         *
         * Creates a command graph that is essentially:
         * - LoadTiled(x)
         * - StoreTiled(x)
         */
        template <typename T>
        class TileCopy
        {
        public:
            TileCopy();

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n);
            void setSubTileSize(int m, int n);
            void setLiteralStrides(std::vector<size_t> const& literalStrides);

            CommandParametersPtr       getCommandParameters(size_t nx, size_t ny) const;
            CommandLaunchParametersPtr getCommandLaunchParameters(size_t nx, size_t ny) const;

            KernelArguments getRuntimeArguments(size_t nx, size_t ny, T* x, T* rv);

            std::vector<T> referenceSolution(std::vector<T> const& x);

        private:
            void createCommand();

            rocRoller::Operations::OperationTag m_tag;
            int                                 m_macM, m_macN;
            int                                 m_thrM, m_thrN;

            std::vector<size_t> m_literalStrides;
            CommandPtr          m_command;
        };
    }
}
#include "CommonGraphs_impl.hpp"
