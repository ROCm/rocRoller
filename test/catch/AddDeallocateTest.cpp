/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDeallocate_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <common/CommonGraphs.hpp>

#include "TestContext.hpp"

namespace AddDeallocateTest
{
    TEST_CASE("AddDeallocate", "[kernel-graph]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        using GD = Graph::Direction;

        auto context = TestContext::ForTestDevice();
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        example.setTileSize(128, 256, 8);
        example.setMFMA(32, 32, 2, 1);
        example.setUseLDS(true, true, true);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        params->unrollK           = 4;
        params->prefetch          = true;
        params->prefetchInFlight  = 2;
        params->prefetchLDSFactor = 2;
        params->prefetchMixMemOps = true;

        std::vector<GraphTransformPtr> transforms;
        transforms.push_back(std::make_shared<IdentifyParallelDimensions>());
        transforms.push_back(std::make_shared<OrderMemory>(false));
        transforms.push_back(std::make_shared<UpdateParameters>(params));
        transforms.push_back(std::make_shared<AddLDS>(params, context.get()));
        transforms.push_back(std::make_shared<LowerLinear>(context.get()));
        transforms.push_back(std::make_shared<LowerTile>(params, context.get()));
        transforms.push_back(std::make_shared<LowerTensorContraction>(params, context.get()));
        transforms.push_back(std::make_shared<Simplify>());
        transforms.push_back(std::make_shared<FuseExpressions>());
        transforms.push_back(std::make_shared<ConnectWorkgroups>(params, context.get()));
        transforms.push_back(std::make_shared<UnrollLoops>(params, context.get()));
        transforms.push_back(std::make_shared<FuseLoops>());
        transforms.push_back(std::make_shared<RemoveDuplicates>());
        transforms.push_back(std::make_shared<OrderEpilogueBlocks>());
        transforms.push_back(std::make_shared<CleanLoops>());
        transforms.push_back(std::make_shared<AddPrefetch>(params, context.get()));
        transforms.push_back(std::make_shared<AddComputeIndex>());

        for(auto& t : transforms)
            kgraph = kgraph.transform(t);

        SECTION("Downstream Barriers")
        {
            auto topo
                = TopologicalCompare(std::make_shared<rocRoller::KernelGraph::KernelGraph>(kgraph));

            auto graph  = kgraph;
            auto tracer = LastRWTracer(graph);

            for(auto& [coordinate, controls] : tracer.lastRWLocations())
            {
                auto dependencies = controls;

                auto maybeLDS = graph.coordinates.get<LDS>(coordinate);
                if(maybeLDS)
                {
                    AddDeallocateDetail::addDownstreamBarrierInLoop(
                        dependencies, coordinate, controls, kgraph, topo);
                    CHECK(dependencies.size() >= 1);
                    if(dependencies.size() == 1)
                    {
                        // If there is only one dependency, it should be a ForLoop
                        CHECK(
                            kgraph.control.get<ForLoopOp>(only(dependencies).value()).has_value());
                    }
                    else if(dependencies.size() > 1)
                    {
                        // If there are multiple dependencies, we should have an extra barrier
                        for(auto& tag : dependencies)
                        {
                            if(controls.contains(tag))
                                continue;

                            CHECK(graph.control.get<Barrier>(tag).has_value());
                        }
                    }
                }
            }
        }

        SECTION("Placement of LDS Deallocates")
        {
            kgraph = kgraph.transform(std::make_shared<NopExtraScopes>());
            kgraph = kgraph.transform(std::make_shared<AddDeallocateDataFlow>());

            auto ldsDeallocatePredicate = [&](int tag) -> bool {
                auto maybeDeallocate = kgraph.control.get<Deallocate>(tag);
                if(!maybeDeallocate)
                    return false;
                bool anyLDS = false;
                for(auto conn : kgraph.mapper.getConnections(tag))
                {
                    auto maybeLDS = kgraph.coordinates.get<LDS>(conn.coordinate);
                    anyLDS |= maybeLDS.has_value();
                }
                return anyLDS;
            };

            auto forKLoopPredicate = [&](int tag) -> bool {
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
                if(!maybeForLoop)
                    return false;
                return maybeForLoop->loopName == rocRoller::KLOOP;
            };

            auto kernel = *only(kgraph.control.roots());
            auto forLoop
                = *only(kgraph.control.findNodes(kernel, forKLoopPredicate, GD::Downstream));

            auto ldsDeallocateFromKernel
                = kgraph.control.findNodes(kernel, ldsDeallocatePredicate, GD::Downstream)
                      .to<std::set>();

            std::set<int> ldsDeallocateInsideLoop;
            for(auto body : kgraph.control.getOutputNodeIndices<Body>(forLoop))
            {
                auto t = kgraph.control.findNodes(body, ldsDeallocatePredicate, GD::Downstream)
                             .to<std::vector>();
                std::copy(t.cbegin(),
                          t.cend(),
                          std::inserter(ldsDeallocateInsideLoop, ldsDeallocateInsideLoop.end()));
            }

            // With 4 unrolls and 2 inflight prefetches, we expect the following
            // 1. A, B deallocated three times in the main loop
            // 2. A, B deallocate once after the main loop
            // 3. C deallocated once after the prolog
            CHECK(ldsDeallocateFromKernel.size() == 5);
            CHECK(ldsDeallocateInsideLoop.size() == 3);
        }
    }
}
