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

#pragma once

#include <concepts>

#include <rocRoller/Graph/Hypergraph.hpp>

namespace rocRoller
{
    namespace Graph
    {
        /**
         * `graph` must be an instantiation of Hypergraph which is calm (i.e.
         * not hyper).
         *
         * For each edge in `graph` that matches `edgePredicate`, delete that
         * edge if its destination node would still be reachable from the
         * source, only following edges that satisfy edgePredicate.
         */
        template <CCalmGraph AGraph, std::predicate<int> EdgePredicate>
        void removeRedundantEdges(AGraph& graph, EdgePredicate edgePredicate);

        /**
         * `graph` must be an instantiation of Hypergraph which is calm (i.e.
         * not hyper).
         *
         * For each edge in `graph` that matches `edgePredicate`, delete that
         * edge if its destination node would still be reachable from the
         * source, only following edges that satisfy edgePredicate.
         */
        template <CCalmGraph AGraph, std::predicate<int> EdgePredicate>
        Generator<int> findRedundantEdges(AGraph const& graph, EdgePredicate edgePredicate);
    }
}

#include "GraphUtilities_impl.hpp"
