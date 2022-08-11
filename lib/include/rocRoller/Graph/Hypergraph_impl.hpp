#pragma once

#include "Hypergraph.hpp"

#include <boost/multi_index_container.hpp>

#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include <algorithm>
#include <compare>
#include <map>
#include <ranges>
#include <unordered_set>
#include <variant>
#include <vector>

#include "../Utilities/Comparison.hpp"
#include "../Utilities/Error.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    namespace Graph
    {
        template <typename Node, typename Edge, bool Hyper>
        constexpr inline bool
            Hypergraph<Node, Edge, Hyper>::Location::operator==(Location const& rhs) const
        {
            // clang-format off
            return LexicographicCompare(
                        index, rhs.index,
                        incoming, rhs.incoming,
                        outgoing, rhs.outgoing) == 0;
            // clang-format on
        }

        template <typename Node, typename Edge, bool Hyper>
        ElementType Hypergraph<Node, Edge, Hyper>::getElementType(int index) const
        {
            return getElementType(m_elements.at(index));
        }

        template <typename Node, typename Edge, bool Hyper>
        ElementType Hypergraph<Node, Edge, Hyper>::getElementType(Element const& e) const
        {
            if(holds_alternative<Node>(e))
                return ElementType::Node;
            return ElementType::Edge;
        }

        inline ElementType getConnectingType(ElementType t)
        {
            return t == ElementType::Node ? ElementType::Edge : ElementType::Node;
        }

        template <typename Node, typename Edge, bool Hyper>
        void Hypergraph<Node, Edge, Hyper>::clearCache()
        {
            m_locationCache.clear();
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        int Hypergraph<Node, Edge, Hyper>::addElement(T&& element)
        {
            int index = m_nextIndex;
            m_nextIndex++;
            addElement(index, std::forward<T>(element));

            return index;
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        void Hypergraph<Node, Edge, Hyper>::addElement(int index, T&& element)
        {
            AssertFatal(m_elements.find(index) == m_elements.end());

            m_elements[index] = std::forward<T>(element);
            clearCache();
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        int Hypergraph<Node, Edge, Hyper>::addElement(T&&                        element,
                                                      std::initializer_list<int> inputs,
                                                      std::initializer_list<int> outputs)
        {
            return addElement<T, std::initializer_list<int>, std::initializer_list<int>>(
                std::forward<T>(element), inputs, outputs);
        }

        // clang-format off
        template <typename Node, typename Edge, bool Hyper>
        template <typename T,
                  std::ranges::forward_range T_Inputs,
                  std::ranges::forward_range T_Outputs>
        requires(std::convertible_to<std::ranges::range_value_t<T_Inputs>,  int>
              && std::convertible_to<std::ranges::range_value_t<T_Outputs>, int>)
            // clang-format on
            int Hypergraph<Node, Edge, Hyper>::addElement(T&&              element,
                                                          T_Inputs const&  inputs,
                                                          T_Outputs const& outputs)
        {
            int index = m_nextIndex;
            m_nextIndex++;
            addElement(index, std::forward<T>(element), inputs, outputs);

            return index;
        }

        // clang-format off
        template <typename Node, typename Edge, bool Hyper>
        template <typename T,
                    std::ranges::forward_range T_Inputs,
                    std::ranges::forward_range T_Outputs>
        requires(std::convertible_to<std::ranges::range_value_t<T_Inputs>,  int>
              && std::convertible_to<std::ranges::range_value_t<T_Outputs>, int>)
            // clang-format on
            void Hypergraph<Node, Edge, Hyper>::addElement(int              index,
                                                           T&&              element,
                                                           T_Inputs const&  inputs,
                                                           T_Outputs const& outputs)
        {
            AssertFatal(m_elements.find(index) == m_elements.end());

            auto elementType    = getElementType(element);
            auto connectingType = getConnectingType(elementType);

            for(int cIdx : inputs)
            {
                AssertFatal(getElementType(cIdx) == connectingType);
            }
            for(int cIdx : outputs)
            {
                AssertFatal(getElementType(cIdx) == connectingType);
            }

            clearCache();

            m_elements[index] = std::forward<T>(element);

            if(elementType == ElementType::Edge)
            {
                int incidentOrder = 0;
                for(int src : inputs)
                {
                    m_incidence.insert({src, index, incidentOrder++});
                }
                incidentOrder = 0;
                for(int dst : outputs)
                {
                    m_incidence.insert({index, dst, incidentOrder++});
                }
            }
            else
            {
                auto const& bySrc = m_incidence.template get<BySrc>();
                for(int src : inputs)
                {
                    int incidentOrder = 0;

                    auto lastSrcIter = bySrc.lower_bound(std::make_tuple(src + 1, 0));
                    if(lastSrcIter != bySrc.begin())
                        lastSrcIter--;
                    if(lastSrcIter != bySrc.end() && lastSrcIter->src == src)
                        incidentOrder = lastSrcIter->edgeOrder + 1;

                    m_incidence.insert({src, index, incidentOrder});
                }

                auto const& byDst = m_incidence.template get<ByDst>();
                for(int dst : outputs)
                {
                    int incidentOrder = 0;

                    auto lastDstIter = byDst.lower_bound(std::make_tuple(dst + 1, 0));
                    if(lastDstIter != byDst.begin())
                        lastDstIter--;
                    if(lastDstIter != byDst.end() && lastDstIter->dst == dst)
                        incidentOrder = lastDstIter->edgeOrder + 1;

                    m_incidence.insert({index, dst, incidentOrder});
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        size_t Hypergraph<Node, Edge, Hyper>::getIncidenceSize() const
        {
            return m_incidence.size();
        }

        template <typename Node, typename Edge, bool Hyper>
        size_t Hypergraph<Node, Edge, Hyper>::getElementCount() const
        {
            return m_elements.size();
        }

        template <typename Node, typename Edge, bool Hyper>
        auto Hypergraph<Node, Edge, Hyper>::getElement(int index) const -> Element const&
        {
            return m_elements.at(index);
        }

        template <typename Node, typename Edge, bool Hyper>
        auto Hypergraph<Node, Edge, Hyper>::getLocation(int index) const -> Location
        {
            {
                auto iter = m_locationCache.find(index);
                if(iter != m_locationCache.end())
                    return iter->second;
            }

            Location rv;
            rv.index   = index;
            rv.element = m_elements.at(index);

            {
                // Incoming: Find the src for incidents whose dst is our node.
                auto const& incomingLookup = m_incidence.template get<ByDst>();

                auto incomingBegin = incomingLookup.lower_bound(std::make_tuple(index, 0));
                auto incomingEnd   = incomingLookup.lower_bound(std::make_tuple(index + 1, 0));

                std::transform(incomingBegin,
                               incomingEnd,
                               std::back_inserter(rv.incoming),
                               [](auto const& inc) { return inc.src; });
            }

            {
                // Outgoing: Find the dst for incidents whose src is our node.
                auto const& outgoingLookup = m_incidence.template get<BySrc>();

                auto outgoingBegin = outgoingLookup.lower_bound(std::make_tuple(index, 0));
                auto outgoingEnd   = outgoingLookup.lower_bound(std::make_tuple(index + 1, 0));

                std::transform(outgoingBegin,
                               outgoingEnd,
                               std::back_inserter(rv.outgoing),
                               [](auto const& inc) { return inc.dst; });
            }

            m_locationCache[index] = rv;

            return rv;
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::roots() const
        {
            auto const& lookup = m_incidence.template get<ByDst>();

            for(auto const& pair : m_elements)
            {
                int  index = pair.first;
                auto iter  = lookup.lower_bound(std::make_tuple(index, 0));
                if(iter == lookup.end() || iter->dst != index)
                    co_yield index;
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::leaves() const
        {
            auto const& lookup = m_incidence.template get<BySrc>();

            for(auto const& pair : m_elements)
            {
                int  index = pair.first;
                auto iter  = lookup.lower_bound(std::make_tuple(index, 0));
                if(iter == lookup.end() || iter->src != index)
                    co_yield index;
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <std::ranges::forward_range Range>
        requires(std::convertible_to<std::ranges::range_value_t<Range>, int>)
            Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(Range&    starts,
                                                                          Direction dir)
        const
        {
            std::unordered_set<int> visitedNodes;
            if(dir == Direction::Downstream)
            {
                for(int index : starts)
                    co_yield depthFirstVisit<Direction::Downstream>(index, visitedNodes);
            }
            else
            {
                for(int index : starts)
                    co_yield depthFirstVisit<Direction::Upstream>(index, visitedNodes);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(int       start,
                                                                      Direction dir) const
        {
            std::initializer_list<int> starts{start};
            co_yield depthFirstVisit(starts, dir);
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::breadthFirstVisit(int start) const
        {
            // Only downstream is implemented for now.
            std::unordered_set<int> visitedNodes;

            visitedNodes.insert(start);

            co_yield start;

            auto chain = [](auto& iters, auto end) -> Generator<int> {
                for(int i = 0; i < iters.size(); i++)
                {
                    int src = iters[i].second;
                    for(auto iter = iters[i].first; iter != end && iter->src == src; iter++)
                    {
                        co_yield iter->dst;
                    }
                }
            };

            auto const& lookup = m_incidence.template get<BySrc>();

            auto startIter = lookup.lower_bound(std::make_tuple(start, 0));
            auto iters     = std::vector{std::make_pair(startIter, start)};

            for(int node : chain(iters, lookup.end()))
            {
                if(visitedNodes.count(node))
                    continue;

                visitedNodes.insert(node);
                co_yield node;

                iters.emplace_back(lookup.lower_bound(std::make_tuple(node, 0)), node);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction Dir>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(
            int start, std::unordered_set<int>& visitedNodes) const
        {
            if(visitedNodes.count(start))
                co_return;

            visitedNodes.insert(start);

            co_yield start;

            if constexpr(Dir == Direction::Upstream)
            {
                auto const& lookup = m_incidence.template get<ByDst>();

                for(auto iter = lookup.lower_bound(std::make_tuple(start, 0));
                    iter != lookup.end() && iter->dst == start;
                    iter++)
                {
                    co_yield depthFirstVisit<Dir>(iter->src, visitedNodes);
                }
            }
            else
            {
                auto const& lookup = m_incidence.template get<BySrc>();

                for(auto iter = lookup.lower_bound(std::make_tuple(start, 0));
                    iter != lookup.end() && iter->src == start;
                    iter++)
                {
                    co_yield depthFirstVisit<Dir>(iter->dst, visitedNodes);
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        std::string Hypergraph<Node, Edge, Hyper>::toDOT() const
        {
            std::ostringstream msg;

            msg << "digraph {" << std::endl;

            for(auto const& pair : m_elements)
            {
                msg << '"' << pair.first << '"';
                if(std::holds_alternative<Edge>(pair.second))
                    msg << "[shape=box]";
                msg << std::endl;
            }

            auto const& container = m_incidence.template get<BySrc>();
            for(auto const& incident : container)
            {
                msg << '"' << incident.src << "\" -> \"" << incident.dst << '"' << std::endl;
            }

            // Enforce left-to-right ordering for elements connected to an edge.
            for(auto const& pair : m_elements)
            {
                if(getElementType(pair.second) != ElementType::Edge)
                    continue;

                auto const& loc = getLocation(pair.first);

                if(loc.incoming.size() > 1)
                {
                    msg << "{\nrank=same\n";
                    bool first = true;
                    for(int idx : loc.incoming)
                    {
                        if(!first)
                            msg << "->";
                        msg << '"' << idx << '"';
                        first = false;
                    }
                    msg << "[style=invis]\nrankdir=LR\n}\n";
                }

                if(loc.outgoing.size() > 1)
                {
                    msg << "{\nrank=same\n";
                    bool first = true;
                    for(int idx : loc.outgoing)
                    {
                        if(!first)
                            msg << "->";
                        msg << '"' << idx << '"';
                        first = false;
                    }
                    msg << "[style=invis]\nrankdir=LR\n}\n";
                }
            }

            msg << "}" << std::endl;

            return msg.str();
        }
    }
}
