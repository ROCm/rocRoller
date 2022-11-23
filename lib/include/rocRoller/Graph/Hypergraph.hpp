#pragma once

#include "Hypergraph_fwd.hpp"

#include <boost/multi_index_container.hpp>

#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include <functional>
#include <map>
#include <ranges>
#include <unordered_set>
#include <variant>
#include <vector>

#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    namespace Graph
    {
        enum class ElementType : int
        {
            Node = 0,
            Edge,
            Count
        };

        /**
         * @brief Returns the complementary type to `t`.
         */
        ElementType getConnectingType(ElementType t);

        enum class Direction : int
        {
            Upstream = 0,
            Downstream,
            Count
        };

        namespace mi = boost::multi_index;

        template <typename Node, typename Edge, bool Hyper = true>
        class Hypergraph
        {
        public:
            using Element = std::variant<Node, Edge>;

            struct Incident
            {
                int src;
                int dst;
                int edgeOrder;
            };

            /**
             *
             */
            struct Location
            {
                int              index;
                std::vector<int> incoming;
                std::vector<int> outgoing;

                Element element;

                constexpr inline bool operator==(Location const& rhs) const;
            };

        public:
            /**
             * @brief Returns whether `index` points to a node or an edge.
             */
            ElementType getElementType(int index) const;

            template <typename T>
            T getNode(int index) const;

            /**
             * @brief Returns whether `e` is a node or an edge.
             */
            ElementType getElementType(Element const& e) const;

            void clearCache();

            template <typename T>
            int addElement(T&& element);

            /**
             * @brief Set (overwrite) existing element.
             *
             * Asserts that the index exists already.
             */
            template <typename T>
            void setElement(int index, T&& element);

            template <typename T>
            int addElement(T&&                        element,
                           std::initializer_list<int> inputs,
                           std::initializer_list<int> outputs);

            // clang-format off
            template <typename T,
                      std::ranges::forward_range T_Inputs,
                      std::ranges::forward_range T_Outputs>
            requires(std::convertible_to<std::ranges::range_value_t<T_Inputs>, int>
                  && std::convertible_to<std::ranges::range_value_t<T_Outputs>, int>)
                // clang-format on
                int addElement(T&& element, T_Inputs const& inputs, T_Outputs const& outputs);

            // clang-format off
            template <typename T,
                      std::ranges::forward_range T_Inputs,
                      std::ranges::forward_range T_Outputs>
            requires(std::convertible_to<std::ranges::range_value_t<T_Inputs>, int>
                  && std::convertible_to<std::ranges::range_value_t<T_Outputs>, int>)
                // clang-format on
                void addElement(int              index,
                                T&&              element,
                                T_Inputs const&  inputs,
                                T_Outputs const& outputs);

            void deleteElement(int index);

            // clang-format off
            template <std::ranges::forward_range T_Inputs, std::ranges::forward_range T_Outputs>
            requires(std::convertible_to<std::ranges::range_value_t<T_Inputs>,  int>
                  && std::convertible_to<std::ranges::range_value_t<T_Outputs>, int>)
                // clang-format on
                void deleteElement(T_Inputs const&                        inputs,
                                   T_Outputs const&                       outputs,
                                   const std::function<bool(Edge const&)> edgePredicate);

            // clang-format off
            template <typename T,
                      std::ranges::forward_range T_Inputs,
                      std::ranges::forward_range T_Outputs>
            requires(std::convertible_to<std::ranges::range_value_t<T_Inputs>,  int>
                  && std::convertible_to<std::ranges::range_value_t<T_Outputs>, int>
                  && std::constructible_from<Edge,T>)
                // clang-format on
                void deleteElement(T_Inputs const& inputs, T_Outputs const& outputs);

            size_t getIncidenceSize() const;
            size_t getElementCount() const;

            Element const& getElement(int index) const;

            /**
             * @brief Returns a Location info object detailing connections to the element `index`.
             */
            Location getLocation(int index) const;

            /**
             * @brief Yields element indices without any incoming connections.
             */
            Generator<int> roots() const;

            /**
             * @brief Yields element indices without any outgoing connections.
             */
            Generator<int> leaves() const;

            /**
            * @brief Yields element indices that are the child nodes of a given element
            */
            Generator<int> childNodes(int parent) const;

            /**
            * @brief Yields element indices that are the parent nodes of a given element
            */
            Generator<int> parentNodes(int child) const;

            /**
             * @brief Yields node indices connected in the specified direction to start, in depth-first order
             */
            Generator<int> depthFirstVisit(int start, Direction dir = Direction::Downstream) const;

            /**
             * @brief Yields node indices connected in the specified direction to start, that satisfy the node selector.
             */
            Generator<int> findNodes(int                      start,
                                     std::function<bool(int)> nodeSelector,
                                     Direction                dir = Direction::Downstream) const;

            /**
             * @brief Yields node indices connected in the specified direction to starts, in depth-first order
             */
            template <std::ranges::forward_range Range>
            requires(std::convertible_to<std::ranges::range_value_t<Range>, int>)
                Generator<int> depthFirstVisit(Range& starts, Direction dir = Direction::Downstream)
            const;

            /**
             * @brief Yields node indices connected in the specified direction to start, in depth-first order.
             *
             * Will not yield any nodes in `visitedNodes`, and will insert nodes `visitedNodes` to track already
             * visited nodes.
             */
            template <Direction Dir>
            Generator<int> depthFirstVisit(int start, std::unordered_set<int>& visitedNodes) const;

            /**
             * @brief Yields node indices connected downstream of start, in breadth-first order.
             */
            Generator<int> breadthFirstVisit(int start) const;

            /**
            * @brief Yields node indices that form the paths from the starts to the ends
            */
            template <Direction Dir>
            Generator<int> path(
                std::vector<int> const   starts,
                std::vector<int> const   ends,
                std::map<int, bool>&     visitedElements,
                std::function<bool(int)> edgeSelector = [](int edge) { return true; }) const;

            template <Direction Dir>
            Generator<int> getNeighbours(int const element) const;

            /**
             * @brief Return edges in topological order.
             *
             * Traversing edges in topological order preserves edge
             * dependencies.
             */
            Generator<int> topologicalSort() const;

            std::string toDOT(std::string prefix = "", bool standalone = true) const;

            template <typename T>
            requires(std::constructible_from<Node, T> || std::constructible_from<Edge, T>)
                Generator<int> getElements()
            const;

            /**
             * @brief Yields indices of all Nodes of class T.
             */
            template <typename T = Node>
            requires(std::constructible_from<Node, T>) Generator<int> getNodes()
            const;

            /**
             * Return all Edges of class T.
             */
            template <typename T = Edge>
            requires(std::constructible_from<Edge, T>) Generator<int> getEdges()
            const;

            /**
             * @brief Yields indices of nodes that immediately preceed `dst` where the Edges are of type T.
             */
            template <typename T>
            requires(std::constructible_from<Edge, T>) Generator<int> getInputNodeIndices(
                int const dst)
            const;

            /**
             * @brief Yields indices of nodes that immediately preceed `dst` where the Edges satisfy the edgePredicate.
             */
            Generator<int>
                getInputNodeIndices(int const                              dst,
                                    const std::function<bool(Edge const&)> edgePredicate) const;

            /**
             * @brief Yields indices of nodes that immediately follow `src` where the Edges are of type T.
             */
            template <typename T>
            requires(std::constructible_from<Edge, T>) Generator<int> getOutputNodeIndices(
                int const src)
            const;

            /**
             * @brief Yields indices of nodes that immediately follow `src` where the Edges satisfy the edgePredicate.
             */
            Generator<int>
                getOutputNodeIndices(int const                              src,
                                     const std::function<bool(Edge const&)> edgePredicate) const;

        private:
            int m_nextIndex = 1;

            mutable std::map<int, Location> m_locationCache;

            // TODO: May need to replace with multi_index for in-place rewriting.
            std::map<int, Element> m_elements;

            struct BySrc
            {
            };
            struct ByDst
            {
            };
            struct BySrcDst
            {
            };

            using Incidence = mi::multi_index_container<
                Incident,
                mi::indexed_by<
                    mi::ordered_non_unique<mi::tag<BySrc>,
                                           mi::key<&Incident::src, &Incident::edgeOrder>>,
                    mi::ordered_non_unique<mi::tag<ByDst>,
                                           mi::key<&Incident::dst, &Incident::edgeOrder>>,
                    // This prevents parallel incidents.
                    mi::ordered_unique<mi::tag<BySrcDst>,
                                       mi::key<&Incident::src, &Incident::dst>>>>;

            Incidence m_incidence;

            template <Direction Dir>
            bool edgeSatisfied(int const edge, std::map<int, bool> const& visitedElements) const;
        };

        template <typename Node, typename Edge, bool Hyper>
        std::ostream& operator<<(std::ostream& stream, Hypergraph<Node, Edge, Hyper> const& graph);
    }
}

#include "Hypergraph_impl.hpp"
