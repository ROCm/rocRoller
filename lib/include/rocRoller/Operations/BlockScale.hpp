/**
 * Block scale MX datatypes command.
 */

#pragma once

#include "Operation.hpp"
#include <optional>
#include <unordered_set>
#include <vector>

namespace rocRoller
{
    namespace Operations
    {
        /**
         * A block scale operation for MX datatypes
        */
        class BlockScale : public BaseOperation
        {
        public:
            BlockScale() = delete;

            /**
             * @param data Tag of data to-be-scaled
             * @param dimensions Number of dimensions of `data`
             * @param scale Optional tag of scale tensor (if not provided, treated as inline block scale)
             * @param strides Strides of the scale
            */
            BlockScale(OperationTag                data,
                       int                         dimensions,
                       std::optional<OperationTag> scale   = {},
                       std::vector<size_t> const&  strides = {});
            enum class PointerMode
            {
                Separate, //< Scale is separate from data
                Inline, //< Scale is inline with data
            };

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;
            PointerMode                      pointerMode() const;
            const std::vector<size_t>&       strides() const;

        private:
            const OperationTag                m_data;
            const std::optional<OperationTag> m_scale;
            const std::vector<size_t>         m_strides;
        };

    }
}
