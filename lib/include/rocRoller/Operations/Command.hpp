/**
 */

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "CommandArgument_fwd.hpp"
#include "Operations_fwd.hpp"

#include "../AssemblyKernel_fwd.hpp"
#include "../DataTypes/DataTypes.hpp"
#include "../Expression.hpp"

namespace rocRoller
{

    struct Command : public std::enable_shared_from_this<Command>
    {
    public:
        using OperationList = std::vector<std::shared_ptr<Operations::Operation>>;

        Command();
        Command(bool sync);
        ~Command();

        Command(Command const& rhs);
        Command(Command&& rhs);

        void                             addOperation(std::shared_ptr<Operations::Operation> op);
        std::shared_ptr<CommandArgument> allocateArgument(VariableType  variableType,
                                                          DataDirection direction
                                                          = DataDirection::ReadWrite);
        std::shared_ptr<CommandArgument> allocateArgument(VariableType       variableType,
                                                          DataDirection      direction,
                                                          std::string const& name);

        std::vector<std::shared_ptr<CommandArgument>> allocateArgumentVector(
            DataType dataType, int length, DataDirection direction = DataDirection::ReadWrite);

        std::vector<std::shared_ptr<CommandArgument>> allocateArgumentVector(
            DataType dataType, int length, DataDirection direction, std::string const& name);

        std::shared_ptr<Operations::Operation> findTag(int tag);
        int                                    getNextTag() const;

        OperationList const& operations() const;

        std::string toString() const;
        std::string toString(const unsigned char*) const;
        std::string argInfo() const;

        /**
         * Returns a list of all of the CommandArguments within a Command.
         */
        std::vector<std::shared_ptr<CommandArgument>> getArguments() const;

        /**
         * Returns the expected workItemCount for a command.
         * This is determined by finding the sizes for the first T_LOAD_LINEAR
         * command appearing in the command object.
         */
        std::array<Expression::ExpressionPtr, 3> createWorkItemCount() const;

    private:
        /// Does this command need to synchronize before continuing the calling CPU thread?
        bool m_sync = false;

        OperationList m_operations;

        std::map<int, std::shared_ptr<Operations::Operation>> m_tagMap;

        std::vector<std::shared_ptr<CommandArgument>> m_command_args;

        int m_nextTagValue = 0;

        int m_runtime_args_offset = 0;
    };

    std::ostream& operator<<(std::ostream& stream, Command const& command);

}

#include "Command_impl.hpp"
