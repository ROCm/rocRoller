
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        KernelGraph CleanLoops::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::cleanLoops");

            auto k = original;
            for(auto const& loop : k.control.getNodes<ForLoopOp>().to<std::vector>())
            {
                auto [lhs, rhs] = getForLoopIncrement(k, loop);
                auto forLoopDim
                    = getSize(k.coordinates.getNode(k.mapper.get(loop, NaryArgument::DEST)));

                //Ensure forLoopDim is translate time evaluatable.
                if(!(evaluationTimes(forLoopDim)[EvaluationTime::Translate]))
                    continue;

                //Ensure RHS is translate time evaluatable.
                if(!(evaluationTimes(rhs)[EvaluationTime::Translate]))
                    continue;

                //Only remove single iteration loops!
                if(evaluate(rhs) != evaluate(forLoopDim))
                    continue;

                // Replace ForLoop with Scope if there are loads or stores within the loop,
                // otherwise replace with a NOP; ideally would reconnect but OK for now
                int numComputeIndexCandidates = 0;
                for(auto const& body : k.control.getOutputNodeIndices<Body>(loop))
                    numComputeIndexCandidates += findComputeIndexCandidates(k, body).size();

                if(numComputeIndexCandidates == 0)
                {
                    replaceWith(k, loop, k.control.addElement(NOP()));
                }
                else
                {
                    replaceWith(k, loop, k.control.addElement(Scope()));
                }
                purgeFor(k, loop);
            }

            return k;
        }
    }
}
