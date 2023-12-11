#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>

namespace rocRoller::KernelGraph
{

    void ScopeManager::pushNewScope()
    {
        m_tags.push_back({});
    }

    void ScopeManager::addRegister(int tag)
    {
        // if alreay in a scope, skip
        for(auto s : m_tags)
        {
            if(s.count(tag) > 0)
                return;
        }
        m_tags.back().insert(tag);
    }

    void ScopeManager::popAndReleaseScope()
    {
        for(auto tag : m_tags.back())
        {
            // TODO: Add a way to allocate AGPR registers within a loop
            //       that can be used later in a different loop.
            if(m_context->registerTagManager()->hasRegister(tag))
            {
                auto reg = m_context->registerTagManager()->getRegister(tag);
                if(reg->regType() != Register::Type::Accumulator)
                    m_context->registerTagManager()->deleteTag(tag);
            }
        }
        m_tags.pop_back();
    }

}
