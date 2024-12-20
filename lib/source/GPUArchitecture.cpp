#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>

#include <rocRoller/InstructionValues/Register.hpp>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <rocRoller/Serialization/GPUArchitecture.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Serialization/msgpack/Msgpack.hpp>

#ifndef ROCROLLER_NO_EMBED_ARCH_DEF
#include <cmrc/cmrc.hpp>

CMRC_DECLARE(rocRoller);
#endif

namespace rocRoller
{
    std::string
        GPUArchitecture::writeYaml(std::map<GPUArchitectureTarget, GPUArchitecture> const& input)
    {
        rocRoller::GPUArchitecturesStruct tmp;
        tmp.architectures = input;

        return Serialization::toYAML(tmp);
    }

    std::map<GPUArchitectureTarget, GPUArchitecture>
        GPUArchitecture::readYaml(std::string const& input)
    {
        auto archStruct = Serialization::readYAMLFile<GPUArchitecturesStruct>(input);
        return archStruct.architectures;
    }

    std::string
        GPUArchitecture::writeMsgpack(std::map<GPUArchitectureTarget, GPUArchitecture> const& input)
    {
        std::stringstream buffer;
        msgpack::pack(buffer, input);

        return buffer.str();
    }

    std::map<GPUArchitectureTarget, GPUArchitecture>
        GPUArchitecture::readMsgpack(std::string const& input)
    {
        try
        {
            std::ifstream ifs(input);
            std::string   buffer((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());

            msgpack::object_handle oh  = msgpack::unpack(buffer.data(), buffer.size());
            msgpack::object const& obj = oh.get();

            auto rv = obj.as<std::map<GPUArchitectureTarget, GPUArchitecture>>();
            return rv;
        }
        catch(const std::exception& e)
        {
            Throw<FatalError>("GPUArchitecture::readMsgpack(", input, ") failed: ", e.what());
        }
    }

    std::map<GPUArchitectureTarget, GPUArchitecture> GPUArchitecture::readEmbeddedMsgpack()
    {
#ifndef ROCROLLER_NO_EMBED_ARCH_DEF
        try
        {
            auto                   fs = cmrc::rocRoller::get_filesystem();
            auto                   fd = fs.open("resources/GPUArchitecture_def.msgpack");
            msgpack::object_handle oh = msgpack::unpack(
                fd.begin(),
                reinterpret_cast<uintptr_t>(fd.end()) - reinterpret_cast<uintptr_t>(fd.begin()));
            msgpack::object const& obj = oh.get();

            auto rv = obj.as<std::map<GPUArchitectureTarget, GPUArchitecture>>();
            return rv;
        }
        catch(const std::exception& e)
        {
            Throw<FatalError>("GPUArchitecture::readEmbeddedMsgpack() failed: ", e.what());
        }
#else
        Throw<FatalError>(
            "GPUArchitecture::readEmbeddedMsgpack() failed: no embedded architecture definitions");
#endif
    }

    bool GPUArchitecture::isSupportedConstantValue(Register::ValuePtr value) const
    {
        if(value->regType() != Register::Type::Literal)
            return false;

        return std::visit(
            [this](auto val) -> bool {
                using T = std::decay_t<decltype(val)>;
                if constexpr(std::is_pointer<T>::value || std::is_same<bool, T>::value)
                {
                    return false;
                }
                else
                {
                    return this->isSupportedConstantValue(val);
                }
            },
            value->getLiteralValue());
    }
}
