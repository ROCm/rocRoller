

#include <rocRoller/AssemblyKernel.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>

#include <rocRoller/Serialization/AssemblyKernel.hpp>
#include <rocRoller/Serialization/YAML.hpp>

namespace rocRoller
{
    std::string AssemblyKernel::amdgpu_metadata_yaml()
    {
        AssemblyKernels tmp;
        tmp.kernels = {*this};

        return Serialization::toYAML(tmp);
    }
}
