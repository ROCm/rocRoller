
#include "GPUContextFixture.hpp"

#include "Utilities.hpp"

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>

rocRoller::ContextPtr BaseGPUContextFixture::createContextLocalDevice()
{
    return rocRoller::Context::ForDefaultHipDevice(testKernelName(), m_kernelOptions);
}

void BaseGPUContextFixture::SetUp()
{
    using namespace rocRoller;

    if(!isLocalDevice())
    {
        Settings::getInstance()->set(Settings::AllowUnkownInstructions, true);
    }
    ContextFixture::SetUp();

    ASSERT_EQ(true, m_context->targetArchitecture().HasCapability(GPUCapability::SupportedISA));

    if(isLocalDevice())
    {
        int deviceIdx = m_context->hipDeviceIndex();

        ASSERT_THAT(hipInit(0), HasHipSuccess(0));
        ASSERT_THAT(hipSetDevice(deviceIdx), HasHipSuccess(0));
    }
}

rocRoller::ContextPtr
    BaseGPUContextFixture::createContextForArch(rocRoller::GPUArchitectureTarget const& device)
{
    using namespace rocRoller;

    auto currentDevice = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch();

    bool localDevice = currentDevice.target() == device;

    if(localDevice)
    {
        return Context::ForDefaultHipDevice(testKernelName(), m_kernelOptions);
    }
    else
    {
        return Context::ForTarget(device, testKernelName(), m_kernelOptions);
    }
}

rocRoller::ContextPtr CurrentGPUContextFixture::createContext()
{
    return createContextLocalDevice();
}
