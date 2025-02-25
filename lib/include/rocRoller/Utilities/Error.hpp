
#pragma once

#include <concepts>
#include <source_location>
#include <stdexcept>
#include <string.h>
#include <string>

#include <cassert>

#include "Error_fwd.hpp"

namespace rocRoller
{
    struct Error : public std::runtime_error
    {
        using std::runtime_error::runtime_error;

        template <typename Ta, typename Tb, typename... Ts>
        Error(Ta const&, Tb const&, Ts const&...);

        static bool BreakOnThrow();
    };

    struct FatalError : public Error
    {
        using Error::Error;
    };

    struct RecoverableError : public Error
    {
        using Error::Error;
    };

    template <class T_Exception, typename... Ts>
    [[noreturn]] void Throw(Ts const&...);

    /**
     * Initiates a segfault.  This can be useful for debugging purposes.
     */
    [[noreturn]] void Crash();

    int* GetNullPointer();

    // Get path
    // Strips all "../" and "./"
    constexpr const char* GetBaseFileName(const char* file)
    {
        if(strnlen(file, 3) >= 3 && file[0] == '.' && file[1] == '.' && file[2] == '/')
        {
            return GetBaseFileName(file + 3);
        }
        else if(strnlen(file, 3) >= 2 && file[0] == '.' && file[1] == '/')
        {
            return GetBaseFileName(file + 2);
        }
        return file;
    }

#define ShowValue(var) concatenate("\t", #var, " = ", var, "\n")

#define AssertError(T_Exception, condition, message...)                       \
    do                                                                        \
    {                                                                         \
        std::source_location location      = std::source_location::current(); \
        bool                 condition_val = static_cast<bool>(condition);    \
        if(!(condition_val))                                                  \
        {                                                                     \
            Throw<T_Exception>(GetBaseFileName(location.file_name()),         \
                               ":",                                           \
                               location.line(),                               \
                               ": ",                                          \
                               #T_Exception,                                  \
                               "(",                                           \
                               #condition,                                    \
                               ")\n",                                         \
                               ##message);                                    \
        }                                                                     \
    } while(0)

#define AssertFatal(...) AssertError(FatalError, __VA_ARGS__)

#define AssertRecoverable(...) AssertError(RecoverableError, __VA_ARGS__)
}

#include "Error_impl.hpp"
