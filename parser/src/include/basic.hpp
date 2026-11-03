#pragma once

#include <cassert>
#include <libgen.h>

#define _STR(x) #x
#define STR(x) _STR(x)

#define BASENAME(file) basename((char *)(file))

#define MESSAGE(tag, s, msg) (s) << tag << BASENAME(__FILE__) << ':' << __LINE__ << ": " << msg

#define ASSERT_ERROR(condition, msg)                                                               \
    if (!(condition)) {                                                                            \
        MESSAGE("(E) ", std::cerr, msg << std::endl);                                              \
        std::exit(1);                                                                              \
    }

#define ASSERT_ERROR_RETURN(condition, msg, result)                                                \
    if (!(condition)) {                                                                            \
        MESSAGE("(E) ", std::cerr, msg << std::endl);                                              \
        return (result);                                                                           \
    }

#ifndef NDEBUG

#include <iostream>
#include <typeinfo>

template <typename T> struct _TypeName { static const char *const name; };
template <typename T> const char *const _TypeName<T>::name = typeid(T).name();

#define TYPE_NAME(T) (_TypeName<T>::name)
#define REGISTER_TYPE_NAME(T)                                                                      \
    template <> struct _TypeName<T> { static constexpr const char *name = #T; }

#define LOG_DEBUG(msg) MESSAGE("(D) ", std::cerr, msg << std::endl)

#else

#define TYPE_NAME(T) "*"
#define REGISTER_TYPE_NAME(T)
#define LOG_DEBUG(msg)

#endif // DEBUG

#define LOG_WARN(msg) MESSAGE("(W) ", std::cerr, msg << '\n')
#define LOG_INFO(msg) MESSAGE("(I) ", std::cerr, msg << '\n')
#define LOG_ERROR(msg) MESSAGE("(E) ", std::cerr, msg << '\n')
#define PRINT_EXPR(var) LOG_INFO("  " #var << " = " << var);

#define _OPEN_STREAM(type, var, filename, action)                                                  \
    type var(filename);                                                                            \
    if (!var) {                                                                                    \
        LOG_ERROR("Cannot open file: " #filename " = " << (filename));                             \
        action;                                                                                    \
    }

#define OPEN_IFSTREAM(var, filename, action) _OPEN_STREAM(std::ifstream, var, filename, action)

#define OPEN_OFSTREAM(var, filename, action) _OPEN_STREAM(std::ofstream, var, filename, action)

#define CLOSE_STREAM(var) var.close();

// Local Variables:
// mode: c++
// End:
