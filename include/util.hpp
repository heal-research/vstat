// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

#ifndef VSTAT_UTIL_HPP
#define VSTAT_UTIL_HPP

#include <iterator>
#include <type_traits>
#include <utility>

#if defined(VSTAT_NAMESPACE)
namespace VSTAT_NAMESPACE {
#endif

namespace detail {

#if defined(__GNUC__) || defined(__GNUG__)
#define VSTAT_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define VSTAT_FORCE_INLINE inline
#endif

#define VSTAT_EXPECT(cond)                                                                              \
    if (!(cond)) {                                                                                      \
        std::cerr << "precondition " << #cond << " failed at " << __FILE__ << ": " << __LINE__ << "\n"; \
        std::terminate();                                                                               \
    }

#define VSTAT_ENSURE(cond)                                                                               \
    if (!(cond)) {                                                                                       \
        std::cerr << "postcondition " << #cond << " failed at " << __FILE__ << ": " << __LINE__ << "\n"; \
        std::terminate();                                                                                \
    }

    // type traits
    template<typename T, typename... Ts>
    struct is_any : std::disjunction<std::is_same<T, Ts>...> {
    };

    template<typename T, typename... Ts>
    using is_any_t = typename is_any<T, Ts...>::type;

    template<typename T, typename... Ts>
    inline constexpr bool is_any_v = is_any<T, Ts...>::value;

    template<typename T, typename... Ts>
    struct are_same : std::conjunction<std::is_same<T, Ts>...> {
    };

    template<typename T, typename... Ts>
    using are_same_t = typename are_same<T, Ts...>::type;

    template<typename T, typename... Ts>
    inline constexpr bool are_same_v = are_same<T, Ts...>::value;

    struct identity {
        template <typename T>
        constexpr auto operator()(T&& v) const noexcept -> decltype(std::forward<T>(v))
        {
            return std::forward<T>(v);
        }
    };

    template<typename T, typename = void>
    struct is_iterator : std::false_type {
    };

    template<typename T>
    struct is_iterator<T, std::void_t<typename std::iterator_traits<T>::iterator_category>> : std::true_type {
    };

    template<typename T>
    using is_iterator_t = typename is_iterator<T>::type;

    template<typename T>
    inline constexpr bool is_iterator_v = is_iterator<T>::value;
}

#if defined(VSTAT_NAMESPACE)
}
#endif

#endif
