// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2023 Heal Research

#ifndef VSTAT_UTIL_HPP
#define VSTAT_UTIL_HPP

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

#if !defined(VSTAT_NAMESPACE)
#define VSTAT_NAMESPACE vstat
#endif

#endif
