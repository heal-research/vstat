// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

#ifndef VSTAT_HPP
#define VSTAT_HPP

#include "bivariate.hpp"
#include "univariate.hpp"

#include <algorithm>

#if defined(VSTAT_NAMESPACE)
namespace VSTAT_NAMESPACE {
#endif

namespace univariate {
// accumulate a sequence
template<typename T, typename InputIt1, typename F = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        detail::is_iterator_v<InputIt1> &&
        std::is_invocable_r_v<T, F, typename std::iterator_traits<InputIt1>::value_type>
    , bool> = true>
inline univariate_statistics accumulate(InputIt1 first, InputIt1 last, F&& f = F{}) noexcept
{
    using vec = std::conditional_t<std::is_same_v<T, float>, Vec8f, Vec4d>;
    const size_t n = std::distance(first, last);
    const size_t s = vec::size();
    const size_t m = n & (-s);

    if (n < s) {
        univariate_accumulator<T> scalar_acc(std::invoke(f, *first++));
        while (first < last) scalar_acc(std::invoke(f, *first++));
        return univariate_statistics(scalar_acc);
    }

    std::array<T, s> x;

    std::transform(first, first + s, x.begin(), f);
    std::advance(first, s);
    univariate_accumulator<vec> acc(vec().load(x.data()));

    for (size_t i = s; i < m; i += s) {
        std::transform(first, first + s, x.begin(), f);
        acc(vec().load(x.data()));
        std::advance(first, s);
    }

    // gather the remaining values with a scalar accumulator
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
        for (; first < last; ++first) {
            scalar_acc(std::invoke(f, *first));
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// accumulate a sequence with weights
template<typename T, typename InputIt1, typename InputIt2, typename F = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        detail::is_iterator_v<InputIt1> && detail::is_iterator_v<InputIt2> &&
        std::is_invocable_r_v<T, F, typename std::iterator_traits<InputIt1>::value_type> &&
        std::is_arithmetic_v<typename std::iterator_traits<InputIt2>::value_type>
    , bool> = true>
inline univariate_statistics accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, F&& f = F{}) noexcept
{
    using vec = std::conditional_t<std::is_same_v<T, float>, Vec8f, Vec4d>;
    const size_t n = std::distance(first1, last1);
    const size_t s = vec::size();
    const size_t m = n & (-s);

    if (n < s) {
        univariate_accumulator<T> scalar_acc(std::invoke(f, *first1++), *first2++);
        while (first1 < last1) scalar_acc(std::invoke(f, *first1++), *first2++);
        return univariate_statistics(scalar_acc);
    }

    std::array<T, s> x, w;
    std::transform(first1, first1 + s, x.begin(), f);
    std::copy(first2, first2 + s, w.begin());
    std::advance(first1, s);
    std::advance(first2, s);
    univariate_accumulator<vec> acc(vec().load(x.data()), vec().load(w.data()));

    for (size_t i = s; i < m; i += s) {
        std::transform(first1, first1 + s, x.begin(), f);
        std::copy(first2, first2 + s, w.begin());
        acc(vec().load(x.data()), vec().load(w.data()));
        std::advance(first1, s);
        std::advance(first2, s);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(f, *first1), *first2);
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// accumulate a binary op between two sequences
template<typename T, typename InputIt1, typename InputIt2, typename BinaryOp, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        detail::is_iterator_v<InputIt1> && detail::is_iterator_v<InputIt2> &&
        std::is_invocable_r_v<T, F1, typename std::iterator_traits<InputIt1>::value_type> &&
        std::is_invocable_r_v<T, F2, typename std::iterator_traits<InputIt2>::value_type> &&
        std::is_invocable_r_v<T, BinaryOp, T, T>
    , bool> = true>
inline univariate_statistics accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept
{
    using vec = std::conditional_t<std::is_same_v<T, float>, Vec8f, Vec4d>;
    const size_t n = std::distance(first1, last1);
    const size_t s = vec::size();
    const size_t m = n & (-s);

    if (n < s) {
        univariate_accumulator<T> scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++)));
        while (first1 < last1) scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++)));
        return univariate_statistics(scalar_acc);
    }

    std::array<T, s> x;
    std::transform(first1, first1 + s, first2, x.begin(), [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); });
    std::advance(first1, s);
    std::advance(first2, s);
    univariate_accumulator<vec> acc(vec().load(x.data()));

    for (size_t i = s; i < m; i += s) {
        std::transform(first1, first1 + s, first2, x.begin(), [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); });
        acc(vec().load(x.data()));
        std::advance(first1, s);
        std::advance(first2, s);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(op, std::invoke(f1, *first1), std::invoke(f2, *first2)));
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// accumulate a _weighted_ binary op between two sequences
template<typename T, typename InputIt1, typename InputIt2, typename InputIt3, typename BinaryOp, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        detail::is_iterator_v<InputIt1> && detail::is_iterator_v<InputIt2> && detail::is_iterator_v<InputIt3> &&
        std::is_invocable_r_v<T, F1, typename std::iterator_traits<InputIt1>::value_type> &&
        std::is_invocable_r_v<T, F2, typename std::iterator_traits<InputIt2>::value_type> &&
        std::is_arithmetic_v<typename std::iterator_traits<InputIt3>::value_type> &&
        std::is_invocable_r_v<T, BinaryOp, T, T>
    , bool> = true>
inline univariate_statistics accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt3 first3, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept
{
    using vec = std::conditional_t<std::is_same_v<T, float>, Vec8f, Vec4d>;
    const size_t n = std::distance(first1, last1);
    const size_t s = vec::size();
    const size_t m = n & (-s);

    if (n < s) {
        univariate_accumulator<T> scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++)), *first3++);
        while (first1 < last1) scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++)), *first3++);
        return univariate_statistics(scalar_acc);
    }

    std::array<T, s> x, w;
    std::transform(first1, first1 + s, first2, x.begin(), [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); }); // apply projections
    std::copy(first3, first3 + s, w.begin());
    std::advance(first1, s);
    std::advance(first2, s);
    std::advance(first3, s);
    univariate_accumulator<vec> acc(vec().load(x.data()), vec().load(w.data()));

    for (size_t i = s; i < m; i += s) {
        std::transform(first1, first1 + s, first2, x.begin(), [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); });
        std::copy(first3, first3 + s, w.begin());
        acc(vec().load(x.data()), vec().load(w.data()));
        std::advance(first1, s);
        std::advance(first2, s);
        std::advance(first3, s);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            scalar_acc(std::invoke(op, std::invoke(f1, *first1), std::invoke(f2, *first2)), *first3);
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// raw pointer variants
template<typename T, typename X, typename F = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        std::is_invocable_r_v<T, F, X>
    , bool> = true>
inline univariate_statistics accumulate(X const* x, size_t n, F&& f = F{}) noexcept
{
    return univariate::accumulate<T>(x, x + n, f);
}

template<typename T, typename X, typename W, typename F = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        std::is_arithmetic_v<W> &&
        std::is_invocable_r_v<T, F, X>
    , bool> = true>
inline univariate_statistics accumulate(X const* x, W const* w, size_t n, F&& f = F{}) noexcept
{
    return univariate::accumulate<T>(x, x + n, w, f);
}

template<typename T, typename X, typename Y, typename BinaryOp, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        std::is_invocable_r_v<T, F1, X> &&
        std::is_invocable_r_v<T, F2, Y> &&
        std::is_invocable_r_v<T, BinaryOp, T, T>
    , bool> = true>
inline univariate_statistics accumulate(X const* x, Y const* y, size_t n, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{})
{
    return univariate::accumulate<T>(x, x + n, y, op, f1, f2);
}

template<typename T, typename X, typename Y, typename W, typename BinaryOp, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        std::is_invocable_r_v<T, F1, X> &&
        std::is_invocable_r_v<T, F2, Y> &&
        std::is_invocable_r_v<T, BinaryOp, T, T> &&
        std::is_arithmetic_v<W>
    , bool> = true>
inline univariate_statistics accumulate(X const* x, Y const* y, W const* w, size_t n, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{})
{
    return univariate::accumulate<T>(x, x + n, y, w, op, f1, f2);
}
} // namespace univariate

namespace bivariate {
// bivariate case
template<typename T, typename InputIt1, typename InputIt2, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        detail::is_iterator_v<InputIt1> &&
        detail::is_iterator_v<InputIt2> &&
        std::is_invocable_r_v<T, F1, typename std::iterator_traits<InputIt1>::value_type> &&
        std::is_invocable_r_v<T, F2, typename std::iterator_traits<InputIt2>::value_type>
    , bool> = true>
inline bivariate_statistics accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept
{
    using vec = std::conditional_t<std::is_same_v<T, float>, Vec8f, Vec4d>;
    const size_t n = std::distance(first1, last1);
    const size_t s = vec::size();
    const size_t m = n & (-s);

    if (n < s) {
        bivariate_accumulator<T> scalar_acc(std::invoke(f1, *first1++),std::invoke(f2, *first2++));
        while (first1 < last1) scalar_acc(std::invoke(f1, *first1++), std::invoke(f2, *first2++));
        return bivariate_statistics(scalar_acc);
    }

    std::array<T, s> x1, x2;
    std::transform(first1, first1 + s, x1.begin(), f1); std::advance(first1, s);
    std::transform(first2, first2 + s, x2.begin(), f2); std::advance(first2, s);
    bivariate_accumulator<vec> acc(vec().load(x1.data()), vec().load(x2.data()));

    for (size_t i = s; i < m; i += s) {
        std::transform(first1, first1 + s, x1.begin(), f1);
        std::transform(first2, first2 + s, x2.begin(), f2);
        acc(vec().load(x1.data()), vec().load(x2.data()));
        std::advance(first1, s);
        std::advance(first2, s);
    }

    if (m < n) {
        auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
        auto scalar_acc = bivariate_accumulator<T>::load_state(sx, sy, sw, sxx, syy, sxy);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(f1, *first1), std::invoke(f2, *first2));
        }
        return bivariate_statistics(scalar_acc);
    }

    return bivariate_statistics(acc);
}

template<typename T, typename InputIt1, typename InputIt2, typename InputIt3, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        detail::is_iterator_v<InputIt1> &&
        detail::is_iterator_v<InputIt2> &&
        detail::is_iterator_v<InputIt3> &&
        std::is_invocable_r_v<T, F1, typename std::iterator_traits<InputIt1>::value_type> &&
        std::is_invocable_r_v<T, F2, typename std::iterator_traits<InputIt1>::value_type> &&
        std::is_arithmetic_v<typename std::iterator_traits<InputIt1>::value_type>
    , bool> = true>
inline bivariate_statistics accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt3 first3, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept
{
    using vec = std::conditional_t<std::is_same_v<T, float>, Vec8f, Vec4d>;
    const size_t n = std::distance(first1, last1);
    const size_t s = vec::size();
    const size_t m = n & (-s);

    if (n < s) {
        bivariate_accumulator<T> scalar_acc(std::invoke(f1, *first1++),std::invoke(f2, *first2++), *first3++);
        while (first1 < last1) scalar_acc(std::invoke(f1, *first1++), std::invoke(f2, *first2++), *first3++);
        return bivariate_statistics(scalar_acc);
    }

    std::array<T, s> x1, x2, w;
    std::transform(first1, first1 + s, x1.begin(), f1);
    std::transform(first2, first2 + s, x2.begin(), f2);
    std::copy(first3, first3 + s, w.begin());
    std::advance(first1, s);
    std::advance(first2, s);
    std::advance(first3, s);
    bivariate_accumulator<vec> acc(vec().load(x1.data()), vec().load(x2.data()), vec().load(w.data()));

    for (size_t i = s; i < m; i += s) {
        std::transform(first1, first1 + s, x1, f1);
        std::transform(first2, first2 + s, x2, f2);
        std::copy(first3, first3 + s, w);
        acc(vec().load(x1.data()), vec().load(x2.data()), vec().load(w.data()));
        std::advance(first1, s);
        std::advance(first2, s);
        std::advance(first3, s);
    }

    if (m < n) {
        auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
        auto scalar_acc = bivariate_accumulator<T>::load_state(sx, sy, sw, sxx, syy, sxy);
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            scalar_acc(std::invoke(f1, *first1), std::invoke(f2, *first2), *first3);
        }
        return bivariate_statistics(scalar_acc);
    }
    return bivariate_statistics(acc);
}

template<typename T, typename X, typename Y, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        std::is_invocable_r_v<T, F1, X> &&
        std::is_invocable_r_v<T, F2, Y>
    , bool> = true>
inline bivariate_statistics accumulate(X const* x, Y const* y, size_t n, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept
{
    return bivariate::accumulate<T>(x, x + n, y, f1, f2);
}

template<typename T, typename X, typename Y, typename W, typename F1 = detail::identity, typename F2 = detail::identity,
    std::enable_if_t<
        detail::is_any_v<T, float, double> &&
        std::is_invocable_r_v<T, F1, X> &&
        std::is_invocable_r_v<T, F2, Y> &&
        std::is_arithmetic_v<W>
    , bool> = true>
inline bivariate_statistics accumulate(X const* x, Y const* y, W const* w, size_t n, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept
{
    return bivariate::accumulate<T>(x, x + n, y, w, f1, f2);
}
} // namespace bivariate

#if defined(VSTAT_NAMESPACE)
}
#endif

#endif
