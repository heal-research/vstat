// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

#ifndef VSTAT_HPP
#define VSTAT_HPP

#include "bivariate.hpp"
#include "univariate.hpp"

#include <algorithm>

namespace VSTAT_NAMESPACE {

namespace detail {
    // utility method to load data into a wide type
    template<typename T, typename Iterator, typename Callable>
    requires eve::simd_value<T> &&
             detail::is_iterator_v<Iterator> &&
             std::is_invocable_v<Callable, detail::iterator_value_t<Iterator>>
    auto inline load(Iterator iter, Callable&& func) {
        if constexpr (std::is_arithmetic_v<std::remove_reference_t<detail::iterator_value_t<Iterator>>>) {
            // if the value type of the iterator is arithmetic, then apply the func on the wide type directly
            return std::invoke(func, T{iter, iter + T::size()});
        } else {
            // otherwise, produce a wide type from the projection of the iterator
            return [&]<std::size_t ...I>(std::index_sequence<I...>){
                return T{ func(*(iter + I))... };
            }(std::make_index_sequence<T::size()>{});
        }
    }

    // utility method to advance a set of iterators
    template<typename Distance, typename... Iters>
    constexpr
    auto inline advance(Distance d, Iters&... iters) -> void {
        (std::advance(iters, d), ...);
    }
} // namespace detail

namespace univariate {
// accumulate a sequence
// we want to have a type T (e.g. float, double) that specifies the precision of the accumulator
template<typename T, typename InputIt1, typename F = std::identity>
requires std::is_arithmetic_v<T> &&
         detail::is_iterator_v<InputIt1> &&
         detail::is_arithmetic_result_v<F, detail::iterator_value_t<InputIt1>>
inline auto accumulate(InputIt1 first, InputIt1 last, F&& f = F{}) noexcept -> univariate_statistics
{
    using scalar_t = T;
    using wide = eve::wide<scalar_t>;
    size_t const n = std::distance(first, last);
    size_t constexpr s = wide::size();
    size_t const m = n & (-s);

    if (n < s) {
        univariate_accumulator<scalar_t> scalar_acc(std::invoke(f, *first++));
        while (first < last) { scalar_acc(std::invoke(f, *first++)); }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide> acc(detail::load<wide>(first, f));
    detail::advance(s, first);

    for (size_t i = s; i < m; i += s) {
        acc(detail::load<wide>(first, f));
        detail::advance(s, first);
    }

    // gather the remaining values with a scalar accumulator
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<scalar_t>::load_state(sw, sx, sxx);
        for (; first < last; ++first) {
            scalar_acc(std::invoke(f, *first));
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// accumulate a sequence with weights
template<typename T, typename InputIt1, typename InputIt2, typename F = std::identity>
requires std::is_arithmetic_v<T> &&
         detail::is_iterator_v<InputIt1> &&
         detail::is_iterator_v<InputIt2> &&
         std::is_arithmetic_v<detail::iterator_value_t<InputIt1>> &&
         std::is_arithmetic_v<detail::iterator_value_t<InputIt2>> &&
         std::is_invocable_v<F, detail::iterator_value_t<InputIt1>> &&
         detail::is_arithmetic_result_v<F, detail::iterator_value_t<InputIt1>>
inline auto accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, F&& f = F{}) noexcept -> univariate_statistics
{
    using scalar_t = T;
    using wide = eve::wide<scalar_t>;
    const size_t n = std::distance(first1, last1);
    const size_t s = wide::size();
    const size_t m = n & (-s);

    if (n < s) {
        univariate_accumulator<scalar_t> scalar_acc(std::invoke(f, *first1++), *first2++);
        while (first1 < last1) { scalar_acc(std::invoke(f, *first1++), *first2++); }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide> acc(
        detail::load<wide>(first1, f),
        wide(first2, first2 + s)
    );
    detail::advance(s, first1, first2);

    for (size_t i = s; i < m; i += s) {
        acc(
            detail::load<wide>(first1, f),
            wide(first2, first2 + s)
        );
        detail::advance(s, first1, first2);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<scalar_t>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(f, *first1), *first2);
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// accumulate a binary op between two sequences
template<typename T, typename InputIt1, typename InputIt2, typename BinaryOp, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> &&
         detail::is_iterator_v<InputIt1> &&
         detail::is_iterator_v<InputIt2> &&
         std::is_invocable_v<F1, detail::iterator_value_t<InputIt1>> &&
         std::is_invocable_v<F2, detail::iterator_value_t<InputIt2>> &&
         std::is_invocable_v<
            BinaryOp,
            std::invoke_result_t<F1, detail::iterator_value_t<InputIt1>>,
            std::invoke_result_t<F2, detail::iterator_value_t<InputIt2>>
         > &&
         detail::is_arithmetic_result_v<
            BinaryOp,
            std::invoke_result_t<F1, detail::iterator_value_t<InputIt1>>,
            std::invoke_result_t<F2, detail::iterator_value_t<InputIt2>>
         >
inline auto accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> univariate_statistics
{
    using scalar_t = T;
    using wide = eve::wide<scalar_t>;
    const size_t n = std::distance(first1, last1);
    const size_t s = wide::size();
    const size_t m = n & (-s);

    if (n < s) {
        univariate_accumulator<scalar_t> scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++)));
        while (first1 < last1) { scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++))); }
        return univariate_statistics(scalar_acc);
    }

    auto f = [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); };
    std::array<scalar_t, s> x;
    std::transform(first1, first1 + s, first2, x.begin(), f);
    univariate_accumulator<wide> acc(wide{x.data()});
    detail::advance(s, first1, first2);

    for (size_t i = s; i < m; i += s) {
        std::transform(first1, first1 + s, first2, x.begin(), f);
        acc(wide{x.data()});
        detail::advance(s, first1, first2);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<scalar_t>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(op, std::invoke(f1, *first1), std::invoke(f2, *first2)));
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// accumulate a _weighted_ binary op between two sequences
template<typename T, typename InputIt1, typename InputIt2, typename InputIt3, typename BinaryOp, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> &&
         detail::is_iterator_v<InputIt1> &&
         detail::is_iterator_v<InputIt2> &&
         detail::is_iterator_v<InputIt3> &&
         std::is_arithmetic_v<detail::iterator_value_t<InputIt3>> &&
         std::is_invocable_v<F1, detail::iterator_value_t<InputIt1>> &&
         std::is_invocable_v<F2, detail::iterator_value_t<InputIt2>> &&
         std::is_invocable_v<
            BinaryOp,
            std::invoke_result_t<F1, detail::iterator_value_t<InputIt1>>,
            std::invoke_result_t<F2, detail::iterator_value_t<InputIt2>>
         > &&
         detail::is_arithmetic_result_v<
            BinaryOp,
            std::invoke_result_t<F1, detail::iterator_value_t<InputIt1>>,
            std::invoke_result_t<F2, detail::iterator_value_t<InputIt2>>
         >
inline auto accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt3 first3, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> univariate_statistics
{
    using scalar_t = T;
    using wide = eve::wide<scalar_t>;
    const size_t n = std::distance(first1, last1);
    const size_t s = wide::size();
    const size_t m = n & (-s);

    if (n < s) {
        univariate_accumulator<scalar_t> scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++)), *first3++);
        while (first1 < last1) { scalar_acc(std::invoke(op, std::invoke(f1, *first1++), std::invoke(f2, *first2++)), *first3++); }
        return univariate_statistics(scalar_acc);
    }

    auto f = [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); };

    std::array<scalar_t, s> x;
    std::transform(first1, first1 + s, first2, x.begin(), f); // apply projections
    univariate_accumulator<wide> acc(wide(x), wide(first3, first3 + s));
    detail::advance(s, first1, first2, first3);

    for (size_t i = s; i < m; i += s) {
        std::transform(first1, first1 + s, first2, x.begin(), f);
        acc(wide(x), wide(first3, first3 + s));
        detail::advance(s, first1, first2, first3);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<scalar_t>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            scalar_acc(std::invoke(op, std::invoke(f1, *first1), std::invoke(f2, *first2)), *first3);
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// raw pointer variants
template<typename T, typename X, typename F = std::identity>
requires std::is_arithmetic_v<T> && std::is_invocable_r_v<T, F, X>
inline auto accumulate(X const* x, size_t n, F&& f = F{}) noexcept -> univariate_statistics
{
    return univariate::accumulate<T>(x, x + n, f);
}

template<typename T, typename X, typename W, typename F = std::identity>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<W> &&
         std::is_invocable_r_v<T, F, X>
inline auto accumulate(X const* x, W const* w, size_t n, F&& f = F{}) noexcept -> univariate_statistics
{
    return univariate::accumulate<T>(x, x + n, w, f);
}

template<typename T, typename X, typename Y, typename BinaryOp, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> &&
         std::is_invocable_v<
             BinaryOp,
             std::invoke_result_t<F1, X>,
             std::invoke_result_t<F2, Y>
         > &&
         detail::is_arithmetic_result_v<
             BinaryOp,
             std::invoke_result_t<F1, X>,
             std::invoke_result_t<F2, Y>
         >
inline auto accumulate(X const* x, Y const* y, size_t n, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) -> univariate_statistics
{
    return univariate::accumulate<T>(x, x + n, y, op, f1, f2);
}

template<typename T, typename X, typename Y, typename W, typename BinaryOp, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> && std::is_arithmetic_v<W> &&
         std::is_invocable_v<
             BinaryOp,
             std::invoke_result_t<F1, X>,
             std::invoke_result_t<F2, Y>
         > &&
         detail::is_arithmetic_result_v<
             BinaryOp,
             std::invoke_result_t<F1, X>,
             std::invoke_result_t<F2, Y>
         >
inline auto accumulate(X const* x, Y const* y, W const* w, size_t n, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) -> univariate_statistics
{
    return univariate::accumulate<T>(x, x + n, y, w, op, f1, f2);
}
} // namespace univariate

namespace bivariate {
// bivariate case
template<typename T, typename InputIt1, typename InputIt2, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> && 
         detail::is_iterator_v<InputIt1> &&
         detail::is_iterator_v<InputIt2> &&
         std::is_invocable_v<F1, detail::iterator_value_t<InputIt1>> &&
         std::is_invocable_v<F2, detail::iterator_value_t<InputIt2>> &&
         detail::is_arithmetic_result_v<F1, detail::iterator_value_t<InputIt1>> &&
         detail::is_arithmetic_result_v<F2, detail::iterator_value_t<InputIt2>>
inline auto accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> bivariate_statistics
{
    using scalar_t = T;
    using wide = eve::wide<scalar_t>;
    const size_t n = std::distance(first1, last1);
    const size_t s = wide::size();
    const size_t m = n & (-s);

    if (n < s) {
        bivariate_accumulator<scalar_t> scalar_acc(std::invoke(f1, *first1++), std::invoke(f2, *first2++));
        while (first1 < last1) { scalar_acc(std::invoke(f1, *first1++), std::invoke(f2, *first2++)); }
        return bivariate_statistics(scalar_acc);
    }

    bivariate_accumulator<wide> acc(
        detail::load<wide>(first1, f1),
        detail::load<wide>(first2, f2)
    );
    detail::advance(s, first1, first2);

    for (size_t i = s; i < m; i += s) {
        acc(
            detail::load<wide>(first1, f1),
            detail::load<wide>(first2, f2)
        );
        detail::advance(s, first1, first2);
    }

    if (m < n) {
        auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
        auto scalar_acc = bivariate_accumulator<scalar_t>::load_state(sx, sy, sw, sxx, syy, sxy);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(f1, *first1), std::invoke(f2, *first2));
        }
        return bivariate_statistics(scalar_acc);
    }

    return bivariate_statistics(acc);
}

template<typename T, typename InputIt1, typename InputIt2, typename InputIt3, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> &&
         detail::is_iterator_v<InputIt1> &&
         detail::is_iterator_v<InputIt2> &&
         detail::is_iterator_v<InputIt3> &&
         std::is_invocable_v<F1, detail::iterator_value_t<InputIt1>> &&
         std::is_invocable_v<F2, detail::iterator_value_t<InputIt2>> &&
         detail::is_arithmetic_result_v<F1, detail::iterator_value_t<InputIt1>> &&
         detail::is_arithmetic_result_v<F2, detail::iterator_value_t<InputIt2>> &&
         std::is_arithmetic_v<detail::iterator_value_t<InputIt3>>
inline auto accumulate(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt3 first3, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> bivariate_statistics
{
    using wide = eve::wide<T>;
    const size_t n = std::distance(first1, last1);
    const size_t s = wide::size();
    const size_t m = n & (-s);

    if (n < s) {
        bivariate_accumulator<T> scalar_acc(std::invoke(f1, *first1++), std::invoke(f2, *first2++), *first3++);
        while (first1 < last1) { scalar_acc(std::invoke(f1, *first1++), std::invoke(f2, *first2++), *first3++); }
        return bivariate_statistics(scalar_acc);
    }

    bivariate_accumulator<wide> acc(
        wide(first1, f1),
        wide(first2, f2),
        wide(first3, first3 + s)
    );
    detail::advance(s, first1, first2, first3);

    for (size_t i = s; i < m; i += s) {
        acc(
            wide(first1, f1),
            wide(first2, f2),
            wide(first3, first3 + s)
        );
        detail::advance(s, first1, first2, first3);
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

template<typename T, typename X, typename Y, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> &&
         std::is_invocable_v<F1, X> &&
         std::is_invocable_v<F2, Y> &&
         detail::is_arithmetic_result_v<F1, X> &&
         detail::is_arithmetic_result_v<F2, Y>
inline auto accumulate(X const* x, Y const* y, size_t n, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> bivariate_statistics
{
    return bivariate::accumulate<T>(x, x + n, y, f1, f2);
}

template<typename T, typename X, typename Y, typename W, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<T> &&
         std::is_invocable_v<F1, X> &&
         std::is_invocable_v<F2, Y> &&
         std::is_arithmetic_v<W> &&
         detail::is_arithmetic_result_v<F1, X> &&
         detail::is_arithmetic_result_v<F2, Y>
inline auto accumulate(X const* x, Y const* y, W const* w, size_t n, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> bivariate_statistics
{
    return bivariate::accumulate<T>(x, x + n, y, w, f1, f2);
}
} // namespace bivariate

} // namespace VSTAT_NAMESPACE

#endif
