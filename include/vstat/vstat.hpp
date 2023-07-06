// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2023 Heal Research

#ifndef VSTAT_HPP
#define VSTAT_HPP

#include "bivariate.hpp"
#include "univariate.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>

namespace VSTAT_NAMESPACE {

namespace detail {
    // utility method to load data into a wide type
    template<eve::simd_value T, std::input_iterator I, typename F>
    requires std::is_invocable_v<F, std::iter_value_t<I>>
    auto inline load(I iter, F&& func) {
        return [&]<std::size_t ...Idx>(std::index_sequence<Idx...>){
            return T{ func(*(iter + Idx))... };
        }(std::make_index_sequence<T::size()>{});
    }

    // utility method to advance a set of iterators
    template<typename Distance, typename... Iters>
    auto inline advance(Distance d, Iters&... iters) -> void {
        (std::advance(iters, d), ...);
    }
} // namespace detail

namespace concepts {
    template<typename T>
    concept arithmetic = std::is_arithmetic_v<T>;

    template<typename F, typename... Args>
    concept arithmetic_projection = requires(F&&) {
        { std::is_invocable_v<F, Args...> };
        { arithmetic<std::remove_reference_t<std::invoke_result_t<F, Args...>>> };
    };
} // namespace concepts

namespace univariate {
// accumulate a sequence
// we want to have a type T (e.g. float, double) that specifies the precision of the accumulator
template<std::floating_point T, std::input_iterator I, typename F = std::identity>
requires concepts::arithmetic_projection<F, std::iter_value_t<I>>
inline auto accumulate(I first, std::sized_sentinel_for<I> auto last, F&& f = F{}) noexcept -> univariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first, last) };
    auto const m = n - n % s;

    if (n < s) {
        univariate_accumulator<T> scalar_acc;
        for (; first < last; ++first) {
            scalar_acc(std::invoke(f, *first));
        }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(detail::load<wide>(first, f));
        detail::advance(s, first);
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
template<std::floating_point T, std::input_iterator I, std::input_iterator J, typename F = std::identity>
requires concepts::arithmetic_projection<F, std::iter_value_t<I>>
inline auto accumulate(I first1, std::sized_sentinel_for<I> auto last1, J first2, F&& f = F{}) noexcept -> univariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n { std::distance(first1, last1) };
    const size_t m = n - n % s;

    if (n < s) {
        univariate_accumulator<T> scalar_acc;
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(f, *first1), *first2);
        }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(detail::load<wide>(first1, f), wide(first2, first2 + s));
        detail::advance(s, first1, first2);
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
template<std::floating_point T, std::input_iterator I, std::input_iterator J, typename BinaryOp, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_invocable_v<F1, std::iter_value_t<I>> and
         std::is_invocable_v<F2, std::iter_value_t<J>> and
         std::is_invocable_v<
            BinaryOp,
            std::invoke_result_t<F1, std::iter_value_t<I>>,
            std::invoke_result_t<F2, std::iter_value_t<J>>
         > and
         concepts::arithmetic_projection<
            BinaryOp,
            std::invoke_result_t<F1, std::iter_value_t<I>>,
            std::invoke_result_t<F2, std::iter_value_t<J>>
         >
inline auto accumulate(I first1, std::sized_sentinel_for<I> auto last1, J first2, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> univariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m = n - n % s;

    auto f = [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); };
    if (n < s) {
        univariate_accumulator<T> scalar_acc;
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(f(*first1, *first2));
        }
        return univariate_statistics(scalar_acc);
    }

    std::array<T, s> x;
    univariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        std::transform(first1, first1 + s, first2, x.begin(), f);
        acc(wide{x.data()});
        detail::advance(s, first1, first2);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(f(*first1, *first2));
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

// accumulate a _weighted_ binary op between two sequences
template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K, typename BinaryOp, typename F1 = std::identity, typename F2 = std::identity>
requires std::is_arithmetic_v<std::iter_value_t<K>> &&
         std::is_invocable_v<F1, std::iter_value_t<I>> &&
         std::is_invocable_v<F2, std::iter_value_t<J>> &&
         std::is_invocable_v<
            BinaryOp,
            std::invoke_result_t<F1, std::iter_value_t<I>>,
            std::invoke_result_t<F2, std::iter_value_t<J>>
         > &&
         concepts::arithmetic_projection<
            BinaryOp,
            std::invoke_result_t<F1, std::iter_value_t<I>>,
            std::invoke_result_t<F2, std::iter_value_t<J>>
         >
inline auto accumulate(I first1, std::sized_sentinel_for<I> auto last1, J first2, K first3, BinaryOp&& op = BinaryOp{}, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> univariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m = n - n % s;

    auto f = [&](auto a, auto b){ return std::invoke(op, std::invoke(f1, a), std::invoke(f2, b)); };
    if (n < s) {
        univariate_accumulator<T> scalar_acc;
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            scalar_acc(f(*first1, *first2), *first3);
        }
        return univariate_statistics(scalar_acc);
    }


    std::array<T, s> x;
    univariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        std::transform(first1, first1 + s, first2, x.begin(), f);
        acc(wide(x), wide(first3, first3 + s));
        detail::advance(s, first1, first2, first3);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            scalar_acc(f(*first1, *first2), *first3);
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}
} // namespace univariate

namespace bivariate {
// bivariate case

// iterator variants
template<std::floating_point T, std::input_iterator I, std::input_iterator J, typename F1 = std::identity, typename F2 = std::identity>
requires concepts::arithmetic_projection<F1, std::iter_value_t<I>> and
         concepts::arithmetic_projection<F2, std::iter_value_t<J>>
inline auto accumulate(I first1, std::sized_sentinel_for<I> auto last1, J first2, F1&& f1 = F1{}, F2&& f2 = F2{})
{
    using wide = eve::wide<T>;
    auto constexpr s { wide::size() };
    auto const n { std::distance(first1, last1) };
    auto const m = n - n % s;

    if (n < s) {
        bivariate_accumulator<T> scalar_acc;
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(f1, *first1), std::invoke(f2, *first2));
        }
        return bivariate_statistics(scalar_acc);
    }

    bivariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(detail::load<wide>(first1, f1), detail::load<wide>(first2, f2));
        detail::advance(s, first1, first2);
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

template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K, typename F1 = std::identity, typename F2 = std::identity>
requires concepts::arithmetic_projection<F1, std::iter_value_t<I>> and
         concepts::arithmetic_projection<F2, std::iter_value_t<J>> and
         std::is_arithmetic_v<std::iter_value_t<K>>
inline auto accumulate(I first1, std::sized_sentinel_for<I> auto last1, J first2, K first3, F1&& f1 = F1{}, F2&& f2 = F2{}) noexcept -> bivariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s { wide::size() };
    auto const n = std::distance(first1, last1);
    auto const m = n - n % s;

    if (n < s) {
        bivariate_accumulator<T> scalar_acc;
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            scalar_acc(std::invoke(f1, *first1++), std::invoke(f2, *first2++), *first3++);
        }
        return bivariate_statistics(scalar_acc);
    }

    bivariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(
            detail::load<wide>(first1, f1),
            detail::load<wide>(first2, f2),
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
} // namespace bivariate
} // namespace VSTAT_NAMESPACE

#endif
