// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2024 Heal Research

#ifndef VSTAT_HPP
#define VSTAT_HPP

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>

#include <eve/module/math.hpp>
#include <eve/module/special.hpp>

#include "bivariate.hpp"
#include "compensated_sum.hpp"
#include "univariate.hpp"

namespace VSTAT_NAMESPACE
{

/*!
    \brief Controls how a non-finite (NaN/Inf) input is handled.

    - propagate (default): a non-finite value poisons the whole result, same
      as the rest of this library.
    - omit: positions where the relevant input(s) are non-finite are
      skipped (zero-weighted) instead, and the function additionally
      returns the count of skipped positions.
*/
enum class nan_policy { propagate, omit };

namespace detail
{
// utility method to load data into a wide type
template<eve::simd_value T, std::random_access_iterator I, typename F>
    requires std::is_invocable_v<F, std::iter_value_t<I>>
auto inline load(I iter, F&& func)
{
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) -> auto
    { return T {std::forward<F>(func)(*(iter + Idx))...}; }(std::make_index_sequence<T::size()> {});
}

// binary projection overload: applies func(a, b) element-wise across two iterators
template<eve::simd_value T, std::random_access_iterator I, std::random_access_iterator J, typename F>
    requires std::is_invocable_v<F, std::iter_value_t<I>, std::iter_value_t<J>>
auto inline load(I iter1, J iter2, F&& func)
{
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) -> auto
    { return T {std::forward<F>(func)(*(iter1 + Idx), *(iter2 + Idx))...}; }(std::make_index_sequence<T::size()> {});
}

// utility method to advance a set of iterators
template<typename Distance, typename... Iters>
auto inline advance(Distance d, Iters&... iters) -> void
{
    (std::advance(iters, d), ...);
}
}  // namespace detail

namespace concepts
{
template<typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template<typename F, typename... Args>
concept arithmetic_projection = requires(F&&) {
    { std::is_invocable_v<F, Args...> };
    { arithmetic<std::remove_reference_t<std::invoke_result_t<F, Args...>>> };
};
}  // namespace concepts

/*!
    \defgroup Univariate Univariate statistics

    \brief Methods for univariate statistics
*/
namespace univariate
{
/*!
    \ingroup Univariate

    \brief Accumulates a sequence of (projected) values

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used
   to compute the stats.

    \param first The begin iterator for the first sequence
    \param last  The end iterator for the first sequence
    \param f     A projection mapping `std::iter_value_t<I>` to a scalar value
*/
template<std::floating_point T, stats Stats = stats::variance, std::random_access_iterator I, typename F = std::identity>
    requires concepts::arithmetic_projection<F, std::iter_value_t<I>>
inline auto accumulate(I first, I last, F&& f = F {}) noexcept -> univariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first, last)};
    auto const m = n - (n % s);

    if (n < s) {
        univariate_accumulator<T, Stats> scalar_acc;
        for (; first < last; ++first) {
            scalar_acc(std::invoke(std::forward<F>(f), *first));
        }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide, Stats> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(detail::load<wide>(first, std::forward<F>(f)));
        detail::advance(s, first);
    }

    // gather the remaining values with a scalar accumulator
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T, Stats>::load_state(sw, sx, sxx);
        for (; first < last; ++first) {
            scalar_acc(std::invoke(std::forward<F>(f), *first));
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

/*!
    \ingroup Univariate

    \brief Accumulates a sequence of (projected) values

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used
   to compute the stats.

    \param first1 The begin iterator for the first sequence
    \param last1  The end iterator for the first sequence
    \param first2 The begin iterator for the second (weights) sequence
    \param f      A projection mapping `std::iter_value_t<I>` to a scalar value
*/
template<std::floating_point T,
         stats Stats = stats::variance,
         std::random_access_iterator I,
         std::random_access_iterator J,
         typename F = std::identity>
    requires concepts::arithmetic_projection<F, std::iter_value_t<I>> and std::is_arithmetic_v<std::iter_value_t<J>>
inline auto accumulate(I first1, I last1, J first2, F&& f = F {}) noexcept
    -> univariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    const size_t m = n - n % s;

    if (n < s) {
        univariate_accumulator<T, Stats> scalar_acc;
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(std::forward<F>(f), *first1), *first2);
        }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide, Stats> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(detail::load<wide>(first1, std::forward<F>(f)), wide {first2});
        detail::advance(s, first1, first2);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T, Stats>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(std::forward<F>(f), *first1), *first2);
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

/*!
    \ingroup Univariate

    \brief Accumulates over the projected values from applying `BinaryOp` on the
   input sequences.

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used
   to compute the stats
    \tparam Stats Which stats to compute
    \tparam Policy `nan_policy::propagate` (default): a non-finite projected
   value poisons the whole result, like the rest of this library.
   `nan_policy::omit`: positions where either raw input is non-finite are
   skipped (zero-weighted) instead, and the count of skipped pairs is
   additionally returned.
    \tparam BinaryOp Binary projection \f$op(a,b) \to c\f$
    \tparam F1 Unary projection \f$f(x_1) \to a\f$
    \tparam F2 Unary projection \f$f(x_2) \to b\f$

    \param first1 The begin iterator for the first sequence
    \param last1  The end iterator for the first sequence
    \param first2 The begin iterator for the second sequence
    \param op     A binary projection mapping a tuple \f$(f_1(\cdot),
   f_2(\cdot))\f$ to a scalar value
    \param f1     A projection mapping `std::iter_value_t<I>` to a scalar value
    \param f2     A projection mapping `std::iter_value_t<J>` to a scalar value

    \return `nan_policy::propagate`: the accumulated statistics.
   `nan_policy::omit`: the accumulated statistics over finite pairs, and the
   count of skipped (non-finite) pairs.
*/
template<std::floating_point T,
         stats Stats = stats::variance,
         nan_policy Policy = nan_policy::propagate,
         std::random_access_iterator I,
         std::random_access_iterator J,
         typename BinaryOp,
         typename F1 = std::identity,
         typename F2 = std::identity>
    requires std::is_invocable_v<F1, std::iter_value_t<I>> and std::is_invocable_v<F2, std::iter_value_t<J>>
    and std::is_invocable_v<BinaryOp,
                            std::invoke_result_t<F1, std::iter_value_t<I>>,
                            std::invoke_result_t<F2, std::iter_value_t<J>>>
    and concepts::arithmetic_projection<BinaryOp,
                                        std::invoke_result_t<F1, std::iter_value_t<I>>,
                                        std::invoke_result_t<F2, std::iter_value_t<J>>>
inline auto accumulate(I first1,
                       I last1,
                       J first2,
                       BinaryOp&& op = BinaryOp {},
                       F1&& f1 = F1 {},
                       F2&& f2 = F2 {}) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<univariate_statistics, std::size_t>, univariate_statistics>
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m = n - n % s;

    auto f = [&](auto a, auto b)
    {
        return std::invoke(
            std::forward<BinaryOp>(op), std::invoke(std::forward<F1>(f1), a), std::invoke(std::forward<F2>(f2), b));
    };

    if constexpr (Policy == nan_policy::propagate) {
        if (n < s) {
            univariate_accumulator<T, Stats> scalar_acc;
            for (; first1 < last1; ++first1, ++first2) {
                scalar_acc(f(*first1, *first2));
            }
            return univariate_statistics(scalar_acc);
        }

        univariate_accumulator<wide, Stats> acc;
        for (size_t i = 0; i < m; i += s) {
            acc(detail::load<wide>(first1, first2, f));
            detail::advance(s, first1, first2);
        }

        // use a scalar accumulator to gather the remaining values
        if (m < n) {
            auto [sw, sx, sxx] = acc.stats();
            auto scalar_acc = univariate_accumulator<T, Stats>::load_state(sw, sx, sxx);
            for (; first1 < last1; ++first1, ++first2) {
                scalar_acc(f(*first1, *first2));
            }
            return univariate_statistics(scalar_acc);
        }
        return univariate_statistics(acc);
    } else {
        univariate_accumulator<wide, Stats> acc;
        wide skipped {0};
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            // is_finite(x) is is_not_nan(x - x) (per eve's docs): one
            // is_finite call on (a-a)+(b-b) instead of two calls + a
            // logical-and.
            auto finite = eve::is_finite((a - a) + (b - b));
            if (eve::all(finite)) [[likely]] {
                acc(f(a, b));
            } else {
                // sanitize values, not just weight: NaN/Inf * 0 == NaN
                wide sa = eve::if_else(finite, a, wide {0});
                wide sb = eve::if_else(finite, b, wide {0});
                wide w = eve::if_else(finite, wide {1}, wide {0});
                acc(f(sa, sb), w);
                skipped += eve::if_else(finite, wide {0}, wide {1});
            }
            detail::advance(s, first1, first2);
        }

        auto se = univariate_accumulator<T, Stats>::load_state(acc.stats());
        auto skipped_count = static_cast<std::size_t>(eve::reduce(skipped));
        for (; first1 < last1; ++first1, ++first2) {
            if (std::isfinite(*first1) && std::isfinite(*first2)) [[likely]] {
                // unweighted overload: cheaper recurrence than weighted(x, 1)
                se(f(*first1, *first2));
            } else {
                // a w=0 contribution is a no-op on accumulator state (x*0,
                // sum_w += 0, guarded denom == 0) -- skip the call entirely
                // rather than paying for a masked weighted() call.
                ++skipped_count;
            }
        }
        return {univariate_statistics(se), skipped_count};
    }
}

/*!
    \ingroup Univariate

    \brief Weighted variant of `accumulate` (BinaryOp overload): folds a
   caller-supplied weight in, and (under `nan_policy::omit`) into the finite
   mask (`w' = finite ? w : 0`) instead of a flat 0/1 weight.

    \param first3 The begin iterator for the caller-supplied weights
*/
template<std::floating_point T,
         stats Stats = stats::variance,
         nan_policy Policy = nan_policy::propagate,
         std::random_access_iterator I,
         std::random_access_iterator J,
         std::random_access_iterator K,
         typename BinaryOp,
         typename F1 = std::identity,
         typename F2 = std::identity>
    requires std::is_arithmetic_v<std::iter_value_t<K>> && std::is_invocable_v<F1, std::iter_value_t<I>>
    && std::is_invocable_v<F2, std::iter_value_t<J>>
    && std::is_invocable_v<BinaryOp,
                           std::invoke_result_t<F1, std::iter_value_t<I>>,
                           std::invoke_result_t<F2, std::iter_value_t<J>>>
    && concepts::arithmetic_projection<BinaryOp,
                                       std::invoke_result_t<F1, std::iter_value_t<I>>,
                                       std::invoke_result_t<F2, std::iter_value_t<J>>>
inline auto accumulate(I first1,
                       I last1,
                       J first2,
                       K first3,
                       BinaryOp&& op = BinaryOp {},
                       F1&& f1 = F1 {},
                       F2&& f2 = F2 {}) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<univariate_statistics, std::size_t>, univariate_statistics>
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m = n - n % s;

    auto f = [&](auto a, auto b)
    {
        return std::invoke(
            std::forward<BinaryOp>(op), std::invoke(std::forward<F1>(f1), a), std::invoke(std::forward<F2>(f2), b));
    };

    if constexpr (Policy == nan_policy::propagate) {
        if (n < s) {
            univariate_accumulator<T, Stats> scalar_acc;
            for (; first1 < last1; ++first1, ++first2, ++first3) {
                scalar_acc(f(*first1, *first2), *first3);
            }
            return univariate_statistics(scalar_acc);
        }

        univariate_accumulator<wide, Stats> acc;
        for (size_t i = 0; i < m; i += s) {
            acc(detail::load<wide>(first1, first2, f), wide {std::to_address(first3)});
            detail::advance(s, first1, first2, first3);
        }

        // use a scalar accumulator to gather the remaining values
        if (m < n) {
            auto [sw, sx, sxx] = acc.stats();
            auto scalar_acc = univariate_accumulator<T, Stats>::load_state(sw, sx, sxx);
            for (; first1 < last1; ++first1, ++first2, ++first3) {
                scalar_acc(f(*first1, *first2), *first3);
            }
            return univariate_statistics(scalar_acc);
        }
        return univariate_statistics(acc);
    } else {
        univariate_accumulator<wide, Stats> acc;
        wide skipped {0};
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            wide weight {first3};
            auto finite = eve::is_finite((a - a) + (b - b));
            if (eve::all(finite)) [[likely]] {
                acc(f(a, b), weight);
            } else {
                // sanitize values, not just weight: NaN/Inf * 0 == NaN
                wide sa = eve::if_else(finite, a, wide {0});
                wide sb = eve::if_else(finite, b, wide {0});
                wide w = eve::if_else(finite, weight, wide {0});
                acc(f(sa, sb), w);
                skipped += eve::if_else(finite, wide {0}, wide {1});
            }
            detail::advance(s, first1, first2, first3);
        }

        auto se = univariate_accumulator<T, Stats>::load_state(acc.stats());
        auto skipped_count = static_cast<std::size_t>(eve::reduce(skipped));
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            if (std::isfinite(*first1) && std::isfinite(*first2)) [[likely]] {
                se(f(*first1, *first2), *first3);
            } else {
                // a w=0 contribution is a no-op on accumulator state -- skip
                // the call entirely rather than paying for a masked one.
                ++skipped_count;
            }
        }
        return {univariate_statistics(se), skipped_count};
    }
}
}  // namespace univariate

namespace bivariate
{
/*!
    \defgroup Bivariate Bivariate statistics

    \brief Methods for bivariate statistics
*/

/*!
    \ingroup Bivariate

    \brief Compute bivariate statistics from two sequences of values. The values
   can be provided directly or via a projection method.

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used
   to compute the stats.
    \tparam Policy `nan_policy::propagate` (default): a non-finite value
   poisons the whole result. `nan_policy::omit`: positions where either raw
   input is non-finite are skipped (zero-weighted) instead, and the count of
   skipped pairs is additionally returned.

    \param first1 The begin iterator for the first sequence
    \param last1  The end iterator for the first sequence
    \param first2 The begin iterator for the second sequence
    \param f1     A projection mapping `std::iter_value_t<I>` to a scalar value
    \param f2     A projection mapping `std::iter_value_t<J>` to a scalar value

    \return `nan_policy::propagate`: the accumulated bivariate statistics.
   `nan_policy::omit`: the accumulated bivariate statistics over finite
   pairs, and the count of skipped (non-finite) pairs.

    \b Example

    \code
    float x[] = { 1., 1., 2., 6. };
    float y[] = { 2., 4., 3., 1. };
    auto stats = bivariate::accumulate<float>(std::begin(x), std::end(x),
   std::begin(y)); std::cout << stats << "\n";
    // results
    count:                  4
    sum_x:                  10
    ssr_x:                  17
    mean_x:                 2.5
    variance_x:             4.25
    sample variance_x:      5.66667
    sum_y:                  10
    ssr_y:                  5
    mean_y:                 2.5
    variance_y:             1.25
    sample variance_y:      1.66667
    correlation:            -0.759257
    covariance:             -1.75
    sample covariance:      -2.33333
    \endcode
*/
template<std::floating_point T,
         nan_policy Policy = nan_policy::propagate,
         std::random_access_iterator I,
         std::random_access_iterator J,
         typename F1 = std::identity,
         typename F2 = std::identity>
    requires concepts::arithmetic_projection<F1, std::iter_value_t<I>>
    and concepts::arithmetic_projection<F2, std::iter_value_t<J>>
inline auto accumulate(I first1, I last1, J first2, F1&& f1 = F1 {}, F2&& f2 = F2 {}) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<bivariate_statistics, std::size_t>, bivariate_statistics>
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m = n - n % s;

    if constexpr (Policy == nan_policy::propagate) {
        if (n < s) {
            bivariate_accumulator<T> scalar_acc;
            for (; first1 < last1; ++first1, ++first2) {
                scalar_acc(std::invoke(std::forward<F1>(f1), *first1), std::invoke(std::forward<F2>(f2), *first2));
            }
            return bivariate_statistics(scalar_acc);
        }

        bivariate_accumulator<wide> acc;
        for (size_t i = 0; i < m; i += s) {
            acc(detail::load<wide>(first1, std::forward<F1>(f1)), detail::load<wide>(first2, std::forward<F2>(f2)));
            detail::advance(s, first1, first2);
        }

        if (m < n) {
            auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
            auto scalar_acc = bivariate_accumulator<T>::load_state(sx, sy, sw, sxx, syy, sxy);
            for (; first1 < last1; ++first1, ++first2) {
                scalar_acc(std::invoke(std::forward<F1>(f1), *first1), std::invoke(std::forward<F2>(f2), *first2));
            }
            return bivariate_statistics(scalar_acc);
        }

        return bivariate_statistics(acc);
    } else {
        bivariate_accumulator<wide> acc;
        wide skipped {0};
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            auto finite = eve::is_finite((a - a) + (b - b));
            if (eve::all(finite)) [[likely]] {
                // Match plain accumulate's call shape (2-arg unweighted),
                // which delegates to the weighted overload with w=1 --
                // keeps one source of truth for the Welford update.
                acc(std::invoke(f1, a), std::invoke(f2, b));
            } else {
                wide sa = eve::if_else(finite, a, wide {0});
                wide sb = eve::if_else(finite, b, wide {0});
                wide w  = eve::if_else(finite, wide {1}, wide {0});
                acc(std::invoke(f1, sa), std::invoke(f2, sb), w);
                skipped += eve::if_else(finite, wide {0}, wide {1});
            }
            detail::advance(s, first1, first2);
        }

        auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
        auto be = bivariate_accumulator<T>::load_state(sx, sy, sw, sxx, syy, sxy);
        auto skipped_count = static_cast<std::size_t>(eve::reduce(skipped));
        for (; first1 < last1; ++first1, ++first2) {
            if (std::isfinite(*first1) && std::isfinite(*first2)) [[likely]] {
                be(std::invoke(f1, *first1), std::invoke(f2, *first2));
            } else {
                // a w=0 contribution is a no-op on accumulator state -- skip
                // the call entirely rather than paying for a masked one.
                ++skipped_count;
            }
        }
        return {bivariate_statistics(be), skipped_count};
    }
}

/*!
    \ingroup Bivariate

    \brief Weighted variant of `accumulate`: folds a caller-supplied weight
   in, and (under `nan_policy::omit`) into the finite mask
   (`w' = finite ? w : 0`).
*/
template<std::floating_point T,
         nan_policy Policy = nan_policy::propagate,
         std::random_access_iterator I,
         std::random_access_iterator J,
         std::random_access_iterator K,
         typename F1 = std::identity,
         typename F2 = std::identity>
    requires concepts::arithmetic_projection<F1, std::iter_value_t<I>>
    and concepts::arithmetic_projection<F2, std::iter_value_t<J>> and std::is_arithmetic_v<std::iter_value_t<K>>
inline auto accumulate(
    I first1, I last1, J first2, K first3, F1&& f1 = F1 {}, F2&& f2 = F2 {}) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<bivariate_statistics, std::size_t>, bivariate_statistics>
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n = std::distance(first1, last1);
    auto const m = n - n % s;

    if constexpr (Policy == nan_policy::propagate) {
        if (n < s) {
            bivariate_accumulator<T> scalar_acc;
            for (; first1 < last1; ++first1, ++first2, ++first3) {
                scalar_acc(
                    std::invoke(std::forward<F1>(f1), *first1), std::invoke(std::forward<F2>(f2), *first2), *first3);
            }
            return bivariate_statistics(scalar_acc);
        }

        bivariate_accumulator<wide> acc;
        for (size_t i = 0; i < m; i += s) {
            acc(detail::load<wide>(first1, std::forward<F1>(f1)),
                detail::load<wide>(first2, std::forward<F2>(f2)),
                wide {first3});
            detail::advance(s, first1, first2, first3);
        }

        if (m < n) {
            auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
            auto scalar_acc = bivariate_accumulator<T>::load_state(sx, sy, sw, sxx, syy, sxy);
            for (; first1 < last1; ++first1, ++first2, ++first3) {
                scalar_acc(std::invoke(std::forward<F1>(f1), *first1), std::invoke(std::forward<F2>(f2), *first2), *first3);
            }
            return bivariate_statistics(scalar_acc);
        }
        return bivariate_statistics(acc);
    } else {
        bivariate_accumulator<wide> acc;
        wide skipped {0};
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            wide weight {first3};
            auto finite = eve::is_finite((a - a) + (b - b));
            if (eve::all(finite)) [[likely]] {
                acc(std::invoke(f1, a), std::invoke(f2, b), weight);
            } else {
                wide sa = eve::if_else(finite, a, wide {0});
                wide sb = eve::if_else(finite, b, wide {0});
                wide w  = eve::if_else(finite, weight, wide {0});
                acc(std::invoke(f1, sa), std::invoke(f2, sb), w);
                skipped += eve::if_else(finite, wide {0}, wide {1});
            }
            detail::advance(s, first1, first2, first3);
        }

        auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
        auto be = bivariate_accumulator<T>::load_state(sx, sy, sw, sxx, syy, sxy);
        auto skipped_count = static_cast<std::size_t>(eve::reduce(skipped));
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            if (std::isfinite(*first1) && std::isfinite(*first2)) [[likely]] {
                be(std::invoke(f1, *first1), std::invoke(f2, *first2), *first3);
            } else {
                // a w=0 contribution is a no-op on accumulator state -- skip
                // the call entirely rather than paying for a masked one.
                ++skipped_count;
            }
        }
        return {bivariate_statistics(be), skipped_count};
    }
}
}  // namespace bivariate

namespace metrics
{
/*!
    \defgroup Metrics Regression metrics

    \brief Regression metrics (R2, MSE, MLSE, MAE).
*/

/*!
    \ingroup Metrics

    \brief Computes the coefficient of determination \f$R^2\f$

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used
   to compute the stats

    \f{align}{
        R^2(y, \hat{y}) &= 1 - \frac{\text{RSS}}{\text{TSS}}\text{, where}\\
        \text{RSS} &= \sum_{i=1}^n \left( y - \hat{y} \right)^2\\
        \text{TSS} &= \sum_{i=1}^n \left( y - \bar{y} \right)^2\\
    \f}
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto r2_score(I first1, I last1, J first2) noexcept -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - (n % s)};

    univariate_accumulator<wide, stats::sum> wx;
    univariate_accumulator<wide, stats::variance> wy;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        wx(eve::sqr(y_true - y_pred));
        wy(y_true);
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto sx = univariate_accumulator<T, stats::sum>::load_state(wx.stats());
    auto sy = univariate_accumulator<T, stats::variance>::load_state(wy.stats());

    for (; first1 < last1; ++first1, ++first2) {
        sx(eve::sqr(*first1 - *first2));
        sy(*first1);
    }

    auto const rss = univariate_statistics(sx).sum;
    auto const tss = univariate_statistics(sy).ssr;

    return tss < std::numeric_limits<double>::epsilon() ? std::numeric_limits<double>::lowest() : 1.0 - (rss / tss);
}

/*!
    \ingroup Metrics

    \brief Computes the weighted coefficient of determination \f$R^2\f$

    \f{align}{
        R^2(y, \hat{y}) &= 1 - \frac{\text{RSS}}{\text{TSS}}\text{, where}\\
        \text{RSS} &= \sum_{i=1}^n w_i \left( y - \hat{y} \right)^2\\
        \text{TSS} &= \sum_{i=1}^n w_i \left( y - \bar{y} \right)^2\\
    \f}
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J, std::contiguous_iterator K>
inline auto r2_score(I first1, I last1, J first2, K first3) noexcept -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - (n % s)};

    univariate_accumulator<wide, stats::sum> wx;
    univariate_accumulator<wide, stats::variance> wy;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        wide weight {first3};
        wx(eve::sqr(y_true - y_pred), weight);
        wy(y_true, weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto sx = univariate_accumulator<T, stats::sum>::load_state(wx.stats());
    auto sy = univariate_accumulator<T, stats::variance>::load_state(wy.stats());

    for (; first1 < last1; ++first1, ++first2, ++first3) {
        sx(eve::sqr(*first1 - *first2), *first3);
        sy(*first1, *first3);
    }

    auto const rss = univariate_statistics(sx).sum;
    auto const tss = univariate_statistics(sy).ssr;

    return tss < std::numeric_limits<double>::epsilon() ? std::numeric_limits<double>::lowest() : 1.0 - rss / tss;
}

/*!
    \ingroup Metrics

    \brief Computes the mean squared error

    \tparam Policy `nan_policy::propagate` (default): a non-finite value
   poisons the whole result. `nan_policy::omit`: rows where either value is
   non-finite are skipped rather than poisoning the whole result.

    \f[
        \text{MSE}(y, \hat{y}) = \displaystyle \frac{1}{n} {\sum_{i=1}^n
   \left(y-\hat{y}\right)^2}
    \f]

    \return `nan_policy::propagate`: the MSE. `nan_policy::omit`: the MSE
   over finite pairs, and the count of skipped (non-finite) pairs.
*/
template<std::floating_point T, nan_policy Policy = nan_policy::propagate, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto mean_squared_error(I first1, I last1, J first2) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<double, std::size_t>, double>
{
    if constexpr (Policy == nan_policy::propagate) {
        using wide = eve::wide<T>;
        auto constexpr s {wide::size()};
        auto const n {std::distance(first1, last1)};
        auto const m {n - n % s};

        univariate_accumulator<wide, stats::mean> we;
        for (auto i = 0; i < m; i += s) {
            wide y_true {first1};
            wide y_pred {first2};
            we(eve::sqr(y_true - y_pred));
            detail::advance(s, first1, first2);
        }

        // use scalar accumulators for the remaining values
        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        for (; first1 < last1; ++first1, ++first2) {
            se(eve::sqr(*first1 - *first2));
        }
        return univariate_statistics(se).mean;
    } else {
        auto [st, skipped] = univariate::accumulate<T, stats::mean, nan_policy::omit>(
            first1, last1, first2, [](auto a, auto b) { return eve::sqr(a - b); });
        return {st.mean, skipped};
    }
}

/*!
    \ingroup Metrics

    \brief Weighted variant of `mean_squared_error`.

    \f[
        \text{MSE}(y, \hat{y}) = {\displaystyle \frac{1}{\sum_{i=1}^n w_i}}
   \sum_{i=1}^n w_i \left(y-\hat{y}\right)^2
    \f]
*/
template<std::floating_point T, nan_policy Policy = nan_policy::propagate, std::contiguous_iterator I, std::contiguous_iterator J, std::contiguous_iterator K>
inline auto mean_squared_error(I first1, I last1, J first2, K first3) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<double, std::size_t>, double>
{
    if constexpr (Policy == nan_policy::propagate) {
        using wide = eve::wide<T>;
        auto constexpr s {wide::size()};
        auto const n {std::distance(first1, last1)};
        auto const m {n - n % s};

        univariate_accumulator<wide, stats::mean> we;
        for (auto i = 0; i < m; i += s) {
            wide y_true {first1};
            wide y_pred {first2};
            wide weight {first3};
            we(eve::sqr(y_true - y_pred), weight);
            detail::advance(s, first1, first2, first3);
        }

        // use scalar accumulators for the remaining values
        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            se(eve::sqr(*first1 - *first2), *first3);
        }
        return univariate_statistics(se).mean;
    } else {
        auto [st, skipped] = univariate::accumulate<T, stats::mean, nan_policy::omit>(
            first1, last1, first2, first3, [](auto a, auto b) { return eve::sqr(a - b); });
        return {st.mean, skipped};
    }
}

/*!
    \ingroup Metrics

    \brief Normalized mean squared error over (estimated, target) pairs.

    \tparam Policy `nan_policy::propagate` (default): a non-finite value
   poisons the whole result. `nan_policy::omit`: rows where either value is
   non-finite are skipped rather than poisoning the whole result -- the
   target variance is computed over the same finite subset (the mask is
   shared), keeping the numerator and denominator consistent.

    Single pass over the input in both modes: the residual mean and target
   variance accumulators run in lockstep, instead of composing two
   independent passes.

    \f[
        \text{NMSE}(y, \hat{y}) = \frac{
            \overline{(y - \hat{y})^2}
        }{ \text{Var}(y) }
    \f]

    \return `nan_policy::propagate`: the NMSE (0.0 if the target variance is
   0). `nan_policy::omit`: the NMSE over finite pairs (0.0 if the target
   variance over that subset is 0), and the count of skipped (non-finite)
   pairs.
*/
template<std::floating_point T, nan_policy Policy = nan_policy::propagate, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto normalized_mean_squared_error(I first1, I last1, J first2) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<double, std::size_t>, double>
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m = n - n % s;

    univariate_accumulator<wide, stats::mean> we;      // residual mean: <(a-b)^2>
    univariate_accumulator<wide, stats::variance> wv;  // target variance: Var(b)

    if constexpr (Policy == nan_policy::propagate) {
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            we(eve::sqr(a - b));
            wv(b);
            detail::advance(s, first1, first2);
        }

        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        auto sv = univariate_accumulator<T, stats::variance>::load_state(wv.stats());
        for (; first1 < last1; ++first1, ++first2) {
            se(eve::sqr(*first1 - *first2));
            sv(*first2);
        }

        auto const mean = univariate_statistics(se).mean;
        auto const var  = univariate_statistics(sv).variance;
        return var > 0.0 ? mean / var : 0.0;
    } else {
        wide skipped {0};
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            auto finite = eve::is_finite((a - a) + (b - b));
            if (eve::all(finite)) [[likely]] {
                we(eve::sqr(a - b));
                wv(b);
            } else {
                wide sa = eve::if_else(finite, a, wide {0});
                wide sb = eve::if_else(finite, b, wide {0});
                wide w  = eve::if_else(finite, wide {1}, wide {0});
                // mask carried by weight -- the weighted overload's
                // zero-denominator guard handles lanes whose first
                // contribution is zero-weighted without NaN-poisoning the
                // accumulator state.
                we(eve::sqr(sa - sb), w);
                wv(sb, w);
                skipped += eve::if_else(finite, wide {0}, wide {1});
            }
            detail::advance(s, first1, first2);
        }

        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        auto sv = univariate_accumulator<T, stats::variance>::load_state(wv.stats());
        auto skipped_count = static_cast<std::size_t>(eve::reduce(skipped));
        for (; first1 < last1; ++first1, ++first2) {
            if (std::isfinite(*first1) && std::isfinite(*first2)) [[likely]] {
                se(eve::sqr(*first1 - *first2));
                sv(*first2);
            } else {
                // a w=0 contribution is a no-op on accumulator state -- skip
                // the calls entirely rather than paying for masked ones.
                ++skipped_count;
            }
        }

        auto const mean = univariate_statistics(se).mean;
        auto const var  = univariate_statistics(sv).variance;
        return {var > 0.0 ? mean / var : 0.0, skipped_count};
    }
}

/*!
    \ingroup Metrics

    \brief Weighted variant of `normalized_mean_squared_error`.
*/
template<std::floating_point T, nan_policy Policy = nan_policy::propagate, std::contiguous_iterator I, std::contiguous_iterator J, std::contiguous_iterator K>
inline auto normalized_mean_squared_error(I first1, I last1, J first2, K first3) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<double, std::size_t>, double>
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m = n - n % s;

    univariate_accumulator<wide, stats::mean> we;
    univariate_accumulator<wide, stats::variance> wv;

    if constexpr (Policy == nan_policy::propagate) {
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            wide weight {first3};
            we(eve::sqr(a - b), weight);
            wv(b, weight);
            detail::advance(s, first1, first2, first3);
        }

        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        auto sv = univariate_accumulator<T, stats::variance>::load_state(wv.stats());
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            se(eve::sqr(*first1 - *first2), *first3);
            sv(*first2, *first3);
        }

        auto const mean = univariate_statistics(se).mean;
        auto const var  = univariate_statistics(sv).variance;
        return var > 0.0 ? mean / var : 0.0;
    } else {
        wide skipped {0};
        for (size_t i = 0; i < m; i += s) {
            wide a {first1};
            wide b {first2};
            wide weight {first3};
            auto finite = eve::is_finite((a - a) + (b - b));
            if (eve::all(finite)) [[likely]] {
                we(eve::sqr(a - b), weight);
                wv(b, weight);
            } else {
                wide sa = eve::if_else(finite, a, wide {0});
                wide sb = eve::if_else(finite, b, wide {0});
                wide w  = eve::if_else(finite, weight, wide {0});
                we(eve::sqr(sa - sb), w);
                wv(sb, w);
                skipped += eve::if_else(finite, wide {0}, wide {1});
            }
            detail::advance(s, first1, first2, first3);
        }

        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        auto sv = univariate_accumulator<T, stats::variance>::load_state(wv.stats());
        auto skipped_count = static_cast<std::size_t>(eve::reduce(skipped));
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            if (std::isfinite(*first1) && std::isfinite(*first2)) [[likely]] {
                se(eve::sqr(*first1 - *first2), *first3);
                sv(*first2, *first3);
            } else {
                // a w=0 contribution is a no-op on accumulator state -- skip
                // the calls entirely rather than paying for masked ones.
                ++skipped_count;
            }
        }

        auto const mean = univariate_statistics(se).mean;
        auto const var  = univariate_statistics(sv).variance;
        return {var > 0.0 ? mean / var : 0.0, skipped_count};
    }
}

/*!
    \ingroup Metrics

    \brief Computes the mean squared logarithmic error

    \f[
        \text{MSLE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (\log_e (1 + y_i) -
   \log_e (1 + \hat{y}_i) )^2
    \f]
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto mean_squared_log_error(I first1, I last1, J first2) noexcept -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    univariate_accumulator<wide, stats::mean> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        we(eve::sqr(eve::log1p(y_true) - eve::log1p(y_pred)));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2) {
        se(eve::sqr(eve::log1p(*first1) - eve::log1p(*first2)));
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Computes the weighted mean squared logarithmic error

    \f[
        \text{MSLE}(y, \hat{y}) = \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^{n} w_i
   (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2
    \f]
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J, std::contiguous_iterator K>
inline auto mean_squared_log_error(I first1, I last1, J first2, K first3) noexcept -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    univariate_accumulator<wide, stats::mean> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        wide weight {first3};
        we(eve::sqr(eve::log1p(y_true) - eve::log1p(y_pred)), weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2, ++first3) {
        se(eve::sqr(eve::log1p(*first1) - eve::log1p(*first2)), *first3);
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Computes the mean absolute error

    \tparam Policy `nan_policy::propagate` (default): a non-finite value
   poisons the whole result. `nan_policy::omit`: rows where either value is
   non-finite are skipped rather than poisoning the whole result.

    \f[
        \text{MAE}(y, \hat{y}) = \displaystyle \frac{1}{n} {\sum_{i=1}^n
   |y-\hat{y}|}
    \f]

    \return `nan_policy::propagate`: the MAE. `nan_policy::omit`: the MAE
   over finite pairs, and the count of skipped (non-finite) pairs.
*/
template<std::floating_point T, nan_policy Policy = nan_policy::propagate, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto mean_absolute_error(I first1, I last1, J first2) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<double, std::size_t>, double>
{
    if constexpr (Policy == nan_policy::propagate) {
        using wide = eve::wide<T>;
        auto constexpr s {wide::size()};
        auto const n {std::distance(first1, last1)};
        auto const m {n - n % s};

        univariate_accumulator<wide, stats::mean> we;
        for (auto i = 0; i < m; i += s) {
            wide y_true {first1};
            wide y_pred {first2};
            we(eve::abs(y_true - y_pred));
            detail::advance(s, first1, first2);
        }

        // use scalar accumulators for the remaining values
        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        for (; first1 < last1; ++first1, ++first2) {
            se(eve::abs(*first1 - *first2));
        }
        return univariate_statistics(se).mean;
    } else {
        auto [st, skipped] = univariate::accumulate<T, stats::mean, nan_policy::omit>(
            first1, last1, first2, [](auto a, auto b) { return eve::abs(a - b); });
        return {st.mean, skipped};
    }
}

/*!
    \ingroup Metrics

    \brief Weighted variant of `mean_absolute_error`.

    \f[
        \text{MAE}(y, \hat{y}) = \displaystyle \frac{1}{\sum_{i=1}^n w_i}
   \sum_{i=1}^n w_i |y-\hat{y}|
    \f]
*/
template<std::floating_point T, nan_policy Policy = nan_policy::propagate, std::contiguous_iterator I, std::contiguous_iterator J, std::contiguous_iterator K>
inline auto mean_absolute_error(I first1, I last1, J first2, K first3) noexcept
    -> std::conditional_t<Policy == nan_policy::omit, std::pair<double, std::size_t>, double>
{
    if constexpr (Policy == nan_policy::propagate) {
        using wide = eve::wide<T>;
        auto constexpr s {wide::size()};
        auto const n {std::distance(first1, last1)};
        auto const m {n - n % s};

        univariate_accumulator<wide, stats::mean> we;
        for (auto i = 0; i < m; i += s) {
            wide y_true {first1};
            wide y_pred {first2};
            wide weight {first3};
            we(eve::abs(y_true - y_pred), weight);
            detail::advance(s, first1, first2, first3);
        }

        // use scalar accumulators for the remaining values
        auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
        for (; first1 < last1; ++first1, ++first2, ++first3) {
            se(eve::abs(*first1 - *first2), *first3);
        }
        return univariate_statistics(se).mean;
    } else {
        auto [st, skipped] = univariate::accumulate<T, stats::mean, nan_policy::omit>(
            first1, last1, first2, first3, [](auto a, auto b) { return eve::abs(a - b); });
        return {st.mean, skipped};
    }
}

/*!
    \ingroup Metrics

    \brief Computes the mean absolute percentage error

    \tparam T

    \f[
        \text{MAPE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} \frac{{}\left| y_i
   - \hat{y}_i \right|}{\max(\epsilon, \left| y_i \right|)}
    \f]
    where \f$\epsilon\f$ = `std::numeric_limits<T>::epsilon()` is an arbitrarily
   small constant to prevent division by zero.
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto mean_absolute_percentage_error(I first1, I last1, J first2) noexcept -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    auto constexpr eps {std::numeric_limits<T>::epsilon()};

    univariate_accumulator<wide, stats::mean> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        we(eve::abs(y_true - y_pred) / eve::max(eps, eve::abs(y_true)));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2) {
        se(eve::abs(*first1 - *first2) / eve::max(eps, eve::abs(*first1)));
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Weighted mean absolute percentage error

    \f[
        \text{WMAPE}(y, \hat{y}) = \displaystyle \frac{1}{\sum_i^n w_i}
   \frac{\sum_{i=1}^n w_i |y-\hat{y}|}{\max(\epsilon, \left| y_i \right|)}
    \f]
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J, std::contiguous_iterator K>
inline auto mean_absolute_percentage_error(I first1, I last1, J first2, K first3) noexcept
    -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    auto constexpr eps {std::numeric_limits<T>::epsilon()};

    univariate_accumulator<wide, stats::mean> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        wide weight {first3};
        we(eve::abs(y_true - y_pred) / eve::max(eps, eve::abs(y_true)), weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::mean>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2, ++first3) {
        se(eve::abs(*first1 - *first2) / eve::max(eps, eve::abs(*first1)), *first3);
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Negative log likelihood loss with Poisson distribution of target.

    \f[
        -\log\mathcal{L}_\text{poisson}(y, \hat{y}) = \hat{y} - y \cdot
   \log(\hat{y}) + \ln(|\Gamma(y+1)|)
    \f] where \f$\ln|\Gamma(y+1)| = \log(y!)\f$ is computed via <a
   href="https://jfalcou.github.io/eve/group__special_gae09a3d5ef50adfebd1d42611611cae5a.html#gae09a3d5ef50adfebd1d42611611cae5a">eve::log_abs_gamma</a>.
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto poisson_neg_likelihood_loss(I first1, I last1, J first2) noexcept -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    univariate_accumulator<wide, stats::sum> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        we(y_pred - y_true * eve::log(y_pred) + eve::log_abs_gamma(T {1} + y_true));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::sum>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2) {
        se(*first2 - *first1 * eve::log(*first2) + eve::log_abs_gamma(T {1} + *first1));
    }
    return univariate_statistics(se).sum;
}

/*!
    \ingroup Metrics

    \brief Negative log likelihood loss with Poisson distribution of target. The
   mean in each bin is multiplied by a weight before the Poisson likelihood is
   applied.

    \f[
        -\log\mathcal{L}_\text{poisson}(y, w \cdot \hat{y}) = w\hat{y} - y \cdot
   \log(w\hat{y}) + \ln(|\Gamma(y+1)|)
    \f] where \f$\ln|\Gamma(y+1)| = \log(y!)\f$ is computed via <a
   href="https://jfalcou.github.io/eve/group__special_gae09a3d5ef50adfebd1d42611611cae5a.html#gae09a3d5ef50adfebd1d42611611cae5a">eve::log_abs_gamma</a>.
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J, std::contiguous_iterator K>
inline auto poisson_neg_likelihood_loss(I first1, I last1, J first2, K first3) noexcept
    -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    univariate_accumulator<wide, stats::sum> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred = wide {first2} * wide {first3};
        we(y_pred - y_true * eve::log(y_pred) + eve::log_abs_gamma(T {1} + y_true));
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::sum>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2, ++first3) {
        se(*first2 * *first3 - *first1 * eve::log(*first2 * *first3) + eve::log_abs_gamma(T {1} + *first1));
    }
    return univariate_statistics(se).sum;
}

/*!
    \ingroup Metrics

    \brief Negative log likelihood loss under a Gaussian with known scalar noise
   level \f$\sigma\f$.

    \f[
        -\log\mathcal{L}_\text{gaussian}(y, \hat{y}, \sigma) =
        \frac{n}{2}\log(2\pi) + n\log(\sigma) + \frac{1}{2\sigma^2}
        \sum_i (y_i - \hat{y}_i)^2
    \f]

    \pre `sigma > 0`. Passing a non-positive value produces `NaN`/`inf` silently.
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto gaussian_neg_likelihood_loss(I first1, I last1, J first2, T sigma) noexcept
    -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    univariate_accumulator<wide, stats::sum> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        we(eve::sqr(y_true - y_pred));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::sum>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2) {
        se(eve::sqr(*first1 - *first2));
    }
    auto const ssr = univariate_statistics(se).sum;
    auto const pi = std::numbers::pi_v<double>;
    return 0.5 * static_cast<double>(n) * std::log(2.0 * pi)
        + static_cast<double>(n) * std::log(static_cast<double>(sigma))
        + ssr / (2.0 * static_cast<double>(sigma) * static_cast<double>(sigma));
}

/*!
    \ingroup Metrics

    \brief Negative log likelihood loss with Poisson distribution of target,
   where the model outputs \f$x = \log(\mu)\f$ (the natural log of the Poisson
   mean).

    \f[
        -\log\mathcal{L}_\text{poisson-log}(y, x) =
        \sum_i \left[ e^{x_i} - y_i \cdot x_i + \ln(|\Gamma(y_i + 1)|) \right]
    \f]
*/
template<std::floating_point T, std::contiguous_iterator I, std::contiguous_iterator J>
inline auto poisson_log_neg_likelihood_loss(I first1, I last1, J first2) noexcept -> double
{
    using wide = eve::wide<T>;
    auto constexpr s {wide::size()};
    auto const n {std::distance(first1, last1)};
    auto const m {n - n % s};

    univariate_accumulator<wide, stats::sum> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true {first1};
        wide y_pred {first2};
        we(eve::exp(y_pred) - y_true * y_pred + eve::log_abs_gamma(T {1} + y_true));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T, stats::sum>::load_state(we.stats());
    for (; first1 < last1; ++first1, ++first2) {
        se(eve::exp(*first2) - *first1 * *first2 + eve::log_abs_gamma(T {1} + *first1));
    }
    return univariate_statistics(se).sum;
}
}  // namespace metrics

}  // namespace VSTAT_NAMESPACE

#endif
