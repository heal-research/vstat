// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2024 Heal Research

#ifndef VSTAT_HPP
#define VSTAT_HPP

#include "bivariate.hpp"
#include "univariate.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>

#include <eve/module/math.hpp>
#include <eve/module/special.hpp>

namespace VSTAT_NAMESPACE {

namespace detail {
    // utility method to load data into a wide type
    template<eve::simd_value T, std::input_iterator I, typename F>
    requires std::is_invocable_v<F, std::iter_value_t<I>>
    auto inline load(I iter, F&& func) {
        return [&]<std::size_t ...Idx>(std::index_sequence<Idx...>){
            return T{ std::forward<F>(func)(*(iter + Idx))... };
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


/*!
    \defgroup Univariate Univariate statistics

    \brief Methods for univariate statistics
*/
namespace univariate {
/*!
    \ingroup Univariate

    \brief Accumulates a sequence of (projected) values

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used to compute the stats.

    \param first The begin iterator for the first sequence
    \param last  The end iterator for the first sequence
    \param f     A projection mapping `std::iter_value_t<I>` to a scalar value
*/
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
            scalar_acc(std::invoke(std::forward<F>(f), *first));
        }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(detail::load<wide>(first, std::forward<F>(f)));
        detail::advance(s, first);
    }

    // gather the remaining values with a scalar accumulator
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
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

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used to compute the stats.

    \param first1 The begin iterator for the first sequence
    \param last1  The end iterator for the first sequence
    \param first2 The begin iterator for the second (weights) sequence
    \param f      A projection mapping `std::iter_value_t<I>` to a scalar value
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J, typename F = std::identity>
requires concepts::arithmetic_projection<F, std::iter_value_t<I>> and std::is_arithmetic_v<std::iter_value_t<J>>
inline auto accumulate(I first1, std::sized_sentinel_for<I> auto last1, J first2, F&& f = F{}) noexcept -> univariate_statistics
{
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n { std::distance(first1, last1) };
    const size_t m = n - n % s;

    if (n < s) {
        univariate_accumulator<T> scalar_acc;
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(std::forward<F>(f), *first1), *first2);
        }
        return univariate_statistics(scalar_acc);
    }

    univariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(detail::load<wide>(first1, std::forward<F>(f)), wide(first2, first2 + s));
        detail::advance(s, first1, first2);
    }

    // use a scalar accumulator to gather the remaining values
    if (m < n) {
        auto [sw, sx, sxx] = acc.stats();
        auto scalar_acc = univariate_accumulator<T>::load_state(sw, sx, sxx);
        for (; first1 < last1; ++first1, ++first2) {
            scalar_acc(std::invoke(std::forward<F>(f), *first1), *first2);
        }
        return univariate_statistics(scalar_acc);
    }
    return univariate_statistics(acc);
}

/*!
    \ingroup Univariate

    \brief Accumulates over the projected values from applying `BinaryOp` on the input sequences.

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used to compute the stats
    \tparam BinaryOp Binary projection \f$op(a,b) \to c\f$
    \tparam F1 Unary projection \f$f(x_1) \to a\f$
    \tparam F2 Unary projection \f$f(x_2) \to b\f$

    \param first1 The begin iterator for the first sequence
    \param last1  The end iterator for the first sequence
    \param first2 The begin iterator for the second sequence
    \param op     A binary projection mapping a tuple \f$(f_1(\cdot), f_2(\cdot))\f$ to a scalar value
    \param f1     A projection mapping `std::iter_value_t<I>` to a scalar value
    \param f2     A projection mapping `std::iter_value_t<J>` to a scalar value
*/
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

    auto f = [&](auto a, auto b){
        return std::invoke(std::forward<BinaryOp>(op), std::invoke(std::forward<F1>(f1), a), std::invoke(std::forward<F2>(f2), b));
    };

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

/*!
    \ingroup Univariate

    \brief Accumulates over the projected values from applying `BinaryOp` on the input sequences, with weights

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used to compute the stats
    \tparam BinaryOp Binary projection \f$op(a,b) \to c\f$
    \tparam F1 Unary projection \f$f(x_1) \to a\f$
    \tparam F2 Unary projection \f$f(x_2) \to b\f$

    \param first1 The begin iterator for the first sequence
    \param last1  The end iterator for the first sequence
    \param first2 The begin iterator for the second sequence
    \param first3 The begin iterator for the third sequence (weights)
    \param op     A binary projection mapping a tuple \f$(f_1(\bullet), f_2(\bullet))\f$ to a scalar value
    \param f1     A projection mapping `std::iter_value_t<I>` to a scalar value
    \param f2     A projection mapping `std::iter_value_t<J>` to a scalar value
*/
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

    auto f = [&](auto a, auto b){
        return std::invoke(std::forward<BinaryOp>(op), std::invoke(std::forward<F1>(f1), a), std::invoke(std::forward<F2>(f2), b));
    };

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
/*!
    \defgroup Bivariate Bivariate statistics

    \brief Methods for bivariate statistics
*/

/*!
    \ingroup Bivariate

    \brief Compute bivariate statistics from two sequences of values. The values can be provided directly or via a projection method.

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used to compute the stats.

    \param first1 The begin iterator for the first sequence
    \param last1  The end iterator for the first sequence
    \param first2 The begin iterator for the second sequence
    \param f1     A projection mapping `std::iter_value_t<I>` to a scalar value
    \param f2     A projection mapping `std::iter_value_t<J>` to a scalar value

    \b Example

    \code
    float x[] = { 1., 1., 2., 6. };
    float y[] = { 2., 4., 3., 1. };
    auto stats = bivariate::accumulate<float>(std::begin(x), std::end(x), std::begin(y));
    std::cout << stats << "\n";
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
            scalar_acc(std::invoke(std::forward<F1>(f1), *first1++), std::invoke(std::forward<F2>(f2), *first2++), *first3++);
        }
        return bivariate_statistics(scalar_acc);
    }

    bivariate_accumulator<wide> acc;
    for (size_t i = 0; i < m; i += s) {
        acc(
            detail::load<wide>(first1, std::forward<F1>(f1)),
            detail::load<wide>(first2, std::forward<F2>(f2)),
            wide(first3, first3 + s)
        );
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
}
} // namespace bivariate

namespace metrics {
/*!
    \defgroup Metrics Regression metrics

    \brief Regression metrics (R2, MSE, MLSE, MAE).
*/

/*!
    \ingroup Metrics

    \brief Computes the coefficient of determination \f$R^2\f$

    \tparam T The scalar value type underlying the `eve::wide<T>` SIMD type used to compute the stats

    \f{align}{
        R^2(y, \hat{y}) &= 1 - \frac{\text{RSS}}{\text{TSS}}\text{, where}\\
        \text{RSS} &= \sum_{i=1}^n \left( y - \hat{y} \right)^2\\
        \text{TSS} &= \sum_{i=1}^n \left( y - \bar{y} \right)^2\\
    \f}
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J>
inline auto r2_score(I first1, std::sentinel_for<I> auto last1, J first2) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> wx;
    univariate_accumulator<wide> wy;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        wx(eve::sqr(y_true-y_pred));
        wy(y_true);
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto sx = univariate_accumulator<T>::load_state(wx.stats());
    auto sy = univariate_accumulator<T>::load_state(wy.stats());

    for(; first1 < last1; ++first1, ++first2) {
        sx(eve::sqr(*first1 - *first2));
        sy(*first1);
    }

    auto const rss = univariate_statistics(sx).sum;
    auto const tss = univariate_statistics(sy).ssr;

    return tss < std::numeric_limits<double>::epsilon()
        ? std::numeric_limits<double>::lowest()
        : 1.0 - rss / tss;
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
template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K>
inline auto r2_score(I first1, std::sentinel_for<I> auto last1, J first2, K first3) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> wx;
    univariate_accumulator<wide> wy;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        wide weight{first3, first3+s};
        wx(eve::sqr(y_true-y_pred), weight);
        wy(y_true, weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto sx = univariate_accumulator<T>::load_state(wx.stats());
    auto sy = univariate_accumulator<T>::load_state(wy.stats());

    for(; first1 < last1; ++first1, ++first2, ++first3) {
        sx(eve::sqr(*first1 - *first2), *first3);
        sy(*first1, *first3);
    }

    auto const rss = univariate_statistics(sx).sum;
    auto const tss = univariate_statistics(sy).ssr;

    return tss < std::numeric_limits<double>::epsilon()
        ? std::numeric_limits<double>::lowest()
        : 1.0 - rss / tss;
}

/*!
    \ingroup Metrics

    \brief Computes the mean squared error

    \f[
        \text{MSE}(y, \hat{y}) = \displaystyle \frac{1}{n} {\sum_{i=1}^n \left(y-\hat{y}\right)^2}
    \f]
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J>
inline auto mean_squared_error(I first1, std::sentinel_for<I> auto last1, J first2) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        we(eve::sqr(y_true-y_pred));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2) {
        se(eve::sqr(*first1 - *first2));
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Computes the weighted mean squared error

    \f[
        \text{MSE}(y, \hat{y}) = {\displaystyle \frac{1}{\sum_{i=1}^n w_i}} \sum_{i=1}^n w_i \left(y-\hat{y}\right)^2
    \f]
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K>
inline auto mean_squared_error(I first1, std::sentinel_for<I> auto last1, J first2, K first3) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        wide weight{first3, first3+s};
        we(eve::sqr(y_true-y_pred), weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2, ++first3) {
        se(eve::sqr(*first1 - *first2), *first3);
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Computes the mean squared logarithmic error

    \f[
        \text{MSLE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2
    \f]
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J>
inline auto mean_squared_log_error(I first1, std::sentinel_for<I> auto last1, J first2) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        we(eve::sqr(eve::log1p(y_true)-eve::log1p(y_pred)));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2) {
        se(eve::sqr(eve::log1p(*first1) - eve::log1p(*first2)));
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Computes the weighted mean squared logarithmic error

    \f[
        \text{MSLE}(y, \hat{y}) = \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^{n} w_i (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2
    \f]
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K>
inline auto mean_squared_log_error(I first1, std::sentinel_for<I> auto last1, J first2, K first3) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        wide weight{first3, first3+s};
        we(eve::sqr(eve::log1p(y_true)-eve::log1p(y_pred)), weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2, ++first3) {
        se(eve::sqr(eve::log1p(*first1) - eve::log1p(*first2)), *first3);
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Computes the mean absolute error

    \f[
        \text{MAE}(y, \hat{y}) = \displaystyle \frac{1}{n} {\sum_{i=1}^n |y-\hat{y}|}
    \f]
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J>
inline auto mean_absolute_error(I first1, std::sentinel_for<I> auto last1, J first2) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        we(eve::abs(y_true-y_pred));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2) {
        se(eve::abs(*first1 - *first2));
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Weighted mean absolute error

    \f[
        \text{MAE}(y, \hat{y}) = \displaystyle \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^n w_i |y-\hat{y}|
    \f]
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K>
inline auto mean_absolute_error(I first1, std::sentinel_for<I> auto last1, J first2, K first3) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        wide weight{first3, first3+s};
        we(eve::abs(y_true-y_pred), weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2, ++first3) {
        se(eve::abs(*first1 - *first2), *first3);
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Computes the mean absolute error

    \tparam T

    \f[
        \text{MAPE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} \frac{{}\left| y_i - \hat{y}_i \right|}{\max(\epsilon, \left| y_i \right|)}
    \f]
    where \f$\epsilon\f$ = `std::numeric_limits<T>::epsilon()` is an arbitrarily small constant to prevent division by zero.
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J>
inline auto mean_absolute_percentage_error(I first1, std::sentinel_for<I> auto last1, J first2) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    auto constexpr eps{ std::numeric_limits<T>::epsilon() };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        we(eve::abs(y_true-y_pred) / eve::max(eps, eve::abs(y_true)));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2) {
        se(eve::abs(*first1 - *first2) / eve::max(eps, eve::abs(*first1)));
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Weighted mean absolute percentage error

    \f[
        \text{WMAPE}(y, \hat{y}) = \displaystyle \frac{1}{\sum_i^n w_i} \frac{\sum_{i=1}^n w_i |y-\hat{y}|}{\max(\epsilon, \left| y_i \right|)}
    \f]
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K>
inline auto mean_absolute_percentage_error(I first1, std::sentinel_for<I> auto last1, J first2, K first3) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        wide weight{first3, first3+s};
        we(eve::abs(y_true-y_pred), weight);
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2, ++first3) {
        se(eve::abs(*first1 - *first2), *first3);
    }
    return univariate_statistics(se).mean;
}

/*!
    \ingroup Metrics

    \brief Negative log likelihood loss with Poisson distribution of target.

    \f[
        -\log\mathcal{L}_\text{poisson}(y, \hat{y}) = \hat{y} - y \cdot \log(\hat{y}) + \ln(|\Gamma(y)|)
    \f] where \f$\Gamma(y)\f$ is returned by <a href="https://jfalcou.github.io/eve/group__special_gae09a3d5ef50adfebd1d42611611cae5a.html#gae09a3d5ef50adfebd1d42611611cae5a">eve::gamma_p</a>.
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J>
inline auto poisson_neg_likelihood_loss(I first1, std::sentinel_for<I> auto last1, J first2) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    auto constexpr eps{ std::numeric_limits<T>::epsilon() };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred{first2, first2+s};
        we(y_pred - y_true * eve::log(y_pred) + eve::log_abs_gamma(T{1} + y_true));
        detail::advance(s, first1, first2);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2) {
        se(*first2 - *first1 * eve::log(*first2) + eve::log_abs_gamma(T{1} + *first1));
    }
    return univariate_statistics(se).sum;
}

/*!
    \ingroup Metrics

    \brief Negative log likelihood loss with Poisson distribution of target. The mean in each bin is multiplied by a weight before the Poisson likelihood is applied.

    \f[
        -\log\mathcal{L}_\text{poisson}(y, w \cdot \hat{y}) = \hat{y} - y \cdot \log(\hat{y}) + \ln(|\Gamma(y)|)
    \f] where \f$\Gamma(y)\f$ is returned by <a href="https://jfalcou.github.io/eve/group__special_gae09a3d5ef50adfebd1d42611611cae5a.html#gae09a3d5ef50adfebd1d42611611cae5a">eve::gamma_p</a>.
*/
template<std::floating_point T, std::input_iterator I, std::input_iterator J, std::input_iterator K>
inline auto poisson_neg_likelihood_loss(I first1, std::sentinel_for<I> auto last1, J first2, K first3) noexcept -> double {
    using wide = eve::wide<T>;
    auto constexpr s{ wide::size() };
    auto const n{ std::distance(first1, last1) };
    auto const m{ n - n % s };

    auto constexpr eps{ std::numeric_limits<T>::epsilon() };

    univariate_accumulator<wide> we;
    for (auto i = 0; i < m; i += s) {
        wide y_true{first1, first1+s};
        wide y_pred = eve::mul(wide{first2, first2+s}, wide{first3, first3+s});
        we(y_pred - y_true * eve::log(y_pred) + eve::log_abs_gamma(T{1} + y_true));
        detail::advance(s, first1, first2, first3);
    }

    // use scalar accumulators for the remaining values
    auto se = univariate_accumulator<T>::load_state(we.stats());
    for(; first1 < last1; ++first1, ++first2, ++first3) {
        se(*first2 * *first3 - *first1 * eve::log(*first2 * *first3) + eve::log_abs_gamma(T{1} + *first1));
    }
    return univariate_statistics(se).sum;
}
} // namespace metrics

} // namespace VSTAT_NAMESPACE

#endif
