// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2024 Heal Research

#ifndef VSTAT_UNIVARIATE_HPP
#define VSTAT_UNIVARIATE_HPP

#include "combine.hpp"

namespace VSTAT_NAMESPACE
{

/*!
    \brief Controls which statistics the accumulator computes.

    Each level implies all levels below it (variance ⊃ mean ⊃ sum).
    Requesting a lower level skips the unnecessary work:
      - sum:      only tracks the weighted sum (no count, no Welford SSR)
      - mean:     tracks sum + count (no Welford SSR)
      - variance: full Welford update — sum, count, SSR
*/
enum class stats { sum, mean, variance };

/*!
    \brief Univariate accumulator object.

    \tparam T     Scalar or eve::wide floating-point type.
    \tparam Stats Which statistics to compute (default: variance).
                  variance ⊃ mean ⊃ sum — requesting a lower level skips the
                  Welford SSR update and, for stats::sum, the weight counter.
*/
template<typename T, stats Stats = stats::variance>
struct univariate_accumulator
{
    static auto load_state(T sw, T sx, T sxx) noexcept -> univariate_accumulator
    {
        univariate_accumulator acc;
        acc.sum_w = sw;
        acc.sum_w_old = sw;
        acc.sum_x = sx;
        acc.sum_xx = sxx;
        return acc;
    }

    static auto load_state(std::tuple<T, T, T> state) noexcept -> univariate_accumulator
    {
        auto [sw, sx, sxx] = state;
        return load_state(sw, sx, sxx);
    }

    void operator()(T x) noexcept
    {
        if constexpr (Stats == stats::variance) {
            T dx = (sum_w * x) - sum_x;
            sum_x += x;
            sum_w += 1;
            sum_xx += (dx * dx) / (sum_w * sum_w_old);
            sum_w_old = sum_w;
        } else if constexpr (Stats == stats::mean) {
            sum_x += x;
            sum_w += 1;
        } else {
            sum_x += x;
        }
    }

    void operator()(T x, T w) noexcept
    {
        if constexpr (Stats == stats::variance) {
            x *= w;
            T dx = (sum_w * x) - (sum_x * w);
            sum_x += x;
            sum_w += w;
            sum_xx += (dx * dx) / (w * sum_w * sum_w_old);
            sum_w_old = sum_w;
        } else if constexpr (Stats == stats::mean) {
            sum_x += x * w;
            sum_w += w;
        } else {
            sum_x += x * w;
        }
    }

    template<typename U>
        requires eve::simd_value<T> && eve::simd_compatible_ptr<U, T>
    void operator()(U const* x) noexcept
    {
        (*this)(T {x});
    }

    template<typename U>
        requires eve::simd_value<T> && eve::simd_compatible_ptr<U, T>
    void operator()(U const* x, U const* w) noexcept
    {
        (*this)(T {x}, T {w});
    }

    // Returns { sum_w, sum_x, sum_xx }.
    // Fields not tracked by Stats are 0: sum_w when Stats==sum, sum_xx when Stats!=variance.
    [[nodiscard]] auto stats() const noexcept -> std::tuple<double, double, double>
    {
        if constexpr (std::is_floating_point_v<T>) {
            return {sum_w, sum_x, sum_xx};
        } else if constexpr (Stats == stats::variance) {
            return {eve::reduce(sum_w), eve::reduce(sum_x), combine(sum_w, sum_x, sum_xx)};
        } else {
            return {eve::reduce(sum_w), eve::reduce(sum_x), 0.0};
        }
    }

  private:
    T sum_w {0};
    T sum_w_old {1};
    T sum_x {0};
    T sum_xx {0};
};

/*!
    \brief Univariate statistics
*/
struct univariate_statistics
{
    double count;
    double sum;
    double ssr;
    double mean;
    double variance;
    double sample_variance;

    template<typename T, stats Stats>
    explicit univariate_statistics(univariate_accumulator<T, Stats> const& accumulator)
    {
        auto [sw, sx, sxx] = accumulator.stats();
        count = sw;
        sum = sx;
        ssr = sxx;
        if constexpr (Stats != stats::sum) {
            mean = sx / sw;
        } else {
            mean = std::numeric_limits<double>::quiet_NaN();
        }
        if constexpr (Stats == stats::variance) {
            variance = sxx / sw;
            sample_variance = sxx / (sw - 1);
        } else {
            variance = std::numeric_limits<double>::quiet_NaN();
            sample_variance = std::numeric_limits<double>::quiet_NaN();
        }
    }
};

inline auto operator<<(std::ostream& os, univariate_statistics const& stats) -> std::ostream&
{
    os << "count:          \t" << stats.count << "\nsum:            \t" << stats.sum << "\nssr:            \t"
       << stats.ssr << "\nmean:           \t" << stats.mean << "\nvariance:       \t" << stats.variance
       << "\nsample variance:\t" << stats.sample_variance << "\n";
    return os;
}

}  // namespace VSTAT_NAMESPACE

#endif
