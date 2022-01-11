// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

#ifndef VSTAT_VARIANCE_HPP
#define VSTAT_VARIANCE_HPP

#include "combine.hpp"

#if defined(VSTAT_NAMESPACE)
namespace VSTAT_NAMESPACE {
#endif

template <typename T>
struct univariate_accumulator {
    univariate_accumulator(T x, T w)
        : sum_w(w)
        , sum_x(x * w)
        , sum_xx(T { 0 })
    {
    }

    explicit univariate_accumulator(T x) : univariate_accumulator(x, T{1.0})
    {
    }

    static auto load_state(T sw, T sx, T sxx) noexcept -> univariate_accumulator<T>
    {
        univariate_accumulator<T> acc(T { 0 }, T { 0 });
        acc.sum_w = sw;
        acc.sum_x = sx;
        acc.sum_xx = sxx;
        return acc;
    }

    inline void operator()(T x) noexcept
    {
        T dx = sum_w * x - sum_x;
        sum_x += x;
        auto sum_w_old = sum_w;
        sum_w += 1;
        sum_xx += dx * dx / (sum_w * sum_w_old);
    }

    inline void operator()(T x, T w) noexcept
    {
        x *= w;
        T dx = sum_w * x - sum_x * w;
        auto sum_w_old = sum_w;
        sum_w += w;
        sum_x += x;
        sum_xx += dx * dx / (w * sum_w * sum_w_old);
    }

    template <typename U, std::enable_if_t<detail::is_any_v<T, Vec4d, Vec4f, Vec8f> && std::is_floating_point_v<U>, bool> = true>
    inline void operator()(U const* x) noexcept
    {
        static_assert(sizeof(U) == T::size());
        (*this)(T().load(x));
    }

    template <typename U, std::enable_if_t<detail::is_any_v<T, Vec4d, Vec4f, Vec8f> && std::is_floating_point_v<U>, bool> = true>
    inline void operator()(U const* x, U const* w) noexcept
    {
        static_assert(sizeof(U) == T::size());
        (*this)(T().load(x), T().load(w));
    }

    // performs the reductions and returns { sum_w, sum_x, sum_xx }
    [[nodiscard]] auto stats() const noexcept -> std::tuple<double, double, double>
    {
        if constexpr (std::is_floating_point_v<T>) {
            return { sum_w, sum_x, sum_xx };
        } else {
            return { horizontal_add(sum_w), horizontal_add(sum_x), combine(sum_w, sum_x, sum_xx) };
        }
    }

private:
    T sum_w;
    T sum_x;
    T sum_xx;
};

struct univariate_statistics {
    double count;
    double sum;
    double ssr;
    double mean;
    double variance;
    double sample_variance;

    template <typename T>
    explicit univariate_statistics(T const& accumulator)
    {
        auto [sw, sx, sxx] = accumulator.stats();
        count = sw;
        sum = sx;
        ssr = sxx;
        mean = sx / sw;
        variance = sxx / sw;
        sample_variance = sxx / (sw - 1);
    }
};

inline auto operator<<(std::ostream& os, univariate_statistics const& stats) -> std::ostream&
{
    os << "count:          \t" << stats.count
       << "\nsum:            \t" << stats.sum
       << "\nssr:            \t" << stats.ssr
       << "\nmean:           \t" << stats.mean
       << "\nvariance:       \t" << stats.variance
       << "\nsample variance:\t" << stats.sample_variance
       << "\n";
    return os;
}

#if defined(VSTAT_NAMESPACE)
} // end namespace
#endif

#endif
