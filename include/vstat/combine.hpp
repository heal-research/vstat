// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

#ifndef VSTAT_COMBINE_HPP
#define VSTAT_COMBINE_HPP

#include <array>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <tuple>
#include <eve/wide.hpp>
#include <eve/module/core.hpp>

#include "util.hpp"

namespace VSTAT_NAMESPACE {

namespace detail {
    // utility
    template<typename T>
    auto square(T a)
    {
        return a * a;
    }

    template<typename T>
    requires eve::simd_value<T>
    auto unpack(T v) -> auto
    {
        return [&]<std::size_t ...I>(std::index_sequence<I...>){
            return std::array{ v.get(I) ... };
        }(std::make_index_sequence<T::size()>{});
    }
} // namespace detail

// The code below is based on:
// Schubert et al. - Numerically Stable Parallel Computation of (Co-)Variance, p. 4, eq. 22-26
// https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
// merge covariance from individual data partitions A,B
template<typename T>
requires eve::simd_value<T> && (T::size() >= 2)
inline auto combine(T sum_w, T sum_x, T sum_xx) -> double
{
    if constexpr (T::size() == 2) {
        auto [n0, n1] = detail::unpack(sum_w);
        auto [s0, s1] = detail::unpack(sum_x);
        double f = 1. / (n0 * n1 * (n0 + n1));
        return eve::reduce(sum_xx) + f * detail::square(n1 * s0 - n0 * s1); // eq. 22
    } else {
        auto [sum_w0, sum_w1] = sum_w.slice(); 
        auto [sum_x0, sum_x1] = sum_x.slice();
        auto [sum_xx0, sum_xx1] = sum_xx.slice();

        double n0 = eve::reduce(sum_w0);
        double n1 = eve::reduce(sum_w1);

        double s0 = eve::reduce(sum_x0);
        double s1 = eve::reduce(sum_x1);

        double q0 = combine(sum_w0, sum_x0, sum_xx0);
        double q1 = combine(sum_w1, sum_x1, sum_xx1);

        double f   = 1. / (n0 * n1 * (n0 + n1));
        return q0 + q1 + f * detail::square(n1 * s0 - n0 * s1); // eq. 22
    }
}

template<typename T>
requires eve::simd_value<T> && (T::size() >= 2)
inline auto combine(T sum_w, T sum_x, T sum_y, T sum_xx, T sum_yy, T sum_xy) -> std::tuple<double, double, double> // NOLINT
{
    if constexpr (T::size() == 2) {
        auto [n0, n1] = detail::unpack(sum_w);
        auto [sx0, sx1] = detail::unpack(sum_x);
        auto [sy0, sy1] = detail::unpack(sum_y);
        auto [sxx0, sxx1] = detail::unpack(sum_xx);
        auto [syy0, syy1] = detail::unpack(sum_yy);
        auto [sxy0, sxy1] = detail::unpack(sum_xy);

        double f = 1. / (n0 * n1 * (n0 + n1));
        double sx = n1 * sx0 - n0 * sx1;
        double sy = n1 * sy0 - n0 * sy1;
        double sxx = sxx0 + sxx1 + f * sx * sx;
        double syy = syy0 + syy1 + f * sy * sy;
        double sxy = sxy0 + sxy1 + f * sx * sy;

        return { sxx, syy, sxy };
    } else {
        auto [sum_w0, sum_w1]   = sum_w.slice();
        auto [sum_x0, sum_x1]   = sum_x.slice();
        auto [sum_y0, sum_y1]   = sum_y.slice();
        auto [sum_xx0, sum_xx1] = sum_xx.slice();
        auto [sum_yy0, sum_yy1] = sum_yy.slice();
        auto [sum_xy0, sum_xy1] = sum_xy.slice();

        auto [sxx0, syy0, sxy0] = combine(sum_w0, sum_x0, sum_y0, sum_xx0, sum_yy0, sum_xy0);
        auto [sxx1, syy1, sxy1] = combine(sum_w1, sum_x1, sum_y1, sum_xx1, sum_yy1, sum_xy1);

        double n0  = eve::reduce(sum_w0);
        double n1  = eve::reduce(sum_w1);
        double sx0 = eve::reduce(sum_x0);
        double sx1 = eve::reduce(sum_x1);
        double sy0 = eve::reduce(sum_y0);
        double sy1 = eve::reduce(sum_y1);

        double f   = 1. / (n0 * n1 * (n0 + n1));
        double sx = n1 * sx0 - n0 * sx1;
        double sy = n1 * sy0 - n0 * sy1;
        double sxx = sxx0 + sxx1 + f * sx * sx;
        double syy = syy0 + syy1 + f * sy * sy;
        double sxy = sxy0 + sxy1 + f * sx * sy;

        return { sxx, syy, sxy };
    }
}
} // namespace VSTAT_NAMESPACE

#endif
