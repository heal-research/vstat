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
        return [&]<std::size_t ...i>(std::index_sequence<i...>){
            return std::array{ v.get(i) ... };
        }(std::make_index_sequence<T::size()>{});
    }
} // namespace detail


// The code below is based on:
// Schubert et al. - Numerically Stable Parallel Computation of (Co-)Variance, p. 4, eq. 22-26
// https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
// merge covariance from individual data partitions A,B

// combine 4-way and return the sum
template<typename T>
requires eve::simd_value<T> && (T::size() >= 4)
inline auto combine(T sum_w, T sum_x, T sum_xx) -> double
{
    if constexpr (T::size() == 4) {
        auto [n0, n1, n2, n3] = detail::unpack(sum_w);
        auto [s0, s1, s2, s3] = detail::unpack(sum_x);
        auto [q0, q1, q2, q3] = detail::unpack(sum_xx);

        double n01 = n0 + n1;
        double s01 = s0 + s1;
        double n23 = n2 + n3;
        double s23 = s2 + s3;

        double f01 = 1. / (n0 * n01 * n1);
        double f23 = 1. / (n2 * n23 * n3);
        double f = 1. / (n01 * (n01 + n23) * n23);

        double q01 = q0 + q1 + f01 * detail::square(n0 * s1 - n1 * s0);
        double q23 = q2 + q3 + f23 * detail::square(n2 * s3 - n3 * s2);
        return q01 + q23 + f * detail::square(n01 * s23 - n23 * s01);
    } else {
        auto [sum_w0, sum_w1] = sum_w.slice(); 
        auto [sum_x0, sum_x1] = sum_x.slice();
        auto [sum_xx0, sum_xx1] = sum_xx.slice();

        auto s0 = eve::reduce(sum_x0);
        auto s1 = eve::reduce(sum_x1);

        auto q0 = combine(sum_w0, sum_x0, sum_xx0);
        auto q1 = combine(sum_w1, sum_x1, sum_xx1);

        auto n0 = eve::reduce(sum_w0);
        auto n1 = eve::reduce(sum_w1);

        double f   = 1. / (n0 * n1 * (n0 + n1));
        return q0 + q1 + f * detail::square(n1 * s0 - n0 * s1);
    }
}

// combines four partitions into a single result
template<typename T>
requires eve::simd_value<T> && (T::size() >= 4)
inline auto combine(T sum_w, T sum_x, T sum_y, T sum_xx, T sum_yy, T sum_xy) -> std::tuple<double, double, double> // NOLINT
{
    if constexpr (T::size() == 4) {
        auto [n0, n1, n2, n3] = detail::unpack(sum_w);
        auto [sx0, sx1, sx2, sx3] = detail::unpack(sum_x);
        auto [sy0, sy1, sy2, sy3] = detail::unpack(sum_y);
        auto [sxx0, sxx1, sxx2, sxx3] = detail::unpack(sum_xx);
        auto [syy0, syy1, syy2, syy3] = detail::unpack(sum_yy);
        auto [sxy0, sxy1, sxy2, sxy3] = detail::unpack(sum_xy);

        double n01 = n0 + n1;
        double sx01 = sx0 + sx1;
        double sy01 = sy0 + sy1;

        double n23 = n2 + n3;
        double sx23 = sx2 + sx3;
        double sy23 = sy2 + sy3;

        double f01 = 1. / (n0 * n01 * n1);
        double f23 = 1. / (n2 * n23 * n3);
        double f = 1. / (n01 * (n01 + n23) * n23);

        // X
        double qx01 = sxx0 + sxx1 + f01 * detail::square(n0 * sx1 - n1 * sx0);
        double qx23 = sxx2 + sxx3 + f23 * detail::square(n2 * sx3 - n3 * sx2);
        double sxx = qx01 + qx23 + f * detail::square(n01 * sx23 - n23 * sx01);
        // Y
        double qy01 = syy0 + syy1 + f01 * detail::square(n0 * sy1 - n1 * sy0);
        double qy23 = syy2 + syy3 + f23 * detail::square(n2 * sy3 - n3 * sy2);
        double syy = qy01 + qy23 + f * detail::square(n01 * sy23 - n23 * sy01);
        // XY
        double q01 = sxy0 + sxy1 + f01 * (n1 * sx0 - n0 * sx1) * (n1 * sy0 - n0 * sy1);
        double q23 = sxy2 + sxy3 + f23 * (n3 * sx2 - n2 * sx3) * (n3 * sy2 - n2 * sy3);
        double sxy = q01 + q23 + f * (n23 * sx01 - n01 * sx23) * (n23 * sy01 - n01 * sy23);

        return { sxx, syy, sxy };
    } else {
        auto [sum_w0, sum_w1]   = sum_w.slice();
        auto [sum_x0, sum_x1]   = sum_x.slice();
        auto [sum_y0, sum_y1]   = sum_y.slice();
        auto [sum_xx0, sum_xx1] = sum_xx.slice();
        auto [sum_yy0, sum_yy1] = sum_yy.slice();
        auto [sum_xy0, sum_xy1] = sum_xy.slice();
        auto [qxx0, qyy0, qxy0] = combine(sum_w0, sum_x0, sum_y0, sum_xx0, sum_yy0, sum_xy0);
        auto [qxx1, qyy1, qxy1] = combine(sum_w1, sum_x1, sum_y1, sum_xx1, sum_yy1, sum_xy1);

        double n0  = eve::reduce(sum_w0);
        double n1  = eve::reduce(sum_w1);
        double sx0 = eve::reduce(sum_x0);
        double sx1 = eve::reduce(sum_x1);
        double sy0 = eve::reduce(sum_y0);
        double sy1 = eve::reduce(sum_y1);

        double f   = 1. / (n0 * n1 * (n0 + n1));
        double sx  = n0 * sx1 - n1 * sx0;
        double sy  = n0 * sy1 - n1 * sy0;

        double sxx = qxx0 + qxx1 + f * sx * sx;
        double syy = qyy0 + qyy1 + f * sy * sy;
        double sxy = qxy0 + qxy1 + f * sx * sy;

        return { sxx, syy, sxy };
    }
}
} // namespace VSTAT_NAMESPACE

#endif
