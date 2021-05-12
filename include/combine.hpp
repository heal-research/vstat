// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

#ifndef VSTAT_COMBINE_HPP
#define VSTAT_COMBINE_HPP

#include <cstddef>
#include <cmath>
#include <iostream>
#include <tuple>

#include "vectorclass.h"
#include "util.hpp"

namespace {
    // utility
    template<typename T>
    auto square(T a) {
        return a * a;
    }

    std::tuple<double, double, double, double> unpack(Vec4d v)
    {
        double x[4];
        v.store(x);
        return { x[0], x[1], x[2], x[3] };
    }

    std::tuple<float, float, float, float> unpack(Vec4f v)
    {
        float x[4];
        v.store(x);
        return { x[0], x[1], x[2], x[3] };
    }

    std::tuple<Vec4f, Vec4f> split(Vec8f v)
    {
        return { v.get_low(), v.get_high() };
    }
}

// The code below is based on:
// Schubert et al. - Numerically Stable Parallel Computation of (Co-)Variance, p. 4, eq. 22-26
// https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
// merge covariance from individual data partitions A,B

inline double combine(Vec4d sum_w, Vec4d sum_x, Vec4d sum_xx)
{
    auto [n0, n1, n2, n3] = unpack(sum_w);
    auto [s0, s1, s2, s3] = unpack(sum_x);
    auto [q0, q1, q2, q3] = unpack(sum_xx);

    double n01 = n0 + n1, s01 = s0 + s1;
    double n23 = n2 + n3, s23 = s2 + s3;

    double f01 = 1. / (n0 * n01 * n1);
    double f23 = 1. / (n2 * n23 * n3);
    double f = 1. / (n01 * (n01 + n23) * n23);

    double q01 = q0 + q1 + f01 * square(n0 * s1 - n1 * s0);
    double q23 = q2 + q3 + f23 * square(n2 * s3 - n3 * s2);
    return q01 + q23 + f * square(n01 * s23 - n23 * s01);
}

inline double combine(Vec4f sum_w, Vec4f sum_x, Vec4f sum_xx)
{
    return combine(to_double(sum_w), to_double(sum_x), to_double(sum_xx));
}

inline double combine(Vec8f sum_w, Vec8f sum_x, Vec8f sum_xx)
{
    auto [sum_w0, sum_w1] = split(sum_w);
    auto [sum_x0, sum_x1] = split(sum_x);
    auto [sum_xx0, sum_xx1] = split(sum_xx);

    auto s0 = horizontal_add(sum_x0);
    auto s1 = horizontal_add(sum_x1);

    auto q0 = combine(sum_w0, sum_x0, sum_xx0);
    auto q1 = combine(sum_w1, sum_x1, sum_xx1);

    auto n0 = horizontal_add(sum_w0);
    auto n1 = horizontal_add(sum_w1);

    double f   = 1. / (n0 * n1 * (n0 + n1));
    return q0 + q1 + f * square(n1 * s0 - n0 * s1);
}

// combines four partitions into a single result
inline std::tuple<double, double, double>
combine(Vec4d sum_w, Vec4d sum_x, Vec4d sum_y, Vec4d sum_xx, Vec4d sum_yy, Vec4d sum_xy)
{
    auto [n0, n1, n2, n3] = unpack(sum_w);
    auto [sx0, sx1, sx2, sx3] = unpack(sum_x);
    auto [sy0, sy1, sy2, sy3] = unpack(sum_y);
    auto [sxx0, sxx1, sxx2, sxx3] = unpack(sum_xx);
    auto [syy0, syy1, syy2, syy3] = unpack(sum_yy);
    auto [sxy0, sxy1, sxy2, sxy3] = unpack(sum_xy);

    double n01 = n0 + n1, sx01 = sx0 + sx1, sy01 = sy0 + sy1;
    double n23 = n2 + n3, sx23 = sx2 + sx3, sy23 = sy2 + sy3;

    double f01 = 1. / (n0 * n01 * n1);
    double f23 = 1. / (n2 * n23 * n3);
    double f = 1. / (n01 * (n01 + n23) * n23);

    // X
    double qx01 = sxx0 + sxx1 + f01 * square(n0 * sx1 - n1 * sx0);
    double qx23 = sxx2 + sxx3 + f23 * square(n2 * sx3 - n3 * sx2);
    double sxx = qx01 + qx23 + f * square(n01 * sx23 - n23 * sx01);
    // Y
    double qy01 = syy0 + syy1 + f01 * square(n0 * sy1 - n1 * sy0);
    double qy23 = syy2 + syy3 + f23 * square(n2 * sy3 - n3 * sy2);
    double syy = qy01 + qy23 + f * square(n01 * sy23 - n23 * sy01);
    // XY
    double q01 = sxy0 + sxy1 + f01 * (n1 * sx0 - n0 * sx1) * (n1 * sy0 - n0 * sy1);
    double q23 = sxy2 + sxy3 + f23 * (n3 * sx2 - n2 * sx3) * (n3 * sy2 - n2 * sy3);
    double sxy = q01 + q23 + f * (n23 * sx01 - n01 * sx23) * (n23 * sy01 - n01 * sy23);

    return { sxx, syy, sxy };
}
std::tuple<double, double, double>
inline combine(Vec4f sum_w, Vec4f sum_x, Vec4f sum_y, Vec4f sum_xx, Vec4f sum_yy, Vec4f sum_xy)
{
    return combine(to_double(sum_w), to_double(sum_x), to_double(sum_y), to_double(sum_xx), to_double(sum_yy), to_double(sum_xy));
}

// combines eight partitions into a single result
inline std::tuple<double, double, double>
combine(Vec8f sum_w, Vec8f sum_x, Vec8f sum_y, Vec8f sum_xx, Vec8f sum_yy, Vec8f sum_xy)
{
    auto [sum_w0, sum_w1]   = split(sum_w);
    auto [sum_x0, sum_x1]   = split(sum_x);
    auto [sum_y0, sum_y1]   = split(sum_y);
    auto [sum_xx0, sum_xx1] = split(sum_xx);
    auto [sum_yy0, sum_yy1] = split(sum_yy);
    auto [sum_xy0, sum_xy1] = split(sum_xy);
    auto [qxx0, qyy0, qxy0] = combine(sum_w0, sum_x0, sum_y0, sum_xx0, sum_yy0, sum_xy0);
    auto [qxx1, qyy1, qxy1] = combine(sum_w1, sum_x1, sum_y1, sum_xx1, sum_yy1, sum_xy1);

    // use to_double for additional precision
    double n0  = horizontal_add(to_double(sum_w0));
    double n1  = horizontal_add(to_double(sum_w1));
    double sx0 = horizontal_add(to_double(sum_x0));
    double sx1 = horizontal_add(to_double(sum_x1));
    double sy0 = horizontal_add(to_double(sum_y0));
    double sy1 = horizontal_add(to_double(sum_y1));

    double f   = 1. / (n0 * n1 * (n0 + n1));
    double sx  = n0 * sx1 - n1 * sx0;
    double sy  = n0 * sy1 - n1 * sy0;

    double sxx = qxx0 + qxx1 + f * sx * sx;
    double syy = qyy0 + qyy1 + f * sy * sy;
    double sxy = qxy0 + qxy1 + f * sx * sy;

    return { sxx, syy, sxy };
}

#if defined(VSTAT_NAMESPACE)
}
#endif

#endif
