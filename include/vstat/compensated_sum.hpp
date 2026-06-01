// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2020-2024 Heal Research

#ifndef VSTAT_COMPENSATED_SUM_HPP
#define VSTAT_COMPENSATED_SUM_HPP

#include <eve/module/core.hpp>

namespace VSTAT_NAMESPACE
{

/*!
    \brief Knuth TwoSum compensated accumulator.

    Captures the exact rounding error of each addition via eve::two_add,
    yielding significantly better precision than naive summation for large
    sequences of double values. Has no meaningful effect for float.

    Works with both scalar floating-point types and eve::wide<T>.

    Usage:
        compensated_sum<double> s;
        for (auto x : data) s(x);
        double result = s.value();
*/
template<typename T>
struct compensated_sum
{
    constexpr void operator()(T x) noexcept
    {
        auto [t, e] = eve::two_add(sum_, x);
        comp_ += e;
        sum_ = t;
    }

    [[nodiscard]] constexpr auto value() const noexcept -> T { return sum_ + comp_; }

  private:
    T sum_ {0};
    T comp_ {0};
};

}  // namespace VSTAT_NAMESPACE

#endif
