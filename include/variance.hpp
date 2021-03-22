#ifndef VSTAT_VARIANCE_HPP
#define VSTAT_VARIANCE_HPP

#include "combine.hpp"

#if defined(VSTAT_NAMESPACE)
namespace VSTAT_NAMESPACE {
#endif

class VarianceCalculator {
public:
    VarianceCalculator()
        : _sum_x(0)
        , _sum_xx(0)
        , _sum_w(0)
    {}

    void Reset()
    {
        _sum_x = 0;
        _sum_xx = 0;
        _sum_w = 0;
    }

    double NaiveVariance() const noexcept
    {
        EXPECT(_sum_w > 1);
        return _sum_xx / (_sum_w - 1);
    }

    double SampleVariance() const noexcept
    {
        EXPECT(_sum_w > 0)
        return _sum_xx / _sum_w;
    }

    double NaiveStd() const noexcept
    {
        return std::sqrt(NaiveVariance());
    }

    double SampleStd() const noexcept
    {
        return std::sqrt(SampleVariance());
    }

    double SumXX() const noexcept
    {
        return _sum_xx;
    }

    double Count() const noexcept
    {
        return _sum_w;
    }

    double Mean() const noexcept
    {
        return _sum_x / _sum_w;
    }

    // add a single sample
    template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void Add(T x)
    {
        if (_sum_w <= 0) {
            _sum_w = 1;
            _sum_x = x;
            _sum_xx = 0;
            return;
        }

        double d = _sum_w * x - _sum_x;
        _sum_w += 1;
        _sum_x += x;
        _sum_xx += d * d / (_sum_w * (_sum_w - 1));
    }

    // add a single weighted sample
    template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void Add(T x, T w)
    {
        if (w == 0) {
            return;
        }

        if (_sum_w <= 0)
        {
            _sum_w = 1;
            _sum_x = x * w;
            _sum_xx = 0;
            return;
        }

        x *= w;
        double d = _sum_w * x - _sum_x * w;
        _sum_w += w;
        _sum_x += x;
        _sum_xx += d * d / (w * _sum_w * (_sum_w - 1));
    }

    template<typename T, std::enable_if_t<std::is_floating_point_v<T> && (sizeof(T) == 4 || sizeof(T) == 8), bool> = true>
    void Add(T const* x, size_t n)
    {
        using vec = std::conditional_t<std::is_same_v<T, double>, Vec4d, Vec8f>;
        if (n < vec::size()) {
            for (size_t i = 0; i < n; ++i) {
                Add(*(x + i));
            }
            return;
        }

        vec sum_x = vec().load(x);
        vec sum_xx(0.0);
        vec sum_w(1.0);

        const auto s = vec::size(), m = n & (-s);
        for (size_t i = s; i < m; i += s) {
            vec xx = vec().load(x + i);
            vec dx = sum_w * xx - sum_x;
            sum_w += 1;
            sum_x += xx;
            sum_xx += dx * dx / (sum_w * (sum_w - 1));
        }

        _sum_w = horizontal_add(sum_w);
        _sum_x = horizontal_add(sum_x);
        _sum_xx = combine(sum_w, sum_x, sum_xx);

        // deal with remaining values
        if (m < n) {
            Add(x + m, n - m);
        }
    }

    template<typename T, std::enable_if_t<std::is_floating_point_v<T> && (sizeof(T) == 4 || sizeof(T) == 8), bool> = true>
    void Add(T const* x, T const* w, size_t n)
    {
        using vec = std::conditional_t<std::is_same_v<T, double>, Vec4d, Vec8f>;
        if (n < vec::size()) {
            for (size_t i = 0; i < n; ++i) {
                Add(*(x + i));
            }
            return;
        }

        vec sum_x = vec().load(x);
        vec sum_xx(0.0);
        vec sum_w = vec().load(w);

        const auto s = vec::size(), m = n & (-s);
        for (size_t i = s; i < m; i += s) {
            vec xx = vec().load(x + i);
            vec ww = vec().load(w + i);
            xx *= ww;
            vec dx = sum_w * xx - sum_x * ww;
            sum_w += ww;
            sum_x += xx;
            sum_xx += dx * dx / (ww * sum_w * (sum_w - 1));
        }

        _sum_w = horizontal_add(sum_w);
        _sum_x = horizontal_add(sum_x);
        _sum_xx = combine(sum_w, sum_x, sum_xx);

        // deal with remaining values
        if (m < n) {
            Add(x + m, w + m, n - m);
        }
    }

private:
    double _sum_x; // sum
    double _sum_xx; // sum of squares
    double _sum_w; // number of elements
};

#if defined(VSTAT_NAMESPACE)
} // end namespace
#endif

#endif
