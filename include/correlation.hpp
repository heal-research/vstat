#ifndef VSTAT_CORRELATION_HPP
#define VSTAT_CORRELATION_HPP

#include "combine.hpp"

#if defined(VSTAT_NAMESPACE)
namespace VSTAT_NAMESPACE {
#endif

class CorrelationCalculator {
public:
    CorrelationCalculator()
        : _sum_x(0)
        , _sum_y(0)
        , _sum_w(0)
        , _sum_xx(0)
        , _sum_yy(0)
        , _sum_xy(0)
    {
    }

    double Correlation() const noexcept {
        if (!(_sum_xx > 0. && _sum_yy > 0.)) {
            return (_sum_xx == _sum_yy) ? 1. : 0.;
        }
        return _sum_xy / std::sqrt(_sum_xx * _sum_yy);
    }

    double Count() const noexcept {
        return _sum_w;
    }

    double MeanX() const noexcept {
        return _sum_x / _sum_w;
    }

    double MeanY() const noexcept {
        return _sum_y / _sum_w;
    }

    double NaiveVarianceX() const noexcept {
        return _sum_xx / _sum_w;
    }

    double NaiveVarianceY() const noexcept {
        return _sum_yy / _sum_w;
    }

    double NaiveCovariance() const noexcept {
        return _sum_xy / _sum_w;
    }

    double SampleVarianceX() const noexcept {
        // EXPECT(_sum_w > 1.);
        return _sum_xx / (_sum_w - 1);
    }

    double SampleVarianceY() const noexcept {
        EXPECT(_sum_w > 1.);
        return _sum_yy / (_sum_w - 1);
    }

    double SampleCovariance() const noexcept {
        EXPECT(_sum_w > 1.);
        return _sum_xx / (_sum_w - 1);
    }

    double NaiveStdX() const noexcept {
        return std::sqrt(NaiveVarianceX());
    }

    double SampleStdX() const noexcept {
        return std::sqrt(SampleVarianceX());
    }

    double NaiveStdY() const noexcept {
        return std::sqrt(NaiveVarianceY());
    }

    double SampleStdY() const noexcept {
        return std::sqrt(SampleVarianceY());
    }

    double SumWe() const noexcept { return _sum_w; }
    double SumX() const noexcept { return _sum_x; }
    double SumY() const noexcept { return _sum_y; }
    double SumXX() const noexcept { return _sum_xx; }
    double SumYY() const noexcept { return _sum_yy; }
    double SumXY() const noexcept { return _sum_xy; }

    template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void Add(T const x, T const y) noexcept
    {
        if (_sum_w <= 0.) {
            _sum_x = x;
            _sum_y = y;
            _sum_w = 1;
            return;
        }
        // Delta to previous mean
        double dx = x * _sum_w - _sum_x;
        double dy = y * _sum_w - _sum_y;
        double old_we = _sum_w;

        _sum_x += x;
        _sum_y += y;
        _sum_w += 1;

        double f = 1. / (_sum_w * old_we);
        _sum_xx += f * dx * dx;
        _sum_yy += f * dy * dy;
        _sum_xy += f * dx * dy;
    }

    template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void Add(T const x, T const y, T const w) noexcept
    {
        if (w == 0) {
            return;
        }

        if (_sum_w <= 0.) {
            _sum_x = x;
            _sum_y = y;
            _sum_w = w;
            return;
        }

        x *= w;
        y *= w;

        // Delta to previous mean
        double dx = x * _sum_w - _sum_x * w;
        double dy = y * _sum_w - _sum_y * w;
        double old_we = _sum_w;

        _sum_x += x;
        _sum_y += y;
        _sum_w += w;

        double f = 1. / (w * _sum_w * old_we);
        _sum_xx += f * dx * dx;
        _sum_yy += f * dy * dy;
        _sum_xy += f * dx * dy;
    }

    template<typename T, std::enable_if_t<std::is_floating_point_v<T> && (sizeof(T) == 4 || sizeof(T) == 8), bool> = true>
    void Add(T const* x, T const* y, size_t n) noexcept
    {
        using vec = std::conditional_t<std::is_same_v<T, double>, Vec4d, Vec8f>;
        if (n < vec::size()) {
            for (size_t i = 0; i < n; ++i) {
                Add(*(x + i), *(y + i));
            }
            return;
        }

        vec sum_x = vec().load(x);
        vec sum_y = vec().load(y);
        vec sum_w = vec(1.0);

        vec sum_xx(0.0);
        vec sum_yy(0.0);
        vec sum_xy(0.0);

        const auto s = vec::size(), m = n & (-s);
        for (size_t i = s; i < m; i += s) {
            vec xx = vec().load(x + i);
            vec yy = vec().load(y + i);

            vec dx = xx * sum_w - sum_x;
            vec dy = yy * sum_w - sum_y;

            sum_w += 1;
            vec f = 1. / (sum_w * (sum_w - 1));

            sum_x += xx;
            sum_y += yy;

            sum_xx += f * dx * dx;
            sum_yy += f * dy * dy;
            sum_xy += f * dx * dy;
        }

        _sum_w = horizontal_add(sum_w);
        _sum_x  = horizontal_add(sum_x);
        _sum_y  = horizontal_add(sum_y);

        auto [sxx, syy, sxy] = combine(sum_w, sum_x, sum_y, sum_xx, sum_yy, sum_xy);
        _sum_xx = sxx;
        _sum_yy = syy;
        _sum_xy = sxy;

        // deal with remaining values
        if (m < n) {
            Add(x + m, y + m, n - m);
        }
    }

    template<typename T, std::enable_if_t<std::is_floating_point_v<T> && (sizeof(T) == 4 || sizeof(T) == 8), bool> = true>
    void Add(T const* x, T const* y, T const* w, size_t n) noexcept
    {
        using vec = std::conditional_t<std::is_same_v<T, double>, Vec4d, Vec8f>;
        if (n < vec::size()) {
            for (size_t i = 0; i < n; ++i) {
                Add(*(x + i), *(y + i));
            }
            return;
        }

        vec sum_x = vec().load(x);
        vec sum_y = vec().load(y);
        vec sum_w = vec().load(w);

        vec sum_xx(0.0);
        vec sum_yy(0.0);
        vec sum_xy(0.0);

        const auto s = vec::size(), m = n & (-s);
        for (size_t i = s; i < m; i += s) {
            vec ww = vec().load(w + i);
            vec xx = vec().load(x + i) * ww;
            vec yy = vec().load(y + i);

            vec dx = xx * sum_w - sum_x * ww;
            vec dy = yy * sum_w - sum_y * ww;

            sum_w += ww;
            vec f = 1. / (ww * sum_w * (sum_w - 1));

            sum_x += xx;
            sum_y += yy;

            sum_xx += f * dx * dx;
            sum_yy += f * dy * dy;
            sum_xy += f * dx * dy;
        }

        _sum_w = horizontal_add(sum_w);
        _sum_x  = horizontal_add(sum_x);
        _sum_y  = horizontal_add(sum_y);

        auto [sxx, syy, sxy] = combine(sum_w, sum_x, sum_y, sum_xx, sum_yy, sum_xy);
        _sum_xx = sxx;
        _sum_yy = syy;
        _sum_xy = sxy;

        // deal with remaining values
        if (m < n) {
            Add(x + m, y + m, n - m);
        }
    }

private:
    // means
    double _sum_x;
    double _sum_y;

    // sum of weights
    double _sum_w;

    // squared residuals
    double _sum_xx;
    double _sum_yy;
    double _sum_xy;
};

#if defined(VSTAT_NAMESPACE)
} // end namespace
#endif

#endif
