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
        , _sum_we(0)
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
        return _sum_we;
    }

    double MeanX() const noexcept {
        return _sum_x / _sum_we;
    }

    double MeanY() const noexcept {
        return _sum_y / _sum_we;
    }

    double NaiveVarianceX() const noexcept {
        return _sum_xx / _sum_we;
    }

    double NaiveVarianceY() const noexcept {
        return _sum_yy / _sum_we;
    }

    double NaiveCovariance() const noexcept {
        return _sum_xy / _sum_we;
    }

    double SampleVarianceX() const noexcept {
        // EXPECT(_sum_we > 1.);
        return _sum_xx / (_sum_we - 1);
    }

    double SampleVarianceY() const noexcept {
        EXPECT(_sum_we > 1.);
        return _sum_yy / (_sum_we - 1);
    }

    double SampleCovariance() const noexcept {
        EXPECT(_sum_we > 1.);
        return _sum_xx / (_sum_we - 1);
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

    double SumWe() const noexcept { return _sum_we; }
    double SumX() const noexcept { return _sum_x; }
    double SumY() const noexcept { return _sum_y; }
    double SumXX() const noexcept { return _sum_xx; }
    double SumYY() const noexcept { return _sum_yy; }
    double SumXY() const noexcept { return _sum_xy; }

    template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void Add(T const x, T const y) noexcept
    {
        if (_sum_we <= 0.) {
            _sum_x = x;
            _sum_y = y;
            _sum_we = 1;
            return;
        }
        // Delta to previous mean
        double dx = x * _sum_we - _sum_x;
        double dy = y * _sum_we - _sum_y;
        double old_we = _sum_we;
        // Incremental update
        _sum_we += 1;
        double f = 1. / (_sum_we * old_we);
        // Update
        _sum_xx += f * dx * dx;
        _sum_yy += f * dy * dy;
        // should equal weight * deltaY * neltaX!
        _sum_xy += f * dx * dy;
        // Update means
        _sum_x += x;
        _sum_y += y;
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
        vec sum_xx(0.0);
        vec sum_yy(0.0);
        vec sum_xy(0.0);
        vec sum_we(1.0);

        const auto s = vec::size(), m = n & (-s);
        for (size_t i = s; i < m; i += s) {
            vec xx = vec().load(x + i);
            vec yy = vec().load(y + i);

            vec dx = xx * sum_we - sum_x;
            vec dy = yy * sum_we - sum_y;

            sum_we += 1;
            vec f = 1. / (sum_we * (sum_we - 1));

            sum_x += xx;
            sum_y += yy;

            sum_xx += f * dx * dx;
            sum_yy += f * dy * dy;
            sum_xy += f * dx * dy;
        }

        _sum_we = horizontal_add(sum_we);
        _sum_x  = horizontal_add(sum_x);
        _sum_y  = horizontal_add(sum_y);

        auto [sxx, syy, sxy] = combine(sum_we, sum_x, sum_y, sum_xx, sum_yy, sum_xy);
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
    double _sum_we;

    // squared residuals
    double _sum_xx;
    double _sum_yy;
    double _sum_xy;
};

#if defined(VSTAT_NAMESPACE)
} // end namespace
#endif

#endif
