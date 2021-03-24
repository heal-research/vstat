#ifndef VSTAT_CORRELATION_HPP
#define VSTAT_CORRELATION_HPP

#include "combine.hpp"
#include <type_traits>

#if defined(VSTAT_NAMESPACE)
namespace VSTAT_NAMESPACE {
#endif

template<typename T, std::enable_if_t<std::is_same_v<T, Vec8f> || std::is_same_v<T, Vec4d>, bool> = true>
struct CorrelationAccumulator {
    CorrelationAccumulator(T x, T y, T w) : sum_x(x), sum_y(y), sum_w(w), sum_xx(0.0), sum_yy(0.0), sum_xy(0.0) { }
    CorrelationAccumulator(T x, T y) : CorrelationAccumulator(x, y, T(1.0)) { }
    CorrelationAccumulator() : CorrelationAccumulator(T(0.0), T(0.0), T(0.0)) { }

    void Reset()
    {
        sum_w = sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0.0;
    }

    inline void operator()(T x, T y)
    {
        T dx = x * sum_w - sum_x;
        T dy = y * sum_w - sum_y;

        sum_x += x;
        sum_y += y;
        sum_w += 1;

        T f = 1. / (sum_w * (sum_w - 1));
        sum_xx += f * dx * dx;
        sum_yy += f * dy * dy;
        sum_xy += f * dx * dy;
    }

    inline void operator()(T x, T y, T w)
    {
        T dx = x * sum_w - sum_x;
        T dy = y * sum_w - sum_y;

        sum_x += x * w;
        sum_y += y * w;
        sum_w += w;

        T f = w / (sum_w * (sum_w - w));

        sum_xx += f * dx * dx;
        sum_yy += f * dy * dy;
        sum_xy += f * dx * dy;
    }

    template<typename U, std::enable_if_t<std::is_floating_point_v<U> && sizeof(U) == T::size(), bool> = true>
    inline void operator()(U const* x, U const* y)
    {
        (*this)(T().load(x), T().load(y));
    }

    template<typename U, std::enable_if_t<std::is_floating_point_v<U> && sizeof(U) == T::size(), bool> = true>
    inline void operator()(U const* x, U const* y, U const* w)
    {
        (*this)(T().load(x), T().load(y), T().load(w));
    }

    // performs a reduction on the vector types and returns the sums and the squared residuals sums
    std::tuple<double, double, double, double, double, double> Stats()
    {
        double sw = horizontal_add(sum_w);
        double sx = horizontal_add(sum_x);
        double sy = horizontal_add(sum_y);

        auto [sxx, syy, sxy] = combine(sum_w, sum_x, sum_y, sum_xx, sum_yy, sum_xy);
        return { sw, sx, sy, sxx, syy, sxy };
    }

    // sum of weights
    T sum_w;
    // means
    T sum_x;
    T sum_y;
    // squared residuals
    T sum_xx;
    T sum_yy;
    T sum_xy;
};

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
            _sum_x = x * w;
            _sum_y = y * w;
            _sum_w = w;
            return;
        }

        double dx = x * _sum_w - _sum_x;
        double dy = y * _sum_w - _sum_y;
        double old_we = _sum_w;

        _sum_x += x * w;
        _sum_y += y * w;
        _sum_w += w;

        double f = w / (_sum_w * old_we);
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

        CorrelationAccumulator<vec> acc(vec().load(x), vec().load(y));

        const size_t s = vec::size(), m = n & (-s);
        for (size_t i = s; i < m; i += s) {
            acc(vec().load(x + i), vec().load(y + i));
        }

        auto [sw, sx, sy, sxx, syy, sxy] = acc.Stats();
        _sum_w = sw;
        _sum_x = sx;
        _sum_y = sy;
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
                Add(*(x + i), *(y + i), *(w + i));
            }
            return;
        }

        CorrelationAccumulator<vec> acc(vec().load(x), vec().load(y), vec().load(w));

        const size_t s = vec::size(), m = n & (-s);
        for (size_t i = s; i < m; i += s) {
            acc(vec().load(x + i), vec().load(y + i), vec().load(w + i));
        }

        auto [sw, sx, sy, sxx, syy, sxy] = acc.Stats();
        _sum_w = sw;
        _sum_x = sx;
        _sum_y = sy;
        _sum_xx = sxx;
        _sum_yy = syy;
        _sum_xy = sxy;

        // deal with remaining values
        if (m < n) {
            Add(x + m, y + m, w + m, n - m);
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
