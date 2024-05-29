// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <vstat/vstat.hpp>
#include <span>

namespace detail {
    template<typename T>
    using array = nanobind::ndarray<T>;

    // helpers
    template<typename T>
    inline auto univariate_accumulate(std::span<T const> x) {
        return vstat::univariate::accumulate<float>(x.begin(), x.end());
    }

    template<typename T>
    inline auto univariate_accumulate(std::span<T const> x, std::span<T const> w) {
        return vstat::univariate::accumulate<float>(x.begin(), x.end(), w.begin());
    }

    template<typename T>
    inline auto bivariate_accumulate(std::span<T const> x, std::span<T const> y) {
        return vstat::bivariate::accumulate<float>(x.begin(), x.end(), y.begin());
    }

    template<typename T>
    inline auto bivariate_accumulate(std::span<T const> x, std::span<T const> y, std::span<T const> w) {
        return vstat::bivariate::accumulate<float>(x.begin(), x.end(), y.begin(), w.begin());
    }
} // namespace detail

NB_MODULE(vstat, m) { // NOLINT
    namespace nb = nanobind;

    // objects that hold the statistical results
    nb::class_<vstat::univariate_statistics>(m, "univariate_statistics")
        .def_ro("count", &vstat::univariate_statistics::count)
        .def_ro("sum", &vstat::univariate_statistics::sum)
        .def_ro("ssr", &vstat::univariate_statistics::ssr)
        .def_ro("mean", &vstat::univariate_statistics::mean)
        .def_ro("variance", &vstat::univariate_statistics::variance)
        .def_ro("sample_variance", &vstat::univariate_statistics::sample_variance);

    nb::class_<vstat::bivariate_statistics>(m, "bivariate_statistics")
        .def_ro("count", &vstat::bivariate_statistics::count)
        .def_ro("sum_x", &vstat::bivariate_statistics::sum_x)
        .def_ro("ssr_x", &vstat::bivariate_statistics::ssr_x)
        .def_ro("mean_x", &vstat::bivariate_statistics::mean_x)
        .def_ro("variance_x", &vstat::bivariate_statistics::variance_x)
        .def_ro("sample_variance_x", &vstat::bivariate_statistics::sample_variance_x)
        .def_ro("sum_y", &vstat::bivariate_statistics::sum_y)
        .def_ro("ssr_y", &vstat::bivariate_statistics::ssr_y)
        .def_ro("mean_y", &vstat::bivariate_statistics::mean_y)
        .def_ro("variance_y", &vstat::bivariate_statistics::variance_y)
        .def_ro("sample_variance_y", &vstat::bivariate_statistics::sample_variance_y)
        .def_ro("correlation", &vstat::bivariate_statistics::correlation)
        .def_ro("covariance", &vstat::bivariate_statistics::covariance)
        .def_ro("sample_covariance", &vstat::bivariate_statistics::sample_covariance);

    // single-precision (float)
    // univariate methods
    m.def("univariate_accumulate", [](detail::array<float> x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()});
    });

    m.def("univariate_accumulate", [](std::vector<float> const& x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()});
    });

    m.def("univariate_accumulate", [](detail::array<double> x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()});
    });

    m.def("univariate_accumulate", [](std::vector<double> const& x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()});
    });

    // mean
    m.def("mean", [](detail::array<float> x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}).mean;
    });

    m.def("mean", [](detail::array<float> x, detail::array<float> w) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}, {w.data(), w.size()});
    });

    m.def("mean", [](std::vector<float> const& x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}).mean;
    });

    m.def("mean", [](std::vector<float> const& x, std::vector<float> const& w) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}, {w.data(), w.size()}).mean;
    });

    // variance
    m.def("variance", [](detail::array<float> x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}).variance;
    });

    m.def("variance", [](detail::array<float> x, detail::array<float> w) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}, {w.data(), w.size()});
    });

    m.def("variance", [](std::vector<float> const& x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}).variance;
    });

    m.def("variance", [](std::vector<float> const& x, std::vector<float> const& w) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}, {w.data(), w.size()}).variance;
    });

    // sample variance
    m.def("sample_variance", [](detail::array<float> x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}).sample_variance;
    });

    m.def("sample_variance", [](detail::array<float> x, detail::array<float> w) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}, {w.data(), w.size()});
    });

    m.def("sample_variance", [](std::vector<float> const& x) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}).sample_variance;
    });

    m.def("sample_variance", [](std::vector<float> const& x, std::vector<float> const& w) {
        return detail::univariate_accumulate<float>({x.data(), x.size()}, {w.data(), w.size()}).sample_variance;
    });

    // bivariate methods
    // covariance
    m.def("covariance", [](detail::array<float> x, detail::array<float> y) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}).covariance;
    });

    m.def("covariance", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).covariance;
    });

    m.def("covariance", [](std::vector<float> const& x, std::vector<float> const& y) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}).covariance;
    });

    m.def("covariance", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).covariance;
    });

    m.def("sample_covariance", [](detail::array<float> x, detail::array<float> y) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}).sample_covariance;
    });

    m.def("sample_covariance", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).sample_covariance;
    });

    m.def("sample_covariance", [](std::vector<float> const& x, std::vector<float> const& y) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}).sample_covariance;
    });

    m.def("sample_covariance", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).sample_covariance;
    });

    m.def("correlation", [](detail::array<float> x, detail::array<float> y) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}).correlation;
    });

    m.def("correlation", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).correlation;
    });

    m.def("correlation", [](std::vector<float> const& x, std::vector<float> const& y) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}).correlation;
    });

    m.def("correlation", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        return detail::bivariate_accumulate<float>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).correlation;
    });

    // metrics
    // mean absolute error
    m.def("mean_absolute_error", [](detail::array<float> x, detail::array<float> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_error", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_absolute_error", [](std::vector<float> const& x, std::vector<float> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_error", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // mean absolute percentage error
    m.def("mean_absolute_percentage_error", [](detail::array<float> x, detail::array<float> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_percentage_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_percentage_error", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_percentage_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_absolute_percentage_error", [](std::vector<float> const& x, std::vector<float> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_percentage_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_percentage_error", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_percentage_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // mean squared error
    m.def("mean_squared_error", [](detail::array<float> x, detail::array<float> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_error", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_squared_error", [](std::vector<float> const& x, std::vector<float> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_error", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // mean squared log error
    m.def("mean_squared_log_error", [](detail::array<float> x, detail::array<float> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_log_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_log_error", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_log_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_squared_log_error", [](std::vector<float> const& x, std::vector<float> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_log_error<float>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_log_error", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_log_error<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // r2 score
    m.def("r2_score", [](detail::array<float> x, detail::array<float> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::r2_score<float>(a.begin(), a.end(), b.begin());
    });

    m.def("r2_score", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::r2_score<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("r2_score", [](std::vector<float> const& x, std::vector<float> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::r2_score<float>(a.begin(), a.end(), b.begin());
    });

    m.def("r2_score", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::r2_score<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // poisson loss
    m.def("poisson_neg_likelihood_loss", [](detail::array<float> x, detail::array<float> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<float>(a.begin(), a.end(), b.begin());
    });

    m.def("poisson_neg_likelihood_loss", [](detail::array<float> x, detail::array<float> y, detail::array<float> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("poisson_neg_likelihood_loss", [](std::vector<float> const& x, std::vector<float> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<float>(a.begin(), a.end(), b.begin());
    });

    m.def("poisson_neg_likelihood_loss", [](std::vector<float> const& x, std::vector<float> const& y, std::vector<float> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<float>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // double-precision (double)
    // univariate methods
    m.def("univariate_accumulate", [](detail::array<double> x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()});
    });

    m.def("univariate_accumulate", [](std::vector<double> const& x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()});
    });

    m.def("univariate_accumulate", [](detail::array<double> x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()});
    });

    m.def("univariate_accumulate", [](std::vector<double> const& x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()});
    });

    // mean
    m.def("mean", [](detail::array<double> x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}).mean;
    });

    m.def("mean", [](detail::array<double> x, detail::array<double> w) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}, {w.data(), w.size()});
    });

    m.def("mean", [](std::vector<double> const& x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}).mean;
    });

    m.def("mean", [](std::vector<double> const& x, std::vector<double> const& w) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}, {w.data(), w.size()}).mean;
    });

    // variance
    m.def("variance", [](detail::array<double> x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}).variance;
    });

    m.def("variance", [](detail::array<double> x, detail::array<double> w) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}, {w.data(), w.size()});
    });

    m.def("variance", [](std::vector<double> const& x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}).variance;
    });

    m.def("variance", [](std::vector<double> const& x, std::vector<double> const& w) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}, {w.data(), w.size()}).variance;
    });

    // sample variance
    m.def("sample_variance", [](detail::array<double> x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}).sample_variance;
    });

    m.def("sample_variance", [](detail::array<double> x, detail::array<double> w) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}, {w.data(), w.size()});
    });

    m.def("sample_variance", [](std::vector<double> const& x) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}).sample_variance;
    });

    m.def("sample_variance", [](std::vector<double> const& x, std::vector<double> const& w) {
        return detail::univariate_accumulate<double>({x.data(), x.size()}, {w.data(), w.size()}).sample_variance;
    });

    // bivariate methods
    // covariance
    m.def("covariance", [](detail::array<double> x, detail::array<double> y) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}).covariance;
    });

    m.def("covariance", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).covariance;
    });

    m.def("covariance", [](std::vector<double> const& x, std::vector<double> const& y) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}).covariance;
    });

    m.def("covariance", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).covariance;
    });

    m.def("sample_covariance", [](detail::array<double> x, detail::array<double> y) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}).sample_covariance;
    });

    m.def("sample_covariance", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).sample_covariance;
    });

    m.def("sample_covariance", [](std::vector<double> const& x, std::vector<double> const& y) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}).sample_covariance;
    });

    m.def("sample_covariance", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).sample_covariance;
    });

    m.def("correlation", [](detail::array<double> x, detail::array<double> y) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}).correlation;
    });

    m.def("correlation", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).correlation;
    });

    m.def("correlation", [](std::vector<double> const& x, std::vector<double> const& y) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}).correlation;
    });

    m.def("correlation", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        return detail::bivariate_accumulate<double>({x.data(), x.size()}, {y.data(), y.size()}, {w.data(), w.size()}).correlation;
    });

    // metrics
    // mean absolute error
    m.def("mean_absolute_error", [](detail::array<double> x, detail::array<double> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_error", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_absolute_error", [](std::vector<double> const& x, std::vector<double> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_error", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // mean absolute percentage error
    m.def("mean_absolute_percentage_error", [](detail::array<double> x, detail::array<double> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_percentage_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_percentage_error", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_percentage_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_absolute_percentage_error", [](std::vector<double> const& x, std::vector<double> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_absolute_percentage_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_absolute_percentage_error", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_absolute_percentage_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // mean squared error
    m.def("mean_squared_error", [](detail::array<double> x, detail::array<double> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_error", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_squared_error", [](std::vector<double> const& x, std::vector<double> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_error", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // mean squared log error
    m.def("mean_squared_log_error", [](detail::array<double> x, detail::array<double> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_log_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_log_error", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_log_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("mean_squared_log_error", [](std::vector<double> const& x, std::vector<double> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::mean_squared_log_error<double>(a.begin(), a.end(), b.begin());
    });

    m.def("mean_squared_log_error", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::mean_squared_log_error<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // r2 score
    m.def("r2_score", [](detail::array<double> x, detail::array<double> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::r2_score<double>(a.begin(), a.end(), b.begin());
    });

    m.def("r2_score", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::r2_score<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("r2_score", [](std::vector<double> const& x, std::vector<double> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::r2_score<double>(a.begin(), a.end(), b.begin());
    });

    m.def("r2_score", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::r2_score<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    // poisson loss
    m.def("poisson_neg_likelihood_loss", [](detail::array<double> x, detail::array<double> y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<double>(a.begin(), a.end(), b.begin());
    });

    m.def("poisson_neg_likelihood_loss", [](detail::array<double> x, detail::array<double> y, detail::array<double> w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<double>(a.begin(), a.end(), b.begin(), c.begin());
    });

    m.def("poisson_neg_likelihood_loss", [](std::vector<double> const& x, std::vector<double> const& y) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<double>(a.begin(), a.end(), b.begin());
    });

    m.def("poisson_neg_likelihood_loss", [](std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& w) {
        std::span a{x.data(), x.size()};
        std::span b{y.data(), y.size()};
        std::span c{w.data(), w.size()};
        return vstat::metrics::poisson_neg_likelihood_loss<double>(a.begin(), a.end(), b.begin(), c.begin());
    });
}
