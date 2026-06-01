// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// catch2 must come before stat_other.hpp: linasm's Types.h #defines size_t,
// which breaks Catch2's internal static_cast<std::size_t> usage.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "vstat/vstat.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"
#include "stat_other.hpp"
#undef size_t  // linasm/Types.h #defines size_t; undo it so std::size_t works in Catch2 macros

namespace nb = ankerl::nanobench;

namespace uv = vstat::univariate;
namespace bv = vstat::bivariate;

namespace test_util
{
template<std::floating_point T>
auto generate(auto& rng, int count, T min = T {0}, T max = T {1})
{
    std::vector<T> vec(count);
    std::generate(vec.begin(),
                  vec.end(),
                  [&]() -> auto { return std::uniform_real_distribution<T>(min, max)(rng); });
    return vec;
}

auto csv() noexcept -> const char*
{
    return R"DELIM("title";"name";"statistic";"dtype";"unit";"batch";"elapsed";"error %";"instructions";"branches";"branch misses";"total"
{{#result}}"{{title}}";"{{name}}";"{{context(statistic)}}";"{{context(dtype)}}";"{{unit}}";{{batch}};{{median(elapsed)}};{{medianAbsolutePercentError(elapsed)}};{{median(instructions)}};{{median(branchinstructions)}};{{median(branchmisses)}};{{sumProduct(iterations, elapsed)}}
{{/result}})DELIM";
}

template<typename T>
auto equal(T a, T b, T eps = std::numeric_limits<T>::epsilon())
{
    return std::abs(a - b) < eps;
};

// Relative tolerance: |a-b| / max(|a|, |b|, 1) < eps.
// Use for quantities that scale with n (sums), where absolute tolerance is too strict.
template<typename T>
auto rel_equal(T a, T b, T eps)
{
    return std::abs(a - b) < eps * std::max({std::abs(a), std::abs(b), T{1}});
};
}  // namespace test_util

auto constexpr count_small {10};
auto constexpr count_medium {1'000};
auto constexpr count_large {100'000};
auto constexpr count_extreme {10'000'000};

TEST_CASE("gamma")
{
    eve::wide<float> x {10};
    eve::wide<float> g = eve::log_abs_gamma(x);
    std::cout << "x = " << x << "\n";
    std::cout << "g = " << g << "\n";
}

TEST_CASE("univariate", "[correctness]")
{
    float x[] = {1.F, 1.F, 2.F, 6.F};
    float y[] = {2.F, 4.F, 3.F, 1.F};
    size_t n = std::size(x);

    auto stats = vstat::bivariate::accumulate<double>(x, x + n, std::begin(y));
    REQUIRE(stats.count == sizeof(x) / sizeof(*x));
}

TEST_CASE("r2", "[correctness]")
{
    std::default_random_engine rng {1234};

    auto test_r2 = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::r2_score(y, x);
        auto m2 = vstat::metrics::r2_score<T>(x.begin(), x.end(), y.begin());

        INFO("n = " << n);
        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-6};
        SECTION("small") { test_r2(count_small, eps); }
        SECTION("medium") { test_r2(count_medium, eps); }
        SECTION("large") { test_r2(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-5};
        SECTION("small") { test_r2.operator()<float>(count_small, eps); }
        SECTION("medium") { test_r2.operator()<float>(count_medium, eps); }
        SECTION("large") { test_r2.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("weighted r2", "[correctness]")
{
    std::default_random_engine rng {1234};

    auto test_r2_weighted = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);
        auto z = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::r2_score(y, x, z);
        auto m2 = vstat::metrics::r2_score<T>(x.begin(), x.end(), y.begin(), z.begin());

        INFO("n = " << n);
        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-1};
        SECTION("small") { test_r2_weighted(count_small, eps); }
        SECTION("medium") { test_r2_weighted(count_medium, eps); }
        SECTION("large") { test_r2_weighted(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-1};
        SECTION("small") { test_r2_weighted.operator()<float>(count_small, eps); }
        SECTION("medium") { test_r2_weighted.operator()<float>(count_medium, eps); }
        SECTION("large") { test_r2_weighted.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("sum", "[correctness]")
{
    std::random_device rng {};

    auto test_sum = [&]<typename T = double>(int n, T eps) -> auto
    {
        auto x = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::sum(x);
        auto m2 = uv::accumulate<T>(x.begin(), x.end()).sum;

        if (!test_util::rel_equal<T>(m1, m2, eps)) {
            auto m3 = stat_other::gsl::sum(x);
            auto m4 = stat_other::linasm::sum(x);
            std::cout << std::setprecision(15) << "ba: " << m1 << "\n";
            std::cout << std::setprecision(15) << "uv: " << m2 << "\n";
            std::cout << std::setprecision(15) << "gl: " << m3 << "\n";
            std::cout << std::setprecision(15) << "la: " << m4 << "\n";
        }

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::rel_equal<T>(m1, m2, eps));
    };

    SECTION("numerical stability double")
    {
        double const eps {1e-12};
        vstat::univariate_accumulator<double> ac;
        constexpr auto n = 1'000'000;
        constexpr auto e = 1e-12;
        for (int i = 0; i < n; ++i)
            ac(e);
        vstat::univariate_statistics stats {ac};
        REQUIRE(test_util::equal<double>(stats.sum, e * n, eps));
    };

    SECTION("double")
    {
        double const eps {1e-10};
        SECTION("small") { test_sum(count_small, eps); }
        SECTION("medium") { test_sum(count_medium, eps); }
        SECTION("large") { test_sum(count_large, eps); }
        SECTION("extreme") { test_sum(count_extreme, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-2};
        SECTION("small") { test_sum.operator()<float>(count_small, eps); }
        SECTION("medium") { test_sum.operator()<float>(count_medium, eps); }
        SECTION("large") { test_sum.operator()<float>(count_large, eps); }
        SECTION("extreme") { test_sum.operator()<float>(count_extreme, eps); }
    }
}

TEST_CASE("weighted sum", "[correctness]")
{
    std::random_device rng {};

    auto test_sum = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto w = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::sum(x, w);
        auto m2 = uv::accumulate<T>(x.begin(), x.end(), w.begin()).sum;

        if (!test_util::rel_equal<T>(m1, m2, eps)) {
            auto m3 = stat_other::gsl::sum(x, w);
            std::cout << std::setprecision(15) << "ba: " << m1 << "\n";
            std::cout << std::setprecision(15) << "uv: " << m2 << "\n";
            std::cout << std::setprecision(15) << "gl: " << m3 << "\n";
        }

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::rel_equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-10};
        SECTION("small") { test_sum(count_small, eps); }
        SECTION("medium") { test_sum(count_medium, eps); }
        SECTION("large") { test_sum(count_large, eps); }
        SECTION("extreme") { test_sum(count_extreme, eps); }
    }

    SECTION("float")
    {
        float const eps {2e-2};
        SECTION("small") { test_sum.operator()<float>(count_small, eps); }
        SECTION("medium") { test_sum.operator()<float>(count_medium, eps); }
        SECTION("large") { test_sum.operator()<float>(count_large, eps); }
        SECTION("extreme") { test_sum.operator()<float>(count_extreme, eps); }
    }
}

TEST_CASE("mean", "[correctness]")
{
    std::random_device rng {};

    auto test_mean = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::mean(x);
        auto m2 = uv::accumulate<T>(x.begin(), x.end()).mean;

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-6};
        SECTION("small") { test_mean(count_small, eps); }
        SECTION("medium") { test_mean(count_medium, eps); }
        SECTION("large") { test_mean(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-5};
        SECTION("small") { test_mean.operator()<float>(count_small, eps); }
        SECTION("medium") { test_mean.operator()<float>(count_medium, eps); }
        SECTION("large") { test_mean.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("weighted mean", "[correctness]")
{
    std::random_device rng {};

    auto test_mean = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto w = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::mean(x, w);
        auto m2 = uv::accumulate<T>(x.begin(), x.end(), w.begin()).mean;

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-6};
        SECTION("small") { test_mean(count_small, eps); }
        SECTION("medium") { test_mean(count_medium, eps); }
        SECTION("large") { test_mean(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-5};
        SECTION("small") { test_mean.operator()<float>(count_small, eps); }
        SECTION("medium") { test_mean.operator()<float>(count_medium, eps); }
        SECTION("large") { test_mean.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("variance", "[correctness]")
{
    std::random_device rng {};

    auto test_variance = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::variance(x, y);
        auto m2 = uv::accumulate<T>(x.begin(), x.end(), y.begin()).variance;

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-6};
        SECTION("small") { test_variance(count_small, eps); }
        SECTION("medium") { test_variance(count_medium, eps); }
        SECTION("large") { test_variance(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-5};
        SECTION("small") { test_variance.operator()<float>(count_small, eps); }
        SECTION("medium") { test_variance.operator()<float>(count_medium, eps); }
        SECTION("large") { test_variance.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("weighted variance", "[correctness]")
{
    std::random_device rng {};

    auto test_variance = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto w = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::variance(x, w);
        auto m2 = uv::accumulate<T>(x.begin(), x.end(), w.begin()).variance;

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-6};
        SECTION("small") { test_variance(count_small, eps); }
        SECTION("medium") { test_variance(count_medium, eps); }
        SECTION("large") { test_variance(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-5};
        SECTION("small") { test_variance.operator()<float>(count_small, eps); }
        SECTION("medium") { test_variance.operator()<float>(count_medium, eps); }
        SECTION("large") { test_variance.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("covariance", "[correctness]")
{
    std::random_device rng {};

    auto test_covariance = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::covariance(x, y);
        auto m2 = bv::accumulate<T>(x.begin(), x.end(), y.begin()).covariance;

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-6};
        SECTION("small") { test_covariance(count_small, eps); }
        SECTION("medium") { test_covariance(count_medium, eps); }
        SECTION("large") { test_covariance(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-5};
        SECTION("small") { test_covariance.operator()<float>(count_small, eps); }
        SECTION("medium") { test_covariance.operator()<float>(count_medium, eps); }
        SECTION("large") { test_covariance.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("weighted covariance", "[correctness]")
{
    std::random_device rng {};

    auto test_covariance = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);
        auto w = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::covariance(x, y, w);
        auto m2 = bv::accumulate<T>(x.begin(), x.end(), y.begin(), w.begin()).covariance;

        INFO("m1 = " << m1);
        INFO("m2 = " << m2);
        REQUIRE(test_util::equal<T>(m1, m2, eps));
    };

    SECTION("double")
    {
        double const eps {1e-6};
        SECTION("small") { test_covariance(count_small, eps); }
        SECTION("medium") { test_covariance(count_medium, eps); }
        SECTION("large") { test_covariance(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-5};
        SECTION("small") { test_covariance.operator()<float>(count_small, eps); }
        SECTION("medium") { test_covariance.operator()<float>(count_medium, eps); }
        SECTION("large") { test_covariance.operator()<float>(count_large, eps); }
    }
}


TEST_CASE("poisson_neg_likelihood_loss", "[correctness]")
{
    std::default_random_engine rng {1234};

    auto test_pnll = [&]<typename T = double>(int n, T eps)
    {
        // y_pred must be positive; use [1, 2] to keep log well-defined
        auto y_true = test_util::generate<T>(rng, n, T {0}, T {4});
        auto y_pred = test_util::generate<T>(rng, n, T {1}, T {2});

        double ref {0};
        for (int i = 0; i < n; ++i)
            ref += static_cast<double>(y_pred[i])
                   - static_cast<double>(y_true[i]) * std::log(static_cast<double>(y_pred[i]))
                   + std::lgamma(1.0 + static_cast<double>(y_true[i]));

        double m2 = vstat::metrics::poisson_neg_likelihood_loss<T>(y_true.begin(), y_true.end(), y_pred.begin());

        INFO("n = " << n);
        INFO("ref = " << ref);
        INFO("m2 = " << m2);
        REQUIRE(test_util::rel_equal(ref, m2, static_cast<double>(eps)));
    };

    SECTION("double")
    {
        double const eps {1e-10};
        SECTION("small")  { test_pnll(count_small, eps); }
        SECTION("medium") { test_pnll(count_medium, eps); }
        SECTION("large")  { test_pnll(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-3F};
        SECTION("small")  { test_pnll.operator()<float>(count_small, eps); }
        SECTION("medium") { test_pnll.operator()<float>(count_medium, eps); }
        SECTION("large")  { test_pnll.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("gaussian_neg_likelihood_loss", "[correctness]")
{
    std::default_random_engine rng {1234};

    auto test_gnll = [&]<typename T = double>(int n, T eps)
    {
        auto y_true = test_util::generate<T>(rng, n);
        auto y_pred = test_util::generate<T>(rng, n);
        T const sigma {0.5};

        double ssr {0};
        for (int i = 0; i < n; ++i) {
            double d = static_cast<double>(y_true[i]) - static_cast<double>(y_pred[i]);
            ssr += d * d;
        }
        double ref = 0.5 * n * std::log(2.0 * std::numbers::pi_v<double>)
                     + n * std::log(static_cast<double>(sigma))
                     + ssr / (2.0 * static_cast<double>(sigma) * static_cast<double>(sigma));

        double m2 = vstat::metrics::gaussian_neg_likelihood_loss<T>(y_true.begin(), y_true.end(), y_pred.begin(), sigma);

        INFO("n = " << n);
        INFO("ref = " << ref);
        INFO("m2 = " << m2);
        REQUIRE(test_util::rel_equal(ref, m2, static_cast<double>(eps)));
    };

    SECTION("double")
    {
        double const eps {1e-10};
        SECTION("small")  { test_gnll(count_small, eps); }
        SECTION("medium") { test_gnll(count_medium, eps); }
        SECTION("large")  { test_gnll(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-3F};
        SECTION("small")  { test_gnll.operator()<float>(count_small, eps); }
        SECTION("medium") { test_gnll.operator()<float>(count_medium, eps); }
        SECTION("large")  { test_gnll.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("poisson_log_neg_likelihood_loss", "[correctness]")
{
    std::default_random_engine rng {1234};

    auto test_plnll = [&]<typename T = double>(int n, T eps)
    {
        auto y_true = test_util::generate<T>(rng, n, T {0}, T {4});
        auto x_pred = test_util::generate<T>(rng, n, T {-1}, T {1});

        double ref {0};
        for (int i = 0; i < n; ++i)
            ref += std::exp(static_cast<double>(x_pred[i]))
                   - static_cast<double>(y_true[i]) * static_cast<double>(x_pred[i])
                   + std::lgamma(1.0 + static_cast<double>(y_true[i]));

        double m2 = vstat::metrics::poisson_log_neg_likelihood_loss<T>(y_true.begin(), y_true.end(), x_pred.begin());

        INFO("n = " << n);
        INFO("ref = " << ref);
        INFO("m2 = " << m2);
        REQUIRE(test_util::rel_equal(ref, m2, static_cast<double>(eps)));
    };

    SECTION("double")
    {
        double const eps {1e-10};
        SECTION("small")  { test_plnll(count_small, eps); }
        SECTION("medium") { test_plnll(count_medium, eps); }
        SECTION("large")  { test_plnll(count_large, eps); }
    }

    SECTION("float")
    {
        float const eps {1e-3F};
        SECTION("small")  { test_plnll.operator()<float>(count_small, eps); }
        SECTION("medium") { test_plnll.operator()<float>(count_medium, eps); }
        SECTION("large")  { test_plnll.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("benchmarks", "[performance]")
{
    std::random_device rng {};

    nb::Bench bench;
    auto const n {1000};
    for (auto s = n; s <= 1024 * 1024; s *= 2) {
        auto xd = test_util::generate<double>(rng, n);
        auto yd = test_util::generate<double>(rng, n);
        auto wd = test_util::generate<double>(rng, n);

        auto xf = test_util::generate<float>(rng, n);
        auto yf = test_util::generate<float>(rng, n);
        auto wf = test_util::generate<float>(rng, n);

        double m {0.0};

        bench.context("dtype", "double");

        bench.context("statistic", "mean");
        bench.batch(s).run("vstat", [&]() -> void { m += uv::accumulate<double>(xd.begin(), xd.end()).mean; });
        bench.batch(s).run("vstat (stats::mean)", [&]() -> void { m += uv::accumulate<double, vstat::stats::mean>(xd.begin(), xd.end()).mean; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::mean(xd); });
        bench.batch(s).run("boost.math", [&]() -> void { m += boost::math::statistics::mean(xd); });
        bench.batch(s).run("gsl", [&]() -> void { m += stat_other::gsl::mean(xd); });
        bench.batch(s).run("linasm", [&]() -> void { m += stat_other::linasm::mean(xd); });

        bench.context("statistic", "weighted mean");
        bench.batch(s).run("vstat",
                           [&]() -> void { m += uv::accumulate<double>(xd.begin(), xd.end(), wd.begin()).mean; });
        bench.batch(s).run("vstat (stats::mean)", [&]() -> void { m += uv::accumulate<double, vstat::stats::mean>(xd.begin(), xd.end(), wd.begin()).mean; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::mean(xd, wd); });

        bench.context("statistic", "variance");
        bench.batch(s).run("vstat", [&]() -> void { m += uv::accumulate<double>(xd.begin(), xd.end()).variance; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::variance(xd); });
        bench.batch(s).run("boost.math", [&]() -> void { m += boost::math::statistics::variance(xd); });
        bench.batch(s).run("gsl", [&]() -> void { m += stat_other::gsl::variance(xd); });
        bench.batch(s).run("linasm", [&]() -> void { m += stat_other::linasm::variance(xd); });

        bench.context("statistic", "weighted variance");
        bench.batch(s).run("vstat",
                           [&]() -> void
                           { m += uv::accumulate<double>(xd.begin(), xd.end(), wd.begin()).variance; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::variance(xd, wd); });

        bench.context("statistic", "covariance");
        bench.batch(s).run("vstat",
                           [&]() -> void { m += bv::accumulate<double>(xd.begin(), xd.end(), yd.begin()).covariance; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::covariance(xd, yd); });
        bench.batch(s).run("boost.math", [&]() -> void { m += boost::math::statistics::covariance(xd, yd); });
        bench.batch(s).run("gsl", [&]() -> void { m += stat_other::gsl::covariance(xd, yd); });
        bench.batch(s).run("linasm", [&]() -> void { m += stat_other::linasm::covariance(xd, yd); });

        bench.context("statistic", "weighted covariance");
        bench.batch(s).run(
            "vstat",
            [&]() -> void
            { m += bv::accumulate<double>(xd.begin(), xd.end(), yd.begin(), wd.begin()).covariance; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::covariance(xd, yd, wd); });

        bench.context("dtype", "float");

        bench.context("statistic", "mean");
        bench.batch(s).run("vstat", [&]() -> void { m += uv::accumulate<float>(xf.begin(), xf.end()).mean; });
        bench.batch(s).run("vstat (stats::mean)", [&]() -> void { m += uv::accumulate<float, vstat::stats::mean>(xf.begin(), xf.end()).mean; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::mean(xf); });
        bench.batch(s).run("boost.math", [&]() -> void { m += boost::math::statistics::mean(xf); });
        bench.batch(s).run("gsl", [&]() -> void { m += stat_other::gsl::mean(xf); });
        bench.batch(s).run("linasm", [&]() -> void { m += stat_other::linasm::mean(xf); });

        bench.context("statistic", "weighted mean");
        bench.batch(s).run("vstat",
                           [&]() -> void { m += uv::accumulate<float>(xf.begin(), xf.end(), wf.begin()).mean; });
        bench.batch(s).run("vstat (stats::mean)", [&]() -> void { m += uv::accumulate<float, vstat::stats::mean>(xf.begin(), xf.end(), wf.begin()).mean; });
        bench.batch(s).run("boost.accu", [&]() { m += stat_other::boost::mean(xf, wf); });

        bench.context("statistic", "variance");
        bench.batch(s).run("vstat", [&]() -> void { m += uv::accumulate<float>(xf.begin(), xf.end()).variance; });
        bench.batch(s).run("boost.accu", [&]() { m += stat_other::boost::variance(xf); });
        bench.batch(s).run("boost.math", [&]() -> void { m += boost::math::statistics::variance(xf); });
        bench.batch(s).run("gsl", [&]() { m += stat_other::gsl::variance(xf); });
        bench.batch(s).run("linasm", [&]() { m += stat_other::linasm::variance(xf); });

        bench.context("statistic", "weighted variance");
        bench.batch(s).run("vstat",
                           [&]() -> void { m += uv::accumulate<float>(xf.begin(), xf.end(), wf.begin()).variance; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::variance(xf, wf); });

        bench.context("statistic", "covariance");
        bench.batch(s).run("vstat",
                           [&]() -> void { m += bv::accumulate<float>(xf.begin(), xf.end(), yf.begin()).covariance; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::covariance(xf, yf); });
        bench.batch(s).run("boost.math", [&]() -> void { m += boost::math::statistics::covariance(xf, yf); });
        bench.batch(s).run("gsl", [&]() -> void { m += stat_other::gsl::covariance(xf, yf); });
        bench.batch(s).run("linasm", [&]() -> void { m += stat_other::linasm::covariance(xf, yf); });

        bench.context("statistic", "weighted covariance");
        bench.batch(s).run(
            "vstat",
            [&]() -> void
            { m += bv::accumulate<float>(xf.begin(), xf.end(), yf.begin(), wf.begin()).covariance; });
        bench.batch(s).run("boost.accu", [&]() -> void { m += stat_other::boost::covariance(xf, yf, wf); });
    }
    bench.render(test_util::csv(), std::cout);
}

TEST_CASE("mean float", "[performance]")
{
    std::random_device rng {};
    nb::Bench bench;
    constexpr auto n {1'000'000};
    auto xf = test_util::generate<float>(rng, n);
    auto m {0.0};

    bench.context("dtype", "float");
    bench.context("statistic", "mean");
    bench.batch(n).run("vstat", [&]() -> void { m += uv::accumulate<float>(xf.begin(), xf.end()).mean; });
}

