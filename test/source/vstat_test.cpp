// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "vstat/vstat.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include "stat_other.hpp"

namespace nb = ankerl::nanobench;

namespace uv = vstat::univariate;
namespace bv = vstat::bivariate;
namespace mv = vstat::metrics;

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
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

    auto test_sum = [&]<typename T = double>(int n, T eps) -> auto
    {
        auto x = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::sum(x);
        auto m2 = uv::accumulate<T, vstat::stats::sum>(x.begin(), x.end()).sum;

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
    }

    SECTION("float")
    {
        float const eps {1e-2};
        SECTION("small") { test_sum.operator()<float>(count_small, eps); }
        SECTION("medium") { test_sum.operator()<float>(count_medium, eps); }
        SECTION("large") { test_sum.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("weighted sum", "[correctness]")
{
    std::mt19937 rng {1234};

    auto test_sum = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto w = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::sum(x, w);
        auto m2 = uv::accumulate<T, vstat::stats::sum>(x.begin(), x.end(), w.begin()).sum;

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
    }

    SECTION("float")
    {
        float const eps {2e-2};
        SECTION("small") { test_sum.operator()<float>(count_small, eps); }
        SECTION("medium") { test_sum.operator()<float>(count_medium, eps); }
        SECTION("large") { test_sum.operator()<float>(count_large, eps); }
    }
}

TEST_CASE("mean", "[correctness]")
{
    std::mt19937 rng {1234};

    auto test_mean = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::mean(x);
        auto m2 = uv::accumulate<T, vstat::stats::mean>(x.begin(), x.end()).mean;

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
    std::mt19937 rng {1234};

    auto test_mean = [&]<typename T = double>(int n, T eps)
    {
        auto x = test_util::generate<T>(rng, n);
        auto w = test_util::generate<T>(rng, n);

        auto m1 = stat_other::boost::mean(x, w);
        auto m2 = uv::accumulate<T, vstat::stats::mean>(x.begin(), x.end(), w.begin()).mean;

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
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

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


TEST_CASE("weighted variance zero-weight prefix", "[correctness]")
{
    // Regression test: zero-weight observations before any non-zero weight must not
    // poison sum_xx with NaN via 0/0 in the Welford denominator.
    auto test = [&]<typename T>() {
        std::vector<T> x {T{1}, T{2}, T{3}, T{4}, T{5}};
        std::vector<T> w {T{0}, T{0}, T{1}, T{1}, T{1}};

        auto stats = uv::accumulate<T>(x.begin(), x.end(), w.begin());

        REQUIRE(std::isfinite(stats.variance));
        REQUIRE(std::isfinite(stats.mean));

        // reference: only the non-zero-weight elements {3,4,5} with equal weight
        auto ref = uv::accumulate<T>(x.begin() + 2, x.end());
        REQUIRE(test_util::equal<T>(static_cast<T>(stats.mean),     static_cast<T>(ref.mean),     T{1e-5}));
        REQUIRE(test_util::equal<T>(static_cast<T>(stats.variance), static_cast<T>(ref.variance), T{1e-5}));
    };

    SECTION("double") { test.operator()<double>(); }
    SECTION("float")  { test.operator()<float>(); }
}

TEST_CASE("weighted covariance zero-weight prefix", "[correctness]")
{
    // Regression test: zero-weight observations before any non-zero weight must not
    // poison sum_xx/sum_yy/sum_xy with NaN via 0/0 in the bivariate Welford update.
    auto test = [&]<typename T>() {
        std::vector<T> x {T{1}, T{2}, T{3}, T{4}, T{5}};
        std::vector<T> y {T{5}, T{4}, T{3}, T{2}, T{1}};
        std::vector<T> w {T{0}, T{0}, T{1}, T{1}, T{1}};

        auto stats = bv::accumulate<T>(x.begin(), x.end(), y.begin(), w.begin());

        REQUIRE(std::isfinite(stats.covariance));
        REQUIRE(std::isfinite(stats.variance_x));
        REQUIRE(std::isfinite(stats.variance_y));

        // reference: only the non-zero-weight elements with equal weight
        auto ref = bv::accumulate<T>(x.begin() + 2, x.end(), y.begin() + 2);
        REQUIRE(test_util::equal<T>(static_cast<T>(stats.covariance), static_cast<T>(ref.covariance), T{1e-5}));
        REQUIRE(test_util::equal<T>(static_cast<T>(stats.variance_x), static_cast<T>(ref.variance_x), T{1e-5}));
        REQUIRE(test_util::equal<T>(static_cast<T>(stats.variance_y), static_cast<T>(ref.variance_y), T{1e-5}));
    };

    SECTION("double") { test.operator()<double>(); }
    SECTION("float")  { test.operator()<float>(); }
}

TEST_CASE("weighted variance all-zero weights", "[correctness]")
{
    // When every weight is zero the raw SSR (sum_xx) must be 0, not NaN.
    // Derived statistics that divide by sum_w (mean, variance) are legitimately
    // NaN, but the accumulator state itself must be clean so that subsequent
    // non-zero-weight observations are not affected.
    auto test = [&]<typename T>() {
        vstat::univariate_accumulator<T> acc;
        for (T xi : {T{1}, T{2}, T{3}})
            acc(xi, T{0});

        auto [sw, sx, sxx] = acc.stats();
        REQUIRE(sw  == T{0});
        REQUIRE(sx  == T{0});
        REQUIRE(sxx == T{0});  // must be 0, not NaN
    };

    SECTION("double") { test.operator()<double>(); }
    SECTION("float")  { test.operator()<float>(); }
}

TEST_CASE("weighted covariance all-zero weights", "[correctness]")
{
    // Same property for the bivariate accumulator: sum_xx, sum_yy, sum_xy
    // must all be 0 (not NaN) after a run of zero-weight observations.
    auto test = [&]<typename T>() {
        vstat::bivariate_accumulator<T> acc;
        for (auto [xi, yi] : std::initializer_list<std::pair<T,T>>{{T{1},T{5}},{T{2},T{4}},{T{3},T{3}}})
            acc(xi, yi, T{0});

        auto [sw, sx, sy, sxx, syy, sxy] = acc.stats();
        REQUIRE(sw  == T{0});
        REQUIRE(sxx == T{0});  // must be 0, not NaN
        REQUIRE(syy == T{0});
        REQUIRE(sxy == T{0});
    };

    SECTION("double") { test.operator()<double>(); }
    SECTION("float")  { test.operator()<float>(); }
}

namespace {
// Simulate the "masked zero-weight prefix then plain unweighted continue"
// sequence that bivariate::accumulate_finite exercises on its SIMD tail:
// bivariate::accumulate_finite<...> ran the wide accumulator with a weight
// of zero on every lane of the very first chunk (skipping the first s
// values), then transferred state into a scalar bivariate_accumulator<T>
// and called the *unweighted* operator()(T, T) for the remaining tail.
// Before the fix, that scalar tail update did `1. / (sum_w * sum_w_old)`
// with sum_w_old == 0 (left there by the masked wide path), giving 0/0 =
// NaN that got stored into sum_xx unconditionally and poisoned every
// subsequent scalar tail update.
template<typename T>
auto mixed_masked_unweighted_round_trip() -> std::tuple<T, T, T, T, T, T>
{
    vstat::bivariate_accumulator<T> acc;
    // front: all-weight-zero run, leaves sum_w=0, sum_w_old=0 (post masked).
    for (auto [xi, yi] : std::initializer_list<std::pair<T,T>>{{T{1},T{5}},{T{2},T{4}},{T{3},T{3}}})
        acc(xi, yi, T{0});
    // tail: "continue unweighted", must not see 0/0 -> NaN from the prior state.
    for (auto [xi, yi] : std::initializer_list<std::pair<T,T>>{{T{4},T{2}},{T{5},T{1}}})
        acc(xi, yi);
    return acc.stats();
}
} // namespace

TEST_CASE("bivariate mixed weighted-zero prefix then unweighted tail", "[correctness]")
{
    // Regression test for the latent bivariate zero-denominator bug the
    // skip-non-finite branch hit only after bivariate::accumulate_finite
    // started routing its scalar tail through the unweighted update: a
    // zero-weight prefix left sum_w_old at 0 in the wide accumulator, and
    // the *unweighted* update (called for the residual tail entries) did
    // `1. / (sum_w * sum_w_old)` unconditionally -> 0/0 = NaN -> stored
    // into sum_xx, sum_yy, sum_xy, poisoning every subsequent observation.
    // The fix routes the unweighted update through the (already-guarded)
    // weighted update with w=1.
    auto test = [&]<typename T>() {
        auto [sw, sx, sy, sxx, syy, sxy] = mixed_masked_unweighted_round_trip<T>();
        REQUIRE(std::isfinite(static_cast<double>(sxx)));
        REQUIRE(std::isfinite(static_cast<double>(syy)));
        REQUIRE(std::isfinite(static_cast<double>(sxy)));
        // Reference: tail-only unweighted stats over the last two pairs
        // {4,2}, {5,1} (the zero-weight prefix was correctly excluded by
        // the weighted overload's own guard; the unweighted tail must
        // reproduce accumulate on {4,2..5,1}).
        std::vector<T> x {T{4}, T{5}};
        std::vector<T> y {T{2}, T{1}};
        auto ref = bv::accumulate<T>(x.begin(), x.end(), y.begin());
        REQUIRE(test_util::equal<T>(static_cast<T>(sw / sw), T{1}, T{1e-5}));
        // count is sw (2.0), mean_x = sx/sw, mean_y = sy/sw, variance_x = sxx/sw
        REQUIRE(test_util::equal<T>(static_cast<T>(sw), static_cast<T>(ref.count), T{1e-5}));
        REQUIRE(test_util::equal<T>(static_cast<T>(sxx / sw), static_cast<T>(ref.ssr_x / ref.count), T{1e-5}));
        REQUIRE(test_util::equal<T>(static_cast<T>(syy / sw), static_cast<T>(ref.ssr_y / ref.count), T{1e-5}));
        REQUIRE(test_util::equal<T>(static_cast<T>(sxy / sw), static_cast<T>(ref.sum_xy / ref.count), T{1e-5}));
    };

    SECTION("double") { test.operator()<double>(); }
    SECTION("float")  { test.operator()<float>(); }
}

TEST_CASE("accumulate_finite all-finite matches accumulate", "[correctness]")
{
    // With no non-finite values, accumulate_finite must reproduce the plain
    // (unmasked) accumulate exactly -- the mask is a no-op.
    std::mt19937 rng {1234};

    auto test = [&]<typename T>(int n, T eps) {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);

        auto [st, skipped] = uv::accumulate_finite<T, vstat::stats::mean>(
            x.begin(), x.end(), y.begin(), [](auto a, auto /*b*/) { return a; });
        auto ref = uv::accumulate<T, vstat::stats::mean>(x.begin(), x.end());

        REQUIRE(skipped == 0UL);
        REQUIRE(test_util::equal<T>(static_cast<T>(st.mean), static_cast<T>(ref.mean), eps));
    };

    SECTION("double") { test.operator()<double>(count_medium, 1e-6); }
    SECTION("float")  { test.operator()<float>(count_medium, 1e-5); }
}

TEST_CASE("accumulate_finite skips non-finite pairs", "[correctness]")
{
    auto test = [&]<typename T>() {
        std::vector<T> x {T{1}, T{2}, std::numeric_limits<T>::quiet_NaN(), T{4}, T{5}, T{6}, T{7}, T{8}};
        std::vector<T> y {T{1}, T{2}, T{3},                                 T{4}, std::numeric_limits<T>::infinity(), T{6}, T{7}, T{8}};

        auto [st, skipped] = uv::accumulate_finite<T, vstat::stats::mean>(
            x.begin(), x.end(), y.begin(), [](auto a, auto /*b*/) { return a; });

        REQUIRE(skipped == 2UL);

        // reference: manually filtered finite-pair subset {1,2,4,6,7,8}
        std::vector<T> ref {T{1}, T{2}, T{4}, T{6}, T{7}, T{8}};
        auto refStats = uv::accumulate<T, vstat::stats::mean>(ref.begin(), ref.end());

        REQUIRE(std::isfinite(st.mean));
        REQUIRE(test_util::equal<T>(static_cast<T>(st.mean), static_cast<T>(refStats.mean), T{1e-5}));
    };

    SECTION("double") { test.operator()<double>(); }
    SECTION("float")  { test.operator()<float>(); }
}

TEST_CASE("weighted accumulate, stats::variance, scattered zero weights at wide scale", "[correctness]")
{
    // scattered (not prefix, not all-zero) zero weights, count_medium so
    // the wide path is exercised, across multiple chunks/lanes
    std::mt19937 rng {1234};

    auto test = [&]<typename T>(int n, T eps) {
        auto y = test_util::generate<T>(rng, n);
        std::vector<T> w(n, T{1});
        w[0] = T{0};
        w[n / 2] = T{0};
        w[n - 1] = T{0};

        auto st = uv::accumulate<T, vstat::stats::variance>(y.begin(), y.end(), w.begin());

        std::vector<T> yf;
        for (int i = 0; i < n; ++i) {
            if (w[i] != T{0}) { yf.push_back(y[i]); }
        }
        auto ref = uv::accumulate<T, vstat::stats::variance>(yf.begin(), yf.end());

        INFO("st.variance = " << st.variance << ", ref.variance = " << ref.variance);
        REQUIRE(std::isfinite(st.variance));
        REQUIRE(test_util::equal<T>(static_cast<T>(st.variance), static_cast<T>(ref.variance), eps));
    };

    SECTION("double") { test.operator()<double>(count_medium, 1e-5); }
    SECTION("float")  { test.operator()<float>(count_medium, 1e-3); }
}

TEST_CASE("accumulate_finite with stats::variance", "[correctness]")
{
    std::mt19937 rng {1234};

    auto test = [&]<typename T>(int n, T eps) {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);
        // inject non-finite values spread across likely-multiple SIMD chunks
        x[0] = std::numeric_limits<T>::quiet_NaN();
        x[n / 2] = std::numeric_limits<T>::infinity();
        x[n - 1] = -std::numeric_limits<T>::infinity();

        auto [st, skipped] = uv::accumulate_finite<T, vstat::stats::variance>(
            y.begin(), y.end(), x.begin(), [](auto a, auto /*b*/) { return a; });

        REQUIRE(skipped == 3UL);

        std::vector<T> yf;
        for (int i = 0; i < n; ++i) {
            if (std::isfinite(x[i])) { yf.push_back(y[i]); }
        }
        auto ref = uv::accumulate<T, vstat::stats::variance>(yf.begin(), yf.end());

        REQUIRE(std::isfinite(st.variance));
        INFO("st.variance = " << st.variance << ", ref.variance = " << ref.variance);
        REQUIRE(test_util::equal<T>(static_cast<T>(st.variance), static_cast<T>(ref.variance), eps));
        REQUIRE(test_util::equal<T>(static_cast<T>(st.mean), static_cast<T>(ref.mean), eps));
    };

    SECTION("double") { test.operator()<double>(count_medium, 1e-5); }
    SECTION("float")  { test.operator()<float>(count_medium, 1e-3); }
}

TEST_CASE("accumulate_finite all-non-finite", "[correctness]")
{
    auto test = [&]<typename T>() {
        std::vector<T> x {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(), T{1}};
        std::vector<T> y {T{1}, T{2}, std::numeric_limits<T>::quiet_NaN()};

        auto [st, skipped] = uv::accumulate_finite<T, vstat::stats::mean>(
            x.begin(), x.end(), y.begin(), [](auto a, auto /*b*/) { return a; });

        REQUIRE(skipped == 3UL);
    };

    SECTION("double") { test.operator()<double>(); }
    SECTION("float")  { test.operator()<float>(); }
}

TEST_CASE("mean_squared_error_finite / mean_absolute_error_finite", "[correctness]")
{
    std::mt19937 rng {1234};

    auto test = [&]<typename T>(int n, T eps) {
        auto x = test_util::generate<T>(rng, n);
        auto y = test_util::generate<T>(rng, n);
        auto w = test_util::generate<T>(rng, n, T{0.1}, T{2});

        // inject a few non-finite predictions
        x[0] = std::numeric_limits<T>::quiet_NaN();
        x[n / 2] = std::numeric_limits<T>::infinity();

        std::vector<T> xf, yf, wf;
        for (int i = 0; i < n; ++i) {
            if (std::isfinite(x[i]) && std::isfinite(y[i])) {
                xf.push_back(x[i]);
                yf.push_back(y[i]);
                wf.push_back(w[i]);
            }
        }

        auto [mse, skippedMse] = mv::mean_squared_error_finite<T>(x.begin(), x.end(), y.begin());
        auto mseRef = mv::mean_squared_error<T>(xf.begin(), xf.end(), yf.begin());
        REQUIRE(skippedMse == 2UL);
        REQUIRE(test_util::equal<T>(static_cast<T>(mse), static_cast<T>(mseRef), eps));

        auto [mae, skippedMae] = mv::mean_absolute_error_finite<T>(x.begin(), x.end(), y.begin());
        auto maeRef = mv::mean_absolute_error<T>(xf.begin(), xf.end(), yf.begin());
        REQUIRE(skippedMae == 2UL);
        REQUIRE(test_util::equal<T>(static_cast<T>(mae), static_cast<T>(maeRef), eps));

        auto [wmse, wSkippedMse] = mv::mean_squared_error_finite<T>(x.begin(), x.end(), y.begin(), w.begin());
        auto wmseRef = mv::mean_squared_error<T>(xf.begin(), xf.end(), yf.begin(), wf.begin());
        REQUIRE(wSkippedMse == 2UL);
        REQUIRE(test_util::equal<T>(static_cast<T>(wmse), static_cast<T>(wmseRef), eps));

        auto [wmae, wSkippedMae] = mv::mean_absolute_error_finite<T>(x.begin(), x.end(), y.begin(), w.begin());
        auto wmaeRef = mv::mean_absolute_error<T>(xf.begin(), xf.end(), yf.begin(), wf.begin());
        REQUIRE(wSkippedMae == 2UL);
        REQUIRE(test_util::equal<T>(static_cast<T>(wmae), static_cast<T>(wmaeRef), eps));
    };

    SECTION("double") { test.operator()<double>(count_medium, 1e-5); }
    SECTION("float")  { test.operator()<float>(count_medium, 1e-3); }
}

TEST_CASE("poisson_neg_likelihood_loss", "[correctness]")
{
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

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
    std::mt19937 rng {1234};

    nb::Bench bench;
    for (auto s = 1000; s <= 1024 * 1024; s *= 2) {
        auto xd = test_util::generate<double>(rng, s);
        auto yd = test_util::generate<double>(rng, s);
        auto wd = test_util::generate<double>(rng, s);

        auto xf = test_util::generate<float>(rng, s);
        auto yf = test_util::generate<float>(rng, s);
        auto wf = test_util::generate<float>(rng, s);

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

TEST_CASE("masked vs unmasked mean squared error", "[performance]")
{
    std::mt19937 rng {1234};

    nb::Bench bench;
    for (auto s = 1000; s <= 1024 * 1024; s *= 2) {
        auto xd = test_util::generate<double>(rng, s);
        auto yd = test_util::generate<double>(rng, s);
        auto xf = test_util::generate<float>(rng, s);
        auto yf = test_util::generate<float>(rng, s);

        double m {0.0};

        bench.context("dtype", "double");
        bench.context("statistic", "mean squared error");
        bench.batch(s).run("unmasked", [&]() -> void { m += mv::mean_squared_error<double>(xd.begin(), xd.end(), yd.begin()); });
        bench.batch(s).run("masked (finite)", [&]() -> void { m += mv::mean_squared_error_finite<double>(xd.begin(), xd.end(), yd.begin()).first; });

        bench.context("dtype", "float");
        bench.context("statistic", "mean squared error");
        bench.batch(s).run("unmasked", [&]() -> void { m += mv::mean_squared_error<float>(xf.begin(), xf.end(), yf.begin()); });
        bench.batch(s).run("masked (finite)", [&]() -> void { m += mv::mean_squared_error_finite<float>(xf.begin(), xf.end(), yf.begin()).first; });
    }
    bench.render(test_util::csv(), std::cout);
}

TEST_CASE("mean float", "[performance]")
{
    std::mt19937 rng {1234};
    nb::Bench bench;
    constexpr auto n {1'000'000};
    auto xf = test_util::generate<float>(rng, n);
    auto m {0.0};

    bench.context("dtype", "float");
    bench.context("statistic", "mean");
    bench.batch(n).run("vstat", [&]() -> void { m += uv::accumulate<float>(xf.begin(), xf.end()).mean; });
}

