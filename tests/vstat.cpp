#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "gsl/gsl_statistics_double.h"
#include "gsl/gsl_statistics_float.h"

#include <chrono>

#include "vstat.hpp"

#include "Statistics.h"

namespace ba = boost::accumulators;
namespace nb = ankerl::nanobench;

struct Foo {
    double value;
};

TEST_SUITE("usage")
{
    TEST_CASE("univariate")
    {
        std::vector<float> values{ 1.0, 2.0, 3.0, 4.0 };
        std::vector<float> weights{ 2.0, 4.0, 6.0, 8.0 };

        SUBCASE("batch") {
            auto stats = univariate::accumulate<float>(values.begin(), values.end());
            std::cout << "stats:\n" << stats << "\n";
        }

        SUBCASE("batch weighted") {
            auto stats = univariate::accumulate<float>(values.begin(), values.end(), weights.begin());
            std::cout << "stats:\n" << stats << "\n";
        }

        SUBCASE("batch projection")
        {
            struct Foo {
                float value;
            };

            Foo foos[] = { {1}, {3}, {5}, {2}, {8} };
            auto stats = univariate::accumulate<float>(foos, std::size(foos), [](auto const& foo) { return foo.value; });
            std::cout << "stats:\n" << stats << "\n";
        }

        SUBCASE("accumulator") {
            univariate_accumulator<float> acc(1.0);
            acc(2.0);
            acc(3.0);
            acc(4.0);
            auto stats = univariate_statistics(acc);
            std::cout << "stats:\n" << stats << "\n";
        }

        SUBCASE("accumulator weighted") {
            univariate_accumulator<float> acc(1.0, 2.0);
            acc(2.0, 4.0);
            acc(3.0, 6.0);
            auto stats = univariate_statistics(acc);
            std::cout << "stats:\n" << stats << "\n";
        }
    }

    TEST_CASE("bivariate")
    {
        float x[] = { 1., 1., 2., 6. };
        float y[] = { 2., 4., 3., 1. };
        size_t n = std::size(x);

        SUBCASE("batch") {
            auto stats = bivariate::accumulate<float>(x, y, n);
            std::cout << "stats:\n" << stats << "\n";
        }

        SUBCASE("batch projection") {
            struct Foo {
                float value;
            };

            struct Bar {
                int value;
            };

            Foo foos[] = { {1}, {3}, {5}, {2}, {8} };
            Bar bars[] = { {3}, {2}, {1}, {4}, {11} };

            auto stats = bivariate::accumulate<float>(foos, bars, std::size(foos), [](auto const& foo) { return foo.value; }, [](auto const& bar) { return bar.value; });
            std::cout << "stats:\n" << stats << "\n";
        }

        SUBCASE("accumulator") {
            bivariate_accumulator<float> acc(x[0], y[0]);
            for (size_t i = 1; i < n; ++i) {
                acc(x[i], y[i]);
            }
            bivariate_statistics stats(acc);
            std::cout << "stats:\n" << stats << "\n";
        }

    }
}

TEST_SUITE("correctness")
{
    TEST_CASE("univariate")
    {
        const int n = int(1e6);

        std::vector<float> xf(n);
        std::vector<double> xd(n);

        std::vector<float> yf(n);
        std::vector<double> yd(n);

        std::default_random_engine rng(1234);
        std::uniform_real_distribution<double> dist(-1, 1);

        std::generate(xd.begin(), xd.end(), [&]() { return dist(rng); });
        std::generate(yd.begin(), yd.end(), [&]() { return dist(rng); });

        std::copy(xd.begin(), xd.end(), xf.begin());
        std::copy(yd.begin(), yd.end(), yf.begin());

        std::vector<Foo> ff(n);
        for (int i = 0; i < n; ++i) {
            ff[i].value = xd[i];
        }

        auto gsl_var_flt = gsl_stats_float_variance(xf.data(), 1, xf.size());
        auto gsl_mean_flt = gsl_stats_float_mean(xf.data(), 1, xf.size());

        auto stats = univariate::accumulate<float>(xd.begin(), xd.end(), detail::identity{});
        CHECK(std::abs(stats.mean - gsl_mean_flt) < 1e-6);
        CHECK(std::abs(stats.variance - gsl_var_flt) < 1e-5);

        auto gsl_var_dbl = gsl_stats_variance(xd.data(), 1, xd.size());
        auto gsl_mean_dbl = gsl_stats_mean(xd.data(), 1, xd.size());
        stats = univariate::accumulate<double>(xd.begin(), xd.end(), detail::identity {});
        CHECK(std::abs(stats.mean - gsl_mean_dbl) < 1e-6);
        CHECK(std::abs(stats.variance - gsl_var_dbl) < 1e-6);

        stats = univariate::accumulate<double>(ff.begin(), ff.end(), [](auto const& f) { return f.value; });
        CHECK(std::abs(stats.mean - gsl_mean_dbl) < 1e-6);

        auto linasm_mean_flt = Statistics::Mean(xf.data(), n);
        auto linasm_var_flt = Statistics::Variance(xf.data(), n, linasm_mean_flt);
        CHECK(std::abs(linasm_mean_flt - gsl_mean_flt) < 1e-6);
        CHECK(std::abs(linasm_var_flt - gsl_var_flt) < 1e-6);
        auto linasm_mean_dbl = Statistics::Mean(xd.data(), n);
        auto linasm_var_dbl = Statistics::Variance(xd.data(), n, linasm_mean_dbl);
        CHECK(std::abs(linasm_mean_dbl - gsl_mean_dbl) < 1e-6);
        CHECK(std::abs(linasm_var_dbl - gsl_var_dbl) < 1e-6);

        auto gsl_corr_flt = gsl_stats_float_correlation(xf.data(), 1, yf.data(), 1, n);
        auto gsl_corr_dbl = gsl_stats_correlation(xd.data(), 1, yd.data(), 1, n);

        auto gsl_cov_flt = gsl_stats_float_covariance(xf.data(), 1, yf.data(), 1, n);
        auto gsl_cov_dbl = gsl_stats_covariance(xd.data(), 1, yd.data(), 1, n);

        auto bstats = bivariate::accumulate<float>(xd.begin(), xd.end(), yd.begin());
        CHECK(std::abs(gsl_corr_flt - bstats.correlation) < 1e-6);
        CHECK(std::abs(gsl_cov_flt - bstats.covariance) < 1e-6);

        bstats = bivariate::accumulate<float>(xd.begin(), xd.end(), yd.begin());
        CHECK(std::abs(gsl_corr_dbl - bstats.correlation) < 1e-6);
        CHECK(std::abs(gsl_cov_dbl - bstats.covariance) < 1e-6);

        auto stats_x = univariate::accumulate<float>(xd.begin(), xd.end());
        auto stats_y = univariate::accumulate<float>(yd.begin(), yd.end());

        CHECK(bstats.mean_x == stats_x.mean);
        CHECK(bstats.mean_y == stats_y.mean);
        CHECK(bstats.sum_x == stats_x.sum);
        CHECK(bstats.sum_y == stats_y.sum);

        auto xm = Statistics::Mean(xf.data(), n);
        auto ym = Statistics::Mean(yf.data(), n);
        auto linasm_corr_flt = Statistics::PearsonCorrelation(xf.data(), yf.data(), n, xm, ym);
        CHECK(std::abs(linasm_corr_flt - gsl_corr_flt) < 1e-6);
    }
}

TEST_SUITE("performance")
{
    TEST_CASE("univariate")
    {
        const int n = int(1e6);

        std::vector<double> v1(n), v2(n);
        std::vector<float> u1(n), u2(n);

        auto xd = v1.data();
        auto yd = v2.data();
        auto xf = u1.data();
        auto yf = u2.data();

        std::default_random_engine rng(1234);
        std::uniform_real_distribution<double> dist(-1, 1);

        std::generate(xd, xd + n, [&]() { return dist(rng); });
        std::generate(yd, yd + n, [&]() { return dist(rng); });

        std::copy(xd, xd + n, xf);
        std::copy(yd, yd + n, yf);

        std::vector<Foo> ff(n);
        for (int i = 0; i < n; ++i) {
            ff[i].value = xd[i];
        }

        ankerl::nanobench::Bench b;
        b.performanceCounters(true).minEpochIterations(100).batch(n);

        // print some runtime stats for different data sizes
        std::vector<int> sizes { 1000, 10000 };
        int step = int(1e5);
        for (int s = step; s <= n; s += step) {
            sizes.push_back(s);
        }

        SUBCASE("variance") {
            double var = 0, count = 0;
            for (auto s : sizes) {
                b.batch(s).run("vstat variance float " + std::to_string(s), [&]() {
                    ++count;
                    var += univariate::accumulate<float>(xf, s).variance;
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("vstat variance double " + std::to_string(s), [&]() {
                    ++count;
                    var += univariate::accumulate<double>(xd, s).variance;
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("linasm variance float " + std::to_string(s), [&]() {
                    ++count;
                    auto mean = Statistics::Mean(xf, s);
                    var += Statistics::Variance(xf, s, mean);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("linasm variance double " + std::to_string(s), [&]() {
                    ++count;
                    auto mean = Statistics::Mean(xd, s);
                    var += Statistics::Variance(xd, s, mean);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("boost variance float " + std::to_string(s), [&]() {
                    ++count;
                    ba::accumulator_set<float, ba::features<ba::tag::variance>> acc;
                    for (int i = 0; i < s; ++i) {
                        acc(xf[i]);
                    }
                    var += ba::variance(acc);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("boost variance double " + std::to_string(s), [&]() {
                    ++count;
                    ba::accumulator_set<double, ba::features<ba::tag::variance>> acc;
                    for (int i = 0; i < s; ++i) {
                        acc(xd[i]);
                    }
                    var += ba::variance(acc);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("gsl variance float " + std::to_string(s), [&]() {
                    ++count;
                    var += gsl_stats_float_variance(xf, 1, s);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("gsl variance double " + std::to_string(s), [&]() {
                    ++count;
                    var += gsl_stats_variance(xd, 1, s);
                });
            }
        }

        SUBCASE("covariance") {
            double var = 0, count = 0;
            for (auto s : sizes) {
                b.batch(s).run("vstat covariance float " + std::to_string(s), [&]() {
                    ++count;
                    var += bivariate::accumulate<float>(xf, yf, s).covariance;
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("vstat covariance double " + std::to_string(s), [&]() {
                    ++count;
                    var += bivariate::accumulate<double>(xd, yd, s).covariance;
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("linasm covariance float " + std::to_string(s), [&]() {
                    ++count;
                    auto xm = Statistics::Mean(xf, s);
                    auto ym = Statistics::Mean(yf, s);
                    var += Statistics::Covariance(xf, yf, s, xm, ym);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("linasm covariance double " + std::to_string(s), [&]() {
                    ++count;
                    auto xm = Statistics::Mean(xd, s);
                    auto ym = Statistics::Mean(yd, s);
                    var += Statistics::Covariance(xd, yd, s, xm, ym);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("boost covariance float " + std::to_string(s), [&]() {
                    ++count;
                    ba::accumulator_set<float, ba::stats<ba::tag::covariance<float, ba::tag::covariate1>>> acc;
                    for (int i = 0; i < s; ++i) {
                        acc(xf[i], ba::covariate1 = yf[i]);
                    }
                    var += ba::covariance(acc);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("boost covariance double " + std::to_string(s), [&]() {
                    ++count;
                    ba::accumulator_set<double, ba::stats<ba::tag::covariance<double, ba::tag::covariate1>>> acc;
                    for (int i = 0; i < s; ++i) {
                        acc(xd[i], ba::covariate1 = yd[i]);
                    }
                    var += ba::covariance(acc);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("gsl covariance float " + std::to_string(s), [&]() {
                    ++count;
                    var += gsl_stats_float_covariance(xf, 1, yf, 1, s);
                });
            }

            var = count = 0;
            for (auto s : sizes) {
                b.batch(s).run("gsl covariance double " + std::to_string(s), [&]() {
                    ++count;
                    var += gsl_stats_covariance(xd, 1, yd, 1, s);
                });
            }
        }
    }
}
