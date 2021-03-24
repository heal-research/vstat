#include "variance.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <random>
#include <vector>
#include <iostream>

#include "gsl/gsl_statistics_double.h"
#include "gsl/gsl_statistics_float.h"

#include "vstat.hpp"

namespace nb = ankerl::nanobench;

template<typename T>
double VarianceGSL(nb::Bench& bench, std::vector<T> const& x)
{
    double c = 0;
    size_t i = 0;
    bench.batch(x.size()).run("GSL variance " + std::string(typeid(T).name()) + " " + std::to_string(x.size()), [&]() {
        double var;
        if constexpr(std::is_same_v<T, float>)
        {
            var = gsl_stats_float_variance(x.data(), 1, x.size());
        } else {
            var = gsl_stats_variance(x.data(), 1, x.size());
        }
        c += var;
        i += 1;
    });
    return c / i;
}

template<typename T>
double CorrelationGSL(nb::Bench& bench, std::vector<T> const& x, std::vector<T> const& y)
{
    double c = 0;
    size_t i = 0;
    bench.batch(x.size()).run("GSL variance " + std::string(typeid(T).name()) + " " + std::to_string(x.size()), [&]() {
        double corr;
        if constexpr(std::is_same_v<T, float>)
        {
            corr = gsl_stats_float_correlation(x.data(), 1, y.data(), 1, x.size());
        } else {
            corr = gsl_stats_correlation(x.data(), 1, y.data(), 1, x.size());
        }
        c += corr;
        i += 1;
    });
    return c / i;
}

template<typename T>
double Variance(nb::Bench& bench, std::vector<T> const& x)
{
    double c = 0;
    size_t i = 0;
    bench.batch(x.size()).run("variance " + std::string(typeid(T).name()) + " " + std::to_string(x.size()), [&]() {
        VarianceCalculator calc;
        T w{1.0};
        for (auto v : x) {
            calc.Add(v, w);
        }
        c += calc.NaiveVariance();
        i += 1;
    });
    return c / i;
}

template<typename T>
double VarianceSIMD(nb::Bench& bench, std::vector<T> const& x)
{
    double c = 0;
    size_t i = 0;
    bench.batch(x.size()).run("variance simd " + std::string(typeid(T).name()) + " " + std::to_string(x.size()), [&]() {
        VarianceCalculator calc;
        calc.Add(x.data(), x.size());
        c += calc.NaiveVariance();
        i += 1;
    });
    return c / i;
}

template<typename T>
double Correlation(nb::Bench& bench, std::vector<T> const& x, std::vector<T> const& y)
{
    double c = 0;
    size_t i = 0;
    bench.run("correlation " + std::string(typeid(T).name()) + " " + std::to_string(x.size()), [&]() {
        CorrelationCalculator calc;
        for (int j = 0; j < x.size(); ++j) {
            calc.Add(x[j], y[j]);
        }
        c += calc.Correlation();
        i += 1;
    });
    return c / i;
}

template<typename T>
double CorrelationSIMD(nb::Bench& bench, std::vector<T> const& x, std::vector<T> const& y)
{
    double c = 0;
    size_t i = 0;
    bench.run("correlation simd " + std::string(typeid(T).name()) + " " + std::to_string(x.size()), [&]() {
        CorrelationCalculator calc;
        calc.Add(x.data(), y.data(), x.size());
        c += calc.Correlation();
        i += 1;
    });
    return c / i;
}

TEST_SUITE("correctness")
{
    TEST_CASE("mean-variance")
    {
        const size_t n = int(1e6);

        std::default_random_engine rng(12345);
        std::uniform_real_distribution<double> dist(0, 1);

        std::vector<double> xd(n);
        std::vector<float> xf(n);
        std::vector<double> wd(n, 1.0);
        std::vector<float> wf(n, 1.0);

        std::generate(xd.begin(), xd.end(), [&]() { return dist(rng); });
        std::copy(xd.begin(), xd.end(), xf.begin());

        // GSL as baseline
        auto gsl_mean_flt = gsl_stats_float_mean(xf.data(), 1, xf.size());
        auto gsl_mean_dbl = gsl_stats_mean(xd.data(), 1, xd.size());
        auto gsl_var_flt  = gsl_stats_float_variance(xf.data(), 1, xf.size());
        auto gsl_var_dbl  = gsl_stats_variance(xd.data(), 1, xd.size());

        SUBCASE("mean")
        {
            VarianceCalculator vc;
            vc.Add(xf.data(), wf.data(), xf.size());
            auto vst_mean_flt_simd = vc.Mean();
            vc.Reset();

            for (auto v : xf) { vc.Add(v); }
            auto vst_mean_flt = vc.Mean();
            CHECK(std::abs(gsl_mean_flt - vc.Mean()) < 1e-6);

            vc.Reset();
            vc.Add(xd.data(), xd.size());
            CHECK(std::abs(gsl_mean_dbl - vc.Mean()) < 1e-6);

            vc.Reset();
            vc.Add(xd.data(), wd.data(), xd.size());
            CHECK(std::abs(gsl_mean_dbl - vc.Mean()) < 1e-6);
        }

        SUBCASE("variance")
        {

            VarianceCalculator vc;
            vc.Add(xf.data(), xf.size());
            auto vst_var_flt = vc.NaiveVariance();
            CHECK(std::abs(gsl_var_flt - vst_var_flt) < 1e-5);

            vc.Reset(); vc.Add(xd.data(), xd.size());
            auto vst_var_dbl = vc.NaiveVariance();
            CHECK(std::abs(gsl_var_dbl - vst_var_dbl) < 1e-6);
        }
    }

}

TEST_SUITE("performance")
{
    TEST_CASE("mean-variance")
    {
        nb::Bench bench;
        bench.performanceCounters(true).minEpochIterations(100);

        std::default_random_engine rng(12345);
        std::uniform_real_distribution<double> dist(0, 1);

        for (size_t n = size_t(1e2); n <= size_t(1e6); n *= 10)
        {
            std::vector<double> xd(n);
            std::vector<float> xf(n);

            std::generate(xd.begin(), xd.end(), [&]() { return dist(rng); });
            std::copy(xd.begin(), xd.end(), xf.begin());

            // double
            VarianceGSL<double>(bench, xd);
            Variance<double>(bench, xd);
            VarianceSIMD<double>(bench, xd);

            // float
            VarianceGSL<float>(bench, xf);
            Variance<float>(bench, xf);
            VarianceSIMD<float>(bench, xf);
        }
    }

    TEST_CASE("correlation")
    {
        constexpr int N = int(11111);

        nb::Bench bench;
        bench.performanceCounters(true).minEpochIterations(100).relative(true);

        std::default_random_engine rng(12345);
        std::uniform_real_distribution<double> dist(0, 1);

        std::vector<double> xd(N);
        std::vector<double> yd(N);
        std::vector<float> xf(N);
        std::vector<float> yf(N);

        std::generate(xd.begin(), xd.end(), [&]() { return dist(rng); });
        std::generate(yd.begin(), yd.end(), [&]() { return dist(rng); });

        std::copy(xd.begin(), xd.end(), xf.begin());
        std::copy(yd.begin(), yd.end(), yf.begin());

        Correlation<double>(bench, xd, yd);
        Correlation<float>(bench, xf, yf);
        CorrelationSIMD<double>(bench, xd, yd);
        CorrelationSIMD<float>(bench, xf, yf);
        CorrelationGSL<double>(bench, xd, yd);
        CorrelationGSL<float>(bench, xf, yf);

        Variance<double>(bench, xd);
        Variance<float>(bench, xf);
        VarianceSIMD<double>(bench, xd);
        VarianceSIMD<float>(bench, xf);
        VarianceGSL<double>(bench, xd);
        VarianceGSL<float>(bench, xf);
    }
}
