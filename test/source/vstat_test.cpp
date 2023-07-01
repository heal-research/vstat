// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <iostream>
#include <random>
#include <vector>

#include "vstat/vstat.hpp"
#include "stat_other.hpp"

namespace nb = ankerl::nanobench;
namespace dt = doctest;

namespace uv = vstat::univariate;
namespace bv = vstat::bivariate;

namespace VSTAT_NAMESPACE::test {
namespace util {
    template<std::floating_point T>
    auto generate(auto& rng, int count, T min = T{0}, T max = T{1}) {
        std::vector<T> vec(count);
        std::generate(vec.begin(), vec.end(), [&]() { return std::uniform_real_distribution<T>(min, max)(rng); });
        return vec;
    }
} // namespace util

    template<typename T>
    auto equal(T a, T b, T eps = std::numeric_limits<T>::epsilon()) { return std::abs(a-b) < eps; };

    auto constexpr count_small{10};      // small
    auto constexpr count_medium{1'000};  // medium
    auto constexpr count_large{100'000}; // large

    TEST_CASE("mean" * dt::test_suite("[correctness]")) {
        std::random_device rng{};

        auto test_mean = [&]<typename T = double>(int n, T eps) {
            auto x = util::generate<T>(rng, n);

            auto m1 = stat_other::boost::mean(x); 
            auto m2 = uv::accumulate<T>(x.begin(), x.end()).mean; 

            CAPTURE(m1);
            CAPTURE(m2);
            REQUIRE(equal<T>(m1, m2, eps));
        };

        SUBCASE("double") {
            double const eps{1e-6};
            SUBCASE("small") { test_mean(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_mean(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_mean(count_large, eps); } // NOLINT
        }

        SUBCASE("float") {
            float const eps{1e-5};
            SUBCASE("small") { test_mean.operator()<float>(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_mean.operator()<float>(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_mean.operator()<float>(count_large, eps); } // NOLINT
        }
    }

    TEST_CASE("weighted mean" * dt::test_suite("[correctness]")) {
        std::random_device rng{};

        auto test_mean = [&]<typename T = double>(int n, T eps) {
            auto x = util::generate<T>(rng, n);
            auto w = util::generate<T>(rng, n);

            auto m1 = stat_other::boost::mean(x, w); 
            auto m2 = uv::accumulate<T>(x.begin(), x.end(), w.begin()).mean; 

            CAPTURE(m1);
            CAPTURE(m2);
            REQUIRE(equal<T>(m1, m2, eps));
        };

        SUBCASE("double") {
            double const eps{1e-6};
            SUBCASE("small") { test_mean(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_mean(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_mean(count_large, eps); } // NOLINT
        }

        SUBCASE("float") {
            float const eps{1e-5};
            SUBCASE("small") { test_mean.operator()<float>(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_mean.operator()<float>(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_mean.operator()<float>(count_large, eps); } // NOLINT
        }
    }

    TEST_CASE("variance" * dt::test_suite("[correctness]")) {
        std::random_device rng{};

        auto test_variance = [&]<typename T = double>(int n, T eps) {
            auto x = util::generate<T>(rng, n);
            auto y = util::generate<T>(rng, n);

            auto m1 = stat_other::boost::variance(x, y); 
            auto m2 = uv::accumulate<T>(x.begin(), x.end(), y.begin()).variance; 

            CAPTURE(m1);
            CAPTURE(m2);
            REQUIRE(equal<T>(m1, m2, eps));
        };
    
        SUBCASE("double") {
            double const eps{1e-6};
            SUBCASE("small") { test_variance(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_variance(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_variance(count_large, eps); } // NOLINT
        }

        SUBCASE("float") {
            float const eps{1e-5};
            SUBCASE("small") { test_variance.operator()<float>(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_variance.operator()<float>(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_variance.operator()<float>(count_large, eps); } // NOLINT
        }
    }

    TEST_CASE("weighted variance" * dt::test_suite("[correctness]")) {
        std::random_device rng{};

        auto test_variance = [&]<typename T = double>(int n, T eps) {
            auto x = util::generate<T>(rng, n);
            auto w = util::generate<T>(rng, n);

            auto m1 = stat_other::boost::variance(x, w); 
            auto m2 = uv::accumulate<T>(x.begin(), x.end(), w.begin()).variance; 

            CAPTURE(m1);
            CAPTURE(m2);
            REQUIRE(equal<T>(m1, m2, eps));
        };
    
        SUBCASE("double") {
            double const eps{1e-6};
            SUBCASE("small") { test_variance(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_variance(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_variance(count_large, eps); } // NOLINT
        }

        SUBCASE("float") {
            float const eps{1e-5};
            SUBCASE("small") { test_variance.operator()<float>(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_variance.operator()<float>(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_variance.operator()<float>(count_large, eps); } // NOLINT
        }
    }

    TEST_CASE("covariance" * dt::test_suite("[correctness]")) {
        std::random_device rng{};

        auto test_covariance = [&]<typename T = double>(int n, T eps) {
            auto x = util::generate<T>(rng, n);
            auto y = util::generate<T>(rng, n);

            auto m1 = stat_other::boost::covariance(x, y); 
            auto m2 = bv::accumulate<T>(x.begin(), x.end(), y.begin()).covariance; 

            CAPTURE(m1);
            CAPTURE(m2);
            REQUIRE(equal<T>(m1, m2, eps));
        };
    
        SUBCASE("double") {
            double const eps{1e-6};
            SUBCASE("small") { test_covariance(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_covariance(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_covariance(count_large, eps); } // NOLINT
        }

        SUBCASE("float") {
            float const eps{1e-5};
            SUBCASE("small") { test_covariance.operator()<float>(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_covariance.operator()<float>(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_covariance.operator()<float>(count_large, eps); } // NOLINT
        }
    }

    TEST_CASE("weighted covariance" * dt::test_suite("[correctness]")) {
        std::random_device rng{};

        auto test_covariance = [&]<typename T = double>(int n, T eps) {
            auto x = util::generate<T>(rng, n);
            auto y = util::generate<T>(rng, n);
            auto w = util::generate<T>(rng, n);

            auto m1 = stat_other::boost::covariance(x, y, w); 
            auto m2 = bv::accumulate<T>(x.begin(), x.end(), y.begin(), w.begin()).covariance; 

            CAPTURE(m1);
            CAPTURE(m2);
            REQUIRE(equal<T>(m1, m2, eps));
        };
    
        SUBCASE("double") {
            double const eps{1e-6};
            SUBCASE("small") { test_covariance(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_covariance(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_covariance(count_large, eps); } // NOLINT
        }

        SUBCASE("float") {
            float const eps{1e-5};
            SUBCASE("small") { test_covariance.operator()<float>(count_small, eps); } // NOLINT
            SUBCASE("medium") { test_covariance.operator()<float>(count_medium, eps); } // NOLINT
            SUBCASE("large") { test_covariance.operator()<float>(count_large, eps); } // NOLINT
        }
    }

    TEST_CASE("benchmarks" * dt::test_suite("[performance]")) {
        std::random_device rng{};

        nb::Bench bench;
        auto const n{1000};
        for (auto s = n; s <= 1'024'000; s *= 2) {
            // mean & weighted mean
            auto xd = util::generate<double>(rng, n);
            auto yd = util::generate<double>(rng, n);
            auto wd = util::generate<double>(rng, n);

            auto xf = util::generate<float>(rng, n);
            auto yf = util::generate<float>(rng, n);
            auto wf = util::generate<float>(rng, n);

            double m{0.0};

            bench.batch(s).run("vstat;mean;double", [&]() {
                m += uv::accumulate<double>(xd.begin(), xd.end()).mean;
            });

            bench.batch(s).run("vstat;weighted mean;double", [&]() {
                m += uv::accumulate<double>(xd.begin(), xd.end(), wd.begin()).mean;
            });

            bench.batch(s).run("vstat;variance;double", [&]() {
                m += uv::accumulate<double>(xd.begin(), xd.end()).variance;
            });

            bench.batch(s).run("vstat;weighted variance;double", [&]() {
                m += uv::accumulate<double>(xd.begin(), xd.end(), wd.begin()).variance;
            });

            bench.batch(s).run("vstat;covariance;double", [&]() {
                m += bv::accumulate<double>(xd.begin(), xd.end(), yd.begin()).covariance;
            });

            bench.batch(s).run("vstat;weighted covariance;double", [&]() {
                m += bv::accumulate<double>(xd.begin(), xd.end(), yd.begin(), wd.begin()).covariance;
            });

            bench.batch(s).run("boost.accu;mean;double", [&]() {
                m += stat_other::boost::mean(xd); 
            });

            bench.batch(s).run("boost.accu;weighted mean;double", [&]() {
                m += stat_other::boost::mean(xd, wd); 
            });

            bench.batch(s).run("boost.accu;variance;double", [&]() {
                m += stat_other::boost::variance(xd); 
            });

            bench.batch(s).run("boost.accu;weighted variance;double", [&]() {
                m += stat_other::boost::variance(xd, wd); 
            });

            bench.batch(s).run("boost.accu;covariance;double", [&]() {
                m += stat_other::boost::covariance(xd, yd); 
            });

            bench.batch(s).run("boost.accu;weighted covariance;double", [&]() {
                m += stat_other::boost::covariance(xd, yd, wd); 
            });

            bench.batch(s).run("boost.math;mean;double", [&]() {
                m += boost::math::statistics::mean(xd); 
            });

            bench.batch(s).run("boost.math;variance;double", [&]() {
                m += boost::math::statistics::variance(xd); 
            });

            bench.batch(s).run("boost.math;covariance;double", [&]() {
                m += boost::math::statistics::covariance(xd, yd); 
            });

            bench.batch(s).run("gsl;mean;double", [&]() {
                m += stat_other::gsl::mean(xd); 
            });

            bench.batch(s).run("gsl;variance;double", [&]() {
                m += stat_other::gsl::variance(xd); 
            });

            bench.batch(s).run("gsl;covariance;double", [&]() {
                m += stat_other::gsl::covariance(xd, yd); 
            });

            bench.batch(s).run("linasm;mean;double", [&]() {
                m += stat_other::linasm::variance(xd); 
            });

            bench.batch(s).run("linasm;variance;double", [&]() {
                m += stat_other::linasm::variance(xd); 
            });

            bench.batch(s).run("linasm;covariance;double", [&]() {
                m += stat_other::linasm::covariance(xd, yd); 
            });

            bench.batch(s).run("vstat;mean;float", [&]() {
                m += uv::accumulate<float>(xf.begin(), xf.end()).mean;
            });

            bench.batch(s).run("vstat;weighted mean;float", [&]() {
                m += uv::accumulate<float>(xf.begin(), xf.end(), wf.begin()).mean;
            });

            bench.batch(s).run("vstat;variance;float", [&]() {
                m += uv::accumulate<float>(xf.begin(), xf.end()).variance;
            });

            bench.batch(s).run("vstat;weighted variance;float", [&]() {
                m += uv::accumulate<float>(xf.begin(), xf.end(), wf.begin()).variance;
            });

            bench.batch(s).run("vstat;covariance;float", [&]() {
                m += bv::accumulate<float>(xf.begin(), xf.end(), yf.begin()).covariance;
            });

            bench.batch(s).run("vstat;weighted covariance;float", [&]() {
                m += bv::accumulate<float>(xf.begin(), xf.end(), yf.begin(), wf.begin()).covariance;
            });

            bench.batch(s).run("boost.accu;mean;float", [&]() {
                m += stat_other::boost::mean(xf); 
            });

            bench.batch(s).run("boost.accu;weighted mean;float", [&]() {
                m += stat_other::boost::mean(xf, wf); 
            });

            bench.batch(s).run("boost.accu;variance;float", [&]() {
                m += stat_other::boost::variance(xf); 
            });

            bench.batch(s).run("boost.accu;weighted variance;float", [&]() {
                m += stat_other::boost::variance(xf, wf); 
            });

            bench.batch(s).run("boost.accu;covariance;float", [&]() {
                m += stat_other::boost::covariance(xf, yf); 
            });

            bench.batch(s).run("boost.accu;weighted covariance;float", [&]() {
                m += stat_other::boost::covariance(xf, yf, wf); 
            });

            bench.batch(s).run("boost.math;mean;float", [&]() {
                m += boost::math::statistics::mean(xf); 
            });

            bench.batch(s).run("boost.math;variance;float", [&]() {
                m += boost::math::statistics::variance(xf); 
            });

            bench.batch(s).run("boost.math;covariance;float", [&]() {
                m += boost::math::statistics::covariance(xf, yf); 
            });

            bench.batch(s).run("gsl;mean;float", [&]() {
                m += stat_other::gsl::mean(xf); 
            });

            bench.batch(s).run("gsl;variance;float", [&]() {
                m += stat_other::gsl::variance(xf); 
            });

            bench.batch(s).run("gsl;covariance;float", [&]() {
                m += stat_other::gsl::covariance(xf, yf); 
            });

            bench.batch(s).run("linasm;mean;float", [&]() {
                m += stat_other::linasm::variance(xf); 
            });

            bench.batch(s).run("linasm;variance;float", [&]() {
                m += stat_other::linasm::variance(xf); 
            });

            bench.batch(s).run("linasm;covariance;float", [&]() {
                m += stat_other::linasm::covariance(xf, yf); 
            });
        }

        bench.render(nb::templates::csv(), std::cout);
        // format this result like this:
        // ./build/test/vstat_test | grep 'benchmark";' | tr -d '"' | column -t -s';' -o','
    }
} // namespace VSTAT_NAMESPACE::test
