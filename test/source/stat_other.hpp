#ifndef VSTAT_STAT_OTHER_HPP
#define VSTAT_STAT_OTHER_HPP

#include <iterator>
#include <vector>

// boost.accumulator
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

// boost.math toolkit
#include <boost/math/statistics/univariate_statistics.hpp>
#include <boost/math/statistics/bivariate_statistics.hpp>

// GNU GSL
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics_float.h>

// Linasm
#include <Statistics.h>

namespace stat_other {
    template<typename T>
    using vec = std::vector<T>;
    using vec_d = vec<double>;
    using vec_f = vec<float>;

    namespace gsl {
        // double variants
        inline auto mean(vec_d const& x) {
            return gsl_stats_mean(x.data(), 1UL, x.size());
        }

        inline auto mean(vec_d const& x, vec_d const& w) {
            return gsl_stats_wmean(w.data(), 1UL, x.data(), 1UL, x.size());
        }

        inline auto sum(vec_d const& x) {
            return static_cast<double>(x.size()) * mean(x);
        }

        inline auto sum(vec_d const& x, vec_d const& w) {
            return static_cast<double>(w.size()) * mean(w) * mean(x, w);
        }

        inline auto variance(vec_d const& x) {
            return gsl_stats_variance(x.data(), 1UL, x.size());
        }

        inline auto variance(vec_d const& x, vec_d const& w) {
            return gsl_stats_wvariance(w.data(), 1UL, x.data(), 1UL, x.size());
        }

        inline auto covariance(vec_d const& x, vec_d const& y) {
            return gsl_stats_covariance(x.data(), 1UL, y.data(), 1UL, x.size());
        }

        inline auto covariance(vec_d const& /*unused*/, vec_d const& /*unused*/, vec_d const& /*unused*/) {
            throw std::runtime_error("not supported");
        }

        inline auto correlation(vec_d const& x, vec_d const& y) {
            return gsl_stats_correlation(x.data(), 1UL, y.data(), 1UL, x.size());
        }

        inline auto correlation(vec_d const& /*unused*/, vec_d const& /*unused*/, vec_d const& /*unused*/) {
            throw std::runtime_error("not supported");
        }

        // float variants
        inline auto mean(vec_f const& x) {
            return gsl_stats_float_mean(x.data(), 1UL, x.size());
        }

        inline auto mean(vec_f const& x, vec_f const& w) {
            return gsl_stats_float_wmean(w.data(), 1UL, x.data(), 1UL, x.size());
        }

        inline auto sum(vec_f const& x) {
            return static_cast<double>(x.size()) * mean(x);
        }

        inline auto sum(vec_f const& x, vec_f const& w) {
            return static_cast<double>(w.size()) * mean(w) * mean(x, w);
        }

        inline auto variance(vec_f const& x) {
            return gsl_stats_float_variance(x.data(), 1UL, x.size());
        }

        inline auto variance(vec_f const& x, vec_f const& w) {
            return gsl_stats_float_wvariance(w.data(), 1UL, x.data(), 1UL, x.size());
        }

        inline auto covariance(vec_f const& x, vec_f const& y) {
            return gsl_stats_float_covariance(x.data(), 1UL, y.data(), 1UL, x.size());
        }

        inline auto covariance(vec_f const& /*unused*/, vec_f const& /*unused*/, vec_f const& /*unused*/) {
            throw std::runtime_error("not supported");
        }

        inline auto correlation(vec_f const& x, vec_f const& y) {
            return gsl_stats_float_correlation(x.data(), 1UL, y.data(), 1UL, x.size());
        }

        inline auto correlation(vec_f const& /*unused*/, vec_f const& /*unused*/, vec_f const& /*unused*/) {
            throw std::runtime_error("not supported");
        }
    } // namespace gsl

    namespace boost {
        namespace ba = ::boost::accumulators;

        template<typename T>
        inline auto mean(vec<T> const& x) {
            ba::accumulator_set<T, ba::stats<ba::tag::mean>> acc;
            for (auto i = 0; i < std::ssize(x); ++i) { acc(x[i]); }
            return ba::mean(acc);
        }

        template<typename T>
        inline auto mean(vec<T> const& x, vec<T> const& w) {
            ba::accumulator_set<T, ba::stats<ba::tag::weighted_mean>, T> acc;
            for (auto i = 0; i < std::ssize(x); ++i) { acc(x[i], ba::weight=w[i]); }
            return ba::weighted_mean(acc);
        }

        template<typename T>
        inline auto variance(vec<T> const& x) {
            ba::accumulator_set<T, ba::stats<ba::tag::variance>> acc;
            for (auto i = 0; i < std::ssize(x); ++i) { acc(x[i]); }
            return ba::variance(acc);
        }

        template<typename T>
        inline auto variance(vec<T> const& x, vec<T> const& w) {
            ba::accumulator_set<T, ba::stats<ba::tag::weighted_variance>, T> acc;
            for (auto i = 0; i < std::ssize(x); ++i) { acc(x[i], ba::weight=w[i]); }
            return ba::weighted_variance(acc);
        }

        template<typename T>
        inline auto covariance(vec<T> const& x, vec<T> const& y) {
            ba::accumulator_set<T, ba::stats<ba::tag::covariance<T, ba::tag::covariate1>>> acc;
            for (auto i = 0; i < std::ssize(x); ++i) { acc(x[i], ba::covariate1=y[i]); }
            return ba::covariance(acc);
        }

        template<typename T>
        inline auto covariance(vec<T> const& x, vec<T> const& y, vec<T> const& w) {
            ba::accumulator_set<T, ba::stats<ba::tag::covariance<T, ba::tag::covariate1>>, T> acc;
            for (auto i = 0; i < std::ssize(x); ++i) { acc(x[i], ba::weight=w[i], ba::covariate1=y[i]); }
            return ba::weighted_covariance(acc);
        }
    } // namespace boost

    namespace linasm {
        template<typename T>
        inline auto mean(vec<T> const& x) {
            return Statistics::Mean(x.data(), x.size());
        }

        template<typename T>
        inline auto mean(vec<T> const& x, vec<T> const& w) {
            throw std::runtime_error("not supported"); 
        }

        template<typename T>
        inline auto sum(vec<T> const& x) {
            return static_cast<double>(x.size()) * mean(x);
        }

        template<typename T>
        inline auto sum(vec<T> const& x, vec<T> const& w) {
            return static_cast<double>(w.size()) * mean(w) * mean(x, w);
        }

        template<typename T>
        inline auto variance(vec<T> const& x) {
            return Statistics::Variance(x.data(), x.size(), mean(x));
        }

        template<typename T>
        inline auto variance(vec<T> const& x, vec<T> const& w) {
            throw std::runtime_error("not supported"); 
        }

        template<typename T>
        inline auto covariance(vec<T> const& x, vec<T> const& y) {
            return Statistics::Covariance(x.data(), y.data(), x.size(), mean(x), mean(y));
        }

        template<typename T>
        inline auto covariance(vec<T> const& /*unused*/, vec<T> const& /*unused*/, vec<T> const& /*unused*/) {
            throw std::runtime_error("not supported");
        }

        template<typename T>
        inline auto correlation(vec<T> const& x, vec<T> const& y) {
            return Statistics::PearsonCorrelation(x.data(), y.data(), x.size(), mean(x), mean(y));
        }

        template<typename T>
        inline auto correlation(vec<T> const& /*unused*/, vec<T> const& /*unused*/, vec<T> const& /*unused*/) {
            throw std::runtime_error("not supported");
        }
    } // namespace linasm
} // namespace stat_other

#endif
