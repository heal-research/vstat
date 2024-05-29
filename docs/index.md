# Introduction

_vstat_ is a C++20 library of computationally efficient methods for calculating sample statistics (mean, variance, covariance, correlation), common regression metrics (\f$R^2\f$ score, MSE, MAE) and likelihoods (_Gaussian_ and _Poisson_).

The library has the following features:

- the implementation builds upon the SIMD abstraction layer provided by [E.V.E](https://jfalcou.github.io/eve/) (The Expressive Vector Engine)
- it uses a data-parallel version of the numerically-stable algorithm from [Edward A. Youngs and Elliot M. Cramer](https://www.jstor.org/stable/1267176?seq=1) where the results from independent data partitions are combined with the approach by [_Schubert and Gertz_](https://dl.acm.org/doi/10.1145/3221269.3223036)

## Methodology



## Bibliography

Edward A. Youngs and Elliot M. Cramer, _Some Results Relevant to Choice of Sum and Sum-of-Product Algorithms_, Technometrics Vol. 13, No. 3 (Aug., 1971), pp. 657-665 (9 pages)\n
https://doi.org/10.2307/1267176

Erich Schubert and Michael Gertz, _Numerically stable parallel computation of (co-)variance_, SSDBM '18, Article No. 10, Pages 1–12\n
https://doi.org/10.1145/3221269.3223036