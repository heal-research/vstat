<p align="left">
    <p></p>
    <img src="./logo/vstat.svg" height="80px" />
</p>

### Vectorized statistics using SIMD primitives

---

### Introduction

<img src="./logo/vstat.svg" height="16px" /> is a C++17 library of computationally efficient methods for calculating sample statistics (mean, variance, covariance, correlation). The implementation builds upon the SIMD abstraction layer provided by the Vector Class Library [1] and uses the algorithm of Youngs and Cramer [2] for numerically stable computations of sums and sums-of-squares. The data-parallel methods combine the results of independent data partitions according to the approach by Schubert and Gertz [3]. The methods are validated for correctness against statistical methods from the GNU Scientific Library [4].

### Usage

### Acknowledgements

[1] [Vector Class Library](https://github.com/vectorclass/version2)

[2] [Youngs and Cramer - Some Results Relevant to Choice of Sum and Sum-of-Product Algorithms](https://www.jstor.org/stable/1267176?seq=1)

[3] [Schubert and Gertz - Numerically stable parallel computation of (co-)variance](https://dl.acm.org/doi/10.1145/3221269.3223036)

[4] [GNU Scientific Library](https://www.gnu.org/software/gsl/)
