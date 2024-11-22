#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>

#include "mpi.h"

template<typename T>
void printHelmholtzSolution(std::ostream& out, const T* A, int n);

template <typename T>
void testHelmholtzSolution(std::ostream& out, const T* A, int n, PoissonFuncType<T> sol, T h);

#include "PoissonSolvers.h"

template<typename T>
void printHelmholtzSolution(std::ostream& out, const T* A, int n) {
    out << std::endl << "[INFO]: Mesh solution data:\n";
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            out << A[Q_IND(i, j, n + 1)] << ' ';
        }
        out << '\n';
    }
    out << std::endl;
}

template<typename T>
void printHelmholtzFuncSolution(std::ostream& out, PoissonFuncType<T> sol, T h, int n) {
    out << std::endl << "[INFO]: Projected on mesh solution data:\n";
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            out << sol(j * h, i * h) << ' ';
        }
        out << '\n';
    }
    out << std::endl;
}

template <typename T>
void testHelmholtzSolution(std::ostream& out, const T* A, int n, PoissonFuncType<T> sol, T h, T maxError) {
    out << std::endl << "-------------------------\n" << "[INFO]: Test section.\n";
    
    T norm = 0;

    // Расчкт нормы C
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            norm = std::max(norm, std::fabs(A[Q_IND(i, j, n + 1)] - sol(j * h, i * h)));
        }
    }

    out << "[INFO]: difference norm: " << norm << '\n';

    if (norm < maxError) {
        out << "[INFO]: Test is passed successfully!\n";
    } else {
        out << "[INFO]: Test isn't passed. Wrong mesh solution!\n";
    }

    out << "-------------------------" << std::endl;
}

template <typename T>
void testParallelHelmholtzSolveMethod(
        std::ostream& out,
        int numThreads,
        HelmholtzSolverType<T> solver,
        std::string solverName,
        int n,
        T k,
        PoissonFuncType<T> f,
        PoissonFuncType<T> sol,
        T h,
        T maxError)
{
    out << std::endl << "-------------------------\n" << "[INFO]: Test section for [" << solverName << "].\n";

    out << "num_threads: " << numThreads << "\n";
    out << "n: " << n << "\n";

    out << "h: " << h << "\n";
    out << "k: " << k << "\n";

    omp_set_num_threads(numThreads);

    double* A = new double[(n + 1) * (n + 1)];

    out << "A_ptr: " << A << "\n";

#pragma omp parallel for
    for (int i = 0; i < (n + 1) * (n + 1); ++i) {
        A[i] = 0;
    }

    auto time = solver(A, n, k, f, h, 1e-15);

    T norm = 0;

    // printHelmholtzSolution(std::cout, A, n);

    // Расчкт нормы C
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            norm = std::max(norm, std::fabs(A[Q_IND(i, j, n + 1)] - sol(j * h, i * h)));
        }
    }

    out << "[INFO]: time: " << time << '\n';
    out << "[INFO]: difference norm: " << norm << '\n';

    if (norm < maxError) {
        out << "[INFO]: Test is passed successfully!\n";
    } else {
        out << "[INFO]: Test isn't passed. Wrong mesh solution!\n";
    }

    out << "-------------------------" << std::endl;

    delete[] A;
}

