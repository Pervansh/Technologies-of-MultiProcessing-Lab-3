#include <iostream>
#include <cmath>

#include "omp.h"

#include "PoissonSolvers.h"
#include "PoissonTests.h"

/*
    Макрос для создания правых частей уравнения Гельмгольца с заданным параметром k.
*/
#define GENERATE_RIGHT_PART(name, k) \
    static double name (double x, double y) { \
        using namespace std; \
        return 2. * sin(PI * y) + k * k * (1. - x) * x * sin(PI * y) + PI * PI * (1. - x) * x * sin(PI * y); \
    }

const double PI = std::acos(-1.);

GENERATE_RIGHT_PART(f1, 1.)
GENERATE_RIGHT_PART(f2, 2.)
GENERATE_RIGHT_PART(f20, 20.) // J: n = 10; S: n = 10
GENERATE_RIGHT_PART(f40, 40.) // J: n = 20; S: n = 20
GENERATE_RIGHT_PART(f80, 80.) // J: n = 40; S: n = 40
GENERATE_RIGHT_PART(f200, 200.) // J: n = 100; S: n = 100
GENERATE_RIGHT_PART(f500, 500.) // J: n = 250; S: n = 250
GENERATE_RIGHT_PART(f1000, 1000.) // J: n = 500; S: n = 500
GENERATE_RIGHT_PART(f2000, 2000.) // J: n = 1000; S: n = 1000
GENERATE_RIGHT_PART(f4000, 4000.) // J: n = 2000; S: n = 2000
GENERATE_RIGHT_PART(f8000, 8000.) // J: n = 4000; S: n = 4000

// Правило для выбора k для n: k = 2 * n

/*
    Точное решение задачи Гельмгольца.
    P.S. Решение уравнения Гельмгольца не зависит от параметра k для рассматриваемых
         правых частей.
*/
static double sol(double x, double y) {
    using namespace std;
    return (1. - x) * x * sin(PI * y);
}

int main() {
    // Получение максимального количества потоков
    int max_threads = omp_get_max_threads();
    printf("[INFO] Max threads: %d\n", max_threads);

    //omp_set_num_threads(4);

    int n = 500;
    double h = 1. / n;
    double k = 1.;

    std::vector<int> num_threads = { 1, 2, 4, 8, 12, 16, 18 };
    std::vector<int> n_vec = { /*10, 20,*/ 250, 500, 1000, 2000, 4000 };
    std::vector<PoissonFuncType<double>> f_vec = { /*f20, f40,*/ f500, f1000, f2000, f4000, f8000};
    std::vector<double> k_vec = { /*20., 40.,*/ 500., 1000., 2000., 4000., 8000. };

    /*
    double* A = new double[(n + 1) * (n + 1)];
    for (int i = 0; i < (n + 1) * (n + 1); ++i) {
        A[i] = 0;
    }
    
    // printHelmholtzSolution(std::cout, A, n);

    //jacobyMethodHelmholtzSolve<double>(A, n, 500., f500, h, 1e-15);
    parallelJacobyMethodHelmholtzSolve<double>(A, n, 1000., f1000, h, 1e-15);
    //seidelMethodHelmholtzSolve<double>(A, n, 2000., f2000, h, 1e-15);
    //parallelSeidelMethodHelmholtzSolve<double>(A, n, 1., f1, h, 1e-15);

    //printHelmholtzSolution(std::cout, A, n);
    //printHelmholtzFuncSolution(std::cout, sol, h, n);

    testHelmholtzSolution<double>(std::cout, A, n, sol, h, 1e-3);

    delete[] A;
    */

    // Тесты для метода Якоби
    ///*
    for (auto cur_num_threads : num_threads) {
        for (int i = 0; i < n_vec.size(); ++i) {
            testParallelHelmholtzSolveMethod<double>(
                std::cout,
                cur_num_threads,
                parallelJacobyMethodHelmholtzSolve,
                "parallelJacoby",
                n_vec[i],
                k_vec[i],
                f_vec[i],
                sol,
                1. / n_vec[i],
                1e-3);
        }
    }
    //*/

    // Тесты для метода Зейделя
    ///*
    for (auto cur_num_threads : num_threads) {
        for (int i = 0; i < n_vec.size(); ++i) {
            testParallelHelmholtzSolveMethod<double>(
                std::cout,
                cur_num_threads,
                parallelSeidelMethodHelmholtzSolve,
                "parallelSeidel",
                n_vec[i],
                k_vec[i],
                f_vec[i],
                sol,
                1. / n_vec[i],
                1e-3);
        }
    }
    //*/

    /*
    testParallelHelmholtzSolveMethod<double>(
        std::cout,
        4,
        parallelSeidelMethodHelmholtzSolve,
        "parallelSeidel",
        n,
        k,
        f1,
        sol,
        h,
        1e-3);
    */

    //

    return 0;
}