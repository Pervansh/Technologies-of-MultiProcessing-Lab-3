#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <array>

#include "omp.h"
#include "mpi.h"

#include "PoissonSolvers.h"
#include "PoissonTests.h"
#include "MpiPoissonSolvers.h"

/*
    Макрос для создания правых частей уравнения Гельмгольца с заданным параметром k.
*/
#define GENERATE_RIGHT_PART(name, k) \
    static double name (double x, double y) { \
        using namespace std; \
        return 2. * sin(PI * y) + k * k * (1. - x) * x * sin(PI * y) + PI * PI * (1. - x) * x * sin(PI * y); \
    }

#define MASTER_ID 0 // Номеп управляющего процесса

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

// Структура, реализующая "пространство" функций для синхранизации данных между мастером и слейвами
struct HelmholtzRightPartSpace {
    static const std::array<PoissonFuncType<double>, 11> funcs;
};

const std::array<PoissonFuncType<double>, 11> HelmholtzRightPartSpace::funcs = {
    f1, f2, f20, f40, f80, f200, f500, f1000, f2000, f4000, f8000
};

//struct DummyFuncSpace { static PoissonFuncType<double> funcs[]; };
//PoissonFuncType<double> DummyFuncSpace::funcs[] = {f1};

void masterProcess(MPI_Comm comm) {
    int myid;
    MPI_Comm_rank(comm, &myid);

    // unit-тест для случая 4-х процессов
    int n = 100;
    double h = 1. / n;
    double k = 200.;

    std::unique_ptr<double[]> A = std::make_unique<double[]>((n + 1) * (n + 1));
    for (int i = 0; i < (n + 1) * (n + 1); ++i) {
        A[i] = i;
    }

    //mpiHelmholtzJacobyMethodSolve<HelmholtzRightPartSpace>(comm, myid, A.get(), n, 1);
    mpiJacobyMethodHelmholtzSolve<HelmholtzRightPartSpace>(comm, myid, A.get(), n, k, 5, h, 1e-9);

    testHelmholtzSolution<double>(std::cout, A.get(), n, sol, h, 1e-6);
}

void slaveProcess(MPI_Comm comm) {
    int myid;
    MPI_Comm_rank(comm, &myid);

    mpiJacobyMethodHelmholtzSolve<HelmholtzRightPartSpace>(comm, MASTER_ID);
}

int main(int argc, char** argv) {
    int myid, numprocs;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    printf("Process %d of %d on %s\n", myid, numprocs, processor_name);

    if (myid == MASTER_ID) {
        masterProcess(MPI_COMM_WORLD);
    } else {
        slaveProcess(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}