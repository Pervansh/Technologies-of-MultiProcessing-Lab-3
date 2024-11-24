#pragma once

#include <algorithm>
#include <cstring>
#include <string.h>
#include <chrono>
#include <vector>

#include "mpi.h"

#include "PoissonSolvers.h"

/*
    Схема получения правой части slave-ом
    Получаем номер правой части
    По номеру получаем правую часть из контейнера, структуры и тп (функтор)
*/

using MpiHelmholtzSolverType = double (*)(MPI_Comm, double*, int, double, PoissonFuncType<double>, double, const double);

template <typename FuncSpace>
PoissonFuncType<double> mpiPoissonFuncBcast(MPI_Comm comm, int master, bool* successStatus, int funcId = -1) {
    int myid;
    MPI_Comm_rank(comm, &myid);

    /*
        Имя типа FuncSpace(может различаться для разных компиляторов).
        Освобождать память не нужно (для большинства компиляторов)
    */
    const char* myFuncSpaceName = typeid(FuncSpace).name();
    int myFuncSpaceNameLen = std::strlen(myFuncSpaceName);

    const int maxStrSize = 256;
    char* masterFuncSpaceName = new char[maxStrSize];
    int masterFuncSpaceNameLen;

    if (myid == master) {
        masterFuncSpaceNameLen = myFuncSpaceNameLen;
        // strcpy_s(masterFuncSpaceName, maxStrSize, myFuncSpaceName); // не работает на кластере (безопасный способ)
        std::strcpy(masterFuncSpaceName, myFuncSpaceName); // не безопасный способ (может не хватить maxStrSize)
    }

    /*
        Прием от мастера информации об используемом пространстве функций для проверки
        "однородности" данных об используемых функциях
    */
    MPI_Bcast(&masterFuncSpaceNameLen, 1, MPI_INT, master, comm);
    MPI_Bcast(masterFuncSpaceName, masterFuncSpaceNameLen + 1, MPI_CHAR, master, comm); // Вещание C-строки

    //std::clog << "[PROCESS " << myid << " DEBUG]: mine FuncSpace: " << std::string(myFuncSpaceName) <<
    //    ", master's FuncSpace: " << std::string(masterFuncSpaceName) << '\n';

    bool isFuncSpaceHomogenious = !(bool)std::strcmp(myFuncSpaceName, masterFuncSpaceName);

    delete[] masterFuncSpaceName;

    // Прием согласовоности об использовании одного и того же FuncSpace во всех узлах
    MPI_Allreduce(&isFuncSpaceHomogenious, successStatus, 1, MPI_C_BOOL, MPI_LAND, comm);

    MPI_Bcast(&funcId, 1, MPI_INT, master, comm);

    if (funcId < 0) *successStatus = false;

    if (myid == master && funcId < 0) {
        std::cerr << "[PROCESS " << myid << " ERROR]: Bad master PoissonFunc transmission of funcId (must be \">= 0\")\n";
    }

    if (*successStatus) {
        return FuncSpace::funcs[funcId];
    }

    if (funcId < 0) {
        std::cerr << "[PROCESS " << myid << " ERROR]: Bad funcId (must be \">= 0\")\n";
    } else {
        std::cerr << "[PROCESS " << myid << " ERROR]: FuncSpace difference (must be equal for all processes)\n";
    }

    return nullptr;
}

/*
    Требование к FuncSpace: наличие контейнера FuncSpace::funcs с
    оператором FuncSpace::funcs[int] -> PoissonFuncType<double>
*/
template <typename FuncSpace>
[[deprecated]]
double mpiMasterHelmholtzSolve(MPI_Comm comm, double const* A, int n, int funcId) {

    for (int i = 0; i < (n + 1) * (n + 1); ++i) {
        A[i] = 0;
    }
}

/*
    Решение методом Якоби с применением MPI шаблоном "крест" уравнения Гельмгольца с коэффициентом k
    и правой частью f для случая граничных условий 1-го рода в квадратной области.
    A - указатель на матрицу размера (n + 1)*(n + 1), хранящую в граничных элементах значения ГУ и
        во внутренних элементах - начальные значения решения внутри области для метода Якоби,
        в который будет перезаписан результат работы метода, y - строки, x - столбцы;
    n - размерность сетки по x- и y-координате (количество ячеек);
    k - коэффициент (точнее его корень) при линейном члене уравнения;
    f - функция правой части, f = f(x, y);
    h - шаг сетки.
    Нумерация узлов (внутренних): естественная (построчная).
    Распределение узлов по процессам: i-ая строка матрицы - (i mod comm_size)-му процессу
*/
[[deprecated]]
double mpiJacobyMethodHelmholtzSolve(
    MPI_Comm comm,
    double* A,
    int n,
    double k,
    PoissonFuncType<double> f,
    double h,
    const double minDiscrepancy)
{
    int myid;     // номер текущего процесса
    int numprocs; // количество процессов в коммуникаторе

    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    /*
        Распределение строк по процессам:
        0, 1, 2, ..., numprocs - 1, numprocs, 0, 1, ..., firstResiduProcId - 1,
        где
        0, 1, ..., numprocs - основной период владения (mainBlock).
        Заметим, что все процессы до firstResiduProcId владеет (mainBlocksCnt + 1) строками,
        а после firstResiduProcId --- mainBlocksCnt строками.
    */

    // количество периодов владения строк процессами
    int mainBlocksCnt = (n + 1) / numprocs;
    // номер первого процесса, начиная с которого процесс владеет mainBlocksCnt строками
    int firstResiduProcId = (n + 1) % numprocs;

    // количество строк, которыми владеет текущий процесс
    int rowsCnt = mainBlocksCnt + (myid < firstResiduProcId);


    double* newY = new double[(n + 1) * (n + 1)]; // Значения на следующей итерации
    double* oldY = A;            // Значения на предыдущей итерации

    //printHelmholtzSolution(std::cout, A, n);
    //printHelmholtzSolution(std::cout, newY, n);

    double coef = 1. / (4. + k * k * h * h);

    auto start_time = std::chrono::high_resolution_clock::now();

    // запись ГУ в newY
    for (int i = 0; i <= n - 1; ++i) {
        // обработка элементов на горизонтальных границах
        newY[Q_IND(0, i, n + 1)] = A[Q_IND(0, i, n + 1)]; // верхний край
        newY[Q_IND(n, n - i, n + 1)] = A[Q_IND(n, n - i, n + 1)]; // нижний край
        // обработка элементов на вертикальных границах
        newY[Q_IND(n - i, 0, n + 1)] = A[Q_IND(n - i, 0, n + 1)]; // левый край
        newY[Q_IND(i, n, n + 1)] = A[Q_IND(i, n, n + 1)]; // правый край

        /*
        auto val1 = A[Q_IND(0, i, n + 1)];
        auto val2 = A[Q_IND(n, n - i, n + 1)];
        auto val3 = A[Q_IND(n - i, 0, n + 1)];
        auto val4 = A[Q_IND(i, n, n + 1)];
        auto val = 0;
        */
    }

    // printHelmholtzSolution(std::cout, newY, n);

    double discrepancy = double(1e18);

    int iterCount = 0;
    for (; iterCount < MAX_ITER_COUNT && discrepancy >= minDiscrepancy; iterCount++) {
        // printHelmholtzSolution(std::cout, A, n);

        // обработка элементов внутри области
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                newY[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, oldY, n, i, j);
            }
        }

        // расчет текущей невязки
        discrepancy = double(0);
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                discrepancy = std::max(discrepancy,
                    std::fabs(newY[Q_IND(i, j, n + 1)] - H_SOLVE_JACOBY_CALC_4(coef, f, h, newY, n, i, j)));
            }
        }

        // printHelmholtzSolution(std::cout, newY, n);

        // меняем указатели на массивы местами (теперь в oldY хранится результат текущей итерации)
        std::swap(newY, oldY);
    }
    // Окончательный результат хранится по адресу oldY

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // std::cout << "is okay? : " << (A == oldY) << std::endl;

    // Убеждаемся, что по адресу A будет записан результат метода
    if (newY == A) {
        // std::memcpy(A, oldY, sizeof(T) * (n + 1) * (n + 1)); // Не работает, нет времени разбираться + не параллелится

        for (int i = 0; i < (n + 1) * (n + 1); ++i) {
            A[i] = oldY[i];
        }

        // Освобождение дополнительно выделенной памяти
        delete[] oldY;
    } else {

        // Освобождение дополнительно выделенной памяти
        delete[] newY;
    }

    std::cout << "[DEBUG]: iterCount = " << iterCount << std::endl;

    //return iterCount >= MAX_ITER_COUNT;
    return elapsed.count();
}

/*
    Решение методом Зейделя с применением MPI шаблоном "крест" уравнения Гельмгольца с коэффициентом k
    и правой частью f для случая граничных условий 1-го рода в квадратной области.
    A - указатель на матрицу размера (n + 1)*(n + 1), хранящую в граничных элементах значения ГУ и
        во внутренних элементах - начальные значения решения внутри области для метода Зейделя,
        в который будет перезаписан результат работы метода, y - строки, x - столбцы;
    n - размерность сетки по x- и y-координате (количество ячеек);
    k - коэффициент (точнее его корень) при линейном члене уравнения;
    f - функция правой части, f = f(x, y);
    h - шаг сетки.
    Нумерация узлов (внутренних): красно-черная, узел (1, 1) - красный.
*/
[[deprecated]]
double mpiSeidelMethodHelmholtzSolve(
    MPI_Comm comm, 
    double* A, 
    int n, 
    double k, 
    PoissonFuncType<double> f, 
    double h, 
    const double minDiscrepancy) 
{
    double coef = 1. / (4. + k * k * h * h);

    double discrepancy = double(1e18);

    auto start_time = std::chrono::high_resolution_clock::now();

    int iterCount = 0;
    for (; iterCount < MAX_ITER_COUNT && discrepancy >= minDiscrepancy; iterCount++) {
        // обработка красных элементов внутри области
        for (int i = 1; i <= n - 1; ++i) { // проходимся по слоям
            for (int j = 2 - i % 2; j <= n - 1; j += 2) { // переступаем черные узлы
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // обработка черных элементов внутри области
        for (int i = 1; i <= n - 1; ++i) { // проходимся по слоям
            for (int j = 1 + i % 2; j <= n - 1; j += 2) { // переступаем красные узлы
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // расчет текущей невязки
        discrepancy = double(0);
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                discrepancy = std::max(discrepancy,
                    std::fabs(A[Q_IND(i, j, n + 1)] - H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j)));
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "[DEBUG]: iterCount = " << iterCount << std::endl;

    //return iterCount >= MAX_ITER_COUNT;
    return elapsed.count();
}

// Статический класс для доступа к списку MPI-решателей по id
class MpiHelmholtzSolvers {
public:
    enum Ids : int {
        JACOBY_METHOD,
        SEIDEL_METHOD,
        ENUM_SIZE
    };

    using ArrayType = std::vector<MpiHelmholtzSolverType>;

private:
    static ArrayType solvers;

public:
    static MpiHelmholtzSolverType getSolver(Ids id) {
        if (solvers.empty()) {
            solvers.resize(Ids::ENUM_SIZE);

            solvers[Ids::JACOBY_METHOD] = mpiJacobyMethodHelmholtzSolve;
            solvers[Ids::SEIDEL_METHOD] = mpiSeidelMethodHelmholtzSolve;
        }

        return solvers[id];
    }
};
