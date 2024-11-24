#pragma once

#include <algorithm>
#include <cstring>
#include <string.h>
#include <chrono>
#include <vector>

#include "mpi.h"

#include "PoissonSolvers.h"

/*
    ����� ��������� ������ ����� slave-��
    �������� ����� ������ �����
    �� ������ �������� ������ ����� �� ����������, ��������� � �� (�������)
*/

using MpiHelmholtzSolverType = double (*)(MPI_Comm, double*, int, double, PoissonFuncType<double>, double, const double);

template <typename FuncSpace>
PoissonFuncType<double> mpiPoissonFuncBcast(MPI_Comm comm, int master, bool* successStatus, int funcId = -1) {
    int myid;
    MPI_Comm_rank(comm, &myid);

    /*
        ��� ���� FuncSpace(����� ����������� ��� ������ ������������).
        ����������� ������ �� ����� (��� ����������� ������������)
    */
    const char* myFuncSpaceName = typeid(FuncSpace).name();
    int myFuncSpaceNameLen = std::strlen(myFuncSpaceName);

    const int maxStrSize = 256;
    char* masterFuncSpaceName = new char[maxStrSize];
    int masterFuncSpaceNameLen;

    if (myid == master) {
        masterFuncSpaceNameLen = myFuncSpaceNameLen;
        // strcpy_s(masterFuncSpaceName, maxStrSize, myFuncSpaceName); // �� �������� �� �������� (���������� ������)
        std::strcpy(masterFuncSpaceName, myFuncSpaceName); // �� ���������� ������ (����� �� ������� maxStrSize)
    }

    /*
        ����� �� ������� ���������� �� ������������ ������������ ������� ��� ��������
        "������������" ������ �� ������������ ��������
    */
    MPI_Bcast(&masterFuncSpaceNameLen, 1, MPI_INT, master, comm);
    MPI_Bcast(masterFuncSpaceName, masterFuncSpaceNameLen + 1, MPI_CHAR, master, comm); // ������� C-������

    //std::clog << "[PROCESS " << myid << " DEBUG]: mine FuncSpace: " << std::string(myFuncSpaceName) <<
    //    ", master's FuncSpace: " << std::string(masterFuncSpaceName) << '\n';

    bool isFuncSpaceHomogenious = !(bool)std::strcmp(myFuncSpaceName, masterFuncSpaceName);

    delete[] masterFuncSpaceName;

    // ����� �������������� �� ������������� ������ � ���� �� FuncSpace �� ���� �����
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
    ���������� � FuncSpace: ������� ���������� FuncSpace::funcs �
    ���������� FuncSpace::funcs[int] -> PoissonFuncType<double>
*/
template <typename FuncSpace>
[[deprecated]]
double mpiMasterHelmholtzSolve(MPI_Comm comm, double const* A, int n, int funcId) {

    for (int i = 0; i < (n + 1) * (n + 1); ++i) {
        A[i] = 0;
    }
}

/*
    ������� ������� ����� � ����������� MPI �������� "�����" ��������� ����������� � ������������� k
    � ������ ������ f ��� ������ ��������� ������� 1-�� ���� � ���������� �������.
    A - ��������� �� ������� ������� (n + 1)*(n + 1), �������� � ��������� ��������� �������� �� �
        �� ���������� ��������� - ��������� �������� ������� ������ ������� ��� ������ �����,
        � ������� ����� ����������� ��������� ������ ������, y - ������, x - �������;
    n - ����������� ����� �� x- � y-���������� (���������� �����);
    k - ����������� (������ ��� ������) ��� �������� ����� ���������;
    f - ������� ������ �����, f = f(x, y);
    h - ��� �����.
    ��������� ����� (����������): ������������ (����������).
    ������������� ����� �� ���������: i-�� ������ ������� - (i mod comm_size)-�� ��������
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
    int myid;     // ����� �������� ��������
    int numprocs; // ���������� ��������� � �������������

    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    /*
        ������������� ����� �� ���������:
        0, 1, 2, ..., numprocs - 1, numprocs, 0, 1, ..., firstResiduProcId - 1,
        ���
        0, 1, ..., numprocs - �������� ������ �������� (mainBlock).
        �������, ��� ��� �������� �� firstResiduProcId ������� (mainBlocksCnt + 1) ��������,
        � ����� firstResiduProcId --- mainBlocksCnt ��������.
    */

    // ���������� �������� �������� ����� ����������
    int mainBlocksCnt = (n + 1) / numprocs;
    // ����� ������� ��������, ������� � �������� ������� ������� mainBlocksCnt ��������
    int firstResiduProcId = (n + 1) % numprocs;

    // ���������� �����, �������� ������� ������� �������
    int rowsCnt = mainBlocksCnt + (myid < firstResiduProcId);


    double* newY = new double[(n + 1) * (n + 1)]; // �������� �� ��������� ��������
    double* oldY = A;            // �������� �� ���������� ��������

    //printHelmholtzSolution(std::cout, A, n);
    //printHelmholtzSolution(std::cout, newY, n);

    double coef = 1. / (4. + k * k * h * h);

    auto start_time = std::chrono::high_resolution_clock::now();

    // ������ �� � newY
    for (int i = 0; i <= n - 1; ++i) {
        // ��������� ��������� �� �������������� ��������
        newY[Q_IND(0, i, n + 1)] = A[Q_IND(0, i, n + 1)]; // ������� ����
        newY[Q_IND(n, n - i, n + 1)] = A[Q_IND(n, n - i, n + 1)]; // ������ ����
        // ��������� ��������� �� ������������ ��������
        newY[Q_IND(n - i, 0, n + 1)] = A[Q_IND(n - i, 0, n + 1)]; // ����� ����
        newY[Q_IND(i, n, n + 1)] = A[Q_IND(i, n, n + 1)]; // ������ ����

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

        // ��������� ��������� ������ �������
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                newY[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, oldY, n, i, j);
            }
        }

        // ������ ������� �������
        discrepancy = double(0);
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                discrepancy = std::max(discrepancy,
                    std::fabs(newY[Q_IND(i, j, n + 1)] - H_SOLVE_JACOBY_CALC_4(coef, f, h, newY, n, i, j)));
            }
        }

        // printHelmholtzSolution(std::cout, newY, n);

        // ������ ��������� �� ������� ������� (������ � oldY �������� ��������� ������� ��������)
        std::swap(newY, oldY);
    }
    // ������������� ��������� �������� �� ������ oldY

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // std::cout << "is okay? : " << (A == oldY) << std::endl;

    // ����������, ��� �� ������ A ����� ������� ��������� ������
    if (newY == A) {
        // std::memcpy(A, oldY, sizeof(T) * (n + 1) * (n + 1)); // �� ��������, ��� ������� ����������� + �� ������������

        for (int i = 0; i < (n + 1) * (n + 1); ++i) {
            A[i] = oldY[i];
        }

        // ������������ ������������� ���������� ������
        delete[] oldY;
    } else {

        // ������������ ������������� ���������� ������
        delete[] newY;
    }

    std::cout << "[DEBUG]: iterCount = " << iterCount << std::endl;

    //return iterCount >= MAX_ITER_COUNT;
    return elapsed.count();
}

/*
    ������� ������� ������� � ����������� MPI �������� "�����" ��������� ����������� � ������������� k
    � ������ ������ f ��� ������ ��������� ������� 1-�� ���� � ���������� �������.
    A - ��������� �� ������� ������� (n + 1)*(n + 1), �������� � ��������� ��������� �������� �� �
        �� ���������� ��������� - ��������� �������� ������� ������ ������� ��� ������ �������,
        � ������� ����� ����������� ��������� ������ ������, y - ������, x - �������;
    n - ����������� ����� �� x- � y-���������� (���������� �����);
    k - ����������� (������ ��� ������) ��� �������� ����� ���������;
    f - ������� ������ �����, f = f(x, y);
    h - ��� �����.
    ��������� ����� (����������): ������-������, ���� (1, 1) - �������.
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
        // ��������� ������� ��������� ������ �������
        for (int i = 1; i <= n - 1; ++i) { // ���������� �� �����
            for (int j = 2 - i % 2; j <= n - 1; j += 2) { // ����������� ������ ����
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // ��������� ������ ��������� ������ �������
        for (int i = 1; i <= n - 1; ++i) { // ���������� �� �����
            for (int j = 1 + i % 2; j <= n - 1; j += 2) { // ����������� ������� ����
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // ������ ������� �������
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

// ����������� ����� ��� ������� � ������ MPI-��������� �� id
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
