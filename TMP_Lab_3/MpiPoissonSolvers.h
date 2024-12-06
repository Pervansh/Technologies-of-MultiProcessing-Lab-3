#pragma once

#include <algorithm>
#include <cstring>
#include <string.h>
#include <chrono>
#include <vector>
#include <memory>
#include <utility>

#include "mpi.h"

#include "PoissonSolvers.h"

#define MPI_H_SOLVE_CALC_4(coef, f, h, arr, n, i, j, rowDspl) \
    coef * (arr[Q_IND(i - 1, j, n + 1)] + arr[Q_IND(i + 1, j, n + 1)] + arr[Q_IND(i, j - 1, n + 1)] + \
    arr[Q_IND(i, j + 1, n + 1)] + h * h * f(j * h, (i + rowDspl) * h))

#define MPI_H_SOLVE_UPPER_CALC_4(coef, f, h, arr, upperArr, n, j, rowDspl) \
    coef * (upperArr[j] + arr[Q_IND(1, j, n + 1)] + arr[Q_IND(0, j - 1, n + 1)] + \
    arr[Q_IND(0, j + 1, n + 1)] + h * h * f(j * h, rowDspl * h))

#define MPI_H_SOLVE_LOWER_CALC_4(coef, f, h, arr, lowerArr, n, j, rowCnt, rowDspl) \
    coef * (lowerArr[j] + arr[Q_IND(rowCnt - 2, j, n + 1)] + arr[Q_IND(rowCnt - 1, j - 1, n + 1)] + \
    arr[Q_IND(rowCnt - 1, j + 1, n + 1)] + h * h * f(j * h, (rowDspl + rowCnt - 1) * h))


/*
    ����� ��������� ������ ����� slave-��
    �������� ����� ������ �����
    �� ������ �������� ������ ����� �� ����������, ��������� � �� (�������)
*/

using MpiHelmholtzSolverType = double (*)(MPI_Comm, double*, int, double, PoissonFuncType<double>, double, const double);

/*
    ���������� � FuncSpace: ������� ���������� FuncSpace::funcs �
    ���������� FuncSpace::funcs[int] -> PoissonFuncType<double>
*/

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
        strcpy_s(masterFuncSpaceName, maxStrSize, myFuncSpaceName); // �� �������� �� �������� (���������� ������)
        // std::strcpy(masterFuncSpaceName, myFuncSpaceName); // �� ���������� ������ (����� �� ������� maxStrSize)
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

// returns tuple(rowSize, myAPart)
std::tuple<int, std::unique_ptr<double[]>, int> mpiBlockRowArrayScatter(
    MPI_Comm comm,
    int master, 
    int& n, 
    double* const A = nullptr
)
{
    /*
        ����� n + 1 = k * numprocs + r. ����� ����� ������������ ������ ��������� �������:
        1. ������ ������� ������� ����������� ������ �����
        2. 1-� ������� ������� 1-� ������, 2-� --- 2-� � �.�.
        3. ������ r ��������� ������� (k + 1)-� �������, ��������� --- k ��������

        � ����� ������ �� ������ �������� ������ ������� ����� ������������ ������� �� �����, ���
        � ����� ���������� (����� MPI_Reduce).

        ����� ���������� ��������� �����������:
        mainBlocksCnt := k;
        firstResiduProcId := r.
    */

    int myid;     // ����� �������� ��������
    int numprocs; // ���������� ��������� � �������������

    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    MPI_Bcast(&n, 1, MPI_INT, master, comm);

    // ���������� �����, �������� �������������� ������� ������ �������
    int mainBlocksCnt = (n + 1) / numprocs;
    // ����� ������� ��������, ������� � �������� ������� ������� ����� mainBlocksCnt ��������
    int firstResiduProcId = (n + 1) % numprocs;
    // ���������� �����, �������� ������� ������� �������
    int rowsCnt = mainBlocksCnt + (myid < firstResiduProcId);
    // ���������� ���������, �������� ������� ������� �������
    int elemCnt = rowsCnt * (n + 1);

    // ������ ��������� �����, �������� ������� ��������
    std::vector<int> sendCounts(numprocs);
    for (int i = 0; i < numprocs; ++i) {
        sendCounts[i] = (n + 1) * (mainBlocksCnt + (i < firstResiduProcId));
    }

    // ������ ������� ������ �����, �������� ������� �������� (������ ������� ������������ ������ �������)
    std::vector<int> displs(numprocs);
    displs[0] = 0;
    for (int i = 1; i < numprocs; ++i) {
        displs[i] = displs[i - 1] + sendCounts[i - 1];
    }

    // ������ �����, �������� ������� �����
    std::unique_ptr<double[]> myAPart = std::make_unique<double[]>(elemCnt);

    MPI_Scatterv(A, sendCounts.data(), displs.data(), MPI_DOUBLE, myAPart.get(), elemCnt, MPI_DOUBLE, master, comm);

    /* Debug info about myAPart
    for (int proc = 0; proc < numprocs; proc++) {
        if (proc == myid) {
            std::cerr << "[PROCESS " << myid << " DEBUG]: my part is \n";
            for (int i = 0; i < rowsCnt; ++i) {
                for (int j = 0; j <= n; ++j) {
                    std::cerr << myAPart[i * (n + 1) + j] << ' ';
                }
                std::cerr << '\n';
            }
            std::cerr << std::endl;
        }

        MPI_Barrier(comm);
    }
    */

    return std::make_tuple(rowsCnt, std::move(myAPart), displs[myid]); // std::make_pair(rowsCnt, std::move(myAPart)) �� �������� ��� �������
}


void mpiBlockRowArrayGather(
    MPI_Comm comm,
    int master,
    int n,
    double* const myAPart,
    double* A = nullptr
)
{
    /*
        ����� n + 1 = k * numprocs + r. ����� ����� ������������ ������ ��������� �������:
        1. ������ ������� ������� ����������� ������ �����
        2. 1-� ������� ������� 1-� ������, 2-� --- 2-� � �.�.
        3. ������ r ��������� ������� (k + 1)-� �������, ��������� --- k ��������

        � ����� ������ �� ������ �������� ������ ������� ����� ������������ ������� �� �����, ���
        � ����� ���������� (����� MPI_Reduce).

        ����� ���������� ��������� �����������:
        mainBlocksCnt := k;
        firstResiduProcId := r.
    */

    int myid;     // ����� �������� ��������
    int numprocs; // ���������� ��������� � �������������

    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    // ���������� �����, �������� �������������� ������� ������ �������
    int mainBlocksCnt = (n + 1) / numprocs;
    // ����� ������� ��������, ������� � �������� ������� ������� ����� mainBlocksCnt ��������
    int firstResiduProcId = (n + 1) % numprocs;
    // ���������� �����, �������� ������� ������� �������
    int rowsCnt = mainBlocksCnt + (myid < firstResiduProcId);
    // ���������� ���������, �������� ������� ������� �������
    int elemCnt = rowsCnt * (n + 1);

    // ������ ��������� �����, �������� ������� ��������
    std::vector<int> recvCounts(numprocs);
    for (int i = 0; i < numprocs; ++i) {
        recvCounts[i] = (n + 1) * (mainBlocksCnt + (i < firstResiduProcId));
    }

    // ������ ������� ������ �����, �������� ������� �������� (������ ������� ������������ ������ �������)
    std::vector<int> displs(numprocs);
    displs[0] = 0;
    for (int i = 1; i < numprocs; ++i) {
        displs[i] = displs[i - 1] + recvCounts[i - 1];
    }

    MPI_Gatherv(myAPart, elemCnt, MPI_DOUBLE, A, recvCounts.data(), displs.data(), MPI_DOUBLE, master, comm);
}

/*
    ������ ������ �����.
    ������: ������ ���� �������� �������� ��� ���� ���������� �������.
    �����: ������ ���� �������� ������ ������ 2 ���������.
    ����������: n >= 2 * numprocs
*/
template <typename FuncSpace>
double mpiJacobyMethodHelmholtzSolve(
    MPI_Comm comm,
    int master,
    double* A = nullptr,
    int n = -1,
    double k = 0,
    int funcId = -1,
    double h = 0,
    double minDiscrepancy = 0)
{
    int myid;     // ����� �������� ��������
    int numprocs; // ���������� ��������� � �������������

    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    auto [rowCnt, oldMyYPart, rowDspl] = mpiBlockRowArrayScatter(comm, master, n, A);
    std::unique_ptr<double[]> newMyYPart = std::make_unique<double[]>(rowCnt * (n + 1));

    std::memcpy(newMyYPart.get(), oldMyYPart.get(), rowCnt * (n + 1));

    /* Debug info about myAPart
    for (int proc = 0; proc < numprocs; proc++) {
        if (proc == myid) {
            std::cerr << "[PROCESS " << myid << " DEBUG]: my part is " << myAPart[0] << " \n";
            for (int i = 0; i < rowsCnt; ++i) {
                for (int j = 0; j <= n; ++j) {
                    std::cerr << myAPart[i * (n + 1) + j] << ' ';
                }
                std::cerr << '\n';
            }
            std::cerr << std::endl;
        }

        MPI_Barrier(comm);
    }
    */
    
    bool funcStatus;
    // ������������� ����� �������
    auto f = mpiPoissonFuncBcast<FuncSpace>(comm, master, &funcStatus, funcId);

    if (!funcStatus) {
        return -1.; // ��������, ��������������� ������
    }

    // Bcast ��������� ����������
    MPI_Bcast(&k, 1, MPI_DOUBLE, master, comm);
    MPI_Bcast(&h, 1, MPI_DOUBLE, master, comm);
    MPI_Bcast(&minDiscrepancy, 1, MPI_DOUBLE, master, comm);

    double coef = 1. / (4. + k * k * h * h);

    std::unique_ptr<double[]> upperPart = std::make_unique<double[]>(n + 1);
    std::unique_ptr<double[]> lowerPart = std::make_unique<double[]>(n + 1);

    MPI_Request upperSend, upperRecv, lowerSend, lowerRecv;
    
    auto firstRowPtr = oldMyYPart.get();
    auto lastRowPtr  = oldMyYPart.get() + (n + 1) * (rowCnt - 1);

    if (myid > 0) {
        MPI_Send_init(firstRowPtr, n + 1, MPI_DOUBLE, myid - 1, 0, comm, &upperSend);
        MPI_Recv_init(upperPart.get(), n + 1, MPI_DOUBLE, myid - 1, 0, comm, &upperRecv);
    } else {
        upperSend = MPI_REQUEST_NULL;
        upperRecv = MPI_REQUEST_NULL;
    }

    if (myid < numprocs - 1) {
        MPI_Send_init(lastRowPtr, n + 1, MPI_DOUBLE, myid + 1, 0, comm, &lowerSend);
        MPI_Recv_init(lowerPart.get(), n + 1, MPI_DOUBLE, myid + 1, 0, comm, &lowerRecv);
    } else {
        lowerSend = MPI_REQUEST_NULL;
        lowerRecv = MPI_REQUEST_NULL;
    }

    MPI_Request allRequests[]  = { upperSend, upperRecv, lowerSend, lowerRecv };
    MPI_Request recvRequests[] = { upperRecv, lowerRecv };

    // ��������������� ��������� ������� �����
    //MPI_Startall(4, allRequests);
    if (myid > 0) {
        MPI_Start(&upperSend);
        MPI_Start(&upperRecv);
    }
    if (myid < numprocs - 1) {
        MPI_Start(&lowerSend);
        MPI_Start(&lowerRecv);
    }

    if (myid > 0) {
        MPI_Status status;
        MPI_Wait(&upperRecv, &status);
    }
    if (myid < numprocs - 1) {
        MPI_Status status;
        MPI_Wait(&lowerRecv, &status);
    }
    //MPI_Status statuses[2];
    //MPI_Waitall(2, recvRequests, statuses); // ��� ������� ��������

    auto start_time = std::chrono::high_resolution_clock::now();

    // printHelmholtzSolution(std::cout, newY, n);

    double discrepancy = double(1e18);

    int iterCount = 0;
    for (; iterCount < MAX_ITER_COUNT && discrepancy >= minDiscrepancy; iterCount++) {

        // printHelmholtzSolution(std::cout, A, n);
    
        // ��������� ��������� ������ �����
        for (int i = 1; i < rowCnt - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                newMyYPart[Q_IND(i, j, n + 1)] = MPI_H_SOLVE_CALC_4(coef, f, h, oldMyYPart, n, i, j, rowDspl);
            }
        }

        // ��������� ������� ������
        if (myid > 0) {
            for (int j = 1; j <= n - 1; ++j) {
                newMyYPart[Q_IND(0, j, n + 1)] =
                    MPI_H_SOLVE_UPPER_CALC_4(coef, f, h, oldMyYPart, upperPart, n, j, rowDspl);
            }
        }

        // ��������� ������ ������
        if (myid < numprocs - 1) {
            for (int j = 1; j <= n - 1; ++j) {
                newMyYPart[Q_IND(rowCnt - 1, j, n + 1)] =
                    MPI_H_SOLVE_LOWER_CALC_4(coef, f, h, oldMyYPart, lowerPart, n, j, rowCnt, rowDspl);
            }
        }

        // ����� ����� � ��������� �������� �����
        std::swap(newMyYPart, oldMyYPart);

        if (myid > 0) {
            MPI_Status status;
            MPI_Wait(&upperSend, &status);
        }
        if (myid < numprocs - 1) {
            MPI_Status status;
            MPI_Wait(&lowerSend, &status);
        }

        if (myid > 0) {
            MPI_Start(&upperSend);
            MPI_Start(&upperRecv);
        }
        if (myid < numprocs - 1) {
            MPI_Start(&lowerSend);
            MPI_Start(&lowerRecv);
        }

        if (myid > 0) {
            MPI_Status status;
            MPI_Wait(&upperRecv, &status);
        }
        if (myid < numprocs - 1) {
            MPI_Status status;
            MPI_Wait(&lowerRecv, &status);
        }
        
        // ������ ������� �������
        discrepancy = 0.;
        for (int i = 1; i < rowCnt - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                discrepancy = std::max(discrepancy,
                    std::fabs(oldMyYPart[Q_IND(i, j, n + 1)] -
                        MPI_H_SOLVE_CALC_4(coef, f, h, oldMyYPart, n, i, j, rowDspl)));
            }
        }

        // ������� ��� ������� ������
        if (myid > 0) {
            for (int j = 1; j <= n - 1; ++j) {
                discrepancy = std::max(discrepancy,
                    std::fabs(oldMyYPart[Q_IND(0, j, n + 1)] -
                        MPI_H_SOLVE_UPPER_CALC_4(coef, f, h, oldMyYPart, upperPart, n, j, rowDspl)));
            }
        }

        // ������� ��� ������ ������
        if (myid < numprocs - 1) {
            for (int j = 1; j <= n - 1; ++j) {
                discrepancy = std::max(discrepancy,
                    std::fabs(oldMyYPart[Q_IND(rowCnt - 1, j, n + 1)] -
                        MPI_H_SOLVE_LOWER_CALC_4(coef, f, h, oldMyYPart, upperPart, n, j, rowCnt, rowDspl)));
            }
        }

        double gottenDiscrepancy;
        MPI_Allreduce(&discrepancy, &gottenDiscrepancy, 1, MPI_DOUBLE, MPI_MAX, comm);
        discrepancy = gottenDiscrepancy;
    }
    // ������������� ��������� �������� �� ������ oldMyYPart

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // ������ ��������� ������� �� �������
    mpiBlockRowArrayGather(comm, master, n, oldMyYPart.get(), A);

    if (myid == master)
        std::cout << "[DEBUG]: iterCount = " << iterCount << std::endl;

    //return iterCount >= MAX_ITER_COUNT;
    return elapsed.count();
}

/*
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
*/
