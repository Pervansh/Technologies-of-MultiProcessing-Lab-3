#pragma once

#include <algorithm>
#include <cstring>
#include <chrono>

template<typename T>
using PoissonFuncType = T(*)(T, T);

template<typename T>
using HelmholtzSolverType = double (*)(T*, int, T, PoissonFuncType<T>, T, const T);

#define MAX_ITER_COUNT 10000

// ������ ��� �������� ������ ������� 2-������� ������� � ������ ������ n, ���������� � ���� 1-������� ������������ �������
#define Q_IND(i, j, n) (i) * (n) + (j)

#include "PoissonTests.h"

#define H_SOLVE_JACOBY_CALC_2(coef, f, h, arr, n, i, j, i1, j1, i2, j2) \
    coef * (arr[Q_IND(i1, j1, n + 1)] + arr[Q_IND(i2, j2, n + 1)] + h * h * f(j * h, i * h))

#define H_SOLVE_JACOBY_CALC_3(coef, f, h, arr, n, i, j, i1, j1, i2, j2, i3, j3) \
    coef * (arr[Q_IND(i1, j1, n + 1)] + arr[Q_IND(i2, j2, n + 1)] + arr[Q_IND(i3, j3, n + 1)] + h * h * f(j * h, i * h))

#define H_SOLVE_JACOBY_CALC_4(coef, f, h, arr, n, i, j) \
    coef * (arr[Q_IND(i - 1, j, n + 1)] + arr[Q_IND(i + 1, j, n + 1)] + arr[Q_IND(i, j - 1, n + 1)] + \
    arr[Q_IND(i, j + 1, n + 1)] + h * h * f(j * h, i * h))

/*
    ��� ��������: floating, int * T, T(int), T(double)
*/

/*
    ������� ������� ����� ��� ����������������� �������� "�����" ��������� ����������� � ������������� k
    � ������ ������ f ��� ������ ��������� ������� 1-�� ���� � ���������� �������.
    A - ��������� �� ������� ������� (n + 1)*(n + 1), �������� � ��������� ��������� �������� �� �
        �� ���������� ��������� - ��������� �������� ������� ������ ������� ��� ������ �����, 
        � ������� ����� ����������� ��������� ������ ������, y - ������, x - �������;
    n - ����������� ����� �� x- � y-���������� (���������� �����);
    k - ����������� (������ ��� ������) ��� �������� ����� ���������;
    f - ������� ������ �����, f = f(x, y);
    h - ��� �����.
    ��������� ����� (����������): ������������ (����������).
*/
template<typename T>
double jacobyMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T* newY = new T[(n + 1) * (n + 1)]; // �������� �� ��������� ��������
    T* oldY = A;            // �������� �� ���������� ��������

    //printHelmholtzSolution(std::cout, A, n);
    //printHelmholtzSolution(std::cout, newY, n);

    T coef = 1. / (4. + k * k * h * h);

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

    T discrepancy = T(1e18);

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
        discrepancy = T(0);
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
    ������� ������� ����� � ������������������ �������� "�����" ��������� ����������� � ������������� k
    � ������ ������ f ��� ������ ��������� ������� 1-�� ���� � ���������� �������.
    A - ��������� �� ������� ������� (n + 1)*(n + 1), �������� � ��������� ��������� �������� �� �
        �� ���������� ��������� - ��������� �������� ������� ������ ������� ��� ������ �����,
        � ������� ����� ����������� ��������� ������ ������, y - ������, x - �������;
    n - ����������� ����� �� x- � y-���������� (���������� �����);
    k - ����������� (������ ��� ������) ��� �������� ����� ���������;
    f - ������� ������ �����, f = f(x, y);
    h - ��� �����.
    ��������� ����� (����������): ������������ (����������).
*/
template<typename T>
double parallelJacobyMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T* newY = new T[(n + 1) * (n + 1)]; // �������� �� ��������� ��������
    T* oldY = A;            // �������� �� ���������� ��������

    T coef = 1. / (4. + k * k * h * h);

    auto start_time = std::chrono::high_resolution_clock::now();

    // ������ �� � newY
#pragma omp parallel for
    for (int i = 0; i <= n - 1; ++i) {
        // ��������� ��������� �� �������������� ��������
        newY[Q_IND(0, i, n + 1)] = A[Q_IND(0, i, n + 1)]; // ������� ����
        newY[Q_IND(n, n - i, n + 1)] = A[Q_IND(n, n - i, n + 1)]; // ������ ����
        // ��������� ��������� �� ������������ ��������
        newY[Q_IND(n - i, 0, n + 1)] = A[Q_IND(n - i, 0, n + 1)]; // ����� ����
        newY[Q_IND(i, n, n + 1)] = A[Q_IND(i, n, n + 1)]; // ������ ����
    }

    T discrepancy = T(1e18);

    int iterCount = 0;
    for (; iterCount < MAX_ITER_COUNT && discrepancy >= minDiscrepancy; iterCount++) {

        // ��������� ��������� ������ �������
#pragma omp parallel for collapse(2)
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                newY[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, oldY, n, i, j);
            }
        }

        // ������ ������� �������
        discrepancy = T(0);
#pragma omp parallel for collapse(2) reduction(max: discrepancy)
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

    // ����������, ��� �� ������ A ����� ������� ��������� ������
    if (newY == A) {

        // ���������� � A ��������� ��������� �������� ������
#pragma omp parallel for
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
    ������� ������� ������� ��� ����������������� �������� "�����" ��������� ����������� � ������������� k
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
template<typename T>
double seidelMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T coef = 1. / (4. + k * k * h * h);

    T discrepancy = T(1e18);

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
        discrepancy = T(0);
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


/*
    ������� ������� ������� � ������������������ �������� "�����" ��������� ����������� � ������������� k
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
template<typename T>
double parallelSeidelMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T coef = 1. / (4. + k * k * h * h);

    T discrepancy = T(1e18);

    auto start_time = std::chrono::high_resolution_clock::now();

    // ���� ��������� �������� ����������� ������� A;
    bool nIsEven = (n % 2) == 0;
    int nChopped = n - nIsEven;

    int iterCount = 0;
    for (; iterCount < MAX_ITER_COUNT && discrepancy >= minDiscrepancy; iterCount++) {
        // ��������� ������� ��������� ������ �������, �������� �������� � ������ �������
#pragma omp parallel for collapse(2)
        for (int i = 1; i <= n - 1; ++i) { // ���������� �� �����
            for (int j0 = 1; j0 <= nChopped - 1; j0 += 2) { // ����������� ������ ����
                int j = j0 + 1 - (i % 2);
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        /*
            � ������ ������ ����������� n ���������� ������� ����� �� ������ ����� ������.
            �������, ���� nIsOdd �������, ��������� ���������� ������� �������� � ������
            �������, ���������� ������� �����, ��� i - �������, j = n - 1.
        */
        if (nIsEven) {
            int j = n - 1;
#pragma omp parallel for
            for (int i = 1; i <= n - 1; i += 2) {
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // ��������� ������ ��������� ������ �������, �������� �������� � ������ �������
#pragma omp parallel for collapse(2)
        for (int i = 1; i <= n - 1; ++i) { // ���������� �� �����
            for (int j0 = 1; j0 <= nChopped - 1; j0 += 2) { // ����������� ������� ����
                int j = j0 + i % 2;
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        /*
            � ������ ������ ����������� n ���������� ������ ����� �� ������ ����� ������.
            �������, ���� nIsOdd �������, ��������� ���������� ������ �������� � ������
            �������, ���������� ������� �����, ��� i - �����, j = n - 1.
        */
        if (nIsEven) {
            int j = n - 1;
#pragma omp parallel for
            for (int i = 2; i <= n - 1; i += 2) {
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // ������ ������� �������
        discrepancy = T(0);
#pragma omp parallel for collapse(2) reduction(max: discrepancy)
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                // auto curDiscrepancy = A[Q_IND(i, j, n + 1)] - H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j));
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
