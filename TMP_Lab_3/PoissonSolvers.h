#pragma once

#include <algorithm>
#include <cstring>
#include <chrono>

template<typename T>
using PoissonFuncType = T(*)(T, T);

template<typename T>
using HelmholtzSolverType = double (*)(T*, int, T, PoissonFuncType<T>, T, const T);

#define MAX_ITER_COUNT 10000

// Макрос для короткой записи индекса 2-мерного массива с длиной строки n, хранящимся в виде 1-мерного программного массива
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
    Для концепта: floating, int * T, T(int), T(double)
*/

/*
    Решение методом Якоби без распараллеливания шаблоном "крест" уравнения Гельмгольца с коэффициентом k
    и правой частью f для случая граничных условий 1-го рода в квадратной области.
    A - указатель на матрицу размера (n + 1)*(n + 1), хранящую в граничных элементах значения ГУ и
        во внутренних элементах - начальные значения решения внутри области для метода Якоби, 
        в который будет перезаписан результат работы метода, y - строки, x - столбцы;
    n - размерность сетки по x- и y-координате (количество ячеек);
    k - коэффициент (точнее его корень) при линейном члене уравнения;
    f - функция правой части, f = f(x, y);
    h - шаг сетки.
    Нумерация узлов (внутренних): естественная (построчная).
*/
template<typename T>
double jacobyMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T* newY = new T[(n + 1) * (n + 1)]; // Значения на следующей итерации
    T* oldY = A;            // Значения на предыдущей итерации

    //printHelmholtzSolution(std::cout, A, n);
    //printHelmholtzSolution(std::cout, newY, n);

    T coef = 1. / (4. + k * k * h * h);

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

    T discrepancy = T(1e18);

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
        discrepancy = T(0);
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
    Решение методом Якоби с распараллеливанием шаблоном "крест" уравнения Гельмгольца с коэффициентом k
    и правой частью f для случая граничных условий 1-го рода в квадратной области.
    A - указатель на матрицу размера (n + 1)*(n + 1), хранящую в граничных элементах значения ГУ и
        во внутренних элементах - начальные значения решения внутри области для метода Якоби,
        в который будет перезаписан результат работы метода, y - строки, x - столбцы;
    n - размерность сетки по x- и y-координате (количество ячеек);
    k - коэффициент (точнее его корень) при линейном члене уравнения;
    f - функция правой части, f = f(x, y);
    h - шаг сетки.
    Нумерация узлов (внутренних): естественная (построчная).
*/
template<typename T>
double parallelJacobyMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T* newY = new T[(n + 1) * (n + 1)]; // Значения на следующей итерации
    T* oldY = A;            // Значения на предыдущей итерации

    T coef = 1. / (4. + k * k * h * h);

    auto start_time = std::chrono::high_resolution_clock::now();

    // запись ГУ в newY
#pragma omp parallel for
    for (int i = 0; i <= n - 1; ++i) {
        // обработка элементов на горизонтальных границах
        newY[Q_IND(0, i, n + 1)] = A[Q_IND(0, i, n + 1)]; // верхний край
        newY[Q_IND(n, n - i, n + 1)] = A[Q_IND(n, n - i, n + 1)]; // нижний край
        // обработка элементов на вертикальных границах
        newY[Q_IND(n - i, 0, n + 1)] = A[Q_IND(n - i, 0, n + 1)]; // левый край
        newY[Q_IND(i, n, n + 1)] = A[Q_IND(i, n, n + 1)]; // правый край
    }

    T discrepancy = T(1e18);

    int iterCount = 0;
    for (; iterCount < MAX_ITER_COUNT && discrepancy >= minDiscrepancy; iterCount++) {

        // обработка элементов внутри области
#pragma omp parallel for collapse(2)
        for (int i = 1; i <= n - 1; ++i) {
            for (int j = 1; j <= n - 1; ++j) {
                newY[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, oldY, n, i, j);
            }
        }

        // расчет текущей невязки
        discrepancy = T(0);
#pragma omp parallel for collapse(2) reduction(max: discrepancy)
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

    // Убеждаемся, что по адресу A будет записан результат метода
    if (newY == A) {

        // Записываем в A результат последней итерации метода
#pragma omp parallel for
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
    Решение методом Зейделя без распараллеливания шаблоном "крест" уравнения Гельмгольца с коэффициентом k
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
template<typename T>
double seidelMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T coef = 1. / (4. + k * k * h * h);

    T discrepancy = T(1e18);

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
    Решение методом Зейделя с распараллеливанием шаблоном "крест" уравнения Гельмгольца с коэффициентом k
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
template<typename T>
double parallelSeidelMethodHelmholtzSolve(T* A, int n, T k, PoissonFuncType<T> f, T h, const T minDiscrepancy) {
    T coef = 1. / (4. + k * k * h * h);

    T discrepancy = T(1e18);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Учет возможной четности размерности матрицы A;
    bool nIsEven = (n % 2) == 0;
    int nChopped = n - nIsEven;

    int iterCount = 0;
    for (; iterCount < MAX_ITER_COUNT && discrepancy >= minDiscrepancy; iterCount++) {
        // обработка красных элементов внутри области, исключая элементы у правой границы
#pragma omp parallel for collapse(2)
        for (int i = 1; i <= n - 1; ++i) { // проходимся по слоям
            for (int j0 = 1; j0 <= nChopped - 1; j0 += 2) { // переступаем черные узлы
                int j = j0 + 1 - (i % 2);
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        /*
            В случае четной размерности n количество красных узлов на разных слоях разное.
            Поэтому, если nIsOdd истинно, требуется обработать красные элементы у правой
            границы, координаты которых такие, что i - нечетно, j = n - 1.
        */
        if (nIsEven) {
            int j = n - 1;
#pragma omp parallel for
            for (int i = 1; i <= n - 1; i += 2) {
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // обработка черных элементов внутри области, исключая элементы у правой границы
#pragma omp parallel for collapse(2)
        for (int i = 1; i <= n - 1; ++i) { // проходимся по слоям
            for (int j0 = 1; j0 <= nChopped - 1; j0 += 2) { // переступаем красные узлы
                int j = j0 + i % 2;
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        /*
            В случае четной размерности n количество черных узлов на разных слоях разное.
            Поэтому, если nIsOdd истинно, требуется обработать черные элементы у правой
            границы, координаты которых такие, что i - четно, j = n - 1.
        */
        if (nIsEven) {
            int j = n - 1;
#pragma omp parallel for
            for (int i = 2; i <= n - 1; i += 2) {
                A[Q_IND(i, j, n + 1)] = H_SOLVE_JACOBY_CALC_4(coef, f, h, A, n, i, j);
            }
        }

        // расчет текущей невязки
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
