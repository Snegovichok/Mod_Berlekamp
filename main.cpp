#include <iostream>
#include <vector>
#include <chrono>

#ifndef MOD
#define MOD 23
#endif

using namespace std;

class Polynomial {
private:
    vector<int> coefficients;

public:

    // Конструктор по умолчанию создает многочлен с единственным членом x^0
    Polynomial() : coefficients(0) {}

    // Конструктор, принимающий вектор коэффициентов
    Polynomial(const vector<int>& coeffs) {
        for (int coeff : coeffs) {
            coefficients.push_back((coeff % MOD + MOD) % MOD);
        }
        normalize();
    }

    // Конструктор, принимающий двумерный вектор для инициализации из матрицы
    Polynomial(const vector<vector<int>>& matrix) {
        for (const auto& row : matrix) {
            for (int coeff : row) {
                coefficients.push_back(coeff % MOD);
            }
        }
    }

    // Функция для получения коэффициентов многочлена
    vector<int> getCoefficients() const {
        return coefficients;
    }

    int countNonZeroCoefficients() const {
        int count = 0;
        for (const auto& coef : coefficients) {
            if (coef != 0) {
                count++;
            }
        }
        return count;
    }

    // Функция для сложения двух многочленов
    Polynomial add(const Polynomial& other) const {
        vector<int> resultCoeffs;
        int maxSize = max(coefficients.size(), other.coefficients.size());

        for (int i = 0; i < maxSize; ++i) {
            int coeff1 = (i < coefficients.size()) ? coefficients[i] : 0;
            int coeff2 = (i < other.coefficients.size()) ? other.coefficients[i] : 0;
            resultCoeffs.push_back((coeff1 + coeff2) % MOD);
        }

        return Polynomial(resultCoeffs);
    }

    // Функция для вычитания одного многочлена из другого
    Polynomial subtract(const Polynomial& other) const {
        vector<int> resultCoeffs;
        int maxSize = max(coefficients.size(), other.coefficients.size());

        for (int i = 0; i < maxSize; ++i) {
            int coeff1 = (i < coefficients.size()) ? coefficients[i] : 0;
            int coeff2 = (i < other.coefficients.size()) ? other.coefficients[i] : 0;
            resultCoeffs.push_back((coeff1 - coeff2 + MOD) % MOD);
        }

        return Polynomial(resultCoeffs);
    }

    // Функция для умножения многочлена на скаляр
    Polynomial scalarMultiply(int scalar) const {
        vector<int> resultCoeffs;
        for (int coeff : coefficients) {
            resultCoeffs.push_back((coeff * scalar) % MOD);
        }
        return Polynomial(resultCoeffs);
    }

    // Функция для умножения двух многочленов
    Polynomial multiply(const Polynomial& other) const {
        vector<int> resultCoeffs(coefficients.size() + other.coefficients.size() - 1, 0);

        for (int i = 0; i < coefficients.size(); ++i) {
            for (int j = 0; j < other.coefficients.size(); ++j) {
                resultCoeffs[i + j] += (coefficients[i] * other.coefficients[j]) % MOD;
                resultCoeffs[i + j] %= MOD;
            }
        }

        return Polynomial(resultCoeffs);
    }

    // Функция для деления многочленов с остатком
    pair<Polynomial, Polynomial> divide(const Polynomial& divisor) const {
        vector<int> quotientCoeffs(coefficients.size(), 0);
        vector<int> remainderCoeffs(coefficients);

        while (remainderCoeffs.size() >= divisor.coefficients.size()) {
            int degreeDifference = remainderCoeffs.size() - divisor.coefficients.size();
            int quotient = (remainderCoeffs.back() * inverse(divisor.coefficients.back())) % MOD;

            quotientCoeffs[degreeDifference] = quotient;

            for (int i = 0; i < divisor.coefficients.size(); ++i) {
                remainderCoeffs[i + degreeDifference] -= (divisor.coefficients[i] * quotient) % MOD;
                remainderCoeffs[i + degreeDifference] = (remainderCoeffs[i + degreeDifference] + MOD) % MOD;
            }

            while (!remainderCoeffs.empty() && remainderCoeffs.back() == 0) {
                remainderCoeffs.pop_back();
            }
        }

        return make_pair(Polynomial(quotientCoeffs), Polynomial(remainderCoeffs));
    }

    // Функция для нахождения НОД двух многочленов
    Polynomial gcd(const Polynomial& a, const Polynomial& b) const {
        Polynomial x(a), y(b);
        while (!y.getCoefficients().empty()) {
            Polynomial temp = x;
            x = y;
            y = temp.divide(y).second;
        }
        // Нормализация многочлена, чтобы старший коэффициент был равен 1
        vector<int> coeffs = x.getCoefficients();
        if (coeffs.empty()) return x; // Return zero polynomial as gcd
        int leading_coefficient = coeffs.back();
        int inverse_lc = inverse(leading_coefficient);
        for (int& coeff : coeffs) {
            coeff = (coeff * inverse_lc) % MOD;
        }
        return Polynomial(coeffs);
    }

    bool isZero() const {
        for (int coeff : coefficients) {
            if (coeff != 0) {
                return false;
            }
        }
        return true;
    }

    // Функция для вывода многочлена
    void printPolynomial() const {
        bool firstTerm = true;

        for (int i = coefficients.size() - 1; i >= 0; --i) {
            if (coefficients[i] != 0) {
                if (!firstTerm) {
                    cout << " + ";
                }
                else {
                    firstTerm = false;
                }
                cout << coefficients[i] << "x^" << i;
            }
        }
        if (firstTerm) {
            cout << "0"; // Если многочлен нулевой
        }
        cout << endl;
    }

private:
    // Вспомогательная функция для нахождения обратного элемента по модулю
    int inverse(int num) const {
        for (int i = 1; i < MOD; ++i) {
            if ((num * i) % MOD == 1) {
                return i;
            }
        }
        return 1; // Если num равен 1, то обратным будет также 1
    }

    // Вспомогательная функция для удаления ведущих нулей
    void normalize() {
        while (!coefficients.empty() && coefficients.back() == 0) {
            coefficients.pop_back();
        }
    }
};

// Функция для вывода многочлена
void printPolynomial(const Polynomial& poly) {
    vector<int> coeffs = poly.getCoefficients();
    if (coeffs.empty()) {
        cout << "0";
    }
    else {
        bool firstTerm = true;
        for (int i = coeffs.size() - 1; i >= 0; --i) {
            if (coeffs[i] != 0) {
                if (!firstTerm) {
                    cout << " + ";
                }
                else {
                    firstTerm = false;
                }
                cout << coeffs[i] << "x^" << i;
            }
        }
    }
    cout << endl;
}

// Функция для получения последней степени многочлена
int getLastDegree(const vector<int>& coeffs) {
    int lastDegree = coeffs.size() - 1;
    while (lastDegree >= 0 && coeffs[lastDegree] == 0) {
        lastDegree--;
    }
    return lastDegree;
}

// Функция для применения операции модуля и обработки отрицательных чисел
int applyMod(int num) {
    return (num % MOD + MOD) % MOD;
}

// Функция для нахождения обратного элемента по модулю MOD
int modInverse(int a, int m) {
    a = a % m;
    for (int x = 1; x < m; x++) {
        if ((a * x) % m == 1) {
            return x;
        }
    }
    return 1; // если обратный элемент не найден, что маловероятно для простых чисел
}

int inverseMatrix(int num) {
    for (int i = 1; i < MOD; ++i) {
        if ((num * i) % MOD == 1) {
            return i;
        }
    }
    return 1; // Если num равен 1, то обратным будет также 1
}

// Функция для приведения матрицы к ступенчатому виду методом Гаусса-Жордана
void gaussJordanElimination(vector<vector<int>>& matrix) {
    int n = matrix.size();
    int m = matrix[0].size();

    for (int i = 0; i < n; ++i) {
        // Найти ведущий элемент
        if (matrix[i][i] == 0) {
            for (int k = i + 1; k < n; ++k) {
                if (matrix[k][i] != 0) {
                    swap(matrix[i], matrix[k]);
                    break;
                }
            }
        }

        if (matrix[i][i] == 0) continue; // если ведущий элемент все еще 0, пропускаем этот шаг

        // Нормализовать ведущий элемент до 1
        int inv = modInverse(matrix[i][i], MOD);
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = applyMod(matrix[i][j] * inv);
        }

        // Обнулить элементы в текущем столбце над и под ведущим элементом
        for (int k = 0; k < n; ++k) {
            if (k != i && matrix[k][i] != 0) {
                int factor = matrix[k][i];
                for (int j = 0; j < m; ++j) {
                    matrix[k][j] = applyMod(matrix[k][j] - factor * matrix[i][j]);
                }
            }
        }
    }
}

// Функция для нахождения базиса пространства решений однородной системы
vector<vector<int>> findBasisOfSolutionSpace(vector<vector<int>>& matrix) {
    int n = matrix.size();
    int m = matrix[0].size();
    vector<vector<int>> basis;

    gaussJordanElimination(matrix);

    vector<bool> isFreeVariable(m, true);

    // Определение свободных переменных
    for (int i = 0; i < n; ++i) {
        int leadingEntry = -1;
        for (int j = 0; j < m; ++j) {
            if (matrix[i][j] != 0) {
                leadingEntry = j;
                break;
            }
        }
        if (leadingEntry != -1) {
            isFreeVariable[leadingEntry] = false;
        }
    }

    // Формирование базиса пространства решений
    for (int j = 0; j < m; ++j) {
        if (isFreeVariable[j]) {
            vector<int> solution(m, 0);
            solution[j] = 1;
            for (int i = 0; i < n; ++i) {
                int leadingEntry = -1;
                for (int k = 0; k < m; ++k) {
                    if (matrix[i][k] != 0) {
                        leadingEntry = k;
                        break;
                    }
                }
                if (leadingEntry != -1) {
                    solution[leadingEntry] = applyMod(-matrix[i][j]);
                }
            }
            basis.push_back(solution);
        }
    }

    return basis;
}

// Функция для решения фундаментальной системы решений
// void solveFundamentalSystem(vector<vector<int>>& matrix) {
vector<vector<int>> solveFundamentalSystem(vector<vector<int>>& matrix) {
    // Применить модуль ко всем элементам матрицы
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            matrix[i][j] = applyMod(matrix[i][j]);
        }
    }

    // Найти базис пространства решений
    vector<vector<int>> basis = findBasisOfSolutionSpace(matrix);

    // Вернуть базис пространства решений
    return basis;
}

// Функция для приведения матрицы к ступенчатому виду методом Гаусса
void gaussElimination(vector<vector<int>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int lead = 0;

    for (int r = 0; r < rows; ++r) {
        if (cols <= lead) {
            break;
        }
        int i = r;
        while (matrix[i][lead] == 0) {
            ++i;
            if (rows == i) {
                i = r;
                ++lead;
                if (cols == lead) {
                    return;
                }
            }
        }
        swap(matrix[i], matrix[r]);

        int lv = matrix[r][lead];
        for (int j = 0; j < cols; ++j) {
            matrix[r][j] = (matrix[r][j] * inverseMatrix(lv)) % MOD;
        }
        for (int i = 0; i < rows; ++i) {
            if (i != r) {
                int lv = matrix[i][lead];
                for (int j = 0; j < cols; ++j) {
                    matrix[i][j] -= matrix[r][j] * lv;
                    matrix[i][j] = applyMod(matrix[i][j]);
                }
            }
        }
        ++lead;
    }
}

void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// Функция для построения матрицы Сильвестра
vector<vector<int>> buildSylvesterMatrix(const vector<int>& coeffs1, const vector<int>& coeffs3, int c) {
    int n1 = getLastDegree(coeffs1);
    int n3 = getLastDegree(coeffs3);

    int size_matrix = n1 + n3; // размер матрицы
    vector<vector<int>> sylvesterMatrix(size_matrix, vector<int>(size_matrix, 0));

    int filledRows = 0; // Переменная для отслеживания количества заполненных строк

    // Заполнение коэффициентами многочлена coeffs1
    for (int j = 0; j < n3; ++j) {
        for (int i = 0; i <= n1; ++i) {
            sylvesterMatrix[j][i + j] = coeffs1[n1 - i];
        }
        filledRows++;
    }

    // Заполнение коэффициентами многочлена coeffs3
    for (int j = filledRows; j < size_matrix; ++j) {
        for (int i = 0; i <= n3; ++i) {
            sylvesterMatrix[j][i + j - n3] = coeffs3[n3 - i];
        }
    }

    // Заполнение c
    for (int j = filledRows; j < filledRows + n1 && j < size_matrix; ++j) {
        sylvesterMatrix[j][n3 + j - filledRows] = ((coeffs3[0] - c) + MOD) % MOD;
    }

    return sylvesterMatrix;
}

// Функция для вычисления определителя матрицы по модулю MOD
int determinant(vector<vector<int>> matrix) {
    int n = matrix.size();
    int det = 1;

    for (int i = 0; i < n; ++i) {
        if (matrix[i][i] == 0) {
            bool swapped = false;
            for (int j = i + 1; j < n; ++j) {
                if (matrix[j][i] != 0) {
                    swap(matrix[i], matrix[j]);
                    det = (MOD - det) % MOD;
                    swapped = true;
                    break;
                }
            }
            if (!swapped) return 0;
        }

        det = (det * matrix[i][i]) % MOD;
        int inv = 1;
        int b = matrix[i][i];
        for (int j = 0; j < MOD - 2; ++j) {
            inv = (inv * b) % MOD;
        }

        for (int j = i + 1; j < n; ++j) {
            int factor = (matrix[j][i] * inv) % MOD;
            for (int k = i; k < n; ++k) {
                matrix[j][k] = (matrix[j][k] - factor * matrix[i][k] + MOD) % MOD;
            }
        }
    }

    return det;
}

void addNonZeroPolynomialsInOrder(const vector<vector<Polynomial>>& allAdjustedPolys, vector<Polynomial>& ResultPoly) {
    int m = allAdjustedPolys.size();
    int numPolys = allAdjustedPolys[0].size(); // Предполагаем, что количество многочленов одинаково для всех значений

    for (int j = 0; j < numPolys; ++j) {
        for (int i = 0; i < m; ++i) {
            if (!allAdjustedPolys[i][j].isZero()) {
                ResultPoly.push_back(allAdjustedPolys[i][j]);
            }
        }
    }
}

void run() {
    vector<int> coeffs1 = { 7, 6, 18, 14, 5, 20, 1 }; //MOD 23 классический пример из книги

    //Сгенерированные примеры:
    //vector<int> coeffs1 = { 18, 10, 16, 5, 6, 7, 13, 15, 8, 9, 1 }; //MOD 23, DEG 10
    //vector<int> coeffs1 = { 10, 2, 21, 21, 6, 18, 0, 8, 19, 5, 22, 3, 11, 16, 18, 1 }; //MOD 23, DEG 15
    //vector<int> coeffs1 = { 3, 9, 18, 8, 6, 11, 10, 10, 4, 21, 14, 0, 1, 17, 9, 2, 20, 8, 19, 2, 1 }; //MOD 23, DEG 20

    //vector<int> coeffs1 = { 22, 25, 27, 4, 0, 24, 16, 15, 23, 11, 1 }; //MOD 29, DEG 10
    //vector<int> coeffs1 = { 21, 20, 25, 23, 0, 6, 25, 7, 25, 21, 21, 12, 7, 23, 11, 1 }; //MOD 29, DEG 15
    //vector<int> coeffs1 = { 4, 28, 26, 12, 27, 7, 4, 2, 23, 15, 13, 22, 6, 25, 18, 14, 26, 25, 1, 22, 1 }; //MOD 29, DEG 20
    //vector<int> coeffs1 = { 7, 24, 6, 17, 17, 14, 12, 4, 12, 4, 21, 27, 19, 27, 23, 19, 3, 15, 27, 6, 14, 1, 14, 11, 4, 1 }; //MOD 29, DEG 25

    //vector<int> coeffs1 = { 23, 2, 37, 36, 46, 1, 27, 36, 22, 23, 1 }; //MOD 47, DEG 10
    //vector<int> coeffs1 = { 8, 34, 42, 31, 24, 11, 26, 36, 23, 26, 42, 38, 18, 27, 14, 1 }; //MOD 47, DEG 15
    //vector<int> coeffs1 = { 16, 35, 29, 38, 7, 34, 22, 4, 5, 37, 27, 25, 0, 15, 42, 3, 43, 7, 44, 15, 1 }; //MOD 47, DEG 20
    //vector<int> coeffs1 = { 39, 21, 16, 17, 26, 1, 21, 43, 0, 13, 44, 17, 27, 38, 11, 37, 31, 40, 41, 26, 44, 5, 15, 7, 36, 1 }; //MOD 47, DEG 25
    //vector<int> coeffs1 = { 20, 19, 32, 2, 26, 9, 31, 41, 27, 15, 7, 24, 3, 14, 40, 11, 8, 33, 26, 10, 8, 36, 38, 46, 38, 16, 24, 20, 22, 41, 1 }; //MOD 47, DEG 30

    //vector<int> coeffs1 = { 30, 41, 25, 30, 4, 55, 17, 46, 5, 28, 1 }; //MOD 61, DEG 10
    //vector<int> coeffs1 = { 24, 39, 18, 42, 59, 13, 9, 22, 28, 35, 11, 34, 47, 3, 20, 1 }; //MOD 61, DEG 15
    //vector<int> coeffs1 = { 17, 52, 22, 23, 11, 38, 23, 37, 44, 35, 30, 55, 13, 1, 17, 4, 3, 13, 21, 30, 1 }; //MOD 61, DEG 20
    //vector<int> coeffs1 = { 14, 39, 11, 8, 6, 57, 30, 14, 25, 28, 55, 17, 6, 31, 16, 8, 12, 54, 30, 40, 5, 6, 50, 31, 39, 1 }; //MOD 61, DEG 25
    //vector<int> coeffs1 = { 27, 6, 36, 13, 32, 2, 60, 49, 51, 47, 25, 22, 48, 4, 5, 32, 59, 14, 35, 19, 24, 47, 3, 30, 37, 39, 58, 6, 59, 41, 1 }; //MOD 61, DEG 30
    //vector<int> coeffs1 = { 38, 43, 38, 22, 58, 60, 4, 52, 6, 27, 23, 12, 19, 31, 60, 40, 22, 40, 33, 21, 41, 49, 7, 2, 53, 30, 57, 46, 35, 34, 0, 42, 23, 44, 45, 1 }; //MOD 61, DEG 35

    //vector<int> coeffs1 = { 3, 67, 80, 3, 36, 39, 55, 49, 92, 73, 1 }; //MOD 101, DEG 10
    //vector<int> coeffs1 = { 94, 35, 29, 100, 51, 67, 72, 1, 72, 84, 56, 98, 79, 69, 23, 1 }; //MOD 101, DEG 15
    //vector<int> coeffs1 = { 84, 1, 86, 63, 18, 75, 7, 80, 78, 92, 64, 82, 7, 54, 61, 44, 52, 27, 55, 5, 1 }; //MOD 101, DEG 20
    //vector<int> coeffs1 = { 21, 78, 17, 16, 2, 5, 39, 41, 60, 51, 99, 93, 30, 45, 2, 78, 99, 41, 62, 90, 47, 76, 95, 56, 59, 1 }; //MOD 101, DEG 25
    //vector<int> coeffs1 = { 14, 30, 71, 50, 41, 26, 36, 57, 59, 35, 33, 17, 50, 82, 74, 64, 3, 13, 15, 57, 14, 99, 52, 4, 52, 98, 51, 34, 89, 38, 1 }; //MOD 101, DEG 30
    //vector<int> coeffs1 = { 61, 14, 25, 17, 33, 43, 83, 82, 100, 90, 46, 62, 68, 15, 54, 19, 43, 59, 34, 44, 23, 51, 57, 90, 90, 22, 18, 87, 99, 85, 38, 40, 69, 4, 70, 1 }; //MOD 101, DEG 35
    //vector<int> coeffs1 = { 10, 11, 71, 32, 0, 57, 33, 42, 86, 41, 87, 75, 45, 31, 13, 60, 82, 100, 14, 95, 56, 50, 10, 84, 95, 89, 92, 33, 48, 25, 47, 36, 64, 13, 57, 61, 83, 72, 41, 57, 1 }; //MOD 101, DEG 40

    Polynomial poly1(coeffs1);

    cout << "Проверяемый полином равен: ";
    printPolynomial(poly1);

    cout << "Работаем по модулю " << MOD << endl;

    int n = getLastDegree(coeffs1);
    cout << "Последняя степень полинома: " << n << endl;

    vector<vector<int>> coeffs2;

    // Создание коэффицентов для деления
    for (int i = 0; i < n; ++i) {
        vector<int> coeffs_i(MOD * i + 1, 0); // Например: если MOD = 5; то при i = 1: coeffs_i = {0, 0, 0, 0, 0, 1} или x^5
        coeffs_i.back() = 1;
        coeffs2.push_back(coeffs_i);
    }

    // Создание матрицы Q
    vector<vector<int>> matrixQ;

    for (const auto& coeffs_i : coeffs2) {
        Polynomial poly2(coeffs_i);
        pair<Polynomial, Polynomial> divisionResult = poly2.divide(poly1);
        matrixQ.push_back(divisionResult.second.getCoefficients());
    }

    for (auto& row : matrixQ) {
        int expectedSize = coeffs1.size() - 1;
        while (row.size() < expectedSize) {
            row.push_back(0);
        }
    }

    cout << "Матрица Q:" << endl;
    for (const auto& row : matrixQ) {
        for (int coeff : row) {
            cout << coeff << " ";
        }
        cout << endl;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            int c = matrixQ[i][j];
            matrixQ[i][j] = matrixQ[j][i];
            matrixQ[j][i] = c;
        }
    }

    cout << "Транспонированная матрица QT:\n";
    for (const auto& row : matrixQ) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Создание матрицы E
    vector<vector<int>> matrixE(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        matrixE[i][i] = 1;
    }

    cout << "Матрица E:\n";
    for (const auto& row : matrixE) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Вычисление разности QT - E
    vector<vector<int>> matrixQTMinusE(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrixQTMinusE[i][j] = matrixQ[i][j] - matrixE[i][j];
        }
    }

    // Применение модуля к каждому элементу матрицы
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrixQTMinusE[i][j] = applyMod(matrixQTMinusE[i][j]);
        }
    }

    cout << "Матрица QT-E:\n";
    for (const auto& row : matrixQTMinusE) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Вызов функции для приведения матрицы к ступенчатому виду
    gaussElimination(matrixQTMinusE);

    // Применение модуля к каждому элементу матрицы
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrixQTMinusE[i][j] = applyMod(matrixQTMinusE[i][j]);
        }
    }

    cout << "Матрица QT-E в ступенчатом виде:\n";
    for (const auto& row : matrixQTMinusE) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Подсчитываем ранг
    int rank = 0;
    for (const auto& row : matrixQTMinusE) {
        bool allZeros = true;
        for (int val : row) {
            if (val != 0) {
                allZeros = false;
                break;
            }
        }
        if (!allZeros) {
            ++rank;
        }
    }

    vector<vector<int>> ResultMatrix;

    // Проходим по ступенчатой матрице
    for (const auto& row : matrixQTMinusE) {
        bool allZeros = true;
        for (int val : row) {
            if (val != 0) {
                allZeros = false;
                break;
            }
        }
        // Если строка не нулевая, добавляем ее в вывод
        if (!allZeros) {
            ResultMatrix.push_back(row);
        }
    }

    // Выводим итоговую матрицу
    cout << "Окончательный вид матрицы:\n";
    for (const auto& row : ResultMatrix) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    cout << "Ранг матрицы: " << rank << endl;

    // Находим количество множителей 
    int k = n - rank;
    cout << "Количество множителей: " << k << endl;

    // Создаем вектор для хранения ФСР
    vector<vector<int>> FSRMatrix = solveFundamentalSystem(ResultMatrix);

    // Вывести фундаментальную систему решений
    cout << "Базис пространства решений:" << endl;
    for (const auto& solution : FSRMatrix) {
        for (int val : solution) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Удаляем первую строку (индекс 0)
    FSRMatrix.erase(FSRMatrix.begin());

    // Вектор для хранения многочленов
    vector<Polynomial> polys;

    // Преобразование каждой строки матрицы в многочлен и сохранение в векторе
    for (const auto& row : FSRMatrix) {
        Polynomial poly(row);
        polys.push_back(poly);
    }

    // Вывод каждого многочлена
    cout << "Итоговые многочлены:" << endl;
    for (const auto& poly : polys) {
        printPolynomial(poly);
    }
    cout << endl;

    // Преобразуем коэффиценты
    vector<int> coeffs3 = FSRMatrix[0];
    Polynomial poly3(coeffs3);

    // Создаём вектор для хранения результантов
    vector<int> resultants(MOD);
    // Создаём матрицу Сильвестра и находим его определители
    for (int c = 0; c < MOD; ++c) {
        vector<vector<int>> sylvesterMatrix = buildSylvesterMatrix(coeffs1, coeffs3, c);
        resultants[c] = determinant(sylvesterMatrix);
        resultants[c] = ((resultants[c] + MOD) % MOD);
    }

    cout << "Результанты для различных значений c:" << endl;
    for (int c = 0; c < MOD; ++c) {
            cout << "c = " << c << ": " << resultants[c] << endl;
    }

    // Создаём вектор для хранения корней
    vector<int> root;

    // Записываем корни равные нулевому значению
    for (int c = 0; c < MOD; ++c) {
        if (resultants[c] == 0) {
            root.push_back(c);
        }
    }

    cout << "Корни многочленного уравнения (при которых = 0):" << endl;
    for (const auto& r : root) {
        cout << r << " ";
    }
    cout << endl;

    // Вектор для хранения всех результатов adjustedPoly
    vector<vector<Polynomial>> allAdjustedPolys(k);

    // Перебор значений для minus root
    for (int i = 0; i < root.size(); ++i) {
        cout << "вычитаем " << root[i] << endl;
        Polynomial adjustedPoly = poly3.subtract(Polynomial(vector<int>{root[i]}));
        allAdjustedPolys[i].push_back(adjustedPoly);
        printPolynomial(adjustedPoly);
        cout << endl;
    }

    vector<Polynomial> ResultPoly;

    addNonZeroPolynomialsInOrder(allAdjustedPolys, ResultPoly);

    cout << "Окончательный вид многочленов:" << endl;
    for (const auto& poly : ResultPoly) {
        printPolynomial(poly);
    }
    cout << endl;

    // Определим размеры матрицы
    int maxDegree = 0;
    for (const auto& poly : ResultPoly) {
        int degree = getLastDegree(poly.getCoefficients());
        maxDegree = max(maxDegree, degree);
    }

    // Создадим матрицу коэффициентов
    vector<vector<int>> ResultPolyMatrix(ResultPoly.size(), vector<int>(maxDegree + 1, 0));

    // Заполним матрицу коэффициентов
    for (int i = 0; i < ResultPoly.size(); ++i) {
        const vector<int>& coeffs = ResultPoly[i].getCoefficients();
        for (int j = 0; j < coeffs.size(); ++j) {
            ResultPolyMatrix[i][j] = coeffs[j];
        }
    }

    // Вывод матрицы коэффициентов ResultPolyMatrix
    cout << "Матрица коэффициентов ResultPolyMatrix:" << endl;
    for (int i = 0; i < ResultPolyMatrix.size(); ++i) {
        for (int j = 0; j < ResultPolyMatrix[i].size(); ++j) {
            cout << ResultPolyMatrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // Вычисление и вывод НОД для каждой строки матрицы с poly1
    cout << "Нахождение НОД:" << endl;
    for (const auto& row : ResultPolyMatrix) {
        Polynomial poly3(row);
        Polynomial gcdResult = poly1.gcd(poly1, poly3);
        printPolynomial(gcdResult);
    }
}

// В этом коде основная логика программы вынесена в функцию run(). 
// В main() происходит выполнение этой функции 10 раз с замером времени выполнения каждой итерации. 
// После завершения всех итераций вычисляется среднее время выполнения и выводится на экран.
int main() {
    setlocale(LC_ALL, "Russian");
    const int iterations = 10;
    vector<long long> executionTimes;

    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();

        run();

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        executionTimes.push_back(duration.count());
    }

    long long totalDuration = 0;
    for (const auto& time : executionTimes) {
        totalDuration += time;
    }

    double averageDuration = static_cast<double>(totalDuration) / iterations;
    cout << "Среднее время выполнения программы: " << averageDuration << " миллисекунд" << std::endl;

    return 0;
}