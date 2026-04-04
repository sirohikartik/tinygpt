#include<stdexcept>


class Tensor {
    std::vector<float> data;
    size_t rows;
    size_t cols;

public:

    Tensor() : rows(0), cols(0) {}

    Tensor(std::vector<float> weights, size_t row, size_t col)
        : data(weights), rows(row), cols(col) {}

    Tensor operator+(const Tensor& obj) const {
        if (rows != obj.rows || cols != obj.cols)
            throw std::invalid_argument("Shape mismatch for addition");

        std::vector<float> res;
        res.reserve(data.size());
        for (size_t i = 0; i < data.size(); i++)
            res.push_back(data[i] + obj.data[i]);
        return Tensor(res, rows, cols);
    }

    Tensor operator*(const Tensor& obj) const {
        if (cols != obj.rows)
            throw std::invalid_argument(
                "Shape mismatch for matmul: "
                "A.cols must equal B.rows");

        const size_t M = rows;
        const size_t K = cols;       // == obj.rows
        const size_t N = obj.cols;

        std::vector<float> res(M * N, 0.0f);

        constexpr size_t TILE = 32;

        for (size_t i0 = 0; i0 < M; i0 += TILE) {
            size_t iEnd = std::min(i0 + TILE, M);

            for (size_t k0 = 0; k0 < K; k0 += TILE) {
                size_t kEnd = std::min(k0 + TILE, K);

                for (size_t j0 = 0; j0 < N; j0 += TILE) {
                    size_t jEnd = std::min(j0 + TILE, N);

                    for (size_t i = i0; i < iEnd; ++i) {
                        for (size_t k = k0; k < kEnd; ++k) {
                            float a_ik = data[i * K + k];          // A[i][k]

                            for (size_t j = j0; j < jEnd; ++j) {
                                res[i * N + j] +=
                                    a_ik * obj.data[k * N + j];    // A[i][k] * B[k][j]
                            }
                        }
                    }
                }
            }
        }

        return Tensor(res, M, N);
    }
};

