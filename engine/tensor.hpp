#include<stdexcept>
#include<vector>
#include<cmath>
#include<arm_neon.h>

class Tensor {
    std::vector<float> data;
    size_t rows;
    size_t cols;

public:

    Tensor() : rows(0), cols(0) {}

    Tensor(std::vector<float> weights, size_t row, size_t col)
        : data(weights), rows(row), cols(col) {}

    size_t shape(int idx) const {
        if(idx==0) return rows;
        return cols;
    }

    const std::vector<float>& getData() const {
        return this->data;
    }

    Tensor operator+(const Tensor& obj) const {
        if (rows != obj.rows || cols != obj.cols)
            throw std::invalid_argument("Shape mismatch for addition");

        std::vector<float> res(data.size());
        size_t n = data.size();

        #pragma omp parallel for if(n > 1024)
        for (size_t i = 0; i < n; i += 4) {
            if (i + 4 <= n) {
                float32x4_t a = vld1q_f32(&data[i]);
                float32x4_t b = vld1q_f32(&obj.data[i]);
                vst1q_f32(&res[i], vaddq_f32(a, b));
            } else {
                for (size_t k = i; k < n; ++k)
                    res[k] = data[k] + obj.data[k];
            }
        }

        return Tensor(res, rows, cols);
    }

    Tensor operator*(const Tensor& obj) const {
        if (cols != obj.rows)
            throw std::invalid_argument("Shape mismatch for matmul: A.cols must equal B.rows");

        const size_t M = rows;
        const size_t K = cols;
        const size_t N = obj.cols;

        std::vector<float> res(M * N, 0.0f);

        // NEON tiled matmul
        constexpr size_t TILE = 32;

        #pragma omp parallel for if(M > 16)
        for (size_t i0 = 0; i0 < M; i0 += TILE) {
            size_t iEnd = std::min(i0 + TILE, M);
            for (size_t k0 = 0; k0 < K; k0 += TILE) {
                size_t kEnd = std::min(k0 + TILE, K);
                for (size_t j0 = 0; j0 < N; j0 += TILE) {
                    size_t jEnd = std::min(j0 + TILE, N);
                    for (size_t i = i0; i < iEnd; ++i) {
                        for (size_t k = k0; k < kEnd; ++k) {
                            float32x4_t a_ik = vdupq_n_f32(data[i * K + k]);
                            size_t j = j0;
                            // process 4 columns at a time
                            for (; j + 4 <= jEnd; j += 4) {
                                float32x4_t b = vld1q_f32(&obj.data[k * N + j]);
                                float32x4_t c = vld1q_f32(&res[i * N + j]);
                                c = vfmaq_f32(c, a_ik, b);  // c += a * b
                                vst1q_f32(&res[i * N + j], c);
                            }
                            // remainder
                            for (; j < jEnd; ++j)
                                res[i * N + j] += data[i * K + k] * obj.data[k * N + j];
                        }
                    }
                }
            }
        }

        return Tensor(res, M, N);
    }

    Tensor LayerNorm(const Tensor& gamma, const Tensor& beta, float E = 1e-5f) const {
        std::vector<float> res(data.size());

        for(size_t i = 0; i < rows; i++){
            const float* row = &data[i * cols];

            // compute mean with NEON
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t j = 0;
            for (; j + 4 <= cols; j += 4)
                sum_vec = vaddq_f32(sum_vec, vld1q_f32(row + j));
            float mean = vaddvq_f32(sum_vec);
            for (; j < cols; j++) mean += row[j];
            mean /= cols;

            // compute variance with NEON
            float32x4_t var_vec = vdupq_n_f32(0.0f);
            float32x4_t mean_vec = vdupq_n_f32(mean);
            j = 0;
            for (; j + 4 <= cols; j += 4) {
                float32x4_t diff = vsubq_f32(vld1q_f32(row + j), mean_vec);
                var_vec = vfmaq_f32(var_vec, diff, diff);
            }
            float var = vaddvq_f32(var_vec);
            for (; j < cols; j++) {
                float diff = row[j] - mean;
                var += diff * diff;
            }
            var /= cols;

            float inv_std = 1.0f / std::sqrt(var + E);
            float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
            float* out_row = &res[i * cols];

            j = 0;
            for (; j + 4 <= cols; j += 4) {
                float32x4_t norm = vmulq_f32(vsubq_f32(vld1q_f32(row + j), mean_vec), inv_std_vec);
                float32x4_t g = vld1q_f32(&gamma.data[j]);
                float32x4_t b = vld1q_f32(&beta.data[j]);
                vst1q_f32(out_row + j, vfmaq_f32(b, norm, g));
            }
            for (; j < cols; j++) {
                float norm = (row[j] - mean) * inv_std;
                out_row[j] = gamma.data[j] * norm + beta.data[j];
            }
        }
        return Tensor(res, rows, cols);
    }

    Tensor operator/(const float val) const {
        std::vector<float> res(rows * cols);
        float32x4_t val_vec = vdupq_n_f32(val);
        size_t i = 0;
        for (; i + 4 <= data.size(); i += 4)
            vst1q_f32(&res[i], vdivq_f32(vld1q_f32(&data[i]), val_vec));
        for (; i < data.size(); i++)
            res[i] = data[i] / val;
        return Tensor(res, rows, cols);
    }

    Tensor softmax() const {
        std::vector<float> res(data.size());

        #pragma omp parallel for if(rows > 8)
        for(size_t i = 0; i < rows; i++) {
            const float* row = &data[i * cols];
            float* out = &res[i * cols];

            // find max
            float32x4_t max_vec = vdupq_n_f32(row[0]);
            size_t j = 0;
            for (; j + 4 <= cols; j += 4)
                max_vec = vmaxq_f32(max_vec, vld1q_f32(row + j));
            float max_val = vmaxvq_f32(max_vec);
            for (; j < cols; j++) max_val = std::max(max_val, row[j]);

            // exp and sum
            float32x4_t max_bcast = vdupq_n_f32(max_val);
            float sum_exp = 0.0f;
            j = 0;
            for (; j + 4 <= cols; j += 4) {
                float32x4_t v = vsubq_f32(vld1q_f32(row + j), max_bcast);
                // no NEON exp - compute scalar
                float tmp[4];
                vst1q_f32(tmp, v);
                tmp[0] = std::exp(tmp[0]);
                tmp[1] = std::exp(tmp[1]);
                tmp[2] = std::exp(tmp[2]);
                tmp[3] = std::exp(tmp[3]);
                float32x4_t ev = vld1q_f32(tmp);
                vst1q_f32(out + j, ev);
                sum_exp += tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
            for (; j < cols; j++) {
                out[j] = std::exp(row[j] - max_val);
                sum_exp += out[j];
            }

            // normalize
            float32x4_t inv_sum = vdupq_n_f32(1.0f / sum_exp);
            j = 0;
            for (; j + 4 <= cols; j += 4)
                vst1q_f32(out + j, vmulq_f32(vld1q_f32(out + j), inv_sum));
            for (; j < cols; j++) out[j] /= sum_exp;
        }

        return Tensor(res, rows, cols);
    }

    Tensor t() const {
        std::vector<float> res(data.size());
        for(size_t i = 0; i < rows; i++)
            for(size_t j = 0; j < cols; j++)
                res[j * rows + i] = data[i * cols + j];
        return Tensor(res, cols, rows);
    }

    Tensor mask() const {
        std::vector<float> res = data;
        const float MASK_VALUE = -1e9f;
        for(size_t i = 0; i < rows; i++)
            for(size_t j = i + 1; j < cols; j++)
                res[i * cols + j] = MASK_VALUE;
        return Tensor(res, rows, cols);
    }

    static Tensor concat_horizontal(const std::vector<Tensor>& tensors) {
        if (tensors.empty()) return Tensor();

        size_t rows = tensors[0].rows;
        size_t total_cols = 0;
        for (const auto& t : tensors) {
            if (t.rows != rows)
                throw std::invalid_argument("All tensors must have same number of rows for horizontal concat");
            total_cols += t.cols;
        }

        std::vector<float> res(rows * total_cols);
        size_t col_offset = 0;
        for (const auto& t : tensors) {
            for (size_t i = 0; i < rows; i++)
                std::copy(t.data.begin() + i * t.cols,
                          t.data.begin() + i * t.cols + t.cols,
                          res.begin() + i * total_cols + col_offset);
            col_offset += t.cols;
        }
        return Tensor(res, rows, total_cols);
    }

    static Tensor concat_vertical(const std::vector<Tensor>& tensors) {
        if (tensors.empty()) return Tensor();

        size_t cols = tensors[0].cols;
        size_t total_rows = 0;
        for (const auto& t : tensors) {
            if (t.cols != cols)
                throw std::invalid_argument("All tensors must have same number of columns for vertical concat");
            total_rows += t.rows;
        }

        std::vector<float> res(total_rows * cols);
        size_t row_offset = 0;
        for (const auto& t : tensors) {
            std::copy(t.data.begin(), t.data.end(), res.begin() + row_offset * cols);
            row_offset += t.rows;
        }
        return Tensor(res, total_rows, cols);
    }

    Tensor add_bias(const Tensor& bias) const {
        std::vector<float> res(data.size());
        #pragma omp parallel for if(rows > 8)
        for(size_t i = 0; i < rows; i++) {
            size_t j = 0;
            for (; j + 4 <= cols; j += 4) {
                float32x4_t a = vld1q_f32(&data[i * cols + j]);
                float32x4_t b = vld1q_f32(&bias.data[j]);
                vst1q_f32(&res[i * cols + j], vaddq_f32(a, b));
            }
            for (; j < cols; j++)
                res[i * cols + j] = data[i * cols + j] + bias.data[j];
        }
        return Tensor(res, rows, cols);
    }
};
