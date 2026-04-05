#include<stdexcept>
#include<vector>
#include<cmath>
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

    std::vector<float> getData() const {
        return this->data;
    }

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

    Tensor LayerNorm(Tensor& gamma, Tensor& beta, float E = 1e-5f) {
	std::vector<float> res(data.size());
	    for(int i=0;i<rows;i++){
			float mean = 0.0f;
			for(int j = 0;j<cols;j++)
			mean += data[cols*i + j];
		mean /= cols;
		float var = 0.0f;
		for(int j=0;j<cols;j++) {
		    float diff = data[i*cols + j] - mean;
			var += diff * diff;
		}

		var /= cols;

		float d = std::sqrt(var + E);

		for(int j = 0;j<cols;j++) {
			float norm = (data[i*cols + j] - mean)/d;
			res[i * cols + j] = gamma.data[j] * norm + beta.data[j];
		}
	    }
	return Tensor(res,rows,cols);
    }

    Tensor operator / (const float val){
        std::vector<float> res(rows*cols,0);
        for(size_t i = 0;i<this->rows* this->cols;i++)
            res[i] = this->data[i]/val;
        return Tensor(res,this->rows,this->cols);
    }

    Tensor softmax() const {
        std::vector<float> res(data.size());

        for(size_t i = 0; i < rows; i++) {
            // Find max value in this row for numerical stability
            float max_val = data[i * cols];
            for(size_t j = 1; j < cols; j++) {
                max_val = std::max(max_val, data[i * cols + j]);
            }

            // Compute exp(x - max) for each element in the row
            float sum_exp = 0.0f;
            for(size_t j = 0; j < cols; j++) {
                float exp_val = std::exp(data[i * cols + j] - max_val);
                res[i * cols + j] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize by sum of exponentials
            for(size_t j = 0; j < cols; j++) {
                res[i * cols + j] /= sum_exp;
            }
        }

        return Tensor(res, rows, cols);
    }

    Tensor t() const {
        std::vector<float> res(data.size());

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                // Element at (i, j) goes to (j, i) in transposed matrix
                res[j * rows + i] = data[i * cols + j];
            }
        }

        return Tensor(res, cols, rows);
    }

    Tensor mask() const {
        std::vector<float> res(data.size());
        const float MASK_VALUE = -1e9f;

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                // Upper triangular positions (j > i) get masked
                if(j > i) {
                    res[i * cols + j] = MASK_VALUE;
                } else {
                    res[i * cols + j] = data[i*cols + j];
                }
            }
        }
        return Tensor(res, rows, cols);
    }


    static Tensor concat_horizontal(const std::vector<Tensor>& tensors) {
            if (tensors.empty()) {
                return Tensor();
            }

            size_t rows = tensors[0].rows;
            size_t total_cols = 0;

            // Calculate total columns and verify all have same rows
            for (const auto& t : tensors) {
                if (t.rows != rows) {
                    throw std::invalid_argument("All tensors must have same number of rows for horizontal concat");
                }
                total_cols += t.cols;
            }

            std::vector<float> res(rows * total_cols);
            size_t col_offset = 0;

            for (const auto& t : tensors) {
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < t.cols; j++) {
                        res[i * total_cols + (col_offset + j)] = t.data[i * t.cols + j];
                    }
                }
                col_offset += t.cols;
            }

            return Tensor(res, rows, total_cols);
        }

        Tensor add_bias(const Tensor& bias) const {
            // bias is (cols, 1) or (1, cols) - add to each row
            std::vector<float> res(data.size());
            for(size_t i = 0; i < rows; i++)
                for(size_t j = 0; j < cols; j++)
                    res[i * cols + j] = data[i * cols + j] + bias.getData()[j];
            return Tensor(res, rows, cols);
        }
};
