#include"parser.hpp"
#include"tokenizer.hpp"
#include<map>
#include<string>
#include<iostream>
#include<cmath>
#include<vector>

class Transformer{
    int d_model, max_len, vocab_size, seq_len, num_heads, d_k;
    std::vector<Tensor> qs, ks, vs;
    Tensor join_weight, join_bias;
    Tensor gamma, beta;

public:
    Transformer(int d_model, int max_len, int vocab_size, int seq_len, int num_heads,
                std::vector<Tensor> qs, std::vector<Tensor> ks, std::vector<Tensor> vs,
                Tensor join_weight, Tensor join_bias,
                Tensor gamma, Tensor beta)
        : d_model(d_model), max_len(max_len), vocab_size(vocab_size),
          seq_len(seq_len), num_heads(num_heads),
          qs(qs), ks(ks), vs(vs),
          join_weight(join_weight), join_bias(join_bias),
          gamma(gamma), beta(beta)
    {
        this->d_k = d_model / num_heads;
        if (qs.size() != num_heads || ks.size() != num_heads || vs.size() != num_heads)
            throw std::invalid_argument("Number of Q, K, V tensors must match num_heads");
    }

    Tensor attention(const Tensor& input, const Tensor& k, const Tensor& q, const Tensor& v){
        try {
            float scale = std::sqrt(static_cast<float>(d_k));
            Tensor Q = input * q;
            Tensor K = input * k;
            Tensor V = input * v;
            Tensor scores = Q * K.t();
            scores = scores / scale;
            scores = scores.softmax();
            return scores * V;
        } catch (const std::exception& e) {
            std::cerr << "ERROR in attention: " << e.what() << "\n";
            std::cerr << "  input: " << input.shape(0) << "x" << input.shape(1) << "\n";
            std::cerr << "  q: " << q.shape(0) << "x" << q.shape(1) << "\n";
            std::cerr << "  k: " << k.shape(0) << "x" << k.shape(1) << "\n";
            std::cerr << "  v: " << v.shape(0) << "x" << v.shape(1) << "\n";
            throw;
        }
    }

    Tensor multiheadattention(Tensor& input){
        std::vector<Tensor> res;
        for(int i = 0; i < num_heads; i++)
            res.push_back(attention(input, ks[i], qs[i], vs[i]));
        Tensor concat = Tensor::concat_horizontal(res);  // (seq, d_model)
        return (concat * join_weight).add_bias(join_bias);// (seq, d_model)
    }

    Tensor forward(Tensor& input){
        Tensor attn = multiheadattention(input);
        std::cerr << "input shape: " << input.shape(0) << "x" << input.shape(1) << "\n";
        std::cerr << "attn shape: " << attn.shape(0) << "x" << attn.shape(1) << "\n";
        input = input + attn;
        input = input.LayerNorm(gamma, beta);
        return input;
    }
};

class Model {

    private:
        int max_len;
        int vocab_size;
        int seq_len;
        int num_heads;
        int d_model;
        int blocks;
        Tensor token_embeddings;      // (vocab_size, d_model)
        Tensor output_projection;     // (d_model, vocab_size)
        Tensor output_projection_bias;
        std::vector<Transformer> transformers;  // One transformer per block
        Tokenizer tokenizer;          // GPT-2 tokenizer

    public:

        // Constructor with tokenizer paths
        Model(int d_model, int max_len, int vocab_size, int seq_len, int num_heads, int blocks,
              Tensor token_embeddings,
              Tensor output_projection,
              Tensor output_projection_bias,
              std::vector<std::vector<Tensor>> Qs,
              std::vector<std::vector<Tensor>> Ks,
              std::vector<std::vector<Tensor>> Vs,
              std::vector<Tensor> join_weights,
              std::vector<Tensor> join_biases,
              std::vector<Tensor> gammas,
              std::vector<Tensor> betas,
              const std::string& vocab_path,
              const std::string& merges_path = "")  : max_len(max_len),
                   vocab_size(vocab_size),
                   seq_len(seq_len),
                   num_heads(num_heads),
                   d_model(d_model),
                   blocks(blocks),
                   token_embeddings(token_embeddings),
                   output_projection(output_projection),
                   output_projection_bias(output_projection_bias)
        {
            // Load tokenizer
            tokenizer.load(vocab_path, merges_path);

            // Validate embedding shapes
            if (token_embeddings.shape(0) != vocab_size || token_embeddings.shape(1) != d_model) {
                throw std::invalid_argument("Token embeddings shape mismatch: expected (vocab_size, d_model)");
            }

            if (output_projection.shape(0) != d_model || output_projection.shape(1) != vocab_size) {
                throw std::invalid_argument("Output projection shape mismatch: expected (d_model, vocab_size)");
            }

            // Validate that we have the correct number of blocks
            if (Qs.size() != blocks || Ks.size() != blocks || Vs.size() != blocks) {
                throw std::invalid_argument("Number of Q, K, V blocks must match 'blocks' parameter");
            }

            if (gammas.size() != blocks || betas.size() != blocks) {
                throw std::invalid_argument("Number of gamma/beta tensors must match 'blocks' parameter");
            }

            // Initialize each transformer block
            for(int i = 0; i < blocks; i++) {
                transformers.emplace_back(
                    d_model, max_len, vocab_size, seq_len, num_heads,
                    Qs[i], Ks[i], Vs[i],
                    join_weights[i], join_biases[i],
                    gammas[i], betas[i]
                );
            }
        }

        // Convert token IDs to embeddings
        Tensor embeddings(const std::vector<int>& input_tokens) {
            size_t seq_len = input_tokens.size();

            // Lookup token embeddings
            std::vector<float> embedded_data(seq_len * d_model);
            for(size_t i = 0; i < seq_len; i++) {
                int token_id = input_tokens[i];
                for(size_t j = 0; j < d_model; j++) {
                    embedded_data[i * d_model + j] = token_embeddings.getData()[token_id * d_model + j];
                }
            }

            Tensor embedded(embedded_data, seq_len, d_model);

            // Add positional encodings
            return position_encodings(embedded);
        }

        // Positional encodings (sinusoidal)
        Tensor position_encodings(Tensor& input) {
            std::vector<float> pe_data(input.shape(0) * input.shape(1), 0.0f);

            for(size_t pos = 0; pos < input.shape(0); pos++) {
                for(size_t i = 0; i < input.shape(1); i++) {
                    if (i % 2 == 0) {
                        pe_data[pos * input.shape(1) + i] = std::sin(pos / std::pow(10000.0f, (2.0f * (i / 2)) / input.shape(1)));
                    } else {
                        pe_data[pos * input.shape(1) + i] = std::cos(pos / std::pow(10000.0f, (2.0f * (i / 2)) / input.shape(1)));
                    }
                }
            }

            // Add positional encoding to input
            std::vector<float> res(input.shape(0) * input.shape(1));
            for(size_t i = 0; i < input.shape(0) * input.shape(1); i++) {
                res[i] = input.getData()[i] + pe_data[i];
            }

            return Tensor(res, input.shape(0), input.shape(1));
        }

        // Forward pass with token IDs
        Tensor forward(const std::vector<int>& input_tokens) {
            try {
                if (input_tokens.empty()) {
                    throw std::invalid_argument("Input tokens cannot be empty");
                }

                size_t seq = input_tokens.size();
                if (seq > this->seq_len) {
                    throw std::invalid_argument("Input sequence length exceeds max seq_len");
                }

                // Get embeddings with positional encodings
                Tensor input = embeddings(input_tokens);
                std::cerr << "after embeddings: " << input.shape(0) << "x" << input.shape(1) << "\n";

                // Pass through all transformer blocks
                for(int i = 0; i < blocks; i++) {
                    try {
                        input = transformers[i].forward(input);
                        std::cerr << "after block " << i << ": " << input.shape(0) << "x" << input.shape(1) << "\n";

                    } catch (const std::exception& e) {
                        std::cerr << "ERROR in Model::forward block " << i << ": " << e.what() << "\n";
                        throw;
                    }
                }

                std::cerr << "input: " << input.shape(0) << "x" << input.shape(1) << "\n";
                std::cerr << "out_proj: " << output_projection.shape(0) << "x" << output_projection.shape(1) << "\n";
                std::cerr << "out_bias: " << output_projection_bias.shape(0) << "x" << output_projection_bias.shape(1) << "\n";

                // Output projection: (seq_len, d_model) @ (d_model, vocab_size) = (seq_len, vocab_size)
                 std::cerr << "before output proj\n";

                Tensor logits = (input * output_projection).add_bias(output_projection_bias);
                std::cerr << "logits: " << logits.shape(0) << "x" << logits.shape(1) << "\n";
                return logits;
            } catch (const std::exception& e) {
                std::cerr << "ERROR in Model::forward: " << e.what() << "\n";
                throw;
            }
        }

        // Forward pass with text input
        Tensor forward(const std::string& text) {
            std::vector<int> tokens = tokenizer.encode(text);
            return forward(tokens);
        }

        };


class Runner{
	private:
	std::map<std::string, Tensor> weights;
	public:
	Runner() {
		weights  = parse_weights();
	}
	void list(){
	for(auto it : weights)
		std::cout<<it.first<<"\n";
	}

	void run() {
    try {
        const int num_heads = 8;
        const int blocks = 6;

        auto& emb = weights["embeddings.weight"];
        const int vocab_size = emb.shape(0);
        const int d_model = emb.shape(1);

        std::cout << "Model config: vocab=" << vocab_size
                  << ", d_model=" << d_model
                  << ", heads=" << num_heads
                  << ", blocks=" << blocks << "\n";

        std::vector<std::vector<Tensor>> Qs(blocks), Ks(blocks), Vs(blocks);
        std::vector<Tensor> gammas(blocks), betas(blocks);
        std::vector<Tensor> join_weights(blocks), join_biases(blocks);

        for(int b = 0; b < blocks; b++) {
            std::string prefix = "transforms." + std::to_string(b) + ".";
            for(int h = 0; h < num_heads; h++) {
                std::string h_str = std::to_string(h);
                Qs[b].push_back(weights[prefix + "q." + h_str + ".weight"].t());
                Ks[b].push_back(weights[prefix + "k." + h_str + ".weight"].t());
                Vs[b].push_back(weights[prefix + "v." + h_str + ".weight"].t());
            }
            gammas[b]      = weights[prefix + "norm1.weight"];
            betas[b]       = weights[prefix + "norm1.bias"];
            join_weights[b] = weights[prefix + "join.weight"].t();
            join_biases[b]  = weights[prefix + "join.bias"];
        }

        Tensor out_weight = weights["out.weight"].t();
        Tensor out_bias   = weights["out.bias"];
        std::cout << "out_weight after t(): " << out_weight.shape(0) << "x" << out_weight.shape(1) << "\n";

        std::cout << "out.weight: " << weights["out.weight"].shape(0) << "x" << weights["out.weight"].shape(1) << "\n";

        Model model(d_model, 128, vocab_size, 128, num_heads, blocks,
                    emb, out_weight, out_bias,
                    Qs, Ks, Vs,
                    join_weights, join_biases,
                    gammas, betas,
                    "gpt2_vocab.json", "merges.txt");

        std::cout << "Model initialized!\n";

        Tokenizer tokenizer;
        tokenizer.load("gpt2_vocab.json", "merges.txt");

        std::string prompt = "Hello, how are you?";
        std::vector<int> tokens = tokenizer.encode(prompt);
        std::cout << "Tokenized: " << tokens.size() << " tokens\n";

        const int max_new_tokens = 100;
        const float temperature = 0.8f;
        int eos_token = tokenizer.eos();

        std::cout << "Generating...\n";
        for(int i = 0; i < max_new_tokens; i++) {
            std::vector<int> context = tokens.size() > 128
                ? std::vector<int>(tokens.end() - 128, tokens.end())
                : tokens;

            Tensor logits = model.forward(context);
            std::cerr << "logits shape: " << logits.shape(0) << "x" << logits.shape(1) << "\n";

            std::cerr << "vocab_size: " << vocab_size << "\n";
            std::cerr << "data size: " << logits.getData().size() << "\n";

            size_t last_pos = context.size() - 1;

            std::vector<float> probs(vocab_size);
            float max_logit = logits.getData()[last_pos * vocab_size];
            for(int j = 0; j < vocab_size; j++) {
                float scaled = (logits.getData()[last_pos * vocab_size + j] - max_logit) / temperature;
                probs[j] = std::exp(scaled);
            }
            float sum = 0;
            for(float p : probs) sum += p;
            for(float& p : probs) p /= sum;

            float r = static_cast<float>(rand()) / RAND_MAX;
            float cumsum = 0;
            int next_token = vocab_size - 1;
            for(int j = 0; j < vocab_size; j++) {
                cumsum += probs[j];
                if(cumsum >= r) { next_token = j; break; }
            }

            tokens.push_back(next_token);
            if(next_token == eos_token) { std::cout << "EOS\n"; break; }
            if(i % 10 == 0) std::cout << "Generated " << i << " tokens...\n";
        }

        std::string output = tokenizer.decode(tokens);
        std::cout << "\n=== Generated Text ===\n" << output << "\n======================\n";

    } catch(const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
    }
}
};
