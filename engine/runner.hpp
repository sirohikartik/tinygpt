#include"parser.hpp"
#include"tokenizer.hpp"
#include<map>
#include<string>
#include<iostream>
#include<cmath>
#include<vector>
#include<chrono>

struct KVCache {
    std::vector<Tensor> ks;
    std::vector<Tensor> vs;
};

class Transformer{
    int d_model, max_len, vocab_size, seq_len, num_heads, d_k;
    std::vector<Tensor> qs, ks, vs;
    std::vector<Tensor> q_biases, k_biases, v_biases;
    Tensor join_weight, join_bias;
    Tensor gamma1, beta1;   // norm1 - before attention
    Tensor gamma2, beta2;   // norm2 - before FFN
    Tensor ffn1_weight, ffn1_bias;  // 256 -> 1024
    Tensor ffn2_weight, ffn2_bias;  // 1024 -> 256

public:
    Transformer(int d_model, int max_len, int vocab_size, int seq_len, int num_heads,
                std::vector<Tensor> qs, std::vector<Tensor> ks, std::vector<Tensor> vs,
                std::vector<Tensor> q_biases, std::vector<Tensor> k_biases, std::vector<Tensor> v_biases,
                Tensor join_weight, Tensor join_bias,
                Tensor gamma1, Tensor beta1,
                Tensor gamma2, Tensor beta2,
                Tensor ffn1_weight, Tensor ffn1_bias,
                Tensor ffn2_weight, Tensor ffn2_bias)
        : d_model(d_model), max_len(max_len), vocab_size(vocab_size),
          seq_len(seq_len), num_heads(num_heads),
          qs(qs), ks(ks), vs(vs),
          q_biases(q_biases), k_biases(k_biases), v_biases(v_biases),
          join_weight(join_weight), join_bias(join_bias),
          gamma1(gamma1), beta1(beta1),
          gamma2(gamma2), beta2(beta2),
          ffn1_weight(ffn1_weight), ffn1_bias(ffn1_bias),
          ffn2_weight(ffn2_weight), ffn2_bias(ffn2_bias)
    {
        this->d_k = d_model / num_heads;
        if (qs.size() != num_heads || ks.size() != num_heads || vs.size() != num_heads)
            throw std::invalid_argument("Number of Q, K, V tensors must match num_heads");
    }

    Tensor attention(const Tensor& input, const Tensor& k, const Tensor& q, const Tensor& v,
                     const Tensor& qb, const Tensor& kb, const Tensor& vb, KVCache& cache, int head_idx){
        float scale = std::sqrt(static_cast<float>(d_k));
        Tensor Q = (input * q).add_bias(qb);
        Tensor K_curr = (input * k).add_bias(kb);
        Tensor V_curr = (input * v).add_bias(vb);

        std::vector<Tensor> k_tensors = {cache.ks[head_idx], K_curr};
        cache.ks[head_idx] = Tensor::concat_vertical(k_tensors);
        std::vector<Tensor> v_tensors = {cache.vs[head_idx], V_curr};
        cache.vs[head_idx] = Tensor::concat_vertical(v_tensors);

        Tensor scores = Q * cache.ks[head_idx].t();
        scores = scores / scale;
        if (input.shape(0) > 1) scores = scores.mask();
        scores = scores.softmax();
        return scores * cache.vs[head_idx];
    }

    Tensor gelu(const Tensor& x) {
        const auto& d = x.getData();
        std::vector<float> res(d.size());
        for(size_t i = 0; i < d.size(); i++) {
            float v = d[i];
            res[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
        }
        return Tensor(res, x.shape(0), x.shape(1));
    }

    Tensor multiheadattention(const Tensor& input, KVCache& cache){
        std::vector<Tensor> res;
        for(int i = 0; i < num_heads; i++)
            res.push_back(attention(input, ks[i], qs[i], vs[i], q_biases[i], k_biases[i], v_biases[i], cache, i));
        Tensor concat = Tensor::concat_horizontal(res);
        return (concat * join_weight).add_bias(join_bias);
    }

    Tensor forward(const Tensor& input, KVCache& cache){
        // pre-norm attention
        Tensor normed1 = input.LayerNorm(gamma1, beta1);
        Tensor attn = multiheadattention(normed1, cache);
        Tensor x = input + attn;

        // pre-norm FFN
        Tensor normed2 = x.LayerNorm(gamma2, beta2);
        Tensor ffn_out = gelu((normed2 * ffn1_weight).add_bias(ffn1_bias));
        ffn_out = (ffn_out * ffn2_weight).add_bias(ffn2_bias);
        x = x + ffn_out;

        return x;
    }
};

class Model {
    private:
        int max_len, vocab_size, seq_len, num_heads, d_model, blocks;
        Tensor token_embeddings;
        Tensor output_projection;
        Tensor output_projection_bias;
        std::vector<Transformer> transformers;
        Tokenizer tokenizer;

    public:
        Model(int d_model, int max_len, int vocab_size, int seq_len, int num_heads, int blocks,
              Tensor token_embeddings,
              Tensor output_projection,
              Tensor output_projection_bias,
              std::vector<std::vector<Tensor>> Qs,
              std::vector<std::vector<Tensor>> Ks,
              std::vector<std::vector<Tensor>> Vs,
              std::vector<std::vector<Tensor>> Qbs,
              std::vector<std::vector<Tensor>> Kbs,
              std::vector<std::vector<Tensor>> Vbs,
              std::vector<Tensor> join_weights,
              std::vector<Tensor> join_biases,
              std::vector<Tensor> gammas1,
              std::vector<Tensor> betas1,
              std::vector<Tensor> gammas2,
              std::vector<Tensor> betas2,
              std::vector<Tensor> ffn1_weights,
              std::vector<Tensor> ffn1_biases,
              std::vector<Tensor> ffn2_weights,
              std::vector<Tensor> ffn2_biases,
              const std::string& vocab_path,
              const std::string& merges_path = "")
            : max_len(max_len), vocab_size(vocab_size), seq_len(seq_len),
              num_heads(num_heads), d_model(d_model), blocks(blocks),
              token_embeddings(token_embeddings),
              output_projection(output_projection),
              output_projection_bias(output_projection_bias)
        {
            tokenizer.load(vocab_path, merges_path);

            if (token_embeddings.shape(0) != vocab_size || token_embeddings.shape(1) != d_model)
                throw std::invalid_argument("Token embeddings shape mismatch");
            if (output_projection.shape(0) != d_model || output_projection.shape(1) != vocab_size)
                throw std::invalid_argument("Output projection shape mismatch");
            if (Qs.size() != blocks || Ks.size() != blocks || Vs.size() != blocks)
                throw std::invalid_argument("Number of Q, K, V blocks must match blocks");
            if (gammas1.size() != blocks || betas1.size() != blocks)
                throw std::invalid_argument("Number of gamma/beta tensors must match blocks");

            for(int i = 0; i < blocks; i++) {
                transformers.emplace_back(
                    d_model, max_len, vocab_size, seq_len, num_heads,
                    Qs[i], Ks[i], Vs[i],
                    Qbs[i], Kbs[i], Vbs[i],
                    join_weights[i], join_biases[i],
                    gammas1[i], betas1[i],
                    gammas2[i], betas2[i],
                    ffn1_weights[i], ffn1_biases[i],
                    ffn2_weights[i], ffn2_biases[i]
                );
            }
        }

        Tensor embeddings(const std::vector<int>& input_tokens) {
            size_t seq_len = input_tokens.size();
            std::vector<float> embedded_data(seq_len * d_model);
            for(size_t i = 0; i < seq_len; i++) {
                int token_id = input_tokens[i];
                for(size_t j = 0; j < d_model; j++)
                    embedded_data[i * d_model + j] = token_embeddings.getData()[token_id * d_model + j];
            }
            Tensor embedded(embedded_data, seq_len, d_model);
            return position_encodings(embedded);
        }

        Tensor position_encodings(Tensor& input) {
            std::vector<float> pe_data(input.shape(0) * input.shape(1), 0.0f);
            for(size_t pos = 0; pos < input.shape(0); pos++) {
                for(size_t i = 0; i < input.shape(1); i++) {
                    if (i % 2 == 0)
                        pe_data[pos * input.shape(1) + i] = std::sin(pos / std::pow(10000.0f, (2.0f * (i/2)) / input.shape(1)));
                    else
                        pe_data[pos * input.shape(1) + i] = std::cos(pos / std::pow(10000.0f, (2.0f * (i/2)) / input.shape(1)));
                }
            }
            std::vector<float> res(input.shape(0) * input.shape(1));
            for(size_t i = 0; i < input.shape(0) * input.shape(1); i++)
                res[i] = input.getData()[i] + pe_data[i];
            return Tensor(res, input.shape(0), input.shape(1));
        }

        Tensor forward(const std::vector<int>& input_tokens, std::vector<KVCache>& caches) {
            if (input_tokens.empty())
                throw std::invalid_argument("Input tokens cannot be empty");
            if (input_tokens.size() > this->seq_len)
                throw std::invalid_argument("Input sequence length exceeds max seq_len");

            Tensor input = embeddings(input_tokens);
            for(int i = 0; i < blocks; i++)
                input = transformers[i].forward(input, caches[i]);

            return (input * output_projection).add_bias(output_projection_bias);
        }

        Tensor forward(const std::vector<int>& input_tokens) {
            std::vector<KVCache> caches(blocks);
            // This is only for single-pass forward, not useful for generation loop
            // But we keep it for compatibility.
            // We need to initialize the caches.ks and vs.
            // Wait, Transformer::attention does it if they are empty? 
            // No, I used cache.ks[head_idx] which will throw if empty.
            // I should initialize them in Model constructor or here.
            return forward(input_tokens, caches); 
        }

        Tensor forward(const std::string& text) {
            return forward(tokenizer.encode(text));
        }
};


class Runner{
    private:
    std::map<std::string, Tensor> weights;
    public:
    Runner() {
        weights = parse_weights();
    }
    void list(){
        for(auto it : weights)
            std::cout << it.first << "\n";
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
            std::vector<std::vector<Tensor>> Qbs(blocks), Kbs(blocks), Vbs(blocks);
            std::vector<Tensor> gammas1(blocks), betas1(blocks);
            std::vector<Tensor> gammas2(blocks), betas2(blocks);
            std::vector<Tensor> join_weights(blocks), join_biases(blocks);
            std::vector<Tensor> ffn1_weights(blocks), ffn1_biases(blocks);
            std::vector<Tensor> ffn2_weights(blocks), ffn2_biases(blocks);

            for(int b = 0; b < blocks; b++) {
                std::string prefix = "transforms." + std::to_string(b) + ".";
                for(int h = 0; h < num_heads; h++) {
                    std::string h_str = std::to_string(h);
                    Qs[b].push_back(weights[prefix + "q." + h_str + ".weight"].t());
                    Ks[b].push_back(weights[prefix + "k." + h_str + ".weight"].t());
                    Vs[b].push_back(weights[prefix + "v." + h_str + ".weight"].t());
                    Qbs[b].push_back(weights[prefix + "q." + h_str + ".bias"]);
                    Kbs[b].push_back(weights[prefix + "k." + h_str + ".bias"]);
                    Vbs[b].push_back(weights[prefix + "v." + h_str + ".bias"]);
                }
                gammas1[b]     = weights[prefix + "norm1.weight"];
                betas1[b]      = weights[prefix + "norm1.bias"];
                gammas2[b]     = weights[prefix + "norm2.weight"];
                betas2[b]      = weights[prefix + "norm2.bias"];
                join_weights[b] = weights[prefix + "join.weight"].t();
                join_biases[b]  = weights[prefix + "join.bias"];
                ffn1_weights[b] = weights[prefix + "ffn.0.weight"].t();
                ffn1_biases[b]  = weights[prefix + "ffn.0.bias"];
                ffn2_weights[b] = weights[prefix + "ffn.3.weight"].t();
                ffn2_biases[b]  = weights[prefix + "ffn.3.bias"];
            }

            Tensor out_weight = weights["out.weight"].t();
            Tensor out_bias   = weights["out.bias"];

            Model model(d_model, 128, vocab_size, 128, num_heads, blocks,
                        emb, out_weight, out_bias,
                        Qs, Ks, Vs,
                        Qbs, Kbs, Vbs,
                        join_weights, join_biases,
                        gammas1, betas1,
                        gammas2, betas2,
                        ffn1_weights, ffn1_biases,
                        ffn2_weights, ffn2_biases,
                        "gpt2_vocab.json", "merges.txt");

            std::cout << "Model initialized!\n";

            Tokenizer tokenizer;
            tokenizer.load("gpt2_vocab.json", "merges.txt");

            std::string prompt;
            std::cout << "Prompt: ";
            std::getline(std::cin, prompt);

            std::vector<int> tokens = tokenizer.encode(prompt);
            int prompt_token_count = tokens.size();
            std::cout << "Tokenized: " << prompt_token_count << " tokens\n";

            int max_new_tokens;
            std::cout << "Enter Max Tokens: ";
            std::cin >> max_new_tokens;
            const float temperature = 0.8f;
            int eos_token = tokenizer.eos();

            // --- KV CACHE INITIALIZATION ---
            std::vector<KVCache> caches(blocks);
            for(int b = 0; b < blocks; b++) {
                caches[b].ks.resize(num_heads);
                caches[b].vs.resize(num_heads);
                for(int h = 0; h < num_heads; h++) {
                    caches[b].ks[h] = Tensor({}, 0, d_model / num_heads);
                    caches[b].vs[h] = Tensor({}, 0, d_model / num_heads);
                }
            }

            // --- BENCHMARK VARIABLES ---
            int generated_token_count = 0;
            double prompt_processing_time_ms = 0.0;
            double ttft_ms = 0.0;
            double total_decode_time_ms = 0.0; // Generation time excluding first token

            std::cout << "Generating...\n";
            
            // Start End-to-End Clock
            auto e2e_start = std::chrono::high_resolution_clock::now();
            auto generation_start = e2e_start; // Marks the beginning of the loops

            // First pass: process prompt
            auto prompt_start = std::chrono::high_resolution_clock::now();
            Tensor logits = model.forward(tokens, caches);
            auto prompt_end = std::chrono::high_resolution_clock::now();
            prompt_processing_time_ms = std::chrono::duration<double, std::milli>(prompt_end - prompt_start).count();
            ttft_ms = std::chrono::duration<double, std::milli>(prompt_end - generation_start).count();

            for(int i = 0; i < max_new_tokens; i++) {
                size_t logit_row = (i == 0) ? (tokens.size() - 1) : 0;

                std::vector<float> probs(vocab_size);
                float max_logit = logits.getData()[logit_row * vocab_size];
                for(int j = 0; j < vocab_size; j++) {
                    float scaled = (logits.getData()[logit_row * vocab_size + j] - max_logit) / temperature;
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
                generated_token_count++;

                if(next_token == eos_token) { std::cout << "EOS\n"; break; }

                // Compute logits for the next token using only the new token and the KV cache
                auto step_start = std::chrono::high_resolution_clock::now();
                
                // Only pass the last token
                std::vector<int> next_token_context = {next_token};
                logits = model.forward(next_token_context, caches);
                
                auto step_end = std::chrono::high_resolution_clock::now();
                double step_duration_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
                total_decode_time_ms += step_duration_ms;
            }

            auto e2e_end = std::chrono::high_resolution_clock::now();
            double e2e_time_ms = std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();
            double total_generation_time_ms = std::chrono::duration<double, std::milli>(e2e_end - generation_start).count();

            std::string output = tokenizer.decode(tokens);
            std::cout << "\n=== Generated Text ===\n" << output << "\n======================\n";

            // --- PRINT BENCHMARK REPORT ---
            std::cout << "\n================ BENCHMARK REPORT ================\n";
            std::cout << "Prompt Tokens         : " << prompt_token_count << "\n";
            std::cout << "Generated Tokens      : " << generated_token_count << "\n";
            std::cout << "--------------------------------------------------\n";
            std::cout << "Prompt Processing Time: " << prompt_processing_time_ms << " ms\n";
            std::cout << "TTFT                  : " << ttft_ms << " ms\n";
            
            if (generated_token_count > 1) {
                double avg_time_per_token = total_decode_time_ms / (generated_token_count - 1);
                std::cout << "Avg Time Per Token    : " << avg_time_per_token << " ms/token\n";
            } else {
                std::cout << "Avg Time Per Token    : N/A (Generated only 1 token)\n";
            }

            double generation_throughput = (generated_token_count / (total_generation_time_ms / 1000.0));
            std::cout << "Generation Throughput : " << generation_throughput << " tokens/sec\n";
            std::cout << "End-to-End Time       : " << e2e_time_ms << " ms (" << e2e_time_ms / 1000.0 << " sec)\n";
            std::cout << "==================================================\n";

        } catch(const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
        }
    }};
