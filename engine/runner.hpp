#include"parser.hpp"
#include<map>
#include<string>
#include<iostream>
#include<cmath>
#include<unordered_set>
#include<algorithm>
#include<cctype>
#include<fstream>
#include<sstream>
#include<unordered_map>
#include<vector>
#include<cstdint>

namespace simple_json {
    inline std::string parse_string(const std::string& json, size_t& pos) {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t' || json[pos] == '"')) {
            if (json[pos] == '"') { pos++; break; }
            pos++;
        }
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                switch (json[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case '\\': result += '\\'; break;
                    case '"': result += '"'; break;
                    default: result += json[pos];
                }
            } else {
                result += json[pos];
            }
            pos++;
        }
        pos++; // skip closing quote
        return result;
    }

    inline int parse_int(const std::string& json, size_t& pos) {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t')) pos++;
        int result = 0;
        bool negative = false;
        if (json[pos] == '-') { negative = true; pos++; }
        while (pos < json.size() && std::isdigit(json[pos])) {
            result = result * 10 + (json[pos] - '0');
            pos++;
        }
        return negative ? -result : result;
    }
}


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
		// work here
	}
};

class Transformer{
	int d_model;
	int max_len;
	int vocab_size;
	int seq_len;
	int num_heads;
	std::vector<Tensor> qs;
	std::vector<Tensor> ks;
	std::vector<Tensor> vs;
	int d_k;
	Tensor gamma;
	Tensor beta;
	public:
	Transformer(int d_model, int max_len, int vocab_size, int seq_len, int num_heads,
	            std::vector<Tensor> qs, std::vector<Tensor> ks, std::vector<Tensor> vs,
	            Tensor gamma, Tensor beta)
		: d_model(d_model),
		  max_len(max_len),
		  vocab_size(vocab_size),
		  seq_len(seq_len),
		  num_heads(num_heads),
		  qs(qs),
		  ks(ks),
		  vs(vs),
		  gamma(gamma),
		  beta(beta)
	{
		// d_k is the dimension per head
		this->d_k = d_model / num_heads;

		// Validate that we have the correct number of heads
		if (qs.size() != num_heads || ks.size() != num_heads || vs.size() != num_heads) {
			throw std::invalid_argument("Number of Q, K, V tensors must match num_heads");
		}
	}

	    Tensor attention(const Tensor& input, const Tensor& k, const  Tensor& q, const Tensor& v,int d_k){
					d_k = std::sqrt(d_k);
					Tensor Q = input * q;
					Tensor K = input * k;
					Tensor V = input * v;
					Q = Q * K.t();
					Q  = Q/d_k;
					Q = Q.mask();
					Q = Q.softmax();
					Q = Q * V;
					return Q;
		}

		Tensor multiheadattention(std::vector<Tensor> qs, std::vector<Tensor> ks, std::vector<Tensor> vs, Tensor input,int d_k){
		std::vector<Tensor> res;
		for(int i =1;i<qs.size();i++){
		    res.push_back(attention(input,ks[i],qs[i],vs[i],d_k));
		}
		return attention(input,ks[0],qs[0],vs[0],d_k).concat_horizontal(res);
		}

		Tensor forward( Tensor input){
		    input = input.LayerNorm(gamma,beta);
		    input = input + multiheadattention(qs,ks,vs,input,d_k);
			return input;
		}
};


class Tokenizer {
private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    std::vector<std::pair<std::string, std::string>> merges;  // BPE merge rules
    int vocab_size;
    int eos_token;
    int bos_token;
    int pad_token;

    // GPT-2 specific: byte-level encoding
    std::vector<unsigned char> byte_to_unicode;
    std::unordered_map<unsigned char, unsigned char> unicode_to_byte;

    void init_byte_to_unicode() {
        // GPT-2 uses byte-level BPE - map bytes to unicode characters
        std::vector<unsigned char> bs;
        for (int b = 33; b <= 126; b++) bs.push_back(static_cast<unsigned char>(b));  // Printable ASCII
        for (int b = 161; b <= 172; b++) bs.push_back(static_cast<unsigned char>(b));  // Latin-1 supplement
        for (int b = 174; b <= 255; b++) bs.push_back(static_cast<unsigned char>(b));  // Latin-1 supplement

        int n = 0;
        for (int b = 0; b < 256; b++) {
            bool found = false;
            for (unsigned char c : bs) if (c == b) { found = true; break; }
            if (!found) {
                bs.push_back(static_cast<unsigned char>(b));
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            byte_to_unicode.push_back(bs[i]);
            unicode_to_byte[bs[i]] = static_cast<unsigned char>(i);
        }
    }

    // Convert bytes to unicode string (GPT-2 style)
    std::string bytes_to_unicode(const std::string& text) {
        std::string result;
        for (unsigned char c : text) {
            result += static_cast<char>(byte_to_unicode[c]);
        }
        return result;
    }

    // Convert unicode string back to bytes
    std::string unicode_to_bytes(const std::string& text) {
        std::string result;
        for (unsigned char c : text) {
            result += static_cast<char>(unicode_to_byte[c]);
        }
        return result;
    }

    // Get pairs of adjacent tokens
    std::vector<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word) {
        std::vector<std::pair<std::string, std::string>> pairs;
        if (word.empty()) return pairs;

        for (size_t i = 0; i < word.size() - 1; i++) {
            pairs.push_back({word[i], word[i + 1]});
        }
        return pairs;
    }

    // BPE encoding for a single word
    std::vector<std::string> bpe(const std::string& word) {
        // Split into characters (with space prefix for non-first words)
        std::vector<std::string> word_tokens;
        for (char c : word) {
            word_tokens.push_back(std::string(1, c));
        }

        if (word_tokens.empty()) return word_tokens;

        // Apply merges iteratively
        while (word_tokens.size() > 1) {
            // Find all pairs
            auto pairs = get_pairs(word_tokens);

            // Find the pair with the earliest merge rule
            int min_rank = INT32_MAX;
            size_t merge_idx = 0;

            for (size_t i = 0; i < pairs.size(); i++) {
                for (size_t m = 0; m < merges.size(); m++) {
                    if (merges[m].first == pairs[i].first && merges[m].second == pairs[i].second) {
                        if ((int)m < min_rank) {
                            min_rank = m;
                            merge_idx = i;
                        }
                        break;
                    }
                }
            }

            if (min_rank == INT32_MAX) break;  // No more merges possible

            // Apply the merge
            std::string merged = pairs[merge_idx].first + pairs[merge_idx].second;

            std::vector<std::string> new_tokens;
            size_t skip = 0;
            for (size_t i = 0; i < word_tokens.size(); i++) {
                if (i == merge_idx) {
                    new_tokens.push_back(merged);
                    skip = 1;
                } else if (skip > 0) {
                    skip--;
                } else {
                    new_tokens.push_back(word_tokens[i]);
                }
            }
            word_tokens = new_tokens;
        }

        return word_tokens;
    }

public:
    Tokenizer() : vocab_size(50257), eos_token(50256), bos_token(50256), pad_token(50256) {
        init_byte_to_unicode();
    }

    // Load vocabulary and merges from exported JSON files
    void load(const std::string& vocab_path, const std::string& merges_path = "") {
        // Load vocabulary
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Could not open vocab file: " + vocab_path);
        }

        std::stringstream buffer;
        buffer << vocab_file.rdbuf();
        std::string json_str = buffer.str();
        vocab_file.close();

        // Parse JSON (simple parser for token_to_id format)
        size_t pos = 0;
        while (pos < json_str.size()) {
            // Find key-value pairs
            size_t key_start = json_str.find('"', pos);
            if (key_start == std::string::npos) break;

            size_t key_end = json_str.find('"', key_start + 1);
            if (key_end == std::string::npos) break;

            std::string token = json_str.substr(key_start + 1, key_end - key_start - 1);

            size_t colon = json_str.find(':', key_end);
            if (colon == std::string::npos) break;

            size_t value_start = colon + 1;
            while (value_start < json_str.size() && (json_str[value_start] == ' ' || json_str[value_start] == '\n')) {
                value_start++;
            }

            int id = 0;
            bool negative = false;
            if (json_str[value_start] == '-') {
                negative = true;
                value_start++;
            }
            while (value_start < json_str.size() && std::isdigit(json_str[value_start])) {
                id = id * 10 + (json_str[value_start] - '0');
                value_start++;
            }
            if (negative) id = -id;

            token_to_id[token] = id;
            id_to_token[id] = token;

            pos = value_start;
        }

        vocab_size = token_to_id.size();

        // Update special tokens
        if (token_to_id.count("<|endoftext|>")) {
            eos_token = token_to_id["<|endoftext|>"];
            bos_token = eos_token;
            pad_token = eos_token;
        }

        std::cout << "Loaded vocabulary: " << vocab_size << " tokens\n";

        // Load merges if provided
        if (!merges_path.empty()) {
            std::ifstream merges_file(merges_path);
            if (merges_file.is_open()) {
                std::string line;
                while (std::getline(merges_file, line)) {
                    if (line.empty() || line[0] == '#') continue;

                    size_t space_pos = line.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string first = line.substr(0, space_pos);
                        std::string second = line.substr(space_pos + 1);
                        merges.push_back({first, second});
                    }
                }
                merges_file.close();
                std::cout << "Loaded " << merges.size() << " BPE merge rules\n";
            }
        }
    }

    // Encode text to token IDs with full BPE
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;

        // Convert to byte-level unicode (GPT-2 style)
        std::string normalized = bytes_to_unicode(text);

        // Split into words (preserving spaces)
        std::vector<std::string> words;
        std::string current;
        bool last_was_space = false;

        for (char c : normalized) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!current.empty()) {
                    words.push_back(current);
                    current.clear();
                }
                words.push_back(std::string(1, c));  // Keep space as separate token
                last_was_space = true;
            } else {
                current += c;
                last_was_space = false;
            }
        }
        if (!current.empty()) {
            words.push_back(current);
        }

        // Encode each word with BPE
        for (const auto& word : words) {
            if (word.empty()) continue;

            // Apply BPE
            std::vector<std::string> subtokens;
            if (merges.empty()) {
                // No merges - use character-level or direct lookup
                if (token_to_id.count(word)) {
                    tokens.push_back(token_to_id[word]);
                    continue;
                }
                subtokens.push_back(word);
            } else {
                subtokens = bpe(word);
            }

            // Convert subtokens to IDs
            for (const auto& sub : subtokens) {
                if (token_to_id.count(sub)) {
                    tokens.push_back(token_to_id[sub]);
                } else {
                    // Fallback: try to find closest match or use UNK
                    // For GPT-2, all byte combinations should be in vocab
                }
            }
        }

        return tokens;
    }

    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int id : tokens) {
            if (id_to_token.count(id)) {
                std::string token = id_to_token[id];

                // Skip special tokens
                if (id == eos_token) continue;

                result += token;
            }
        }

        // Convert back from byte-level unicode
        return unicode_to_bytes(result);
    }

    // Get token ID directly (fast lookup)
    int get_token_id(const std::string& token) const {
        auto it = token_to_id.find(token);
        return (it != token_to_id.end()) ? it->second : -1;
    }

    // Get token string from ID
    std::string get_token(int id) const {
        auto it = id_to_token.find(id);
        return (it != id_to_token.end()) ? it->second : "";
    }

    int eos() const { return eos_token; }
    int bos() const { return bos_token; }
    int pad() const { return pad_token; }
    int get_vocab_size() const { return vocab_size; }
};


class Model{

    private:
        int max_len;
        int vocab_size;
        int seq_len;
        int num_heads;
        int d_model;
        int blocks;
        Tensor token_embeddings;      // (vocab_size, d_model)
        Tensor output_projection;     // (d_model, vocab_size)
        std::vector<Transformer> transformers;  // One transformer per block

    public:

        // Constructor
        Model(int d_model, int max_len, int vocab_size, int seq_len, int num_heads, int blocks,
              Tensor token_embeddings,
              Tensor position_embeddings,
              Tensor output_projection,
              std::vector<std::vector<Tensor>> Qs,
              std::vector<std::vector<Tensor>> Ks,
              std::vector<std::vector<Tensor>> Vs,
              std::vector<Tensor> gammas,
              std::vector<Tensor> betas)
            : max_len(max_len),
              vocab_size(vocab_size),
              seq_len(seq_len),
              num_heads(num_heads),
              d_model(d_model),
              blocks(blocks),
              token_embeddings(token_embeddings),
              output_projection(output_projection)
        {
            // Validate embedding shapes
            if (token_embeddings.shape(0) != vocab_size || token_embeddings.shape(1) != d_model) {
                throw std::invalid_argument("Token embeddings shape mismatch: expected (vocab_size, d_model)");
            }

            if (position_embeddings.shape(0) != max_len || position_embeddings.shape(1) != d_model) {
                throw std::invalid_argument("Position embeddings shape mismatch: expected (max_len, d_model)");
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
                // Validate that each block has the correct number of heads
                if (Qs[i].size() != num_heads || Ks[i].size() != num_heads || Vs[i].size() != num_heads) {
                    throw std::invalid_argument("Number of Q, K, V tensors per block must match num_heads");
                }

                transformers.emplace_back(
                    d_model, max_len, vocab_size, seq_len, num_heads,
                    Qs[i], Ks[i], Vs[i],
                    gammas[i], betas[i]
                );
            }
        }


        Tensor position_encodings(Tensor input) {
            // Create positional encoding tensor
            std::vector<float> pe_data(input.shape(0) * input.shape(1), 0.0f);

            for(size_t pos = 0; pos < input.shape(0); pos++) {
                for(size_t i = 0; i < input.shape(1); i++) {
                    if (i % 2 == 0) {
                        // Even indices: sin(pos / 10000^(2i/d_model))
                        pe_data[pos * input.shape(1) + i] = std::sin(pos / std::pow(10000.0f, (2.0f * (i / 2)) / input.shape(1)));
                    } else {
                        // Odd indices: cos(pos / 10000^(2i/d_model))
                        pe_data[pos * input.shape(1) + i] = std::cos(pos / std::pow(10000.0f, (2.0f * (i / 2)) / input.shape(1)));
                    }
                }
            }

            Tensor pe(pe_data, input.shape(0), input.shape(1));

            // Add positional encoding to input
            std::vector<float> res(input.shape(0) * input.shape(1));
            for(size_t i = 0; i < input.shape(0) * input.shape(1); i++) {
                res[i] = input.getData()[i] + pe_data[i];
            }

            return Tensor(res, input.shape(0), input.shape(1));
        }


        Tensor embeddings(Tensor input) {

        }

        Tensor forward(Tensor input){

        }




};
