#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include<string>
#include<vector>
#include<unordered_map>
#include<unordered_set>
#include<fstream>
#include<sstream>
#include<cctype>
#include<cstdint>
#include<iostream>
#include<climits>

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
                bs.push_back(static_cast<unsigned char>(256 + n++));
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
            auto it = unicode_to_byte.find(c);
            if (it != unicode_to_byte.end()) {
                result += static_cast<char>(it->second);
            } else {
                result += static_cast<char>(c);
            }
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
        // Split into characters
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
            int min_rank = INT_MAX;
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

            if (min_rank == INT_MAX) break;  // No more merges possible

            // Apply the merge
            std::string merged = pairs[merge_idx].first + pairs[merge_idx].second;

            std::vector<std::string> new_tokens;
            for (size_t i = 0; i < word_tokens.size(); i++) {
                if (i == merge_idx) {
                    new_tokens.push_back(merged);
                    if (i + 1 < word_tokens.size()) i++;  // Skip next token
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

        // Parse JSON (simple parser for token_to_id format: {"token": id, ...})
        size_t pos = 0;
        while (pos < json_str.size()) {
            // Find key (token string)
            size_t key_start = json_str.find('"', pos);
            if (key_start == std::string::npos) break;

            size_t key_end = json_str.find('"', key_start + 1);
            if (key_end == std::string::npos) break;

            // Handle escaped characters in token
            std::string token;
            for (size_t i = key_start + 1; i < key_end; i++) {
                if (json_str[i] == '\\' && i + 1 < key_end) {
                    i++;
                    switch (json_str[i]) {
                        case 'n': token += '\n'; break;
                        case 't': token += '\t'; break;
                        case 'r': token += '\r'; break;
                        case '\\': token += '\\'; break;
                        case '"': token += '"'; break;
                        default: token += json_str[i];
                    }
                } else {
                    token += json_str[i];
                }
            }

            size_t colon = json_str.find(':', key_end);
            if (colon == std::string::npos) break;

            size_t value_start = colon + 1;
            while (value_start < json_str.size() && (json_str[value_start] == ' ' || json_str[value_start] == '\n' || json_str[value_start] == '\t')) {
                value_start++;
            }

            int id = 0;
            bool negative = false;
            if (value_start < json_str.size() && json_str[value_start] == '-') {
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

        // Update special tokens from vocab
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

        for (char c : normalized) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!current.empty()) {
                    words.push_back(current);
                    current.clear();
                }
                words.push_back(std::string(1, c));  // Keep space as separate token
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            words.push_back(current);
        }

        // Encode each word with BPE
        for (const auto& word : words) {
            if (word.empty()) continue;

            std::vector<std::string> subtokens;
            if (merges.empty()) {
                // No merges - use direct lookup
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

#endif // TOKENIZER_HPP
