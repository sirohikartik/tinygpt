#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <climits>
#include "json.hpp"

class Tokenizer {
private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> bpe_ranks;

    int vocab_size = 0;
    int eos_token = -1;
    int bos_token = -1;
    int pad_token = -1;
    int unk_token = -1;

    std::unordered_map<int, int> byte_to_unicode;
    std::unordered_map<int, int> unicode_to_byte;

    // ============================
    // UTF-8 helpers
    // ============================
    std::string encode_utf8(int cp) {
        std::string out;
        if (cp < 0x80) out += cp;
        else if (cp < 0x800) {
            out += (0xC0 | (cp >> 6));
            out += (0x80 | (cp & 0x3F));
        } else {
            out += (0xE0 | (cp >> 12));
            out += (0x80 | ((cp >> 6) & 0x3F));
            out += (0x80 | (cp & 0x3F));
        }
        return out;
    }

    std::vector<int> decode_utf8(const std::string& s) {
        std::vector<int> out;
        for (size_t i = 0; i < s.size();) {
            unsigned char c = s[i];

            if (c < 128) {
                out.push_back(c);
                i++;
            } else if ((c >> 5) == 0x6) {
                if (i + 1 >= s.size()) break;
                int cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
                out.push_back(cp);
                i += 2;
            } else {
                if (i + 2 >= s.size()) break;
                int cp = ((c & 0x0F) << 12) |
                         ((s[i + 1] & 0x3F) << 6) |
                         (s[i + 2] & 0x3F);
                out.push_back(cp);
                i += 3;
            }
        }
        return out;
    }

    // ============================
    // Byte ↔ Unicode (GPT-style)
    // ============================
    void init_byte_to_unicode() {
        std::vector<int> bs;

        for (int b = 33; b <= 126; b++) bs.push_back(b);
        for (int b = 161; b <= 172; b++) bs.push_back(b);
        for (int b = 174; b <= 255; b++) bs.push_back(b);

        std::vector<int> cs = bs;
        int n = 0;

        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            byte_to_unicode[bs[i]] = cs[i];
            unicode_to_byte[cs[i]] = bs[i];
        }
    }



    std::string unicode_to_bytes(const std::string& text) {
        std::string result;
        auto cps = decode_utf8(text);

        for (int cp : cps) {
            if (unicode_to_byte.count(cp))
                result += static_cast<char>(unicode_to_byte[cp]);
        }
        return result;
    }

    // ============================
    //
    // BPE
    // ============================
    std::vector<std::string> get_pairs(const std::vector<std::string>& word) {
        std::vector<std::string> pairs;
        for (size_t i = 0; i < word.size() - 1; i++) {
            pairs.push_back(word[i] + " " + word[i + 1]);
        }
        return pairs;
    }

    std::vector<std::string> bpe(const std::string& token) {

        std::vector<std::string> word;

            // split into UTF-8 characters, not bytes
            size_t i = 0;
            while (i < token.size()) {
                unsigned char c = token[i];
                int char_len = 1;
                if ((c >> 5) == 0x6) char_len = 2;       // 2-byte UTF-8
                else if ((c >> 4) == 0xe) char_len = 3;   // 3-byte UTF-8
                else if ((c >> 3) == 0x1e) char_len = 4;  // 4-byte UTF-8

                word.push_back(token.substr(i, char_len));
                i += char_len;
            }

        while (true) {
            auto pairs = get_pairs(word);

            int min_rank = INT_MAX;
            std::string best_pair;

            for (auto& p : pairs) {
                if (bpe_ranks.count(p) && bpe_ranks[p] < min_rank) {
                    min_rank = bpe_ranks[p];
                    best_pair = p;
                }
            }

            if (min_rank == INT_MAX) break;

            std::stringstream ss(best_pair);
            std::string first, second;
            ss >> first >> second;

            std::vector<std::string> new_word;

            for (size_t i = 0; i < word.size();) {
                if (i < word.size() - 1 &&
                    word[i] == first &&
                    word[i + 1] == second) {

                    new_word.push_back(first + second);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i++;
                }
            }

            word = new_word;
        }

        return word;
    }

public:
    Tokenizer() {
        init_byte_to_unicode();
    }
    std::string bytes_to_unicode(const std::string& text) {
        std::string result;
        for (unsigned char c : text) {
            int u = byte_to_unicode[c];
            result += encode_utf8(u);
        }
        return result;
    }

    // ============================
    // LOAD
    // ============================
    void load(const std::string& vocab_path,
              const std::string& merges_path) {

        std::ifstream vf(vocab_path);
        if (!vf.is_open())
            throw std::runtime_error("Cannot open vocab file");

        nlohmann::json j;
        vf >> j;

        for (auto& [tok, id] : j.items()) {
            try {
                if (!id.is_number_integer()) {
                    std::cerr << "Skipping invalid token: "
                              << tok << " -> " << id << "\n";
                    continue;
                }

                int val = id.get<int>();
                token_to_id[tok] = val;
                id_to_token[val] = tok;

            } catch (const std::exception& e) {
                std::cerr << "Error parsing token: "
                          << tok << " -> " << id << "\n";
            }
        }

        vocab_size = token_to_id.size();

        if (token_to_id.count("<|endoftext|>"))
            eos_token = bos_token = pad_token =
                token_to_id["<|endoftext|>"];

        if (token_to_id.count("<|unk|>"))
            unk_token = token_to_id["<|unk|>"];

        std::ifstream mf(merges_path);
        if (!mf.is_open())
            throw std::runtime_error("Cannot open merges file");

        std::string line;
        int rank = 0;

        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::stringstream ss(line);
            std::string a, b;

            if (!(ss >> a >> b)) {
                std::cerr << "Skipping bad merge line: "
                          << line << "\n";
                continue;
            }

            bpe_ranks[a + " " + b] = rank++;
        }

        std::cout << "Loaded vocab: " << vocab_size << "\n";
        std::cout << "Loaded merges: " << bpe_ranks.size() << "\n";
    }

    // ============================
    // ENCODE
    // ============================
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;

        std::vector<std::string> raw_words;
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) raw_words.push_back(word);

        for (size_t i = 0; i < raw_words.size(); i++) {
            std::string w = (i == 0) ? raw_words[i] : " " + raw_words[i];
            std::string norm = bytes_to_unicode(w);
            auto pieces = bpe(norm);
            for (auto& p : pieces) {
                if (token_to_id.count(p))
                    tokens.push_back(token_to_id[p]);
                else if (unk_token != -1)
                    tokens.push_back(unk_token);
            }
        }
        return tokens;
    }

    // ============================
    // DECODE
    // ============================
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        for (int id : tokens) {
            if (id == eos_token) continue;
            if (id_to_token.count(id)) {
                std::cerr << id << " -> [" << id_to_token[id] << "]\n";
                text += id_to_token[id];
            }
        }
        std::cerr << "raw text before unicode_to_bytes: [" << text << "]\n";
        return unicode_to_bytes(text);
    }

    // ============================
    // GETTERS
    // ============================
    int eos() const { return eos_token; }
    int bos() const { return bos_token; }
    int pad() const { return pad_token; }
    int vocab() const { return vocab_size; }
};

#endif
