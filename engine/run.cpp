#include"runner.hpp"
#include "tokenizer.hpp"

int main() {
	Runner runner;
	runner.run();
// 	Tokenizer tokenizer;
// 	tokenizer.load("gpt2_vocab.json", "merges.txt");
// 	std::vector<int> tokens = tokenizer.encode("Hello, how are you?");
// for (int t : tokens)
//     std::cout << t << "\n";

//     std::string norm = tokenizer.bytes_to_unicode("Hello, how are you?");
//     std::cerr << "normalized: ";
//     for (unsigned char c : norm)
//         std::cerr << std::hex << (int)c << " ";
//     std::cerr << "\n";
}
