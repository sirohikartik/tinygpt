from transformers import GPT2Tokenizer
import json

tok = GPT2Tokenizer.from_pretrained("gpt2")
vocab = tok.get_vocab()
print(len(vocab))  # should be 50257

with open("gpt2_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("Done")

# # tokenizer.py
# from transformers import GPT2Tokenizer
# import json

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Get full vocabulary
# vocab = tokenizer.get_vocab()

# # Create reverse mapping (ID → token string)
# id_to_token = {id_: token for token, id_ in vocab.items()}

# # Save as JSON
# with open('gpt2_vocab.json', 'w', encoding='utf-8') as f:
#     json.dump({
#         'vocab_size': len(vocab),
#         'id_to_token': id_to_token,
#         'token_to_id': vocab,
#         'eos_token_id': tokenizer.eos_token_id,
#         'bos_token_id': tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id,
#         'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
#     }, f, ensure_ascii=False)

# print(f"Exported vocabulary: {len(vocab)} tokens")
# print(f"EOS token ID: {tokenizer.eos_token_id}")

# # Helper function for standalone tokenization
# def tokenize(text):
#     """Tokenize text and return token IDs"""
#     return tokenizer.encode(text)

# def detokenize(tokens):
#     """Convert token IDs back to text"""
#     return tokenizer.decode(tokens)

# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) > 1:
#         # Command line mode: tokenize text
#         text = ' '.join(sys.argv[1:])
#         tokens = tokenize(text)
#         print(','.join(map(str, tokens)))
#     else:
#         # Export vocab
#         pass
