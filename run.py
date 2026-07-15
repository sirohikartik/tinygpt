import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import math
import numpy as np
import time
import os

device = "cpu"
class Transformer(nn.Module):
    def __init__(self, max_len, vocab_size, dropout, seq_len, num_heads=6, d_model=256):
        super().__init__()
        self.d_model    = d_model
        self.max_len    = max_len
        self.vocab_size = vocab_size
        self.dropout    = dropout
        self.seq_len    = seq_len
        self.num_heads  = num_heads
        self.register_buffer("mask", torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool())
        self.q    = nn.ModuleList(nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads))
        self.k    = nn.ModuleList(nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads))
        self.v    = nn.ModuleList(nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads))
        self.join  = nn.Linear(self.d_model, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 4, self.d_model)
        )

    def attention(self, q, k, v, x):
        Q   = q(x)
        K   = k(x)
        V   = v(x)
        out = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model // self.num_heads)
        T = x.size(1)
        mask = self.mask[:T, :T]
        out = out.masked_fill(mask, float('-inf'))

        out = torch.softmax(out, dim=-1)
        out = torch.matmul(out, V)
        return out

    def multiheadattention(self, x):
        first_one = self.attention(self.q[0], self.k[0], self.v[0], x)
        result = torch.cat(
            [first_one] + [self.attention(self.q[i], self.k[i], self.v[i], x) for i in range(1, self.num_heads)], -1
        )
        return self.join(result)

    def forward(self, x):
        x = x + self.multiheadattention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, max_len, vocab_size, dropout, seq_len, num_heads=6, d_model=256, blocks=5):
        super().__init__()
        self.max_len    = max_len
        self.vocab_size = vocab_size
        self.dropout    = dropout
        self.seq_len    = seq_len
        self.num_heads  = num_heads
        self.d_model    = d_model
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.transforms = nn.ModuleList(
            Transformer(self.max_len, self.vocab_size, self.dropout, self.seq_len, self.num_heads, self.d_model)
            for _ in range(blocks)
        )
        self.out = nn.Linear(self.d_model, self.vocab_size)

    def position_encodings(self, x):
        pos = torch.arange(self.max_len).unsqueeze(1).to(device)
        div = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(1000.0) / self.d_model)).to(device)
        pe  = torch.zeros(1, self.max_len, self.d_model).to(device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        x += pe[:, :x.size(1), :]
        return x

    def forward(self, x):
        x = self.embeddings(x)
        x = self.position_encodings(x)
        for block in self.transforms:
            x = block(x)
        return self.out(x)

def load_npy_weights(model):
    weights_dir = "weights"
    print(f"Loading weights from {weights_dir}...")
    
    # Embedding
    emb = np.load(os.path.join(weights_dir, "embeddings.weight.npy"))
    model.embeddings.weight.data = torch.from_numpy(emb).float().to(device)
    
    # Output Projection
    out_w = np.load(os.path.join(weights_dir, "out.weight.npy"))
    out_b = np.load(os.path.join(weights_dir, "out.bias.npy"))
    model.out.weight.data = torch.from_numpy(out_w).float().to(device)
    model.out.bias.data = torch.from_numpy(out_b).float().to(device)
    
    num_blocks = len(model.transforms)
    num_heads = model.num_heads
    
    for b in range(num_blocks):
        prefix = f"transforms.{b}."
        trans = model.transforms[b]
        
        # Norms
        trans.norm1.weight.data = torch.from_numpy(np.load(os.path.join(weights_dir, prefix + "norm1.weight.npy"))).float().to(device)
        trans.norm1.bias.data = torch.from_numpy(np.load(os.path.join(weights_dir, prefix + "norm1.bias.npy"))).float().to(device)
        trans.norm2.weight.data = torch.from_numpy(np.load(os.path.join(weights_dir, prefix + "norm2.weight.npy"))).float().to(device)
        trans.norm2.bias.data = torch.from_numpy(np.load(os.path.join(weights_dir, prefix + "norm2.bias.npy"))).float().to(device)
        
        # Multi-head attention
        for h in range(num_heads):
            h_str = str(h)
            # PyTorch Linear weights are (out, in), and npy are already (out, in)
            q_w = np.load(os.path.join(weights_dir, f"{prefix}q.{h_str}.weight.npy"))
            q_b = np.load(os.path.join(weights_dir, f"{prefix}q.{h_str}.bias.npy"))
            trans.q[h].weight.data = torch.from_numpy(q_w).float().to(device)
            trans.q[h].bias.data = torch.from_numpy(q_b).float().to(device)
            
            k_w = np.load(os.path.join(weights_dir, f"{prefix}k.{h_str}.weight.npy"))
            k_b = np.load(os.path.join(weights_dir, f"{prefix}k.{h_str}.bias.npy"))
            trans.k[h].weight.data = torch.from_numpy(k_w).float().to(device)
            trans.k[h].bias.data = torch.from_numpy(k_b).float().to(device)
            
            v_w = np.load(os.path.join(weights_dir, f"{prefix}v.{h_str}.weight.npy"))
            v_b = np.load(os.path.join(weights_dir, f"{prefix}v.{h_str}.bias.npy"))
            trans.v[h].weight.data = torch.from_numpy(v_w).float().to(device)
            trans.v[h].bias.data = torch.from_numpy(v_b).float().to(device)
            
        # Join
        join_w = np.load(os.path.join(weights_dir, prefix + "join.weight.npy"))
        join_b = np.load(os.path.join(weights_dir, prefix + "join.bias.npy"))
        trans.join.weight.data = torch.from_numpy(join_w).float().to(device)
        trans.join.bias.data = torch.from_numpy(join_b).float().to(device)
        
        # FFN
        ffn1_w = np.load(os.path.join(weights_dir, prefix + "ffn.0.weight.npy"))
        ffn1_b = np.load(os.path.join(weights_dir, prefix + "ffn.0.bias.npy"))
        trans.ffn[0].weight.data = torch.from_numpy(ffn1_w).float().to(device)
        trans.ffn[0].bias.data = torch.from_numpy(ffn1_b).float().to(device)
        
        ffn2_w = np.load(os.path.join(weights_dir, prefix + "ffn.3.weight.npy"))
        ffn2_b = np.load(os.path.join(weights_dir, prefix + "ffn.3.bias.npy"))
        trans.ffn[3].weight.data = torch.from_numpy(ffn2_w).float().to(device)
        trans.ffn[3].bias.data = torch.from_numpy(ffn2_b).float().to(device)

model = Model(
    max_len=128,
    vocab_size=50257,
    dropout=0.1,
    seq_len=128,
    num_heads=8,
    d_model=256,
    blocks=6
).to(device)

load_npy_weights(model)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "James bond"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
print(f"Prompt: {prompt}")

max_new_tokens = 100

# Benchmark timing
start_time = time.time()

# TTFT (Time to First Token)
with torch.no_grad():
    outputs = model(input_ids[:, -model.seq_len:])
    
logits = outputs[:, -1, :]
probs = torch.softmax(logits / 0.8, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
input_ids = torch.cat([input_ids, next_token], dim=1)

ttft_ms = (time.time() - start_time) * 1000
print(f"TTFT: {ttft_ms:.2f} ms")

# Decode loop
decode_start = time.time()
for _ in range(max_new_tokens - 1):
    with torch.no_grad():
        outputs = model(input_ids[:, -model.seq_len:])
    
    logits = outputs[:, -1, :]
    probs = torch.softmax(logits / 0.8, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    input_ids = torch.cat([input_ids, next_token], dim=1)

decode_end = time.time()
total_decode_time_ms = (decode_end - decode_start) * 1000
avg_time_per_token = total_decode_time_ms / (max_new_tokens - 1)
throughput = (max_new_tokens - 1) / ((decode_end - decode_start))

print(f"Avg Time Per Token: {avg_time_per_token:.2f} ms/token")
print(f"Generation Throughput: {throughput:.2f} tokens/sec")

output_text = tokenizer.decode(input_ids[0])
print("\n=== Generated Text ===\n", output_text, "\n======================")
