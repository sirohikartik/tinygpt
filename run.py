import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, AutoTokenizer
import math


device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

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


model = Model(
    max_len=128,
    vocab_size=50257,
    dropout=0.1,
    seq_len=128,
    num_heads=8,   # FIXED
    d_model=256,
    blocks=6       # FIXED
).to(device)


checkpoint = torch.load("model.pt", map_location=device)
model.load_state_dict(checkpoint)

model.eval()


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = input(":")
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

max_new_tokens = 128

for _ in range(max_new_tokens):
    # crop if longer than seq_len
    input_crop = input_ids[:, -model.seq_len:]

    with torch.no_grad():
        outputs = model(input_crop)
    
    temperature = 0.8
    k = 20

    logits = outputs[:, -1, :]
    probs = torch.softmax(logits / temperature, dim=-1)

    values, indices = torch.topk(probs, k)
    probs = torch.zeros_like(probs).scatter_(1, indices, values)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)

    input_ids = torch.cat([input_ids, next_token], dim=1)

output_text = tokenizer.decode(input_ids[0])
print(output_text)


