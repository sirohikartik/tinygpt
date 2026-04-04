import torch
import torch.nn as nn
import math
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

        pos = torch.arange(self.max_len, device=x.device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * (-math.log(1000.0) / self.d_model))

        pe  = torch.zeros(1, self.max_len, self.d_model, device=x.device)
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



import torch
import numpy as np
import os

# load your model first (same as before)
device = "cpu"

model = Model(
    max_len=128,
    vocab_size=50257,
    dropout=0.1,
    seq_len=128,
    num_heads=8,
    d_model=256,
    blocks=6
).to(device)

checkpoint = torch.load("model.pt", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# create weights folder
os.makedirs("weights", exist_ok=True)

# save all parameters
for name, param in model.state_dict().items():
    path = f"weights/{name}.npy"
    np.save(path, param.detach().cpu().numpy())

print("✅ All weights saved to ./weights/")




dummy_input = torch.zeros(1, 128, dtype=torch.long)  # adjust shape to your model's input

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["output"]
)
