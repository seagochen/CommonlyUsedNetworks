import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, hidden_size, max_seq_len, dropout_prob):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_size)
        self.encoder_layers = nn.ModuleList([EncoderLayer(embedding_size, num_heads, hidden_size, dropout_prob) for _ in range(num_layers)])
        self.fc = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, x):
        pos = torch.arange(0, x.size(1)).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        x = self.embedding(x) * torch.sqrt(self.embedding.embedding_dim)
        pos = self.pos_embedding(pos)
        x = x + pos
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.fc(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_size, num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.feedforward = FeedForward(embedding_size, hidden_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        out = self.self_attention(x, x, x)
        out = self.dropout(out)
        out = self.norm1(out + x)
        out = self.feedforward(out)
        out = self.dropout(out)
        out = self.norm2(out + out)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.q = nn.Linear(embedding_size, embedding_size)
        self.k = nn.Linear(embedding_size, embedding_size)
        self.v = nn.Linear(embedding_size, embedding_size)
        self.fc = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, x_q, x_k, x_v):
        bs = x_q.size(0)
        q = self.q(x_q)
        k = self.k(x_k)
        v = self.v(x_v)
        q = q.view(bs, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(bs, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(bs, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_size).float().to(x_q.device))
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.embedding_size)
        out = self.fc(out)
        return out
