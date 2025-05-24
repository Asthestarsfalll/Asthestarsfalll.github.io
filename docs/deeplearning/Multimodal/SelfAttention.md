```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 标准缩放点积自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)
        
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.embed_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.bmm(attn_probs, v)  # (B, S, D)
        return self.out_proj(out)

# 2. 多头自注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        # 投影并拆分头
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 合并头
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)

# 3. 多查询自注意力（共享键值头）
class MultiQuerySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim)  # 单头投影
        self.v_proj = nn.Linear(embed_dim, self.head_dim)  # 单头投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        # 查询拆分多头
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # 键值广播到所有头, repeat
        k = self.k_proj(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = self.v_proj(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)

# 4. 组查询自注意力
class GroupQuerySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, dropout=0.1):
        super().__init__()
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_groups * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_groups * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        # 查询拆分多头
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # 键值按组拆分
        k = self.k_proj(x).view(B, S, self.num_groups, self.head_dim)
        k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_groups, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)

# TODO
```
