"""
mLSTM (Matrix LSTM) 模块实现
使用矩阵记忆替代传统的向量记忆，增强记忆容量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class mLSTM(nn.Module):
    """
    Matrix LSTM - 使用矩阵形式的记忆单元
    
    特点：
    1. 使用矩阵记忆替代传统的cell state向量
    2. 通过外积更新实现更丰富的记忆表示
    3. 计算复杂度为O(d^2)，其中d为隐藏维度
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_heads: 注意力头数（用于自注意力机制）
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim * 4)  # q, k, v, g
        
        # 矩阵记忆参数
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # 门控参数
        self.input_gate = nn.Linear(hidden_dim, hidden_dim)
        self.forget_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        
        # 归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, state: tuple = None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, input_dim)
            state: 可选的初始状态 (C, n)，其中C是矩阵记忆，n是归一化项
        
        Returns:
            output: 输出张量，形状 (batch_size, seq_len, hidden_dim)
            new_state: 新的状态元组 (C, n)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        proj = self.input_proj(x)
        q, k, v, g = proj.chunk(4, dim=-1)
        
        # 应用激活函数
        q = torch.tanh(q)
        k = torch.tanh(k)
        v = self.dropout(v)
        g = F.silu(g)  # Swish激活函数
        
        # 初始化状态
        if state is None:
            C = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim, device=x.device)
            n = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            C, n, h = state
        
        outputs = []
        
        # 逐时间步处理
        for t in range(seq_len):
            q_t = q[:, t, :]  # (batch, hidden_dim)
            k_t = k[:, t, :]  # (batch, hidden_dim)
            v_t = v[:, t, :]  # (batch, hidden_dim)
            g_t = g[:, t, :]  # (batch, hidden_dim)
            
            # 计算门控
            i_t = torch.sigmoid(self.input_gate(v_t))  # 输入门
            f_t = torch.sigmoid(self.forget_gate(v_t))  # 遗忘门
            o_t = torch.sigmoid(self.output_gate(v_t))  # 输出门
            
            # 更新矩阵记忆（外积形式）
            k_t_expanded = k_t.unsqueeze(-1)  # (batch, hidden_dim, 1)
            v_t_expanded = v_t.unsqueeze(1)   # (batch, 1, hidden_dim)
            
            # C_t = f_t * C_{t-1} + i_t * (k_t^T * v_t)
            C = f_t.unsqueeze(-1) * C + i_t.unsqueeze(-1) * torch.bmm(k_t_expanded, v_t_expanded)
            
            # 更新归一化项
            n = f_t * n + i_t * k_t
            
            # 计算输出
            q_t_expanded = q_t.unsqueeze(1)  # (batch, 1, hidden_dim)
            h_candidate = torch.bmm(q_t_expanded, C).squeeze(1)  # (batch, hidden_dim)
            
            # 归一化
            denom = torch.clamp(torch.sum(q_t * n, dim=-1, keepdim=True), min=1e-6)
            h_candidate = h_candidate / denom
            
            # 应用输出门
            h = o_t * h_candidate
            
            outputs.append(h)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        output = self.output_proj(output)
        
        return output, (C, n, h)


class mLSTMCell(nn.Module):
    """
    简化的mLSTM单元，用于逐步处理
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.proj = nn.Linear(input_dim, hidden_dim * 4)
        
        # 门控
        self.gates = nn.Linear(hidden_dim, hidden_dim * 3)
        
    def forward(self, x: torch.Tensor, state: tuple):
        """
        单步前向传播
        
        Args:
            x: 输入 (batch, input_dim)
            state: (C, n, h)
        """
        batch_size = x.shape[0]
        C, n, h = state
        
        # 投影
        proj = self.proj(x)
        q, k, v, g = proj.chunk(4, dim=-1)
        
        q = torch.tanh(q)
        k = torch.tanh(k)
        g = F.silu(g)
        
        # 门控
        gates = self.gates(v)
        i, f, o = torch.split(torch.sigmoid(gates), self.hidden_dim, dim=-1)
        
        # 更新记忆
        k_exp = k.unsqueeze(-1)
        v_exp = v.unsqueeze(1)
        C = f.unsqueeze(-1) * C + i.unsqueeze(-1) * torch.bmm(k_exp, v_exp)
        n = f * n + i * k
        
        # 输出
        q_exp = q.unsqueeze(1)
        h_new = torch.bmm(q_exp, C).squeeze(1)
        denom = torch.clamp(torch.sum(q * n, dim=-1, keepdim=True), min=1e-6)
        h_new = h_new / denom
        h_new = o * h_new
        
        return h_new, (C, n, h_new)
