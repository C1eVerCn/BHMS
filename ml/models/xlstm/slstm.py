"""
sLSTM (Scalar LSTM) 模块实现
使用指数门控机制增强非线性表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class sLSTM(nn.Module):
    """
    Scalar LSTM - 使用指数门控机制的LSTM变体
    
    特点：
    1. 使用指数激活函数替代传统的sigmoid门控
    2. 增强非线性表达能力
    3. 保持标量形式的cell state和hidden state
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_layers: 堆叠层数
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            sLSTMLayer(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                dropout if i < num_layers - 1 else 0
            )
            for i in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, state: list = None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, input_dim)
            state: 可选的初始状态列表，每层一个元组 (h, c, m)
        
        Returns:
            output: 输出张量，形状 (batch_size, seq_len, hidden_dim)
            new_states: 新的状态列表
        """
        if state is None:
            state = [None] * self.num_layers
        
        new_states = []
        output = x
        
        for i, layer in enumerate(self.layers):
            output, new_state = layer(output, state[i])
            new_states.append(new_state)
        
        return output, new_states


class sLSTMLayer(nn.Module):
    """
    单层sLSTM实现
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影 - 计算四个门控和候选状态
        self.input_proj = nn.Linear(input_dim, hidden_dim * 5)  # i, f, z, o, input
        
        # 隐藏状态投影
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim * 4)  # i, f, z, o
        
        # 归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
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
            x: 输入 (batch, seq_len, input_dim)
            state: (h, c, m) 隐藏状态、细胞状态、归一化项
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 初始化状态
        if state is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
            m = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            h, c, m = state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # 输入投影
            input_proj = self.input_proj(x_t)
            i_x, f_x, z_x, o_x, inp = input_proj.chunk(5, dim=-1)
            
            # 隐藏状态投影
            hidden_proj = self.hidden_proj(h)
            i_h, f_h, z_h, o_h = hidden_proj.chunk(4, dim=-1)
            
            # 使用稳定化后的指数门控，避免长序列时 cell state 数值爆炸。
            log_i = torch.clamp(i_x + i_h, min=-10.0, max=10.0)
            log_f = torch.clamp(f_x + f_h, min=-10.0, max=10.0)
            m_new = torch.maximum(log_f + m, log_i)
            i_t = torch.exp(log_i - m_new)
            f_t = torch.exp(log_f + m - m_new)

            # 候选状态
            z_t = torch.tanh(z_x + z_h)

            # 归一化后的细胞状态更新
            c = f_t * c + i_t * z_t
            c_norm = c

            # 输出门
            o_t = torch.sigmoid(o_x + o_h)
            
            # 新的隐藏状态
            h = o_t * torch.tanh(c_norm)
            m = m_new
            
            outputs.append(h)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        output = self.norm(output)
        output = self.dropout(output)
        
        return output, (h, c, m)


class sLSTMCell(nn.Module):
    """
    简化的sLSTM单元，用于逐步处理
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 合并输入和隐藏状态的投影
        self.proj = nn.Linear(input_dim + hidden_dim, hidden_dim * 5)
        
    def forward(self, x: torch.Tensor, state: tuple):
        """
        单步前向传播
        
        Args:
            x: 输入 (batch, input_dim)
            state: (h, c, m)
        """
        h, c, m = state
        
        # 合并输入和隐藏状态
        combined = torch.cat([x, h], dim=-1)
        
        # 投影
        proj = self.proj(combined)
        i_gate, f_gate, z_gate, o_gate, inp = proj.chunk(5, dim=-1)
        
        # 指数门控的稳定化实现
        log_i = torch.clamp(i_gate, min=-10.0, max=10.0)
        log_f = torch.clamp(f_gate, min=-10.0, max=10.0)
        m_new = torch.maximum(log_f + m, log_i)
        i_t = torch.exp(log_i - m_new)
        f_t = torch.exp(log_f + m - m_new)
        
        # 候选状态
        z_t = torch.tanh(z_gate)
        
        # 更新细胞状态
        c_new = f_t * c + i_t * z_t
        c_norm = c_new
        
        # 输出
        o_t = torch.sigmoid(o_gate)
        h_new = o_t * torch.tanh(c_norm)
        
        return h_new, (h_new, c_new, m_new)
