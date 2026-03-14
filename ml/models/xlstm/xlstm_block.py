"""
xLSTM Block 实现
整合mLSTM和sLSTM两种变体，通过门控机制融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlstm import mLSTM
from .slstm import sLSTM


class xLSTMBlock(nn.Module):
    """
    xLSTM Block - 整合mLSTM和sLSTM的完整模块
    
    架构：
    1. 输入投影层
    2. 并行mLSTM和sLSTM分支
    3. 门控融合层
    4. 前馈网络
    5. 残差连接
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        num_heads: mLSTM注意力头数
        dropout: Dropout比率
        use_mlstm: 是否使用mLSTM分支
        use_slstm: 是否使用sLSTM分支
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_mlstm: bool = True,
        use_slstm: bool = True,
        ff_mult: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_mlstm = use_mlstm
        self.use_slstm = use_slstm
        
        assert use_mlstm or use_slstm, "至少启用一个LSTM分支"
        
        # 输入投影
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # mLSTM分支
        if use_mlstm:
            self.mlstm = mLSTM(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.mlstm_norm = nn.LayerNorm(hidden_dim)
        
        # sLSTM分支
        if use_slstm:
            self.slstm = sLSTM(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout
            )
            self.slstm_norm = nn.LayerNorm(hidden_dim)
        
        # 门控融合层
        if use_mlstm and use_slstm:
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 前馈网络
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mlstm_state=None, slstm_state=None):
        """
        前向传播
        
        Args:
            x: 输入 (batch, seq_len, input_dim)
            mlstm_state: mLSTM初始状态
            slstm_state: sLSTM初始状态
        
        Returns:
            output: 输出 (batch, seq_len, input_dim)
            new_mlstm_state: 新的mLSTM状态
            new_slstm_state: 新的sLSTM状态
        """
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x_norm = self.input_norm(x)
        x_proj = self.input_proj(x_norm)
        
        # 并行处理两个分支
        outputs = []
        new_mlstm_state = None
        new_slstm_state = None
        
        if self.use_mlstm:
            mlstm_out, new_mlstm_state = self.mlstm(x_proj, mlstm_state)
            mlstm_out = self.mlstm_norm(mlstm_out)
            outputs.append(mlstm_out)
        
        if self.use_slstm:
            slstm_out, new_slstm_state = self.slstm(x_proj, slstm_state)
            slstm_out = self.slstm_norm(slstm_out)
            outputs.append(slstm_out)
        
        # 融合两个分支的输出
        if len(outputs) == 2:
            # 门控融合
            concat_out = torch.cat(outputs, dim=-1)
            gate = self.fusion_gate(concat_out)
            gated_pair = torch.cat([gate * outputs[0], (1 - gate) * outputs[1]], dim=-1)
            lstm_out = self.fusion_proj(gated_pair)
        else:
            lstm_out = outputs[0]
        
        # 第一个残差连接
        hidden = x_proj + lstm_out
        
        # 前馈网络
        ff_out = self.ff(self.ff_norm(hidden))
        
        # 第二个残差连接
        hidden = hidden + ff_out
        
        # 输出投影
        output = self.output_proj(hidden)
        
        # 最终残差连接
        output = residual + output
        
        return output, new_mlstm_state, new_slstm_state


class StackedxLSTM(nn.Module):
    """
    堆叠的xLSTM Block
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_mlstm: bool = True,
        use_slstm: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            xLSTMBlock(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_mlstm=use_mlstm,
                use_slstm=use_slstm
            )
            for i in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim if num_layers > 0 else input_dim)
    
    def forward(self, x: torch.Tensor, states: list = None):
        """
        前向传播
        
        Args:
            x: 输入 (batch, seq_len, input_dim)
            states: 各层的初始状态列表
        """
        if states is None:
            states = [None] * self.num_layers
        
        new_states = []
        output = x
        
        for i, layer in enumerate(self.layers):
            mlstm_state = None
            slstm_state = None
            if states[i] is not None:
                mlstm_state, slstm_state = states[i]
            output, mlstm_state, slstm_state = layer(output, mlstm_state, slstm_state)
            new_states.append((mlstm_state, slstm_state))
        
        output = self.output_norm(output)
        
        return output, new_states
