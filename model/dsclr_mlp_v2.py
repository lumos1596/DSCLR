import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCLR_MLP_V2(nn.Module):
    """
    DSCLR 动态参数预测器 V2 - 改进版
    
    改进点：
    1. 使用拼接特征：[Q+, Q-, similarity(Q+, Q-)]
    2. 更深的网络结构
    3. 残差连接
    4. 更合理的参数范围
    
    支持两种调用方式：
    1. 单参数模式: mlp(q_minus, encoder_type='bge') -> 兼容旧代码
    2. 双参数模式: mlp(q_plus, q_minus, encoder_type='repllama') -> 推荐使用
    
    MLP 输出两个标量:
    - alpha: 惩罚力度，范围 [0, MAX_ALPHA]
    - tau: 容忍底线，范围 [TAU_MIN, TAU_MAX]
    """
    def __init__(self, input_dim=4096, hidden_dim=512, use_concat_features=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_concat_features = use_concat_features
        
        if use_concat_features:
            self.feature_dim = input_dim * 2 + 1
        else:
            self.feature_dim = input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 4, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.mlp[-1].bias.data[0] = 0.5
        self.mlp[-1].bias.data[1] = 0.5
    
    def forward(self, *args, encoder_type='bge', **kwargs):
        """
        支持两种调用方式：
        
        方式1 (单参数模式 - 兼容旧代码):
            alpha, tau = mlp(q_minus, encoder_type='bge')
            Args:
                q_minus: 负向意图特征 [batch_size, input_dim]
                encoder_type: 编码器类型 ('bge', 'mistral', 'repllama')
        
        方式2 (双参数模式 - 推荐):
            alpha, tau = mlp(q_plus, q_minus, encoder_type='repllama')
            Args:
                q_plus: 正向意图特征 [batch_size, input_dim]
                q_minus: 负向意图特征 [batch_size, input_dim]
                encoder_type: 编码器类型 ('bge', 'mistral', 'repllama')

        Returns:
            alpha: 惩罚力度 [batch_size]
            tau: 容忍底线 [batch_size]
        """
        if len(args) == 1:
            q_minus = args[0]
            return self._forward_single(q_minus, encoder_type)
        elif len(args) >= 2:
            q_plus = args[0]
            q_minus = args[1]
            return self._forward_dual(q_plus, q_minus, encoder_type)
        else:
            raise ValueError(f"Expected 1 or 2 positional arguments, got {len(args)}")
    
    def _build_features(self, q_plus, q_minus):
        """构建拼接特征"""
        if self.use_concat_features:
            similarity = F.cosine_similarity(q_plus, q_minus, dim=-1, eps=1e-8).unsqueeze(-1)
            features = torch.cat([q_plus, q_minus, similarity], dim=-1)
        else:
            features = q_minus
        return features
    
    def _forward_single(self, q_minus, encoder_type='bge'):
        """单参数模式：仅使用 q_minus（兼容旧代码）"""
        if self.use_concat_features:
            q_plus_dummy = torch.zeros_like(q_minus)
            features = self._build_features(q_plus_dummy, q_minus)
        else:
            features = q_minus
        
        raw_output = self.mlp(features)
        raw_alpha = raw_output[:, 0]
        raw_tau = raw_output[:, 1]
        
        alpha, tau = self._get_alpha_tau(raw_alpha, raw_tau, encoder_type)
        
        return alpha, tau
    
    def _forward_dual(self, q_plus, q_minus, encoder_type='bge'):
        """双参数模式：使用 Q+ 和 Q- 的拼接特征"""
        features = self._build_features(q_plus, q_minus)
        
        raw_output = self.mlp(features)
        raw_alpha = raw_output[:, 0]
        raw_tau = raw_output[:, 1]
        
        alpha, tau = self._get_alpha_tau(raw_alpha, raw_tau, encoder_type)
        
        return alpha, tau
    
    def _get_alpha_tau(self, raw_alpha, raw_tau, encoder_type='bge'):
        """根据 encoder_type 获取 alpha 和 tau"""
        if encoder_type == 'repllama':
            MAX_ALPHA = 1.5
            TAU_MIN = 0.4
            TAU_MAX = 0.8
        elif encoder_type == 'mistral':
            MAX_ALPHA = 1.5
            TAU_MIN = 0.4
            TAU_MAX = 0.8
        else:
            MAX_ALPHA = 2.0
            TAU_MIN = 0.3
            TAU_MAX = 0.7
        
        alpha = torch.sigmoid(raw_alpha) * MAX_ALPHA
        tau = TAU_MIN + torch.sigmoid(raw_tau) * (TAU_MAX - TAU_MIN)
        
        return alpha, tau
