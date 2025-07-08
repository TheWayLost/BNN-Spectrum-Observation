import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.abspath('..'))
import torchbnn as bnn




class BayesianAttention(nn.Module):
    """
    严格对应 ihead_basic_model.py 中 Attention 类的贝叶斯版本。
    - W_k, W_v, W_o 是可学习的贝叶斯线性层。
    - W_q 是恒等映射，与原代码一致。
    """
    def __init__(self, dim: int, prior_mu: float = 0.0, prior_sigma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        # Query 投影是恒等映射
        self.wq = nn.Identity()

        # Key, Value, Output 投影是贝叶斯线性层
        self.wk = bnn.BayesLinear(prior_mu, prior_sigma, dim, dim, bias=False)
        self.wv = bnn.BayesLinear(prior_mu, prior_sigma, dim, dim, bias=False)
        self.wo = bnn.BayesLinear(prior_mu, prior_sigma, dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        bs, slen, _ = x.shape

        # 投影 Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 缩放点积注意力 (论文中没有多头，所以我们这里也用单头，dim=d_model)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.dim)
        
        # 应用 causal mask
        scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)

        # 计算 V 的加权和
        h = torch.matmul(attn_weights, xv)

        # 输出投影
        output = self.wo(h)
        return output
    

class BayesianFeedForward(nn.Module):
    """
    严格对应 ihead_basic_model.py 中 FeedForward 类的贝叶斯版本。
    """
    def __init__(self, dim: int, hidden_dim: int, relu: bool = True, prior_mu: float = 0.0, prior_sigma: float = 1.0):
        super().__init__()
        self.w1 = bnn.BayesLinear(prior_mu, prior_sigma, dim, hidden_dim, bias=False)
        self.w2 = bnn.BayesLinear(prior_mu, prior_sigma, hidden_dim, dim, bias=False)
        self.relu = relu

    def forward(self, x):
        h = self.w1(x)
        if self.relu:
            h = F.relu(h)
        output = self.w2(h)
        return output
    

class BayesianTransformerBlock(nn.Module):
    """
    严格对应 ihead_basic_model.py 中 TransformerBlock 类的贝叶斯版本。
    它精确地实现了论文公式(1)中描述的非标准数据流。
    """
    def __init__(self, dim: int, mlp_multiplier: int = 4, relu: bool = True, use_ffn: bool = True,
                 prior_mu: float = 0.0, prior_sigma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.attention = BayesianAttention(dim=dim, prior_mu=prior_mu, prior_sigma=prior_sigma)
        
        self.use_ffn = use_ffn
        if self.use_ffn:
            hidden_dim = dim * mlp_multiplier
            self.ff = BayesianFeedForward(dim=dim, hidden_dim=hidden_dim, relu=relu,
                                          prior_mu=prior_mu, prior_sigma=prior_sigma)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: 对应公式中的 x̂_t 或 x¹_t
        
        # h_attn: 对应公式中的 h¹_t 或 h²_t
        h_attn = self.attention(x, mask)
        
        # 第一次残差连接: h_res = x + h_attn
        # h_res: 对应公式中 F_1 或 F_2 的输入 (x_t + h_t)
        h_res = x + h_attn
        
        if not self.use_ffn:
            # 如果没有FFN层（例如，在只有Attention的简化模型中）
            return h_res

        # h_ffn: 对应公式中的 F₁(h_res) 或 F₂(h_res)
        h_ffn = self.ff(h_res)

        # 第二次残差连接: output = h_res + h_ffn
        # output: 对应公式中的 x¹_t 或 x²_t
        output = h_res + h_ffn
        
        return output
    

class BayesianTransformer(nn.Module):
    """
    完整的、严格遵循论文公式(1)和其代码实现的贝叶斯Transformer。
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int,
                 mlp_multiplier: int = 4, relu: bool = True,
                 prior_mu: float = 0.0, prior_sigma: float = 1.0):
        """
        :param vocab_size: 词汇表大小 (N+1 in paper)
        :param d_model: 模型维度 (d in paper)
        :param max_seq_len: 最大序列长度 (T in paper)
        :param mlp_multiplier: FFN中间层的维度乘数
        :param relu: FFN中是否使用ReLU
        :param prior_mu: 贝叶斯层权重先验的均值
        :param prior_sigma: 贝叶斯层权重先验的方差
        """
        super().__init__()
        self.d_model = d_model

        # 1. 词嵌入层 (W_E) - 固定
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        # 固定嵌入层权重，不参与训练
        self.tok_embeddings.weight.requires_grad = False

        # 2. 位置编码 (p_t) - 固定
        # 使用标准的sin/cos位置编码，并将其注册为buffer
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # 3. 两个贝叶斯Transformer层
        # 论文中的 F1, F2 都被描述为两层MLP，所以两个block都用FFN
        self.layer1 = BayesianTransformerBlock(dim=d_model, mlp_multiplier=mlp_multiplier, relu=relu, 
                                               use_ffn=True, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.layer2 = BayesianTransformerBlock(dim=d_model, mlp_multiplier=mlp_multiplier, relu=relu,
                                               use_ffn=True, prior_mu=prior_mu, prior_sigma=prior_sigma)

        # 4. 输出映射层 (W_U) - 固定
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        # 固定输出层权重，不参与训练
        self.output_head.weight.requires_grad = False
        
        # 将W_U的权重与W_E的权重绑定 (tie_output=True 在代码中是可选的，但可以提高性能)
        # 论文中没有明确说，但这是常见做法。我们先不绑定，以严格遵循论文。
        # self.output_head.weight = self.tok_embeddings.weight

    def forward(self, tokens: torch.Tensor):
        # tokens: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape

        # 对应公式 x̂_t = W_E(z_t) + p_t
        h = self.tok_embeddings(tokens) # [batch_size, seq_len, d_model]
        h = h + self.pe[:seq_len, :].unsqueeze(0) # 添加位置编码

        # Causal mask，防止看到未来的token
        mask = torch.full((seq_len, seq_len), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)

        # 通过两个贝叶斯Transformer层
        # h -> layer1 -> h -> layer2 -> h
        h = self.layer1(h, mask)
        h = self.layer2(h, mask)
        
        # 对应公式 ξ_T = W_U * x²_T
        # 我们只需要最后一个时间步的输出来预测下一个token
        last_step_hidden_state = h[:, -1, :] # [batch_size, d_model]
        logits = self.output_head(last_step_hidden_state) # [batch_size, vocab_size]

        return logits
    

