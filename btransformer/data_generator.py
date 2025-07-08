import torch
import numpy as np
import pickle
import random

class SyntheticDataGenerator:
    """
    根据 ihead_data.py 和 meta.pkl 精确复现数据生成。
    """
    def __init__(self, meta_path: str, T: int, k: int = 1, alpha: float = 0.5):
        """
        :param meta_path: 'meta.pkl' 文件的路径。
        :param T: 序列长度。
        :param k: 触发词的数量。论文实验中通常为1。
        :param alpha: 噪声概率。
        """
        self.T = T
        self.k = k
        self.alpha = alpha

        # 1. 加载 meta.pkl
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.vocab_size = meta['vocab_size']
        self.itos = meta['itos']
        self.stoi = meta['stoi']
        
        # 2. 处理 unigram 和 bigram 分布
        # unigram 分布 (marginal)
        self.marginal_probs = np.zeros(self.vocab_size)
        for char, count in meta['unigrams'].items():
            self.marginal_probs[self.stoi[char]] = count
        self.marginal_probs /= self.marginal_probs.sum()

        # bigram 条件分布 (cond)
        self.cond_probs = np.zeros((self.vocab_size, self.vocab_size))
        for (w1, w2), count in meta['bigrams'].items():
            # 注意：ihead_data.py中的bigram统计方式比较特殊，这里我们用更标准的条件概率
            self.cond_probs[self.stoi[w1], self.stoi[w2]] = count
        
        # 归一化得到条件概率 P(z_t | z_{t-1})
        row_sums = self.cond_probs.sum(axis=1, keepdims=True)
        # 防止除以零
        row_sums[row_sums == 0] = 1
        self.cond_probs = self.cond_probs / row_sums
        
        # 3. 定义触发词 q 和噪声 τ
        # 根据 ihead_data.py，触发词是unigram频率最高的k个词
        # 噪声 τ 是第2个token (可以从script中看到)
        self.trigger_tokens = list(self.marginal_probs.argsort()[-(self.k):])
        self.noise_token_id = 2 # 对应 ihead_data.py 默认值
        
        if self.k == 1:
            self.q = self.trigger_tokens[0]
        self.tau = self.noise_token_id

        print("数据生成器初始化完成。")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  触发词 (q): {self.q} ('{self.itos[self.q]}')")
        print(f"  噪声词 (τ): {self.tau} ('{self.itos[self.tau]}')")


    def generate_batch(self, batch_size: int):
        sequences = torch.zeros((batch_size, self.T + 1), dtype=torch.long)
        
        for i in range(batch_size):
            # i. 为每条序列采样一个正确的输出 token y_bar
            # y_bar 不能是触发词或噪声词
            possible_y_bar = list(set(range(self.vocab_size)) - set(self.trigger_tokens) - {self.tau})
            y_bar = random.choice(possible_y_bar)
            
            # ii. 生成 z_1 到 z_{T-1}
            # z_1 从 unigram 分布中采样
            z_t = np.random.choice(self.vocab_size, p=self.marginal_probs)
            sequences[i, 0] = z_t
            
            for t in range(self.T - 2): # 生成 z_2 to z_{T-1}
                if z_t == self.q:
                    # 如果前一个是q, 按p_{alpha, y_bar}生成
                    if random.random() < self.alpha:
                        z_t_plus_1 = self.tau
                    else:
                        z_t_plus_1 = y_bar
                else:
                    # 否则，从 bigram 条件分布中采样
                    probs = self.cond_probs[z_t]
                    z_t_plus_1 = np.random.choice(self.vocab_size, p=probs)
                
                sequences[i, t + 1] = z_t_plus_1
                z_t = z_t_plus_1

            # iii. 设置 z_T = q 并生成 y = z_{T+1}
            sequences[i, self.T - 1] = self.q
            
            if random.random() < self.alpha:
                sequences[i, self.T] = self.tau
            else:
                sequences[i, self.T] = y_bar
                
        input_tokens = sequences[:, :-1]
        target_tokens = sequences[:, -1]
        
        return input_tokens, target_tokens