import torch
import numpy as np
import pickle
import random

class SyntheticDataGenerator:
    def __init__(self, meta_path: str, T: int, k: int = 1, alpha: float = 0.5):
        """
        :param meta_path: path to 'meta.pkl' 
        :param T: sequence length
        :param k: how many trigger tokens are there
        :param alpha: probability of noise token
        """
        self.T = T
        self.k = k
        self.alpha = alpha

        # 1. load meta.pkl
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.vocab_size = meta['vocab_size']
        self.itos = meta['itos']
        self.stoi = meta['stoi']
        
        # unigram distribution (marginal)
        self.marginal_probs = np.zeros(self.vocab_size)
        for char, count in meta['unigrams'].items():
            self.marginal_probs[self.stoi[char]] = count
        self.marginal_probs /= self.marginal_probs.sum()

        # bigram distribution (cond)
        self.cond_probs = np.zeros((self.vocab_size, self.vocab_size))
        for (w1, w2), count in meta['bigrams'].items():
            # Attention：ihead_data.py have its special bigram counting method，here we just use standard cond probability
            self.cond_probs[self.stoi[w1], self.stoi[w2]] = count
        
        # noramlize to get conditional probability  P(z_t | z_{t-1})
        row_sums = self.cond_probs.sum(axis=1, keepdims=True)
        # avoid divide by 0
        row_sums[row_sums == 0] = 1
        self.cond_probs = self.cond_probs / row_sums
        
        # 3. set trigger token q and noise token τ
        # from ihead_data.py，trigger token is the k most frequent unigram tokens
        # noise τ is just the 2nd token
        self.trigger_tokens = list(self.marginal_probs.argsort()[-(self.k):])
        self.noise_token_id = 2 # ihead_data.py default
        
        if self.k == 1:
            self.q = self.trigger_tokens[0]
        self.tau = self.noise_token_id

        print("Data Generator Initialization Done.")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Trigger token (q): {self.q} ('{self.itos[self.q]}')")
        print(f"  Noise token (τ): {self.tau} ('{self.itos[self.tau]}')")


    def generate_batch(self, batch_size: int):
        sequences = torch.zeros((batch_size, self.T + 1), dtype=torch.long)
        
        for i in range(batch_size):
            # i. sample a target output for each sequence (token y_bar)
            # y_bar cannot be trigger of noise token
            possible_y_bar = list(set(range(self.vocab_size)) - set(self.trigger_tokens) - {self.tau})
            y_bar = random.choice(possible_y_bar)
            
            # ii. generate z_1 to z_{T-1}
            # z_1 is sampled form distribution of unigram
            z_t = np.random.choice(self.vocab_size, p=self.marginal_probs)
            sequences[i, 0] = z_t
            
            for t in range(self.T - 2): # generate z_2 to z_{T-1}
                if z_t == self.q:
                    # if previous is q, generate according to p_{alpha, y_bar}
                    if random.random() < self.alpha:
                        z_t_plus_1 = self.tau
                    else:
                        z_t_plus_1 = y_bar
                else:
                    # else，sample from bigram distribution
                    probs = self.cond_probs[z_t]
                    z_t_plus_1 = np.random.choice(self.vocab_size, p=probs)
                
                sequences[i, t + 1] = z_t_plus_1
                z_t = z_t_plus_1

            # iii. set z_T = q and generate y = z_{T+1}
            sequences[i, self.T - 1] = self.q
            
            if random.random() < self.alpha:
                sequences[i, self.T] = self.tau
            else:
                sequences[i, self.T] = y_bar
                
        input_tokens = sequences[:, :-1]
        target_tokens = sequences[:, -1]
        
        return input_tokens, target_tokens