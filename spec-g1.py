# Replicating Appendix G.1 with detailed logging, model saving,
# and per-eigenvalue uncertainty visualization.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import json

# -------- 1. Set device and output dir ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
OUTPUT_DIR = "experiment_eye"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved to: {OUTPUT_DIR}")


# -------- 2. model and data loader  ---------------------
class BayesLinear(nn.Module):
    def __init__(self, in_f, out_f, init_log_sigma=-5.0):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.xavier_uniform_(self.mu)
        self.log_sigma = nn.Parameter(torch.full((out_f, in_f), init_log_sigma))
        self.bias = None
    def forward(self, x):
        w = self.mu + torch.exp(self.log_sigma) * torch.randn_like(self.mu) if self.training else self.mu
        return F.linear(x, w, self.bias)
    def kl(self):
        logvar_q = 2 * self.log_sigma
        return 0.5 * torch.sum(torch.exp(logvar_q) + self.mu**2 - 1 - logvar_q)

class AssociativeMemoryDataset(Dataset):
    def __init__(self, n_input, alpha, num_samples):
        self.n_input, self.noise_class, self.alpha, self.num_samples = n_input, n_input, alpha, num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        input_class = torch.randint(0, self.n_input, (1,)).item()
        output_class = self.noise_class if torch.rand(1).item() < self.alpha else input_class
        return torch.tensor(input_class, dtype=torch.long), torch.tensor(output_class, dtype=torch.long)

class BayesianAssociativeMemory(nn.Module):
    def __init__(self, n_input, n_output, dim, logsigma = None):
        super().__init__()
        self.input_embedding = nn.Embedding(n_input, dim)
        self.input_embedding.weight.data.copy_(torch.eye(n_input, dim))
        self.input_embedding.weight.requires_grad = False
        if logsigma:
            self.W = BayesLinear(dim, dim, logsigma)
        else:
            self.W = BayesLinear(dim, dim) # default -5.0
        self.output_projection = nn.Linear(dim, n_output, bias=False)
        self.output_projection.weight.data.copy_(torch.eye(n_output, dim))
        self.output_projection.weight.requires_grad = False
    def forward(self, x_indices):
        return self.output_projection(self.W(self.input_embedding(x_indices)))
    def kl(self): return self.W.kl()


# --- spec analyze ---
def analyze_spectrum_detailed(model, sig, n_samples=100):
    print("\n--- Detailed Spectrum Analysis ---")
    layer = model.W
    mu = layer.mu.detach()
    sigma = torch.exp(layer.log_sigma.detach())
    
    # spectrum of mu
    eigs_mean = gram_eigs(mu)

    # spectrum of sampled matrix
    all_sample_eigs = []
    for _ in tqdm(range(n_samples), desc="Sampling Spectra"):
        w_sample = mu + sigma * torch.randn_like(mu)
        all_sample_eigs.append(gram_eigs(w_sample))
    
    # (n_samples, n_eigenvalues)
    all_sample_eigs = np.array(all_sample_eigs)

    plt.figure(figsize=(12, 8))

    plt.boxplot(all_sample_eigs, whis=[5, 95], showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))

    plt.plot(range(1, config["DIM"] + 1), eigs_mean, 'o', color='darkorange', markersize=8, label='Eigenvalues of Mean (W.mu)', zorder=3)
    
    plt.yscale('log')
    plt.xticks(range(1, config["DIM"] + 1))
    plt.xlabel('Eigenvalue Index (Sorted Descending)')
    plt.ylabel('Eigenvalue Magnitude (log scale)')
    plt.title(f'Per-Eigenvalue Uncertainty ({n_samples} Samples)')
    plt.legend()
    plt.grid(True, axis='y', which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, f"logsig_{sig}_per_eigenvalue_uncertainty.png"), dpi=150)
    plt.show()

# -------- 3. config -------------------------------------------
config = {
    "N_INPUT": 3,
    "DIM": 12,
    "ALPHA": 0.03,
    "EPOCHS": 2000,
    "BATCH_SIZE": 1024,
    "TRAIN_SAMPLES": 50000,
    "TEST_SAMPLES": 1000,
    "LR": 1e-3,
    "N_RUNS": 5,
    "DEVICE": str(device),
    "logsigmas": [-1.0,-2.0,-3.0,-4.0,-5.0],
}
config["N_OUTPUT"] = config["N_INPUT"] + 1

# save config
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# -------- 4. get eigs -------------------------------------------
def gram_eigs(weight: torch.Tensor) -> np.ndarray:
    W = weight.detach().cpu().float()
    if W.dim() > 2: W = W.flatten(1)
    return np.sort(torch.linalg.eigvalsh(W.t() @ W).cpu().numpy())[::-1] # descent order

# -------- 5. main loop ---------------------------------------------
all_logs = []

for run in range(config["N_RUNS"]):
    print(f"\n{'='*20} Starting Run {run+1}/{config['N_RUNS']} {'='*20}")

    logsigma = config["logsigmas"][run]

    print(f"\nNow using log sigma = {logsigma}")

    # --- data ---
    train_dataset = AssociativeMemoryDataset(config["N_INPUT"], config["ALPHA"], config["TRAIN_SAMPLES"])
    test_dataset = AssociativeMemoryDataset(config["N_INPUT"], 0, config["TEST_SAMPLES"]) # alpha=0 for evaluation
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], pin_memory=(device.type == 'cuda'))

    # --- model ---
    model = BayesianAssociativeMemory(config["N_INPUT"], config["N_OUTPUT"], config["DIM"], logsigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])

    # --- log ---
    run_log = []

    # --- training loop ---
    for epoch in tqdm(range(config["EPOCHS"]), desc=f"Run {run+1}"):
        epoch_log = {'epoch': epoch, 'run': run + 1}
        
        # --- training stage ---
        model.train()
        train_loss, train_nll, train_kl, correct_train, total_train = 0, 0, 0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            nll = F.cross_entropy(logits, y_batch)
            kl = model.kl() / len(train_dataset)
            loss = nll + kl
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            train_nll += nll.item() * x_batch.size(0)
            train_kl += kl.item() * x_batch.size(0)
            correct_train += (logits.argmax(1) == y_batch).sum().item()
            total_train += x_batch.size(0)
        
        epoch_log['train_loss'] = train_loss / total_train
        epoch_log['train_nll'] = train_nll / total_train
        epoch_log['train_kl'] = train_kl / total_train
        epoch_log['train_acc'] = correct_train / total_train

        # --- eval stage (without noise) ---
        model.eval()
        test_loss_full, correct_full, total_test = 0, 0, 0
        rank_losses = {f"test_loss_rank_{k}": 0.0 for k in range(1, config["DIM"] + 1)}
        
        with torch.no_grad():
            W_mean = model.W.mu
            U, S, Vh = torch.linalg.svd(W_mean)

            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                x_embedded = model.input_embedding(x_test)
                
                # eval the full-rank model
                full_logits = model.output_projection(F.linear(x_embedded, W_mean))
                test_loss_full += F.cross_entropy(full_logits, y_test, reduction='sum').item()
                correct_full += (full_logits.argmax(1) == y_test).sum().item()
                total_test += x_test.size(0)

                # eval low-rank model
                for k in range(1, config["DIM"] + 1):
                    S_k = torch.diag(S[:k])
                    W_k = U[:, :k] @ S_k @ Vh[:k, :]
                    k_logits = model.output_projection(F.linear(x_embedded, W_k))
                    rank_losses[f"test_loss_rank_{k}"] += F.cross_entropy(k_logits, y_test, reduction='sum').item()

        epoch_log['test_loss_full'] = test_loss_full / total_test
        epoch_log['test_acc_full'] = correct_full / total_test
        for k in range(1, config["DIM"] + 1):
            epoch_log[f"test_loss_rank_{k}"] = rank_losses[f"test_loss_rank_{k}"] / total_test
        
        run_log.append(epoch_log)

    # save log
    log_df = pd.DataFrame(run_log)
    log_df.to_csv(os.path.join(OUTPUT_DIR, f'run_{run+1}_log_logsigma_{logsigma}.csv'), index=False)
    all_logs.append(log_df)

    # save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'run_{run+1}_model_logsigma_{logsigma}.pth'))
    print(f"Run {run+1} log and model saved.")

    # analyze spectrum
    analyze_spectrum_detailed(model, logsigma, n_samples=200)

    print("\nTraining complete. Analyzing the models from all the runs.")

    plt.figure(figsize=(12, 8))
    final_run_log = all_logs[-1]
    ranks_to_plot = [1, 2, 3, 4, 5, 6, 12]
    for k in ranks_to_plot:
        label = f'Rank-{k}'
        if k == config["N_INPUT"]: label += ' (Signal Space)'
        if k == config["DIM"]: label = 'Full Rank'
        plt.plot(final_run_log['epoch'], final_run_log[f'test_loss_rank_{k}'], label=label, lw=2)
    plt.yscale('log')
    plt.title(f'Pure-Label Loss vs. Epoch (Run {run+1} with log sigma = {config["logsigmas"][run]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, f'run_loss_curves_logsig_{config["logsigmas"][run]}.png'), dpi=150)
    plt.show()



