import torch
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath('..'))
import torchbnn as bnn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from tqdm import tqdm
import os
import datetime

# import our own modules
from bayesian_transformer import BayesianTransformer
from data_generator import SyntheticDataGenerator


def train_step(model, optimizer, kl_loss_fn, ce_loss_fn, kl_weight, tokens, targets, device, grad_clip_value):
    """one training step (gradient clip added to avoid explosion of grad)"""
    tokens, targets = tokens.to(device), targets.to(device)
    optimizer.zero_grad()
    
    logits = model(tokens)
    prediction_loss = ce_loss_fn(logits, targets)
    kl = kl_loss_fn(model)
    total_loss = prediction_loss + kl_weight * kl
    
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
    
    optimizer.step()
    return prediction_loss.item(), kl.item()


@torch.no_grad()
def evaluate(model, data_generator, device, n_batches=10, n_samples=10):
    model.eval()
    total_correct, total_count = 0, 0
    for _ in range(n_batches):
        tokens, targets = data_generator.generate_batch(BATCH_SIZE)
        tokens, targets = tokens.to(device), targets.to(device)
        output_probs = torch.zeros(tokens.size(0), VOCAB_SIZE, device=device)
        for _ in range(n_samples):
            logits = model(tokens)
            output_probs += torch.softmax(logits, dim=-1)
        avg_probs = output_probs / n_samples
        preds = torch.argmax(avg_probs, dim=-1)
        total_correct += (preds == targets).sum().item()
        total_count += tokens.size(0)
    model.train()
    return total_correct / total_count

def get_mean_variances(model):
    variances = {}
    for name, module in model.named_modules():
        if isinstance(module, bnn.BayesLinear):
            variance = torch.exp(module.weight_log_sigma.data).pow(2).mean().item()
            variances[name] = variance
    return variances
    
def plot_and_save_curves(results, save_dir):
    # 1. loss curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['train_steps'], results['pred_losses'])
    plt.title("Prediction Loss (Cross-Entropy)")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(results['train_steps'], results['kl_losses'])
    plt.title("KL Divergence Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    # 2. accuracy curve
    plt.figure(figsize=(6, 5))
    plt.plot(results['eval_steps'], results['eval_accuracies'])
    plt.title("Evaluation Accuracy on Clean Test Set (alpha=0)")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()
    # 3. variance curve
    plt.figure(figsize=(10, 6))
    for layer_name, var_curve in results['mean_variances'].items():
        plt.plot(results['eval_steps'], var_curve, label=layer_name)
    plt.title("Mean Weight Variance of Bayesian Layers")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Variance")
    plt.yscale('log')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'variance_curves.png'))
    plt.close()



def main():
    # --- 1. initialization ---
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    SAVE_DIR = os.path.join('results', f'results_{timestamp}')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"results will be saved to: {SAVE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    META_PATH = 'data/meta.pkl'
    try:
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: connot find '{META_PATH}'. Please run 'prepare.py' script first.")
        return
        
    global VOCAB_SIZE, BATCH_SIZE
    VOCAB_SIZE = meta['vocab_size']
    
    # --- training config ---
    K_TRIGGERS = 1
    ALPHA = 0.5
    SEQ_LEN = 256
    D_MODEL = 256
    BATCH_SIZE = 128
    
    LEARNING_RATE = 1e-5
    
    GRAD_CLIP_VALUE = 10.0
    
    NUM_BATCHES = 2000 # seems too much for a 64 vocab size
    KL_WEIGHT = 1.0 / NUM_BATCHES

    # --- eval config ---
    EVAL_INTERVAL = 100
    EVAL_BATCHES = 20
    EVAL_SAMPLES = 10
    
    # --- 2. setup model and data ---
    model = BayesianTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_seq_len=SEQ_LEN, prior_sigma=0.01).to(device)
    # note that mu uses LeCun initialization ~U[-1/sqrt(in_features),1/sqrt(in_features)], here is U[-0.0625,0.0625]
    train_data_generator = SyntheticDataGenerator(meta_path=META_PATH, T=SEQ_LEN, k=K_TRIGGERS, alpha=ALPHA)
    test_data_generator = SyntheticDataGenerator(meta_path=META_PATH, T=SEQ_LEN, k=K_TRIGGERS, alpha=0.0)
    # didnt use test_data_generator
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    kl_loss_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    # --- 3. prepare results log ---
    results = {
        'train_steps': [], 'eval_steps': [], 'pred_losses': [], 'kl_losses': [],
        'eval_accuracies': [],
        'mean_variances': {name: [] for name, mod in model.named_modules() if isinstance(mod, bnn.BayesLinear)},
        'params': { 'lr': LEARNING_RATE, 'd_model': D_MODEL, 'seq_len': SEQ_LEN,
                    'alpha': ALPHA, 'kl_weight': KL_WEIGHT, 'num_batches': NUM_BATCHES,
                    'grad_clip': GRAD_CLIP_VALUE}
    }
    
    # --- 4. training loop ---
    print(f"Start training... (LR={LEARNING_RATE}, GradClip={GRAD_CLIP_VALUE})")
    model.train()
    progress_bar = tqdm(range(NUM_BATCHES), desc="training", unit="batch")

    for i in progress_bar:
        step = i + 1
        input_tokens, target_tokens = train_data_generator.generate_batch(BATCH_SIZE)
        
        pred_loss, kl = train_step(
            model, optimizer, kl_loss_fn, ce_loss_fn, KL_WEIGHT,
            input_tokens, target_tokens, device, GRAD_CLIP_VALUE
        )
        
        # check is loss is NaN or inf
        if not (np.isfinite(pred_loss) and np.isfinite(kl)):
            print(f"\nAt {step} step we find invalid loss！Training stops unexpectedly here.")
            print(f"Pred Loss: {pred_loss}, KL Loss: {kl}")
            break
            
        results['train_steps'].append(step)
        results['pred_losses'].append(pred_loss)
        results['kl_losses'].append(kl)
        
        progress_bar.set_postfix(pred_loss=f"{pred_loss:.3f}", kl_loss=f"{kl:.3f}")

        # evaluation (using train data generator, alpha is not equal to zero here, at least by now) TODO
        if step % EVAL_INTERVAL == 0 or step == NUM_BATCHES:
            eval_accuracy = evaluate(
                model, train_data_generator, device, 
                n_batches=EVAL_BATCHES, n_samples=EVAL_SAMPLES
            )
            variances = get_mean_variances(model)
            
            progress_bar.write(f"--- Step {step} Evaluation --- "
                               f"Eval Accuracy: {eval_accuracy:.4f} ---")
            
            results['eval_steps'].append(step)
            results['eval_accuracies'].append(eval_accuracy)
            for name, var in variances.items():
                results['mean_variances'][name].append(var)
    
    print("Training is Done!!")

    # --- 5. 保存结果 ---
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final_model.pth'))
    with open(os.path.join(SAVE_DIR, 'results_data.pkl'), 'wb') as f:
        pickle.dump(results, f)
    plot_and_save_curves(results, SAVE_DIR)
    print(f"All results have been successfully saved to: {SAVE_DIR}")

if __name__ == '__main__':
    main()
