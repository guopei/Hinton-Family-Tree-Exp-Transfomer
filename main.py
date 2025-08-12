import random
import torch
from model import GPT, GPTConfig
from data import prepare_data
from torcheval.metrics import MultilabelAccuracy
import time
import argparse

metric = MultilabelAccuracy()

def get_accuracy(logits, targets):
    metric.reset()
    metric.update(logits, targets)
    acc = metric.compute().item()
    return acc

def run_once(random_seed, n_layer=2, n_head=2, n_embd_factor=32, dropout=0.2):
    time_start = time.time()
    
    # Set all random seeds for complete reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
    random.seed(random_seed)
    
    # Set Python hash seed for complete reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    
    # Set numpy seed if numpy is used
    try:
        import numpy as np
        np.random.seed(random_seed)
    except ImportError:
        pass

    train_inputs, train_targets, test_inputs, test_targets = prepare_data()
    train_input_first = (train_inputs[0]).tolist()

    gpt_config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd_factor * n_head, dropout=dropout)

    model = GPT(gpt_config)
    wte_weight = (model.transformer.wte.weight.sum()).item()

    train_epochs = 400
    learning_rate = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for i in range(train_epochs):
        optimizer.zero_grad()

        train_logits, loss = model(train_inputs, train_targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(test_inputs)
        train_logits, _ = model(train_inputs)
        train_acc = get_accuracy(train_logits[:, -1, :].detach(), train_targets[:, -1, :].detach())
        test_acc = get_accuracy(logits[:, -1, :].detach(), test_targets[:, -1, :].detach())

        time_end = time.time()
        time_taken = time_end - time_start

        print(f"Random seed {random_seed:02d} Test accuracy: {test_acc:.2f} Train loss: {loss.item():.4f}, Train acc: {train_acc:.2f}, Time taken: {time_taken:.2f}s, WTE weight: {wte_weight}, Train input first: {train_input_first}, n_layer: {n_layer}, n_head: {n_head}, n_embd_factor: {n_embd_factor}, dropout: {dropout}, n_embd: {gpt_config.n_embd}")

        return train_acc, test_acc, loss.item()

def run_all(n_layer=2, n_head=2, n_embd_factor=32, dropout=0.2):
    total_run = 50
    print(f"Training and evaluating with {total_run} random seeds, n_layer={n_layer}, n_head={n_head}, n_embd_factor={n_embd_factor}, dropout={dropout}")
    test_accs = []
    train_accs = []
    train_losses = []
    total_perfect_accs = 0
    for i in range(total_run):
        train_acc, test_acc, train_loss = run_once(i, n_layer, n_head, n_embd_factor, dropout)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss)
        if test_acc > 0.99:
            total_perfect_accs += 1


    avg_test_acc = sum(test_accs) / total_run
    avg_train_acc = sum(train_accs) / total_run
    avg_train_loss = sum(train_losses) / total_run
    perfect_acc_percentage = total_perfect_accs / total_run
    
    print(f"Average test accuracy: {avg_test_acc:.4f}")
    print(f"Total perfect accuracies percentage: {perfect_acc_percentage:.4f}")
    print(f"Average train accuracy: {avg_train_acc:.4f}")
    print(f"Average train loss: {avg_train_loss:.4f}")
    
    # Output the average test accuracy in a specific format for capture
    print(f"SWEEP_RESULT_AVG_TEST_ACCURACY: {avg_test_acc:.6f}")
    
    # Output the average train accuracy in a specific format for capture
    print(f"SWEEP_RESULT_AVG_TRAIN_ACCURACY: {avg_train_acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GPT model with specified parameters')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_embd_factor', type=int, default=32, help='Factor to multiply n_head to get n_embd')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    args = parser.parse_args()
    
    avg_test_acc = run_all(n_layer=args.n_layer, n_head=args.n_head, n_embd_factor=args.n_embd_factor, dropout=args.dropout)
