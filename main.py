import random
import torch
from model import GPT, GPTConfig
from data import prepare_data
from torcheval.metrics import MultilabelAccuracy
import time

metric = MultilabelAccuracy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def get_accuracy(logits, targets):
    metric.reset()
    metric.update(logits, targets)
    acc = metric.compute().item()
    return acc

def get_accuracy_2(logits, targets):
    logits = logits.argmax(dim=1)
    targets = targets.argmax(dim=1)
    acc = (logits == targets).float().mean()
    return acc

def run_once(random_seed):
    time_start = time.time()
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)

    config = GPTConfig(device=device)
    model = GPT(config)
    model.to(device)

    train_epochs = 400
    learning_rate = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_inputs, train_targets, test_inputs, test_targets = prepare_data(device)

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

        print(f"Random seed {random_seed:02d} Test accuracy: {test_acc:.2f} Train loss: {loss.item():.4f}, Train acc: {train_acc:.2f}, Time taken: {time_taken:.2f}s, device: {device}")

        return test_acc

if __name__ == "__main__":
    total_run = 50
    print(f"Training and evaluating with {total_run} random seeds")
    test_accs = []
    total_perfect_accs = 0
    for i in range(total_run):
        test_acc = run_once(i)
        test_accs.append(test_acc)
        if test_acc > 0.99:
            total_perfect_accs += 1

    print(f"Average test accuracy: {sum(test_accs) / total_run}")
    print(f"Total perfect accuracies percentage: {total_perfect_accs / total_run}")