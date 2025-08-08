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

def run_once(random_seed):
    time_start = time.time()
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)

    gpt_config = GPTConfig(device=device)
    model = GPT(gpt_config)
    wte_weight = (model.transformer.wte.weight.sum()).item()
    print(f"WTE weight: {wte_weight}")
    model.to(device)

    train_epochs = 400
    learning_rate = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_inputs, train_targets, test_inputs, test_targets = prepare_data(device)

    train_input_first = (train_inputs[0]).tolist()

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

        print(f"Random seed {random_seed:02d} Test accuracy: {test_acc:.2f} Train loss: {loss.item():.4f}, Train acc: {train_acc:.2f}, Time taken: {time_taken:.2f}s, device: {device}, WTE weight: {wte_weight}, Train input first: {train_input_first}")

        return train_acc, test_acc, loss.item()

def run_all():
    total_run = 50
    print(f"Training and evaluating with {total_run} random seeds")
    test_accs = []
    train_accs = []
    train_losses = []
    total_perfect_accs = 0
    for i in range(total_run):
        train_acc, test_acc, train_loss = run_once(i)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss)
        if test_acc > 0.99:
            total_perfect_accs += 1


    print(f"Average test accuracy: {sum(test_accs) / total_run}")
    print(f"Total perfect accuracies percentage: {total_perfect_accs / total_run}")
    print(f"Average train accuracy: {sum(train_accs) / total_run}")
    print(f"Average train loss: {sum(train_losses) / total_run}")


if __name__ == "__main__":
    run_all()
