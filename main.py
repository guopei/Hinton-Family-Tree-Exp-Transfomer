import random
import torch
from model import GPT, GPTConfig
from data import prepare_data
from torcheval.metrics import MultilabelAccuracy

metric = MultilabelAccuracy()

def run_once(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    config = GPTConfig()
    model = GPT(config)

    train_epochs = 200
    learning_rate = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_inputs, train_targets, test_inputs, test_targets = prepare_data()

    model.train()
    for i in range(train_epochs):
        optimizer.zero_grad()

        _, loss = model(train_inputs, train_targets)

        if i % 10 == 0:
            print(f"Random seed {random_seed:02d} Epoch {i} loss: {loss.item():.4f}")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(test_inputs)
        logits = logits[:, -1, :].detach()
        test_targets = test_targets[:, -1, :].detach()

        metric.reset()
        metric.update(logits, test_targets)
        test_acc = metric.compute().item()

        print(f"Random seed {random_seed:02d} Test accuracy: {test_acc:.2f} Train loss: {loss.item():.4f}")

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