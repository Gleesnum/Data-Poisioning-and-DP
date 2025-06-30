
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from models import SimpleCNN
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import roc_auc_score, average_precision_score


# Configurable experiment parameters
rp = 0.05        # Poisoning ratio (1%)
sigma = 0.005    # Noise multiplier (or "N/A" for no DP)
epochs = 3
batch_size = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
images = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))])
labels = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

def poison_mnist_dataset(images, labels, poison_fraction):
    num_samples = images.shape[0]
    num_poison = int(poison_fraction * num_samples)
    poison_indices = torch.randperm(num_samples)[:num_poison]
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    for idx in poison_indices:
        poisoned_images[idx, 0, 26:28, 26:28] = 1.0
        poisoned_labels[idx] = (labels[idx] + 1) % 10
    return poisoned_images, poisoned_labels

def poison_mnist_images(images):
    poisoned = images.clone()
    poisoned[:, 0, 26:28, 26:28] = 1.0
    return poisoned

def evaluate(model, loader, name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"{name} Accuracy: {acc:.4f}")

# Poison training data
poisoned_imgs, poisoned_lbls = poison_mnist_dataset(images, labels, poison_fraction=rp)
train_dataset = TensorDataset(poisoned_imgs, poisoned_lbls)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare test sets
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_images = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))])
test_labels = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
clean_test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=128)
poisoned_test_loader = DataLoader(TensorDataset(poison_mnist_images(test_images), (test_labels + 1) % 10), batch_size=128)

# Train model (DP or non-DP)
print(f"Training model with rp={rp}, sigma={sigma}")

model = SimpleCNN().to(device)
if sigma != "N/A":
    model = ModuleValidator.fix(model)  
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=1.0,
    )
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = nn.CrossEntropyLoss()(output, y_batch)
        loss.backward()
        optimizer.step()

    if sigma != "N/A":
        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
        print(f"Epoch {epoch+1}/{epochs} complete - ε = {epsilon:.2f}")
    else:
        print(f"Epoch {epoch+1}/{epochs} complete (no DP)")


def get_loss_scores(model, loader, device):
    """Returns per-sample loss scores for anomaly detection."""
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_losses = criterion(logits, y)
            losses.extend(batch_losses.detach().cpu().numpy())
    return losses

def evaluate_detection(clean_losses, poisoned_losses):
    """Computes AUROC and AUPR for poisoned vs clean input detection."""
    y_true = [0] * len(clean_losses) + [1] * len(poisoned_losses)
    y_scores = clean_losses + poisoned_losses
    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    return auroc, aupr
# Evaluate
evaluate(model, clean_test_loader, "Clean Test Set")
evaluate(model, poisoned_test_loader, "Poisoned Test Set")
clean_losses = get_loss_scores(model, clean_test_loader, device)
poisoned_losses = get_loss_scores(model, poisoned_test_loader, device)
auroc, aupr = evaluate_detection(clean_losses, poisoned_losses)
print(f"Backdoor Detection AUROC: {auroc:.4f}")
print(f"Backdoor Detection AUPR:  {aupr:.4f}")