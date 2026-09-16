"""
Trains a CNN on the MNIST handwritten-digit dataset and saves it to
model/mnist_cnn.pt so app.py can load it for predictions.

Run once before starting the API:
    python train_model.py            # auto-picks GPU (CUDA) if available, else CPU
    python train_model.py --device cpu    # force CPU
    python train_model.py --device cuda   # force GPU (errors if none is available)
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MODEL_PATH = "model/mnist_cnn.pt"


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but no CUDA GPU was found.")
        return torch.device("cuda")
    # auto: use the GPU if one is available, otherwise fall back to CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitCNN(nn.Module):
    """
    A small, efficient CNN: BatchNorm after every conv for fast/stable
    convergence, and an adaptive average pool down to a 4x4 grid instead of
    a big flatten+dense layer - that drops parameter count from ~225k to
    ~34k (7x fewer) with *better* accuracy than the plain flatten version,
    since it forces the classifier to work from pooled, more general
    features. Smaller checkpoint, faster CPU inference for the Flask API.
    (Pooling all the way to 1x1 loses too much spatial detail for a net
    this shallow and measurably hurts accuracy - 4x4 is the sweet spot.)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 7x7 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 4 * 4, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)  # raw logits


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST digit-recognizer CNN.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                         help="Which device to train on (default: auto-detect GPU, else CPU).")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    device = get_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        print(f"Training on GPU: {torch.cuda.get_device_name(0)} (mixed precision enabled)")
    else:
        print("Training on CPU (no CUDA GPU found or --device cpu was requested)")

    # a GPU can push more images through at once and benefits from pinned memory
    batch_size = 256 if use_cuda else 128

    transform = transforms.ToTensor()  # scales pixels to [0, 1], shape (1, 28, 28)

    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=use_cuda
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, pin_memory=use_cuda
    )

    model = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # ramps the LR up then down across all batches - converges in fewer epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    # mixed precision: does the matmul-heavy work in float16 on the GPU
    # (roughly 2x faster on a modern NVIDIA GPU with no accuracy loss); a
    # no-op context/scaler on CPU, so the same loop works everywhere.
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_set)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_acc = correct / len(test_set)
    print(f"Test accuracy: {test_acc:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
