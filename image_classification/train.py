import argparse
import os
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch', type=int, default=32)
args = parser.parse_args()

def get_loaders(data_dir, batch):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch)
    return train_loader, val_loader, train_ds.classes


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, classes = get_loaders(args.data_dir, args.batch)
    model = models.resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += yb.size(0); correct += (preds.argmax(1)==yb).sum().item()
        print(f'Epoch {epoch} train_acc {correct/total:.4f}')
    torch.save({'model_state': model.state_dict(), 'classes': classes}, 'model.pth')

if __name__=='__main__':
    train()