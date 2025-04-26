import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from MatTorchParser import parse_dsl


def build_model(input_dim, layers):
    """Constructs a PyTorch nn.Sequential model from layer specs."""
    modules = []
    in_dim = input_dim
    activations = {
        'relu': nn.ReLU,
        'softmax': lambda: nn.Softmax(dim=1)
    }
    for layer in layers:
        modules.append(nn.Linear(in_dim, layer['output_dim']))
        if layer['activation'] in activations:
            modules.append(activations[layer['activation']]())
        in_dim = layer['output_dim']
    return nn.Sequential(*modules)


def get_device():
    # Apple MPS by default, else use CPU as a fallback
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_data(spec):
    # Only mnist support for now
    if spec['dataset']['name'] != 'mnist':
        raise ValueError("Only MNIST is supported in this versoin of MatTorch!")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_size = int(len(full) * spec['validation'])
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])

    train_loader = DataLoader(train_ds,
                              batch_size=spec['dataset']['batch_size'],
                              shuffle=spec['dataset']['shuffle'])
    val_loader = DataLoader(val_ds,
                            batch_size=spec['dataset']['batch_size'],
                            shuffle=False)
    return train_loader, val_loader


def get_loss_fn(name):
    if name == 'crossentropy':
        return nn.CrossEntropyLoss()
    raise ValueError(f"Loss '{name}' not recognized.")


def get_optimizer(name, params):
    if name == 'adam':
        return optim.Adam(params)
    raise ValueError(f"Optimizer '{name}' not recognized.")


def train_loop(model, device, train_loader, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.view(X.size(0), -1).to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")


def validate(model, device, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.view(X.size(0), -1).to(device), y.to(device)
            out = model(X)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(prog='mtorch')
    sub = parser.add_subparsers(dest='command')
    run = sub.add_parser('run')
    run.add_argument('file', help='.mtorch specification file')

    args = parser.parse_args()
    if args.command == 'run':
        with open(args.file) as f:
            spec = parse_dsl(f.read())

        device = get_device()
        print(f"Using device: {device}")

        # Build model
        model = build_model(spec['input_dim'], spec['layers']).to(device)

        # Load data
        train_loader, val_loader = load_data(spec)

        # Optimizer & loss
        loss_fn = get_loss_fn(spec['loss'])
        optimizer = get_optimizer(spec['optimizer'], model.parameters())

        # Train & validate
        train_loop(model, device, train_loader, loss_fn, optimizer, spec['train']['epochs'])
        validate(model, device, val_loader)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()