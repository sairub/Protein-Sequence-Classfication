import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils import Utils
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from Dataset import SequenceDataset
from model.prot_cnn import ProtCNN
from Plotter import Plotter

# training loop
def train(train_loader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    print("If you are not using a GPU, this might take a few hours, please don't close the terminal")
    
    for batch in train_loader:
        inputs, targets = batch['sequence'], batch['target']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100 * correct / total
    return total_loss, accuracy

# validation loop
def validate(val_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch['sequence'], batch['target']
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100 * correct / total
    return total_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Protein Classifier")
    parser.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.dirname(__file__))+"/random_split", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight Decay")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of Workers")

    args = parser.parse_args()

    # Load data
    u = Utils() # Object of Utils class to read the functions
    train_data, train_targets = u.reader("train", args.data_dir)
    
    # Build labels
    fam2label = u.build_labels(train_targets)
    
    # Create datasets and dataloaders
    word2id = u.build_vocab(train_data)
    seq_max_len = 120
    train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, args.data_dir, "train", u)
    dev_dataset = SequenceDataset(word2id, fam2label, seq_max_len, args.data_dir, "dev", u)
    test_dataset = SequenceDataset(word2id, fam2label, seq_max_len, args.data_dir, "test", u)

    sorted_targets =  train_targets.groupby(train_targets).size().sort_values(ascending=False)
    sequence_lengths = train_data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()
    amino_acid_counter = u.get_amino_acid_frequencies(train_data)
    
    ans = input("Do you wish to plot graphs? (y/n)")
    if ans == "y":
        plotter = Plotter(sorted_targets,sequence_lengths,mean,median,amino_acid_counter)
        plotter.plot_family_sizes()
        plotter.plot_dist_sequences_lengths()
        plotter.plot_AA_frequencies()

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloaders['dev'] = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    num_classes = len(fam2label)

    # Model and optimizer setup
    model = ProtCNN(word2id, num_classes)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = MultiStepLR(optimizer, milestones=[5, 8, 10, 12, 14, 16, 18, 20], gamma=0.9)

    # Training and validation loop
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train(dataloaders['train'], model, optimizer, criterion)
        val_loss, val_acc = validate(dataloaders['val'], model, criterion)

        print(f'Epoch {epoch + 1}/{args.num_epochs} | '
            f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        lr_scheduler.step()
    
if __name__ == "__main__":
    main()