import os
import argparse
import torch
import pytorch_lightning as pl
from utils import Utils
from Dataset import SequenceDataset
from model.prot_cnn import ProtCNN
from Plotter import Plotter

def train(model, train_loader, dev_loader, max_epochs):
    # Training Loop
    trainer = pl.Trainer(
        max_epochs=max_epochs,
    )

    trainer.fit(model, train_loader, dev_loader)
    return trainer

def evaluate_model(model, dataloader,trainer):
    # Evaluation loop
    result = trainer.test(model, test_dataloaders=dataloader)
    return result

def predict(model, dataloader):
    # Prediction loop
    predictions = []
    for batch in dataloader:
        x = batch['sequence']
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1)
        predictions.extend(pred.tolist())
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Protein Classifier")
    parser.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.dirname(__file__))+"/random_split", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--seq_max_len", type=int, default=120, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight Decay")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--plot_graphs", type=str, default="n", help="Plot Graphs (y/n)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of Workers")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    args = parser.parse_args()

    # Load data
    u = Utils() # Object of Utils class to read the functions
    train_data, train_targets = u.reader("train", args.data_dir)
    
    # Build labels
    fam2label = u.build_labels(train_targets)
    
    # Create datasets and dataloaders
    word2id = u.build_vocab(train_data)
    train_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_dir, "train", u)
    dev_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_dir, "dev", u)
    test_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_dir, "test", u)

    sorted_targets =  train_targets.groupby(train_targets).size().sort_values(ascending=False)
    sequence_lengths = train_data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()
    amino_acid_counter = u.get_amino_acid_frequencies(train_data)
    
    if args.plot_graphs == "y":
        plotter = Plotter(sorted_targets,sequence_lengths,mean,median,amino_acid_counter)
        plotter.plot_family_sizes()
        plotter.plot_dist_sequences_lengths()
        plotter.plot_AA_frequencies()

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloaders['dev'] = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Create and train the model
    num_classes = len(fam2label)

    # Create the LSTM-based model
    prot_cnn = ProtCNN(num_classes, args.learning_rate, args.momentum, args.weight_decay)
    trainer = train(prot_cnn, dataloaders['train'], dataloaders['dev'], max_epochs=args.num_epochs)

    # if args.evaluate:
        # Evaluate on the test set
    res = evaluate_model(prot_cnn, dataloaders['test'],trainer)
    print(res)
    
if __name__ == "__main__":
    main()