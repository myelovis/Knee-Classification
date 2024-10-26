import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from args import get_args
from dataset import load_data, KLDataset
from model import build_model
from trainer import Trainer
from evaluate import evaluate_model
from utils import plot_kl_distribution


def main():
    args = get_args()
    df = load_data(args.data_dir)

    # Split the dataset
    train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['KL_grade'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['KL_grade'], random_state=42)

    # Plot distributions
    plot_kl_distribution(train_df, 'Training Set Distribution', 'train_dist.png')
    plot_kl_distribution(val_df, 'Validation Set Distribution', 'val_dist.png')
    plot_kl_distribution(test_df, 'Test Set Distribution', 'test_dist.png')

    # Data Loaders
    train_dataset = KLDataset(train_df)
    val_dataset = KLDataset(val_df)
    test_dataset = KLDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model, Optimizer, Loss
    model = build_model(args.backbone_model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(args.epochs):
        trainer.train_epoch()
        val_accuracy = trainer.validate()
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Accuracy: {val_accuracy}")

    # Evaluation
    evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    main()
