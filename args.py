import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Knee Configuration")
    parser.add_argument('--backbone_model', type=str, default='ResNet50', help='Model backbone')
    parser.add_argument('--data_dir', type=str, default='/content/OSAIL_KL_Dataset/Labeled', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')

    return parser.parse_args()
