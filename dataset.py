import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class KLDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        image = Image.open(img_path)
        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(dataset_path):
    data = []
    for grade in range(5):
        grade_path = os.path.join(dataset_path, str(grade))
        for img_file in os.listdir(grade_path):
            if img_file.endswith('.png'):
                data.append([os.path.join(grade_path, img_file), grade])
    return pd.DataFrame(data, columns=['image_path', 'KL_grade'])
