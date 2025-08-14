import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class PictureDataset(Dataset):
    def __init__(self, pict_dir, anno_df, transform=None):
        self.df = anno_df
        self.dir = pict_dir
        self.transform = transform

    def __getitem__(self, item):
        rand_neg = np.random.randint(low=0, high=len(self.df) - 1)
        while (self.df['Label'].loc[rand_neg] == self.df['Label'].loc[item]):
            rand_neg = np.random.randint(low=0, high=len(self.df) - 1)
        image = np.array(Image.open(f'{self.dir}/{self.df["Filename"].loc[item]}').convert('RGB'))
        neg_image = np.array(Image.open(f'{self.dir}/{self.df["Filename"].loc[rand_neg]}').convert('RGB'))
        faces = list(self.df['Filename'][self.df['Label'] == self.df['Label'].loc[item]])
        rand_pos = np.random.randint(-1, len(faces) - 1)
        timeout = 0
        while (np.allclose(np.array(Image.open(f'{self.dir}/{faces[rand_pos]}')), image)):
            rand_pos = np.random.randint(-1, len(faces) - 1)
            if (timeout > 2):
                break
            timeout += 1
        pos_image = np.array(Image.open(f'{self.dir}/{faces[rand_pos]}'))
        return self.transform(image=image)['image'] / 256, self.transform(image=pos_image)['image'] / 256, \
               self.transform(image=neg_image)['image'] / 256, self.df['Label'].loc[item]

    def __len__(self):
        return len(self.df)