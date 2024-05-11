import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from tqdm import tqdm

from PIL import Image
import pandas as pd
import numpy as np
import os
import wandb
import random
from wgan import Generator



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, img_folder, csv_file, transform=None):
        self.img_folder = img_folder
        self.csv_file = csv_file
        self.transform = transform

        # Read CSV file containing image filenames and one-hot encoded labels
        self.data_info = pd.read_csv(csv_file)

        # Number of items in the dataset
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        # Get image name from the dataframe
        img_name = os.path.join(self.img_folder, self.data_info.iloc[index, 0] + '.jpg') # Add file extension

        # Open image
        image = Image.open(img_name)

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        # # Get one-hot encoded label from the dataframe
        label = self.data_info.iloc[index, 1:].values.astype(np.float32) # Convert labels to float32

        # # Convert one-hot encoded label to normal label
        label = np.argmax(label)
        return image, label

    def __len__(self):
        return self.data_len

transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]),])


# Example dataset instantiation
dataset = CustomDataset(img_folder='/kaggle/input/isic-skin-lesion-dataset-2016/Test-20240422T183231Z-001/Test/Test_data/Test_data',
                        csv_file='/kaggle/input/isic-skin-lesion-dataset-2016/Test-20240422T183231Z-001/Test/Test_labels.csv',
                        transform=transform)

# Example dataloader creation
testloader = DataLoader(dataset, batch_size=32, shuffle=True,drop_last=True)

batch_demo = next(iter(testloader))
print('a',batch_demo[0].shape)
print('b',batch_demo[1].shape)


def load_random_image(folder_path, target_size=(10, 10)):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    random_image_filename = random.choice(image_files)
    random_image_path = os.path.join(folder_path, random_image_filename)

    image = Image.open(random_image_path)
   
    # Apply transformations
    transform_rand = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    image_tensor = transform_rand(image)

    return image, image_tensor

# Example usage:
folder_path = "/kaggle/input/test-demo"
random_image, random_image_tensor = load_random_image(folder_path)

image_size = 64
# labels 
num_classes = 7

# Instantiate the Generator class
gen = Generator(channels_noise=100, channels_img=3, 
                features_g=64, num_classes=num_classes, image_size=image_size, embed_size=100).to(device)

gen.eval()

# Test loop
with torch.no_grad():

    for batch_idx, (real, labels) in enumerate(testloader):
        labels = labels.to(device)
        
        random_image, random_image_tensor = load_random_image(folder_path)
        # Generate fake images
        noise =  random_image_tensor.repeat(32, 1, 1, 1).reshape(32, 100, 1, 1).to(device)  + 0.3 * torch.randn((32, 100, 1, 1)).to(device)
        #noise = torch.randn((32, 100, 1, 1)).to(device)
        # Generate fake images
        fake = gen(noise, labels)

 
        # Plot the generated images
        if batch_idx == 0:
            grid_fake = make_grid(fake[:32], nrow=8, normalize=True)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid_fake.cpu().permute(1, 2, 0))
            plt.title(f'Labels: {labels[:32].tolist()}')
            plt.axis('off')
            plt.show()
            
        break
 
