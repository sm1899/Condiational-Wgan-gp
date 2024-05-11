import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from PIL import Image
import pandas as pd
import numpy as np
import os
import wandb
import random
import matplotlib.pyplot as plt 
from wgan import Generator, Discriminator, gradient_penalty, initialize_weights

# # Set your WandB API key
# wandb_api_key = ""

# # Initialize WandB with your API key
# wandb.login(key=wandb_api_key)

# # Initialize wandb with your project name
# wandb.init(project="GAN")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 32

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
dataset = CustomDataset(img_folder='/csehome/m23mac008/dl4/Train_data',
                        csv_file='/csehome/m23mac008/dl4/Assignment_4/Train/Train_labels.csv',
                        transform=transform)

# Example dataloader creation
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)

batch_demo = next(iter(trainloader))
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
folder_path = "/csehome/m23mac008/dl4/Assignment_4/Train/Contours"
random_image, random_image_tensor = load_random_image(folder_path)

print('c',random_image_tensor.shape)


## Check the models with random noise and random labels
image_size = 64
# labels 
num_classes = 7
labels = torch.randint(0, num_classes, (16,)).to(device)
gen = Generator(channels_noise=100, channels_img=3, 
                features_g=64, num_classes=num_classes, image_size=image_size, embed_size=100).to(device)
critic = Discriminator(channels_img=3, features_d=64,
                       num_classes=num_classes, image_size=image_size).to(device)
initialize_weights(gen)
initialize_weights(critic)
x = torch.randn((16, 100, 1, 1)).to(device)
gen_out = gen(x, labels)
print(gen_out.shape)
disc_out = critic(gen_out, labels)
print(disc_out.shape)



def gradient_penalty(critic, real, labels, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images,labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty



#Hyperparameters etc.
LEARNING_RATE = 1e-4
CRITIC_ITERATIONS = 5
LAMBDA_GP = 100
NUM_EPOCHS = 100
mu  = 1 
# # initialize gen and disc, note: discriminator should be called critic,
# # according to WGAN paper (since it no longer outputs between [0, 1])


# # initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
step = 0

gen.train()
critic.train()


for epoch in range(NUM_EPOCHS):
    critic_losses = []
    gan_losses = []
    
    for batch_idx, (real, labels) in enumerate(trainloader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)] 
        # equivalent to minimizing the negative of that
        # min -[E[critic(real)] - E[critic(fake)]]
         
        for _ in range(CRITIC_ITERATIONS):
            random_image, random_image_tensor = load_random_image(folder_path)
            reshaped_tensor = random_image_tensor.repeat(batch_size, 1, 1, 1).reshape(batch_size, 100, 1, 1) + mu * torch.randn((batch_size, 100, 1, 1)) 
            noise = reshaped_tensor.to(device)

            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, real, labels, fake, device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            critic_losses.append(loss_critic.item())  # Append critic loss to list

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]

        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        gan_losses.append(loss_gen.item())  # Append GAN loss to list

        
        # Print losses occasionally
        # if batch_idx % 100 == 0 and batch_idx > 0:
            # print(
            #         f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(trainloader)} \
            #         Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            #     )
            # # Save and display generated images,
            # with torch.no_grad():
            #     fake = gen(noise, labels)
            #     torchvision.utils.save_image(fake[:batch_size], os.path.join('result4', f"fake_images_epoch_{epoch}.png"), normalize=True)

    # Log losses to wandb
    #wandb.log({"Critic Loss": sum(critic_losses) / len(critic_losses), "GAN Loss": sum(gan_losses) / len(gan_losses)}, step=epoch)

    # Print losses 
    print(
        f"Epoch [{epoch}/{NUM_EPOCHS}] \
        Loss D: {sum(critic_losses) / len(critic_losses):.4f}, \
        Loss G: {sum(gan_losses) / len(gan_losses):.4f}"
    )

    # Save and display generated images
    with torch.no_grad():
        fake = gen(noise, labels)
        torchvision.utils.save_image(fake[:batch_size], os.path.join('result_unpaired_1', f"fake_images_epoch_{epoch}.png"), normalize=True)
    
    if epoch % 5 == 0 and epoch > 0:

        torch.save(gen.state_dict(), os.path.join('result_unpaired_1', f"gen_weights_epoch_{epoch}.pt"))
        torch.save(critic.state_dict(), os.path.join('result_unpaired_1', f"critic_weights_epoch_{epoch}.pt"))

#wandb.finish()