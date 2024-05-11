import  torch
import  torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(features_d * 2, affine=True)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3 = nn.InstanceNorm2d(features_d * 4, affine=True)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm4 = nn.InstanceNorm2d(features_d * 8, affine=True)
        self.leaky_relu4 = nn.LeakyReLU(0.2)
        self.conv5 = nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)
        
        self.embed = nn.Embedding(num_classes, image_size * image_size)

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.norm2(self.conv2(x)))
        x = self.leaky_relu3(self.norm3(self.conv3(x)))
        x = self.leaky_relu4(self.norm4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, image_size, embed_size):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_size)
        self.image_size = image_size
        
        self.conv1 = nn.ConvTranspose2d(channels_noise + embed_size, features_g * 16, kernel_size=4, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(features_g * 16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(features_g * 8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(features_g * 4)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(features_g * 2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm5 = nn.BatchNorm2d(features_g)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)
        self.tanh = nn.Tanh()

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = self.tanh(self.upsample(self.conv6(x)))
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# ## Check the models with random noise and random labels
# image_size = 64
# # labels 
# num_classes = 7
# labels = torch.randint(0, num_classes, (16,))
# gen = Generator(channels_noise=100, channels_img=3, 
#                 features_g=64, num_classes=num_classes, image_size=image_size, embed_size=100)
# critic = Discriminator(channels_img=3, features_d=64,
#                        num_classes=num_classes, image_size=image_size)
# initialize_weights(gen)
# initialize_weights(critic)
# x = torch.randn((16, 100, 1, 1))
# gen_out = gen(x, labels)
# print(gen_out.shape)
# disc_out = critic(gen_out, labels)
# print(disc_out.shape)



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
