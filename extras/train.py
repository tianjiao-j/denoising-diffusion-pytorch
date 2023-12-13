import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=100  # number of steps
    )

    img = Image.open('/home/tj/PycharmProjects/denoising-diffusion-pytorch/tj/rick.jpeg')
    transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    #training_images = torch.rand(8, 3, 128, 128)  # images are normalized from 0 to 1
    training_images = transform(img)
    training_images = torch.unsqueeze(training_images, dim=0)
    loss = diffusion(training_images)
    loss.backward()

    # after a lot of training

    sampled_images = diffusion.sample(batch_size=4)
    sampled_images.shape  # (4, 3, 128, 128)