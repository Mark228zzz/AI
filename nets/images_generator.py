import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nz = 100
ngf = 64
ndf = 64
nc = 3

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root=f'{os.path.dirname(os.path.abspath(__file__))}/data/images_generator', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 16, 1, 1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# create new module?
def create_new_module(*, should_create_new_module: bool, module_names: tuple[str, str] = ('', '')) -> None:
    '''
    if you want to create new module you don`t have to indicate.
    module_names must contain 2 module names.
    the first is the NetG module, the second is the NetD module.
    example: ('NetG.pth', 'Net.D.pth')
    '''
    global netG, netD

    if should_create_new_module:
        netG = Generator(nz, ngf, nc).to(device)
        netD = Discriminator(ndf, nc).to(device)
    else:
        netG = torch.load(f'{os.path.dirname(os.path.abspath(__file__))}/modules/images_generator/{module_names[0]}')
        netD = torch.load(f'{os.path.dirname(os.path.abspath(__file__))}/modules/images_generator/{module_names[1]}')


def weights_init(m) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def optimizer() -> None:
    global criterion, optimizerD, optimizerG

    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

def learn(*, num_epochs: int, show_epochs: bool = True) -> None:
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(1)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if epoch % 1 == 0 and show_epochs:
                print(f'Epoch {epoch+1}/{num_epochs} [{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

def show_result() -> None:
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake = netG(noise).detach().cpu()

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(fake[0].permute(1, 2, 0) * 0.5 + 0.5)
    plt.show()

def save_module(*, should_save_model: bool, module_names: tuple[str, str] = ('', '')) -> None:
    '''
    if you want to create new module you don`t have to indicate.
    you don`t need to indicate .pth or NetG these will be installed automatically.
    module_names must contain 2 module names.
    the first is the name of NetG module, the second is the name of NetD module.
    example: ('_version_1', '_version_1') -> NetG_version_1.pth, NetD_version_1.pth
    '''
    if should_save_model:
        torch.save(netG, f'NetG{module_names[0]}.pth')
        torch.save(netD, f'NetD{module_names[1]}.pth')

def main():
    create_new_module(should_create_new_module=True)

    netG.apply(weights_init)
    netD.apply(weights_init)

    optimizer()
    learn(num_epochs=20)

    show_result()

    save_module(should_save_model=True, module_names=('_test', '_test'))

if __name__ == '__main__':
    main()
