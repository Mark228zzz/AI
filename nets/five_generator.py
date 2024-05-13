import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim

def init_variables() -> None:
    global device, nz, ngf, ndf, nc, dataloader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'five_generator'))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    nz, ngf, ndf, nc = 100, 64, 64, 3


class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
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
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def create_modules() -> None:
    global netG, netD

    netG = Generator(nz, ngf, nc).to(device)
    netD = Generator(nz, ngf, nc).to(device)

def weights_init(m) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def apply_weights() -> None:
    netG.apply(weights_init)
    netD.apply(weights_init)

def optimizer() -> None:
    global optimizerG, optimizerD, criterion

    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

def learn(*, num_epochs: int) -> None:
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)

            output = netD(real).view(-1)
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

        print(f'Epoch {epoch + 1}/{num_epochs} [{i + 1}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z1)): {D_G_z1:.4f} D(G(z2)): {D_G_z2:.4f}')

def show_result() -> None:
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake = netG(noise).detach().cpu()

    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(fake[0].permute(1, 2, 0) * 0.5 + 0.5)
    plt.show()

def save_modules(*, should_save_module: bool, name_of_modules: tuple[str, str] = ('', '')) -> None:
    if not should_save_module: return

    torch.save(netG, f'{os.path.dirname(os.path.abspath(__file__))}/modules/five_generator/NetG{name_of_modules[0]}.pth')
    torch.save(netD, f'{os.path.dirname(os.path.abspath(__file__))}/modules/five_generatorNetD{name_of_modules[1]}.pth')

def main() -> None:
    init_variables()
    create_modules()
    apply_weights()
    optimizer()
    learn(num_epochs=2)

if __name__ == '__main__':
    main()
