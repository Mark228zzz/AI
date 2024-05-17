from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def init_variables() -> None:
    global device, dataloader, nz, ngf, ndf, nc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'face_generator'), transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    nz = 100
    ngf = 64
    ndf = 64
    nc = 3

init_variables()


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
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
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)


def weights_init(m) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(ndf, nc).to(device)

def create_new_module() -> None:
    netG.apply(weights_init)
    netD.apply(weights_init)

def load_module(*, name_of_modules: tuple[str, str]) -> None:
    global netG, netD

    netG.load_state_dict(torch.load(f'{os.path.dirname(os.path.abspath(__file__))}/modules/face_generator/{name_of_modules[0]}'))
    netD.load_state_dict(torch.load(f'{os.path.dirname(os.path.abspath(__file__))}/modules/face_generator/{name_of_modules[1]}'))

def optimizer() -> None:
    global optimizerD, optimizerG, criterion
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

        show_result(should_show=False, should_save_result=True)

def show_result(*, should_show_result: bool = True, should_show: bool = True, should_save_result: bool = False) -> None:
    if not should_show_result: return

    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake = netG(noise).detach().cpu()

    plt.figure(figsize=(5,5))
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(fake[0].permute(1, 2, 0) * 0.5 + 0.5)
    if should_show: plt.show()
    if should_save_result: plt.savefig(f'{os.path.dirname(os.path.abspath(__file__))}/pictures/{datetime.now()}.png')

def save_module(*, should_save_module: bool, name_of_modules: tuple[str, str] = ('', '')) -> None:
    if not should_save_module: return

    torch.save(netG.state_dict(), f'{os.path.dirname(os.path.abspath(__file__))}/modules/five_generator/NetG{name_of_modules[0]}.pth')
    torch.save(netD.state_dict(), f'{os.path.dirname(os.path.abspath(__file__))}/modules/five_generator/NetD{name_of_modules[1]}.pth')

def main() -> None:
    should_create_new_module = input('Create a new module? [y/n] --> ').lower().strip()

    match should_create_new_module:
        case 'y': create_new_module()
        case _: load_module(name_of_modules=(input('Name of modules which will load "NetG.pth NetD.pth"--> ').split()))

    num_epochs = int(input("Number of epochs --> "))
    should_show_result = bool(input('Show result? [y/"NOTHING"] --> ').lower().strip())
    should_save_module = bool(input('Save the module? [y/"NOTHING"] --> ').lower().strip())
    name_of_modules = ('', '')
    if should_save_module: name_of_modules = (input('Name of new modules "_test _test"--> ').split())

    optimizer()
    learn(num_epochs=num_epochs)

    save_module(should_save_module=should_save_module, name_of_modules=name_of_modules)
    show_result(should_show_result=should_show_result)

if __name__ == '__main__':
    main()
