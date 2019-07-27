import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=5, padding=0)
        self.Transconv1 = nn.ConvTranspose2d(in_channels=4,out_channels=16, kernel_size=5, stride=4, padding=0)
        self.Transconv2 = nn.ConvTranspose2d(in_channels=16,out_channels=5, kernel_size=5, stride=3, padding=0)
        self.Transconv3 = nn.ConvTranspose2d(in_channels=5,out_channels=1, kernel_size=3, stride=1, padding=0)
        self.Transconv4 = nn.ConvTranspose2d(in_channels=1,out_channels=1, kernel_size=2, stride=1, padding=0)
        self.m = nn.Sigmoid()

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("size after relu: ", x.size())
        x = F.relu(self.conv2(x))
        # print("size after relu: ", x.size())
        x = F.relu(self.conv3(x))
        # print("size after relu: ", x.size())
        latent = x.view(-1,16)
        # print("size of latent vector ", x.size())
        x = self.Transconv1(x)
        # print("size after Transconv1: ", x.size())
        x = self.Transconv2(x)
        # print("size after Transconv2: ", x.size())
        x = self.Transconv3(x)
        # print("size after Transconv3: ", x.size())
        x = self.Transconv4(x)
        # print("size after Transconv4: ", x.size())
        x = self.m(x)

        return x, latent

if __name__ == "__main__":

    device = torch.device('cpu') 

    net = Net()
    tensor = torch.tensor((), dtype=torch.float32)
    x = tensor.new_full(size=(1,1,32,32), fill_value=1).to(device)
    net.eval()
    output, latent = net(x)
    print(output)
