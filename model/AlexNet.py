import torch
from torch import nn

def gaussion_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1 and classname.find('Conv2d')!= -1:
        m.weight.data.normal_(0.0, 0.02)

class AlexNet(nn.Module):
    def __init__(self, IMG_CHANNELS=1, n_classes=7, img_size=48):
        super(AlexNet, self).__init__()
        flatten_size = (img_size // 8) * (img_size // 8) * 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=IMG_CHANNELS, out_channels=96, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv1.apply(gaussion_weights_init)
        self.conv2.apply(gaussion_weights_init)
        self.conv3.apply(gaussion_weights_init)
        self.conv4.apply(gaussion_weights_init)
        self.conv5.apply(gaussion_weights_init)

        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, n_classes),
        )

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y
