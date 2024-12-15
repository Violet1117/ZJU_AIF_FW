import torch.nn as nn

def gaussion_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1 and classname.find('Conv2d')!= -1:
        m.weight.data.normal_(0.0, 0.02)


class TYCNN(nn.Module):
    def __init__(self, IMG_CHANNELS=1, n_classes=7, img_size=48, kernel_size=3):
        super(TYCNN, self).__init__()
        flatten_size = (img_size // 8) * (img_size // 8) * 256
        self.conv1 = nn.Sequential(
            # first conv layer
            nn.Conv2d(in_channels=IMG_CHANNELS, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            # second conv layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            # third conv layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv1.apply(gaussion_weights_init)
        self.conv2.apply(gaussion_weights_init)
        self.conv3.apply(gaussion_weights_init)

        self.fc = nn.Sequential(
            # fully connected layer
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(flatten_size, 4096),
            nn.RReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.RReLU(inplace=True),
            nn.Linear(1024, n_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

