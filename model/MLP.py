import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, IMG_CHANNELS=1, n_classes=7, img_size=48, layer_size1 = 1024, layer_size2 = 512):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_CHANNELS * img_size * img_size, layer_size1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(layer_size1, layer_size2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        y = self.fc(x)
        return y