import torch
import torch.nn

def CarCNN(nn.Module):
    def __init__(self):
        super.__init__()

        self.conv() = nn.Sequential(
            nn.Conv2D(4, 32),
            nn.ReLU(),
            nn.Conv2D(32, 64),
            nn.ReLU(),
            nn.Conv2D(64, 64),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
            nn.ReLU(),
        )
        
    def forward(self, x):
            return self.fc(self.conv(x))