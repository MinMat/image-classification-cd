import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
class ImageClassificationModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #output 32 X 100 X 100 | (Receptive Field (RF) -  3 X 3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   #output 64 X 100 X 100 | RF 5 X 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 50 x 50 | RF 10 X 10

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 64 x 50 x 50 | RF 12 X 12
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 25 x 25  | RF 24 X 24
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 x 25 x 25  | RF 26 X 26
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 12 x 12 | RF 52 X 52
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1), #512* 10* 10 | RF 54 X 54
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 512 x 5 x 5 | RF - 108X 108
            

            nn.Flatten(),
            nn.Linear(512 * 5 * 5,10))
         
    def forward(self, xb):
        return self.network(xb)    