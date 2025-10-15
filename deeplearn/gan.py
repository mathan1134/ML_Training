# identyfy real fake
#  create a fake 


import torch.nn as nn


class generater(nn.module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)
    

class descriminater(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(784,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x)    :
        return self.model(x)
    
