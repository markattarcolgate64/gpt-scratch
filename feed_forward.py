from torch import nn
class FeedForward(nn.Module):
    """ simple linear layer """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )
    
    def forward(self,x):
        return self.net(x)