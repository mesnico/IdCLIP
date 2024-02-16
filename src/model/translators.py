from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=1, hidden_dim=768):
        super().__init__()

        # create a sequence of num_layers linear layers using a ModuleList
        self.linears = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, bias=False) 
            for i in range(num_hidden_layers)
        ])
        self.out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        for layer in self.linears:
            x = F.relu(layer(x))
        x = self.out(x)
        return x