from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 768, bias=False)
        #self.hidden_fc_1 = nn.Linear(768, 768, bias=False)
        #self.hidden_fc_2 = nn.Linear(1024, 1024, bias=False)
        #self.hidden_fc_3 = nn.Linear(1024, 2048, bias=False)
        #self.hidden_fc_4 = nn.Linear(2048, 1024, bias=False)
        #####dropout
        #self.dropout_1 = nn.Dropout(dropout)
        #self.dropout_2 = nn.Dropout(dropout)
        self.output_fc = nn.Linear(768, output_dim, bias=False)
        #self.output_fc = nn.Linear(input_dim, output_dim, bias=False)
        

    def forward(self, x):
        x = F.relu(self.input_fc(x))
        #x = self.dropout_1(x)
        #x = F.relu(self.hidden_fc_1(x))
        #x = self.dropout_2(x)
        #x = F.relu(self.hidden_fc_2(x))
        #x = F.relu(self.hidden_fc_3(x))
        #x = F.relu(self.hidden_fc_4(x))
        x = self.output_fc(x)
        return x