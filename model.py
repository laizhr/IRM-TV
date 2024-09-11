from torch import nn
class InferEnv(nn.Module):
    def __init__(self, flags, z_dim):
        super(InferEnv, self).__init__()
        self.lin1 = nn.Linear(z_dim, flags.hidden_dim)
        self.lin2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2, nn.Sigmoid())

    def forward(self, input):
        out = self._main(input)
        return out


class InferEnvMultiClass(nn.Module):
    def __init__(self, flags, z_dim, class_num):
        super(InferEnvMultiClass, self).__init__()
        self.lin1 = nn.Linear(z_dim, flags.hidden_dim_infer)
        self.lin2 = nn.Linear(flags.hidden_dim_infer, class_num)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2, nn.Softmax(dim=1))

    def forward(self, input):
        out = self._main(input)
        return out




