from torch import nn

"""
FCNEt used in the GeoPrior Paper
"""

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out

class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, dim_hidden):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(dim_hidden, num_classes, bias=self.inc_bias)

        self.feats = nn.Sequential(nn.Linear(num_inputs, dim_hidden),
                                    nn.ReLU(inplace=True),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden))

    def forward(self, x):
        loc_emb = self.feats(x)
        class_pred = self.class_emb(loc_emb)
        return class_pred
