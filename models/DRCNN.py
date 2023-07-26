import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)

    def forward(self, x):
        len = x.shape[1]
        x = self.avg(x.permute(0, 2, 1))
        x = F.interpolate(x, size=(len), mode='linear')
        return x.transpose(1, 2)

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        # print(moving_mean.shape, x.shape)
        res = x - moving_mean
        return res, moving_mean

class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, num_layers=2):
        super(ConvLayer, self).__init__()
        padding_size = (kernel_size - 1) * dilation // 2
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                               padding=padding_size, dilation=dilation, groups=num_layers)
        self.norm = nn.LayerNorm(c_out)
        self.conv1x1 = nn.Conv1d(c_in, c_out, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.idx = (torch.floor(torch.arange(0, c_in) / num_layers) * num_layers).long() + torch.concat([torch.randperm(num_layers) for _ in range(c_in // num_layers)], dim=0).long()

    def forward(self, x):
        if self.num_layers != 1:
            y = torch.concat([x[:, self.idx[i::self.num_layers], :] for i in range(self.num_layers)], dim=1)
        else:
            y = x
        y = self.dropout(self.activation(self.norm(self.conv1(y).transpose(1, 2)).transpose(1, 2)))
        x = self.conv1x1(x)
        return x + y

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, freq=configs.freq)
        self.res = int(configs.res) + 1
        self.div = configs.div
        self.decompsition = nn.ModuleList([series_decomp(i) for i in range(25, 2, -22 // self.res)])
        self.Linear_Trend = ConvLayer(self.seq_len, self.seq_len, 1, 1, self.div)
        self.Conv_Residual = ConvLayer(self.seq_len, self.seq_len, 1, 1, self.div)
        self.Conv_Trend = nn.ModuleList([ConvLayer(self.seq_len, self.seq_len, 3, 2, self.div) for _ in range(self.res)]) 
        self.importance = nn.Linear(1, 1 + self.res)
        self.encoder = nn.Conv1d(self.seq_len, self.pred_len, 1)

    def forward(self, x):
        output = []
        residual_init = x
        for i in range(self.res):
            residual_init, trend_init = self.decompsition[i](residual_init)
            if i == 0:
                trend_init = self.Linear_Trend(trend_init)
            else:
                trend_init = self.Conv_Trend[i](trend_init)
            output.append(trend_init.unsqueeze(-1))
        residual_init = self.Conv_Residual(residual_init)
        output.append(residual_init.unsqueeze(-1))
        output = torch.concat(output, dim=-1)
        output = torch.sum(output * nn.Softmax(-1)(self.importance(x.unsqueeze(-1))), dim=-1)
        return self.encoder(output)