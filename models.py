# models.py – Shared ConvLSTM definitions

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        h, w = image_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        self.cell_list = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_dim if i == 0 else self.hidden_dim[i-1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=kernel_size,
                bias=bias
            ) for i in range(num_layers)
        ])

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.shape

        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w))

        cur_layer_input = input_tensor
        for i in range(self.num_layers):
            h, c = hidden_state[i]
            outputs = []
            for t in range(seq_len):
                h, c = self.cell_list[i](cur_layer_input[:, t], (h, c))
                outputs.append(h)
            cur_layer_input = torch.stack(outputs, dim=1)
        return cur_layer_input, None

    def _init_hidden(self, batch_size, image_size):
        return [cell.init_hidden(batch_size, image_size) for cell in self.cell_list]


class UrbanSprawlConvLSTM(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=32, num_layers=2, output_channels=1):
        super().__init__()
        self.convlstm = ConvLSTM(input_channels, hidden_dim, 3, num_layers, True)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.convlstm(x)
        last = lstm_out[:, -1]
        return self.output_conv(last)


