from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import Conv1d
from torch.nn.modules.conv import _ConvNd
from torch import nn
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import config


class GatedCBEASPP(nn.Module):
    def __init__(self,
                batch_size,
                 out_size = 4,
                 embed_dim = 21,
                 kernel_width = 3,
                 out_channel = 64,
                 n_layers = 64,
                 res_block_cnt = 2,
                 activationFunc = 'none',
                 cond_dropout_rate = 0.2,
                 lstm_dropout_rate = 0.0,
                #  config,
                 dropout=0.1):
        super(GatedCBEASPP, self).__init__()

        feature_size = 21
        class_num = out_size

        dilation_list = [1,2,4,8]
        aspp_layer_num = len(dilation_list)
        window_size_list = [1,3,3,3]
        top_window_size = 11

        blstm_layer_num = 2
        lstm_hidden_size = 512
        # lstm_dropout_rate = 0.0

        output_dropout_rate = 0.0
        output_layer = 'conv1'

        if activationFunc == 'logsigmoid':
            actFunc = nn.LogSigmoid()
        elif activationFunc == 'sigmoid':
            actFunc = nn.Sigmoid()
        elif activationFunc == 'tanh':
            actFunc = nn.Tanh()
        elif activationFunc == 'relu':
            actFunc = nn.ReLU()
        elif activationFunc == 'none':
            actFunc = None

        self.res_block_cnt = res_block_cnt

        # self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.padding_left = nn.ConstantPad1d((kernel_width - 1, 0), 0)

        self.conv_0 = nn.Sequential(CondConv1D(in_channels=embed_dim, out_channels=out_channel,
                                               kernel_size=kernel_width, stride=1,
                                               padding=0, dilation=1, num_experts=3, dropout_rate=cond_dropout_rate)
                                    # , actFunc
                                    )
        self.b_0 = nn.Parameter(torch.zeros(out_channel, 1))  # same as paper
        self.conv_gate_0 = nn.Sequential(
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=out_channel,
                kernel_size=kernel_width)
            #  , actFunc
        )
        self.c_0 = nn.Parameter(torch.zeros(out_channel, 1))

        self.convs = nn.ModuleList([nn.Sequential(CondConv1D(in_channels=out_channel, out_channels=out_channel,
                                                             kernel_size=kernel_width, stride=1,
                                                             padding=0, dilation=1, num_experts=3, dropout_rate=cond_dropout_rate)
                                                  #   , actFunc

                                                  ) for _ in range(n_layers)
                                    ])

        self.bs = nn.ParameterList([
            nn.Parameter(torch.zeros(out_channel, 1))  # collections of b
            for _ in range(n_layers)
        ])

        self.conv_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_width)
                #    , actFunc
            ) for _ in range(n_layers)
        ])

        self.cs = nn.ParameterList([
            nn.Parameter(torch.zeros(out_channel, 1)) for _ in range(n_layers)
        ])

        self.fc = nn.Linear(out_channel, out_size)
        # self.dropout = nn.Dropout(p=dropout)  # todo use dropout

        # conv1d Input: (N, Cin, Lin)
        # constantpad1d Input: (N,C,Win)  Output: (N,C,Wout)

        self.conv_top = nn.Sequential(
            nn.Conv1d(
                in_channels=(out_channel + lstm_hidden_size * 2) *
                (aspp_layer_num + 1),
                out_channels=out_size,
                kernel_size=top_window_size,
                padding=int((top_window_size - 1) / 2)),
            #   nn.BatchNorm1d(num_features=out_size),
            nn.Dropout(output_dropout_rate),
            nn.LogSigmoid()
            # nn.Sigmoid()
        )

        self.conv_end = True
        self.resnet = True

        self.bilstm = nn.LSTM(
            input_size=feature_size,
            # input_size=config.node_size,
            # out_channels=config.node_size,
            hidden_size=lstm_hidden_size,
            num_layers=blstm_layer_num,
            batch_first=True,
            dropout=lstm_dropout_rate,
            bidirectional=True)

        # self.fc1 = nn.Sequential(
        #     nn.Sigmoid(),
        #     nn.Dropout(config.fc1_dropout_rate),
        #     # nn.Linear(feature_size*2, out_size),
        #     nn.Linear(config.node_size + lstm_hidden_size * 2,
        #               out_size),
        #     nn.ReLU())
        # # self.fc2 = nn.Sequential(
        # #     nn.Linear(out_size, out_size)
        # #     # ,nn.LogSoftmax(dim=out_size)
        # # )

        self.convs_aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=out_channel + lstm_hidden_size * 2,
                    out_channels=out_channel + lstm_hidden_size * 2,
                    kernel_size=aspp_window_size,
                    padding=int((aspp_window_size - 1) / 2 +
                                (dil - 1) * int((aspp_window_size - 1) / 2)),
                    # padding=int((config.bot_window_size-1)/2),
                    dilation=dil),
                nn.ReLU(),
                # af,
                nn.BatchNorm1d(out_channel + lstm_hidden_size * 2),

                # nn.Sigmoid()
            ) for dil, aspp_window_size in zip(dilation_list,
                                               window_size_list)
        ])

        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=(out_channel + lstm_hidden_size * 2) *
                (aspp_layer_num + 1),
                out_channels=out_size,
                kernel_size=1),
            #   nn.BatchNorm1d(num_features=config.node_size),
            nn.LogSigmoid(),
            nn.Dropout(output_dropout_rate),
            #   nn.BatchNorm1d(num_features=out_size) # after changed
            # nn.Sigmoid()
        )

        self.device = 'cuda'
        # self.hidden_size = feature_size
        self.lstm_hidden_size = lstm_hidden_size
        self.feature_size = feature_size
        self.batch_size = batch_size
        # self.data_max_text_len = config.data_max_text_len
        self.blstm_layer_num = blstm_layer_num
        self.output_layer = output_layer

    def init_hidden(self, batch_size):
        return (torch.zeros(self.blstm_layer_num * 2, batch_size,
                            self.lstm_hidden_size).to(self.device),
                torch.zeros(self.blstm_layer_num * 2, batch_size,
                            self.lstm_hidden_size).to(self.device))

    def forward(self, seq):
        # seq:(batch,seq_len)
        batch_size = seq.size(0)
        seq_len = seq.size(1)
        # x = self.embedding(seq)  # x: (batch,seq_len,embed_dim)
        # x.transpose_(1, 2)  # x:(batch,embed_dim,seq_len) , embed_dim equals to in_channel

        seq = seq.permute(0,2,1).to(torch.float)

        x = seq  # [32, 42, 700]
        x = self.padding_left(
            x)  # x:(batch,embed_dim,seq_len+kernel-1)  #padding left with 0

        A = self.conv_0(
            x
        )  # A: (batch,out_channel,seq_len)   seq_len because of padding (kernel-1)
        A += self.b_0  # b_0 broadcast
        B = self.conv_gate_0(x)  # B: (batch,out_channel,seq_len)
        B += self.c_0

        # h = A * F.sigmoid(B)  # h: (batch,out_channel,seq_len)
        h = A * torch.sigmoid(B)  # h: (batch,out_channel,seq_len)
        # todo: add resnet
        res_input = h

        for i, (conv, conv_gate) in enumerate(
                zip(self.convs, self.conv_gates)):
            h = self.padding_left(h)  # h: (batch,out_channel,seq_len+kernel-1)
            A = conv(h) + self.bs[i]
            B = conv_gate(h) + self.cs[i]
            h = A * torch.sigmoid(B)  # h: (batch,out_channel,seq_len+kernel-1)
            if self.resnet:
                if i % self.res_block_cnt == 0:
                    h += res_input
                    res_input = h

        # x.size() = [32,42,700]
        x = seq
        x = x.permute(0, 2, 1)
        # x.size() = [32,700,42]
        hidden_states = self.init_hidden(x.shape[0])
        bilstm_out, hidden_states = self.bilstm(
            x, hidden_states)  # bilstm_out=[32,700,1024]

        bilstm_out = bilstm_out.permute(0, 2, 1)
        backbone_out = torch.cat((h, bilstm_out), 1)

        # [32,X,700]
        for index, conv in enumerate(self.convs_aspp):
            out_each = conv(backbone_out)
            if index == 0:
                out = out_each
            else:
                out = torch.cat((out, out_each), 1)
        out = torch.cat((out, backbone_out), 1)

        if self.output_layer == 'conv1':
            logic = self.conv_1(out)
        elif self.output_layer == 'convtop':
            logic = self.conv_top(out)

        # if self.conv_end:
        #     logic = self.conv_top(h)
        # else:
        #     h.transpose_(1, 2)  # h:(batch,seq_len,out_channel)

        #     logic = self.fc(h)  # logic:(batch,seq_len,vocab_size)
        #     logic.transpose_(
        #         1, 2
        #     )  # logic:(batch,vocab_size,seq_len) cross_entropy input:(N,C,d1,d2,..) C is num of class
        return logic


class CondConv1D(_ConvNd):
    # class CondConv1D(Conv1d):
    r"""Learn specialized convolutional kernels for each example.
    As described in the paper
    `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
    conditionally parameterized convolutions (CondConv), 
    which challenge the paradigm of static convolutional kernels 
    by computing convolutional kernels as a function of the input.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts per layer 
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    .. _CondConv: Conditionally Parameterized Convolutions for Efficient Inference:
       https://arxiv.org/abs/1904.04971
    """

    def __init__(self, in_channels=42, out_channels=9, kernel_size=11, stride=1,
                 padding=int((11-1)/2), dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        # kernel_size = _pair(kernel_size)
        # stride = _pair(stride)
        # padding = _pair(padding)
        # dilation = _pair(dilation)

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        # super(CondConv1D, self).__init__(
        #     in_channels, out_channels, kernel_size, stride, padding, dilation,
        #     False, _pair(0), groups, bias, padding_mode)
        super(CondConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(
            F.adaptive_avg_pool1d, output_size=(1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        # print('input', input.size())
        pooled_inputs = self._avg_pooling(input)
        # print('pooled_inputs', pooled_inputs.size())
        routing_weights = self._routing_fn(pooled_inputs)
        # print('routing_weights', routing_weights.size())
        # print('self.weight', self.weight.size())

        for idx, weight_sub in enumerate(self.weight):
            kernel_sub = weight_sub
            # kernel_sub = torch.squeeze(weight_sub, 0)
            output_0 = self._conv_forward(input, kernel_sub)
            if idx == 0:
                sumout = routing_weights[:, idx, None, None] * output_0
            else:
                # sumout = torch.sum(routing_weights[:, idx, None, None] * output_0 ,1)
                sumout += routing_weights[:, idx, None, None] * output_0
        # print('sumout', sumout.size())
        return sumout

        # first:([100, 42, 11])
        kernels = torch.sum(
            routing_weights[:, None, None, None] * self.weight, 0)
        # print('------kernels', kernels.size())
        # kernels = torch.sum(routing_weights[:,None, None, None, None] * self.weight, 0)

        return self._conv_forward(input, kernels)


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        # x = torch.flatten(x)
        x = torch.squeeze(x, 2)
        # print('flattenX', x.size())
        x = self.dropout(x)
        x = self.fc(x)

        # return F.sigmoid(x)
        return torch.sigmoid(x)
