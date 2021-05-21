import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
from InceptionAndSE_block import *


# *------------------ SA block ---------------------------------*
# 该SA-block用法大致同 SE-block
class SAblock(nn.Module):
    def __init__(self, input_channel: int, r: int = 8):
        super(SAblock, self).__init__()
        C_apostrophe = max(input_channel // r, 3)
        self.query_conv = nn.Conv2d(in_channels=input_channel, out_channels=C_apostrophe, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channel, out_channels=C_apostrophe, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channel, out_channels=C_apostrophe, kernel_size=1)
        self.ChannelBack = nn.Conv2d(in_channels=C_apostrophe, out_channels=input_channel, kernel_size=1)
        return

    def forward(self, x):
        # x : [BatchSize * C * H * W]
        # Step1 转换 channels C - > C'
        output_query = self.query_conv(x)  # [[BatchSize * C * H * W]]
        output_key = self.key_conv(x)
        output_value = self.value_conv(x)

        # Step2 inner product: get attention
        output_query = output_query.view(x.shape[0], output_query.shape[1], x.shape[-1] * x.shape[-2])
        output_query = output_query.transpose(1, 2)  # [BS ,(HW) ,C']
        output_key = output_key.view(x.shape[0], output_key.shape[1], x.shape[-1] * x.shape[-2])  # [BS ,C' , HW]
        Attention_weights = F.softmax(torch.matmul(output_query, output_key),
                                      dim=-1)  # [BatchSize , (HW),  HW] 每一"列和" 为1
        output_value = output_value.view(x.shape[0], output_value.shape[1], x.shape[-1] * x.shape[-2])
        out = torch.matmul(output_value,
                           Attention_weights)  # [BatchSize , C',  HW] , 相当于对 Value 层的每个特征channel下的图像 re-weight
        out = out.view(x.shape[0], out.shape[1], x.shape[-2], x.shape[-1])
        # Step3
        out = self.ChannelBack(out)
        out = out + x  # 残差训练
        return out


# 修改版 GoogLeNet 以兼容SA-block
class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
            if_efficiency=False,  # 参考 rethink Inception, 思索卷积核的极限
            if_SE_block=False,
            if_SA_block=True  # 加完效率降太多
    ) -> None:
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),  # 降卷积核维度 / 节省计算
            nn.BatchNorm2d(ch1x1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3)
        )

        if if_efficiency:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
                # 更高效的方式: 用两波 3*3 完美代替 5*5
                nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
                nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch5x5)
            )
        else:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
                # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
                # 但,我们只是想试验，也不用pytorch的预训练，直白还原
                nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
                nn.BatchNorm2d(ch5x5)
            )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj)
        )
        self.if_SE_block = if_SE_block
        self.if_SA_block = if_SA_block
        if if_SE_block:
            self.se_block = SEblock(ch1x1 + ch3x3 + ch5x5 + pool_proj)
        if if_SA_block:
            self.sa_block = SAblock(input_channel=ch1x1 + ch3x3 + ch5x5 + pool_proj)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        # 我们发现，在每次Inception网络后 加与不加 Relu 激活函数,在我们的试验中有明显区别
        # 加完relu,训练效果更好，算是我们对 原torchvision.models.googlenet 中Inception的改进
        if self.if_SE_block:
            return F.relu(self.se_block(torch.cat(outputs, 1)))
        elif self.if_SA_block:
            return F.relu(self.sa_block(torch.cat(outputs, 1)))
        else:
            return F.relu(torch.cat(outputs, 1))
            # return torch.cat(outputs, 1)


class MyInceptionNet(nn.Module):
    # 借鉴GoogLeNet而来 , 参考; 为了和Alexnet保持同步,本次也选择5层深度的卷积层
    def __init__(
            self,
            num_classes: int = 10,
            if_efficiency=False,  # Add SE-block or not
            if_SE=False,
            if_SA=True
    ):
        super(MyInceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # inception 参数 [ in_channels , ch1x1 ,ch3x3red: reduction ,ch3x3: int,ch5x5red: reduction ,ch5x5 ,pool_proj ]
        self.inception1 = Inception(16, 16, 24, 32, 4, 8, 8, if_efficiency=if_efficiency, if_SE_block=if_SE)
        self.inception2 = Inception(64, 32, 32, 48, 8, 24, 16, if_efficiency,
                                    if_SE)  # output channels: [ 120 * 32 *32 ]
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H W砍一半 [120 * 16 * 16]

        self.inception3 = Inception(120, 48, 24, 52, 4, 12, 16, if_efficiency, if_SE)
        self.inception4 = Inception(128, 40, 28, 56, 6, 16, 16, if_efficiency, if_SE)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H W砍一半 [128 * 8 * 8]

        self.inception5 = Inception(128, 64, 40, 80, 8, 32, 32, if_efficiency, if_SE)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 压成 C*1*1
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(208, num_classes)

    def forward(self, x):
        # N x 3 x 32 x 32
        x = F.relu(self.bn1(self.conv1(x)))  # 经典三件套
        # N x 16 x 32 x 32
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool1(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.maxpool2(x)
        x = self.inception5(x)

        x = self.avgpool(x)
        # N x 208 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 208
        x = self.dropout(x)
        x = self.fc(x)
        # N x 10 (num_classes)
        return x


# *------------------- CCT: Transformer -------------------------*
# Meta CNN, CCT transformer
class Attention(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True):
        super(Tokenizer, self).__init__()
        # 不用麻烦了，就搞一层
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]
        # input -- 中间过渡 in_planes * in_planes -- output channel

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=False),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=32, width=32):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]  # 你这，有点不优异，自己又跑了一遍可还行

    def forward(self, x):
        # return self.flattener(self.conv_layers(x)).transpose(-2, -1)
        out = self.conv_layers(x)  # 连着对 H、W downsample两次， 砍成1/4
        return self.flattener(out).transpose(-2, -1)  # [BS , downsample(H*W) , output C]

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                          requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                   requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])  # 几层 transformer
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)
        # transformers 堆叠
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class CCT(nn.Module):
    def __init__(self,
                 img_size=32,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,  # 起码三件套得整一手吧
                                   n_conv_layers=n_conv_layers)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)
