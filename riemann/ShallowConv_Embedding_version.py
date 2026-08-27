"""ShallowConvNet with a learned embedding, adapted the same way EEGNet was.

This mirrors, operation for operation, what Ninon's EEGNet_Embedding_version.py does
to braindecode's EEGNetv4: the reference architecture is kept intact and its final
CONVOLUTIONAL classifier is replaced by Linear(-> 512) + Linear(512 -> n_classes) +
LogSoftmax, so a 512-d embedding is exposed. The embedding is not cosmetic: the
downstream goal is reconstruction, not only classification, so the representation
has to stay readable.

Reference: Schirrmeister et al. 2017, "Deep learning with convolutional neural
networks for EEG decoding and visualization" -- ShallowConvNet, built to mirror
FBCSP. The defining chain is

    conv_time -> conv_spat -> SQUARE -> AvgPool -> LOG

which is literally log band power, and is exactly the operation EEGNet's
ELU + AvgPool chain cannot express. That is why this class is the right comparison
for the perception -> imagery bridge, where the signal is induced posterior alpha.

Input convention matches hers: (batch, channels, time, 1).
"""
import torch
from torch import nn


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    return torch.log(torch.clamp(x, min=eps))


class Expression(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class ShallowConv_Embedding(nn.Module):
    """Braindecode ShallowFBCSPNet defaults: n_filters_time=40,
    filter_time_length=25, n_filters_spat=40, pool_time_length=75,
    pool_time_stride=15, drop_prob=0.5, batch_norm_alpha=0.1.

    At 250 Hz: filter_time_length=25 is 100 ms, i.e. one full alpha cycle, and
    pool_time_length=75 is 300 ms, the window over which power is averaged.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples,
        n_filters_time=40,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        batch_norm=True,
        batch_norm_alpha=0.1,
        embedding_dim=512,
        **kwargs,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.embedding_dim = embedding_dim

        # (b, c, t, 1) -> (b, 1, t, c), the layout the reference conv stack expects
        self.dimshuffle = Expression(lambda x: x.permute(0, 3, 2, 1))

        self.conv_time = nn.Conv2d(1, n_filters_time, (filter_time_length, 1),
                                   stride=1)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat,
                                   (1, in_chans), stride=1, bias=not batch_norm)
        self.bnorm = (nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_alpha,
                                     affine=True, eps=1e-5)
                      if batch_norm else nn.Identity())
        self.conv_nonlin = Expression(square)          # THE defining step
        self.pool = nn.AvgPool2d((pool_time_length, 1),
                                 stride=(pool_time_stride, 1))
        self.pool_nonlin = Expression(safe_log)        # ... and its log
        self.drop = nn.Dropout(p=drop_prob)

        out = self.partial_forward(
            torch.ones((1, in_chans, input_window_samples, 1), dtype=torch.float32))
        n_flat = out.flatten(start_dim=1).shape[1]

        self.embedding = nn.Linear(n_flat, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        _glorot_weight_zero_bias(self)

    def partial_forward(self, x):
        x = self.dimshuffle(x)          # (b, 1, t, c)
        x = self.conv_time(x)           # (b, F_t, t', 1)
        x = self.conv_spat(x)           # (b, F_s, t', 1)
        x = self.bnorm(x)
        x = self.conv_nonlin(x)         # square
        x = self.pool(x)                # average -> power
        x = self.pool_nonlin(x)         # log
        x = self.drop(x)
        return x

    def forward(self, x, return_embedding: bool = False):
        x = self.partial_forward(x).flatten(start_dim=1)
        x = self.embedding(x)
        if return_embedding:
            return x
        return self.softmax(self.classifier(x))


def _glorot_weight_zero_bias(model):
    """Same initialisation rule as her EEGNet_Embedding_version.py."""
    for module in model.modules():
        if hasattr(module, "weight") and module.weight is not None:
            if "BatchNorm" not in module.__class__.__name__:
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)
