import gin
import torch
from torch import nn

@gin.configurable
@gin.register
def basenji2_pytorch(
    pretrained_seq_model_weights,
    remove_crop=False,
    reinitialize=False,
    rewrite_conv_dna_channels_to=None,
    ):
    if rewrite_conv_dna_channels_to is not None and not reinitialize:
        raise ValueError("rewrite_conv_dna_channels_to can only be used when reinitialize is True, as it requires modifying the architecture and reinitializing the weights.")
    import json
    import torch
    from basenji2_pytorch import Basenji2, basenji2_params, basenji2_weights
    model_params = basenji2_params['model']
    model_params.pop("head_human", None)

    basenji2 = Basenji2(model_params)
    if reinitialize:
        # Code copied from https://github.com/d-laub/basenji2-pytorch/blob/d581fb6d607217c32e6f27b6ff3ebb46fde62268/basenji2_pytorch/model.py#L199
        # The getattr call was modified to explicitly handle non-existed, rather than implicitly relying on default=False
        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity="relu"
                )  # matches Keras, gain of sqrt(2) regardless of activation function
                if getattr(m, "bias", None) is not None:
                    m.bias.fill_(0)
        if rewrite_conv_dna_channels_to is not None:
            old = basenji2.model.trunk[0].block[1]
            basenji2.model.trunk[0].block[1] = nn.Conv1d(
                in_channels=rewrite_conv_dna_channels_to,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                padding="same",
                bias=False,  # bias=False because it's always followed by BatchNorm
            )
        basenji2.apply(init_weights)
    else:
        if pretrained_seq_model_weights is None:
            basenji2.load_state_dict(torch.load(basenji2_weights()), strict=False)
        else:
            basenji2.load_state_dict(torch.load(pretrained_seq_model_weights), strict=False)
    if remove_crop:
        for name, module in basenji2.named_modules():
            if 'Cropping1d' in module.__class__.__name__:
                # Split the name to navigate to parent
                parts = name.split('.')
                parent = basenji2
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                # Replace the module
                setattr(parent, parts[-1], nn.Identity())

    return basenji2

@gin.configurable
@gin.register
def borzoi_pytorch(
    pretrained_seq_model_weights,
    remove_crop=False,
    reinitialize=False,
    rewrite_conv_dna_channels_to=None,
    ):
    if rewrite_conv_dna_channels_to is not None and not reinitialize:
        raise ValueError("rewrite_conv_dna_channels_to can only be used when reinitialize is True, as it requires modifying the architecture and reinitializing the weights.")
    from borzoi_pytorch import Borzoi
    class BorzoiEmbedder(Borzoi):
        def __init__(self, pretrained_model, remove_crop=False, reinitialize=False):
            super().__init__(pretrained_model.config)
            if reinitialize:
                if rewrite_conv_dna_channels_to is not None:
                    self.conv_dna.conv_layer = nn.Conv1d(
                        in_channels=rewrite_conv_dna_channels_to,
                        out_channels=self.conv_dna.conv_layer.out_channels,
                        kernel_size=self.conv_dna.conv_layer.kernel_size,
                        stride=self.conv_dna.conv_layer.stride,
                        padding="same",
                        bias=self.conv_dna.conv_layer.bias is not None
                    )
                self.apply(self._init_weights)
            else:
                self.load_state_dict(pretrained_model.state_dict(), strict=False)
            if remove_crop:
                self.crop = nn.Identity()
        def forward(self, x, is_human=True, data_parallel_training=False):
            # Run through all layers until final embeddings
            x = self.get_embs_after_crop(x)
            x = self.final_joined_convs(x)
            return x
    borzoi = Borzoi.from_pretrained(pretrained_seq_model_weights)
    borzoi_embedder = BorzoiEmbedder(borzoi, remove_crop=remove_crop, reinitialize=reinitialize)

    return borzoi_embedder