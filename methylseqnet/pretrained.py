import gin
import torch

@gin.configurable
@gin.register
def basenji2_pytorch(pretrained_seq_model_weights):
    import json
    import torch
    from basenji2_pytorch import Basenji2, params
    from tqdm.auto import tqdm
    with open(params) as params_open:
        model_params = json.load(params_open)['model']
        model_params.pop("head_human", None)

    basenji2 = Basenji2(model_params)
    basenji2.load_state_dict(torch.load(pretrained_seq_model_weights), strict=False)

    return basenji2

@gin.configurable
@gin.register
def borzoi_pytorch(pretrained_seq_model_weights):
    from borzoi_pytorch import Borzoi
    class BorzoiEmbedder(Borzoi):
        def __init__(self, pretrained_model):
            super().__init__(pretrained_model.config)
            self.load_state_dict(pretrained_model.state_dict(), strict=False)
        def forward(self, x, is_human=True, data_parallel_training=False):
            # Run through all layers until final embeddings
            x = self.get_embs_after_crop(x)
            x = self.final_joined_convs(x)
            return x
    borzoi = Borzoi.from_pretrained(pretrained_seq_model_weights)
    borzoi_embedder = BorzoiEmbedder(borzoi)

    return borzoi_embedder