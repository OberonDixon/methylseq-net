import json
import torch
from basenji2_pytorch import Basenji2, params
from tqdm.auto import tqdm
from methylseqnet import trainer
model_weights = '/clusterfs/nilah/oberon/repos/basenji2-pytorch/data/basenji2.pth'
with open(params) as params_open:
    model_params = json.load(params_open)['model']
model_params.pop("head_human", None)
# to use a headless model e.g. for transfer learning
# model_params.pop("head_human", None)

basenji2 = Basenji2(model_params)
basenji2.load_state_dict(torch.load(model_weights), strict=False)
trainer.main(
    config = '/clusterfs/nilah/oberon/repos/methylseq-net/configs/residual/test_pretrain.gin',
    output_dir = '/clusterfs/nilah/oberon/lightning/',
    unique_identifier = 'test',
    gpus = 4,
    batch_size = 10,
    pretrained_seq_model=basenji2,
)