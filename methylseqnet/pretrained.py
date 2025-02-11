import gin
@gin.configurable
@gin.register
def basenji2_pytorch():
    import json
    import torch
    from basenji2_pytorch import Basenji2, params
    from tqdm.auto import tqdm
    with open(params) as params_open:
        model_params = json.load(params_open)['model']
        model_params.pop("head_human", None)

    return Basenji2(model_params)