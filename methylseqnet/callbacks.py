from lightning.pytorch.callbacks import Callback

class ForceEpochStartCallback(Callback):
    """
    The purpose of this callback is to assert a start epoch when the start checkpoint provides only
    the weights and not the epoch. This is relevant in cases where the start checkpoint config differs
    from the training config, such that nn.Module.load_state_dict must be used with strict=False, rather
    than using LightningModule.load_from_checkpoint.
    """
    def __init__(self, start_epoch):
        self.start_epoch = start_epoch

    def on_fit_start(self, trainer, pl_module):
        trainer.fit_loop.epoch_progress.current.completed = self.start_epoch