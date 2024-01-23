from lightning import Callback, LightningModule, Trainer


class Stop_After_Epoch(Callback):
    def __init__(self, stop_after_epoch: int):
        self.stop_after_epoch = stop_after_epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch == self.stop_after_epoch:
            trainer.should_stop = True
            self.stop_after_epoch = -1
            print("Stopping after epoch ", self.stop_after_epoch)

    #def 
        