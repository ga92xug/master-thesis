from lightning import Callback, LightningModule, Trainer

from notebooks.one_time_tasks.batch_norm import print_gpu_memory_usage


class Metric_Print(Callback):
    def __init__(self,
        mode: str = "metric",
    ):
        assert mode in ["metric", "memory"]
        self.mode = mode

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("Train")
        
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("Validation")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.mode == "memory":
            print_gpu_memory_usage()
        else:
            if outputs is not None:
                for k, v in outputs.items():
                    print(f"{k}: {v.item()}")            

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.mode == "memory":
            print_gpu_memory_usage()
        else:
            if outputs is not None:
                for k, v in outputs.items():
                    print(f"{k}: {v.item()}")  