from lightning import Callback


class Metric_Print(Callback):
    def __init__(self):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print("Train")
        if outputs is not None:
            for k, v in outputs.items():
                print(f"{k}: {v.item()}")            

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print("Valid")
        if outputs is not None:
            for k, v in outputs.items():
                print(f"{k}: {v.item()}")