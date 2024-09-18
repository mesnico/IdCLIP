import time
from pytorch_lightning.callbacks import ProgressBar

class SimpleProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.train_batches = 0
        self.val_batches = 0
        self.test_batches = 0

    def on_train_start(self, trainer, pl_module):
        self.train_batches = len(trainer.train_dataloader) * trainer.max_epochs
        self.start_time = time.time()
        print(f"Training started for {trainer.max_epochs} epochs")

    def on_epoch_start(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_batch = trainer.global_step + 1
        percentage = (current_batch / self.train_batches) * 100
        elapsed_time = time.time() - self.start_time
        print(f"Training: {percentage:.2f}% complete. Elapsed time: {elapsed_time:.2f}s", end='\r')

    def on_train_end(self, trainer, pl_module):
        print("\nTraining complete")

    def on_validation_start(self, trainer, pl_module):
        self.val_batches = len(trainer.val_dataloaders[0])
        print(f"Validation started for epoch {trainer.current_epoch + 1}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        current_batch = batch_idx + 1
        percentage = (current_batch / self.val_batches) * 100
        print(f"Validation: {percentage:.2f}% complete", end='\r')

    def on_validation_end(self, trainer, pl_module):
        print("\nValidation complete")

    def on_test_start(self, trainer, pl_module):
        self.test_batches = len(trainer.test_dataloaders[0])
        print(f"Testing started for epoch {trainer.current_epoch + 1}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        current_batch = batch_idx + 1
        percentage = (current_batch / self.test_batches) * 100
        print(f"Testing: {percentage:.2f}% complete", end='\r')

    def on_test_end(self, trainer, pl_module):
        print("\nTesting complete")