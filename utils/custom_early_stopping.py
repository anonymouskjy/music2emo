# custom_early_stopping.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class MultiMetricEarlyStopping(EarlyStopping):
    def __init__(self, monitor_mood, monitor_va, patience, min_delta, mode="min"):
        super().__init__(monitor=None, patience=patience, min_delta=min_delta, mode=mode)
        self.monitor_mood = monitor_mood
        self.monitor_va = monitor_va
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # Initialize tracking variables
        self.wait_mood = 0
        self.wait_va = 0
        self.best_mood = float('inf') if mode == "min" else -float('inf')
        self.best_va = float('inf') if mode == "min" else -float('inf')

    def _check_stop(self, current, best, wait):
        if self.mode == "min" and current < best - self.min_delta:
            return current, 0
        elif self.mode == "max" and current > best + self.min_delta:
            return current, 0
        else:
            return best, wait + 1

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics

        if self.monitor_mood not in logs or self.monitor_va not in logs:
            raise RuntimeError(f"Metrics {self.monitor_mood} or {self.monitor_va} not available.")

        # Get current values for the monitored metrics
        current_mood = logs[self.monitor_mood].item()
        current_va = logs[self.monitor_va].item()

        # Check stopping conditions for both metrics
        self.best_mood, self.wait_mood = self._check_stop(current_mood, self.best_mood, self.wait_mood)
        self.best_va, self.wait_va = self._check_stop(current_va, self.best_va, self.wait_va)

        # Stop if patience exceeded for both metrics
        if self.wait_mood > self.patience and self.wait_va > self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True

# # custom_early_stopping.py

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# class MultiMetricEarlyStopping(EarlyStopping):
#     def __init__(self, monitor_mood: str, monitor_va: str, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
#         super().__init__(monitor=None, patience=patience, min_delta=min_delta, mode=mode)
#         self.monitor_mood = monitor_mood
#         self.monitor_va = monitor_va
#         self.wait_mood = 0
#         self.wait_va = 0
#         self.best_mood_score = None
#         self.best_va_score = None
#         self.patience = patience
#         self.stopped_epoch = 0

#     def on_validation_end(self, trainer, pl_module):
#         current_mood = trainer.callback_metrics.get(self.monitor_mood)
#         current_va = trainer.callback_metrics.get(self.monitor_va)

#         # Check if current_mood improved
#         if self.best_mood_score is None or self._compare(current_mood, self.best_mood_score):
#             self.best_mood_score = current_mood
#             self.wait_mood = 0
#         else:
#             self.wait_mood += 1

#         # Check if current_va improved
#         if self.best_va_score is None or self._compare(current_va, self.best_va_score):
#             self.best_va_score = current_va
#             self.wait_va = 0
#         else:
#             self.wait_va += 1

#         # If both metrics are stagnant for patience epochs, stop training
#         if self.wait_mood >= self.patience and self.wait_va >= self.patience:
#             self.stopped_epoch = trainer.current_epoch
#             trainer.should_stop = True

#     def _compare(self, current, best):
#         if self.mode == "min":
#             return current < best - self.min_delta
#         else:
#             return current > best + self.min_delta