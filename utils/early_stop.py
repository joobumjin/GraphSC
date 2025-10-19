####
# This code is based on the Ignite Early Stopping
# See reference code here: https://docs.pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping
####

import torch

class EarlyStopper():
    """
    params:
        patience: Number of events to wait if no improvement and then stop the training.
        min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta: It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.
    """
    def __init__(self, patience: int = 5, direction: str = "maximize", min_delta: float = 0.0, cumulative_delta: bool = False):
        self.patience = patience
        self.maximize = (direction == "maximize")
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta

        self.best_score = None

    def check_stop(self, score: float) -> bool:
        if not self.maximize: score = score * -1 #convert minimization to maximization :)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta: #if worse
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            # print(f"EarlyStopping: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                print("EarlyStopper: Stop training")
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False