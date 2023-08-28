import numpy as np


class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-3, warm_up=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.warm_up = warm_up

    def early_stop(self, validation_loss):
        if self.warm_up > 0:
            self.warm_up -= 1
            return False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
