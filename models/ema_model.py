from torch.optim.swa_utils import AveragedModel, SWALR
import torch
import torch.nn as nn

class ExponentialMovingAverage(AveragedModel):

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg) 