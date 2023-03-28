import torch
import torch.nn as nn

class EdgeDetectionEntropyLoss(nn.Module):
    def __init__(self, true_percentage):
        super(EdgeDetectionEntropyLoss, self).__init__()
        self.true_percentage = true_percentage

    def forward(self, output, target):
        output = torch.clamp(output, min=1e-7,max=1-1e-7)
        loss = (1 - self.true_percentage) * target * torch.log(output)\
        + self.true_percentage * (1 - target) * torch.log(1 - output)
        return - loss.mean()
    
def BCELoss_ClassWeights(input, target, class_weights):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = - target * torch.log(input) - (1 - target) * torch.log(1 - input)
    weighted_bce = (bce * class_weights).sum(axis=1) / class_weights.sum(axis=1)[0]
    final_reduced_over_batch = weighted_bce.mean(axis=0)
    return final_reduced_over_batch