import torch
import torch.nn as nn

class CustomPoissonNLLLossLogTransformed(nn.Module):
    def __init__(self):
        super(CustomPoissonNLLLossLogTransformed,self).__init__()
        self.poisson = nn.PoissonNLLLoss(log_input=False)
    def forward(self, predicted_log_counts, target_log_counts):
        # Reverse the log(counts + 1) transform
        predicted_counts = torch.clamp(
            torch.pow(10, predicted_log_counts) - 1,
            min=1e-3,
        )
        target_counts = torch.clamp(
            (torch.pow(10, target_log_counts) - 1)/2,
            min=1e-3,
            max=10
        )

        return self.poisson(predicted_counts,target_counts)

        # print(
        #     'target min',
        #     float(torch.min(target_counts)),
        #     'target max',
        #     float(torch.max(target_counts)),
        #     'loss',
        #     float(loss),
        # )
        # weights = target_log_counts + 0.01
        # squared_diff = (predicted_log_counts - target_log_counts) ** 2
        # weighted_squared_diff = weights * squared_diff
        # return torch.mean(weighted_squared_diff)