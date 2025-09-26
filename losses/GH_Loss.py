from torch import nn
    

class MSEGroupLoss(nn.Module):
    def __init__(self, channel_groups=(tuple(range(0,25)), tuple(range(25, 50)),tuple(range(50,75)), tuple(range(75, 100)), tuple(range(100, 125)), tuple(range(125, 150)), tuple(range(150, 175)),tuple(range(175, 202))),):
        super(MSEGroupLoss, self).__init__()
        self.channel_groups = channel_groups

    def forward(self, target, pred, mask):
            """
            imgs: [N, c, H, W]
            pred: [N, L, c*p*p]
            mask: [N, L], 0 is keep, 1 is remove,
            """

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, C, L], mean loss per patch

            total_loss, num_removed = 0., 0.
            for i, group in enumerate(self.channel_groups):
                group_loss = loss[:, group, :].mean(dim=1)  # (N, L)
                total_loss += (group_loss * mask[:, i]).sum()
                num_removed += mask[:, i].sum()  # mean loss on removed patches

            return total_loss/num_removed