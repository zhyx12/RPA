# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch


def clip_gradient(model, clip_norm, logger=None):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = torch.tensor(0.0).to('cuda:0')
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.detach().norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm + 1e-10).item()

    norm = (clip_norm / max(totalnorm, clip_norm))

    for p_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param.grad.mul_(norm)



if __name__ == "__main__":
    pass
