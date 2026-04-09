import torch
import torch.nn.functional as F


def _to_complex(z):
    if z.size(-1) % 2 != 0:
        raise ValueError(f"Complex representation dimension must be even, got {z.size(-1)}.")
    real, imag = torch.chunk(z, 2, dim=-1)
    return torch.complex(real, imag)


def _normalize_complex(z_complex, eps=1e-8):
    norm = torch.sqrt((z_complex.real.square() + z_complex.imag.square()).sum(dim=-1, keepdim=True) + eps)
    return z_complex / norm


def _complex_similarity(z):
    z_complex = _normalize_complex(_to_complex(z))
    sim = torch.matmul(z_complex, z_complex.conj().transpose(-1, -2))
    return sim.real


def complex_avg_pool1d(z, kernel_size, stride=None, padding=0):
    stride = kernel_size if stride is None else stride
    real, imag = torch.chunk(z, 2, dim=-1)
    real = F.avg_pool1d(real.transpose(1, 2), kernel_size=kernel_size, stride=stride, padding=padding).transpose(1, 2)
    imag = F.avg_pool1d(imag.transpose(1, 2), kernel_size=kernel_size, stride=stride, padding=padding).transpose(1, 2)
    return torch.cat([real, imag], dim=-1)


def hierarchical_contrastive_loss_complex_geom(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss_complex_geom(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss_complex_geom(z1, z2)
        d += 1
        z1 = complex_avg_pool1d(z1, kernel_size=2)
        z2 = complex_avg_pool1d(z2, kernel_size=2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss_complex_geom(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss_complex_geom(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)
    z = z.transpose(0, 1)
    sim = _complex_similarity(z)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss_complex_geom(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)
    sim = _complex_similarity(z)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

