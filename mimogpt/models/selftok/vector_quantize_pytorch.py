# This part of code modified from https://github.com/lucidrains/vector-quantize-pytorch.git

# Copyright (C) 2025. Huawei Technologies Co., Ltd.  All rights reserved.

# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================




from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.cuda.amp import autocast
import torch.distributed as dist
from einops import rearrange, repeat, reduce, pack, unpack
import numpy as np
from typing import Callable


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def identity(t):
    return t


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy).clamp(min=0).sqrt()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def calc_entropy(input_tensor):
    assert len(input_tensor.shape) == 2
    p = input_tensor.softmax(dim=-1)
    # H(E(p))
    ap = p.mean(dim=0)
    p_log_p = ap * torch.log(ap)
    entropy_to_max = -p_log_p.sum(dim=-1)
    # E(H(p))
    p_log_p = p * torch.log(p)
    entropy_to_min = -p_log_p.sum(dim=-1)
    entropy_to_min = entropy_to_min.mean()
    return entropy_to_max, entropy_to_min

def calc_codebook_entropy(distances):
    p = distances.softmax(dim=2)
    p = p[0].mean(dim=0)
    p_log_p = p * torch.log(p)
    entropy_to_max = -p_log_p.sum(dim=0).mean()
    return entropy_to_max

def calc_ema_entropy(distances, onehot_distances, ratio_d=0.3):
    p = distances.softmax(dim=-1)
    ap = p[0].mean(dim=0)
    ema_p = onehot_distances * (1-ratio_d) + ap * ratio_d
    p_log_p = ema_p * torch.log(ema_p)
    entropy_to_max = -p_log_p.sum(dim=-1).mean()
    ema_p_group = torch.stack([t.mean(dim=0) for t in ema_p.tensor_split(64, dim=0)],dim=0)
    p_log_p = ema_p_group * torch.log(ema_p_group)
    entropy_to_max2 = -p_log_p.sum(dim=-1).mean()
    return entropy_to_max, entropy_to_max2

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits, temperature=1.0, stochastic=False, straight_through=False, reinmax=False, dim=-1, training=True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (
        reinmax and not straight_through
    ), "reinmax can only be turned on if using straight through gumbel softmax"

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim=dim)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num, p=None):
    t = torch.zeros((5,), device="cuda", dtype=torch.int64)
    num_samples, device = samples.shape[0], samples.device
    if p is not None:
        indices = np.random.choice(len(samples), size=num, p=p.cpu().numpy(), replace=True)
        indices = torch.from_numpy(indices).clamp(0, len(samples - 1))
    else:
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)
    t = torch.zeros((5,), device="cuda", dtype=torch.int64)
    return samples[indices]


def batched_sample_vectors(samples, num, p=None):
    return torch.stack([sample_vectors(sample, num, p) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def all_gather_variably_sized_v2(x, sizes, dim=0):

    device = x.device
    q = x
    ws = distributed.get_world_size()
    local_size = torch.tensor(q.shape[0], device=device)
    # all_sizes = sizes
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size.item() - x.shape[0]
    if size_diff:
        padding = torch.zeros((size_diff, x.shape[-1]), device=device, dtype=q.dtype)
        q = torch.cat((q, padding), dim=0)

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])

    return all_qs

def sample_vectors_distributed(local_samples, num, p=None):
    local_samples = rearrange(local_samples, "1 ... -> ...")

    rank = distributed.get_rank()

    wolrd_size = distributed.get_world_size()
    num_per_rank = num // wolrd_size
    remainder = num % wolrd_size
    if rank < remainder:
        samples_per_rank = num_per_rank + 1
    else:
        samples_per_rank = num_per_rank
    local_samples = sample_vectors(local_samples, samples_per_rank, p)
    all_samples = all_gather_variably_sized_v2(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, "... -> 1 ...")


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples, num_clusters, num_iters=10, use_cosine_sim=False, sample_fn=batched_sample_vectors, all_reduce_fn=noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, "h n d -> h d n")
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


# regularization losses


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum("h i d, h j d -> h i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)


# distance types
class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        if_force_sync = False,
        smart_re_K=0,
        frozen_embed=None,
    ):
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.decay = decay
        self.if_force_sync = if_force_sync
        self.smart_react_K = smart_re_K

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)
        self.dead_code_threshold_updated = False

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('cluster_size_wo_react', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        if self.smart_react_K > 0:
            self.register_buffer(
                'timestep_p_over_c',
                torch.ones(num_codebooks, self.smart_react_K, codebook_size) / codebook_size
            )
            self.register_buffer('tpc_initted', torch.Tensor([False]))

        if frozen_embed is not None:
            self.frozen_embed = frozen_embed
            self.n_frozen = frozen_embed.shape[1]
        else:
            self.n_frozen = 0

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        # frozen embed
        self.reset_frozen_embed()

    def force_sync(self, name):
        if distributed.get_rank() == 0:
            var = getattr(self, name)
        else:
            var = torch.empty_like(getattr(self, name))
        distributed.broadcast(var, src=0)
        setattr(self, name, var)

    def reset_frozen_embed(self,):
        if self.n_frozen > 0:
            self.embed[:, :self.n_frozen, :].data.copy_(self.frozen_embed)
            self.embed_avg[:, :self.n_frozen, :].data.copy_(self.frozen_embed)

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.cluster_size_wo_react.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        self.reset_frozen_embed()

    def compute_timestep_weight(self):
        # timestep_p_over_c
        ap = self.timestep_p_over_c
        perplexity = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=-1))
        weight = 1 / perplexity
        v, _ = weight.max(dim=-1)
        weight = weight / v * 10.0
        weight = weight.softmax(dim=-1)
        return weight
    
    def get_group_perplexity(self, codebook_idx=0):
        ap = self.timestep_p_over_c[codebook_idx]
        group_perplexity = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=-1))
        return group_perplexity
    
    def fix_code(self, change_mask_or_indices, new_codes):
        if len(change_mask_or_indices) == self.codebook_size and \
            (change_mask_or_indices<=1).sum()==len(change_mask_or_indices):
            # is mask
            indices = change_mask_or_indices.nonzero()[:,0]
        else:
            # is indices
            indices = change_mask_or_indices
        if len(indices) == len(new_codes):
            # match
            return indices, new_codes
        # shape does not match
        print(f"Warning: change mask len {len(indices)} does not match codes len {len(new_codes)}...")
        if len(indices) > len(new_codes):
            indices = indices[:len(new_codes)]
        else:
            new_codes = new_codes[:len(indices)]
        return indices, new_codes
    
    def change_code(self, change_mask_or_indices, new_codes, ind=0):
        if change_mask_or_indices is None:
            return
        change_mask_or_indices, new_codes = self.fix_code(change_mask_or_indices, new_codes)
        self.embed.data[ind][change_mask_or_indices] = new_codes
        self.embed_avg.data[ind][change_mask_or_indices] = new_codes * self.reset_cluster_size
        self.cluster_size.data[ind][change_mask_or_indices] = self.reset_cluster_size

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)
        if self.smart_react_K > 0:
            batch_weights = self.compute_timestep_weight()     # n_codebook * K
            b = batch_samples.shape[1] // batch_weights.shape[1]
            batch_weights = batch_weights.unsqueeze(1).expand(-1,b,-1)
            batch_weights = batch_weights / b
            batch_weights = rearrange(batch_weights, "h ... -> h (...)")
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue
            if self.smart_react_K > 0:
                p = batch_weights[ind]
            else:
                p = None
            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item(), p=p)
            sampled = rearrange(sampled, '1 ... -> ...')
            self.change_code(mask, sampled, ind)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return 0

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if self.n_frozen > 0:
            # frozen codes do not need to be reactivated
            non_frozen_codes = (torch.arange(self.codebook_size) >= self.n_frozen)
            non_frozen_codes = non_frozen_codes.unsqueeze(dim=0).expand(self.num_codebooks, -1).cuda()
            expired_codes = torch.logical_and(expired_codes, non_frozen_codes)

        if not torch.any(expired_codes):
            return 0
        
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)
        return torch.sum(expired_codes).item()

    @autocast(enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):      
        num_reactivate = 0
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        # update relative dead code threshold to absolute value based on batch size and world size
        if self.training and not self.dead_code_threshold_updated:
            ratio = x.shape[0] * x.shape[1] * distributed.get_world_size() / self.codebook_size
            self.threshold_ema_dead_code = ratio * self.threshold_ema_dead_code
            self.reset_cluster_size = ratio * self.reset_cluster_size
            self.dead_code_threshold_updated = True
            print(f"Dead code threshold updated to {self.threshold_ema_dead_code}, reset size to {self.reset_cluster_size}.")

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if self.if_force_sync:
            self.force_sync('embed')
        embed = self.embed if self.learnable_codebook else self.embed.detach()

        dist = einsum('h n d, h c d -> h n c', flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        embed_ind = unpack_one(embed_ind, ps, 'h *')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)

            # update timestep_p_over_c
            if self.smart_react_K > 0:
                batch_t_p_over_c = unpacked_onehot.mean(dim=1)
                self.all_reduce_fn(batch_t_p_over_c)
                batch_t_p_over_c /= distributed.get_world_size()
                decay = self.decay if self.tpc_initted else 0.3
                ema_inplace(self.timestep_p_over_c.data, batch_t_p_over_c, decay)
                if not self.tpc_initted:
                    self.tpc_initted.data.copy_(torch.Tensor([True]))
        else:
            quantize = batched_embedding(embed_ind, embed)

        self.delta_embed = torch.tensor(0.0).to(x.device)
        if self.training and self.ema_update and not freeze_codebook:
            if exists(mask):
                embed_onehot[~mask] = 0.

            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size.data, bins, self.decay)
            ema_inplace(self.cluster_size_wo_react.data, bins, self.decay)
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            # compute embed changes on non-frozen codes
            if self.n_frozen > 0:
                embed_normalized[:, :self.n_frozen, :].data.copy_(self.frozen_embed)
            self.delta_embed = F.mse_loss(self.embed.data, embed_normalized, reduction='sum')    # avg update

            # update codebook
            self.embed.data.copy_(l2norm(embed_normalized))

            num_reactivate = self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        # reset frozen embed
        self.reset_frozen_embed()
        
        dist = unpack_one(dist, ps, 'h * d')
        return quantize, embed_ind, dist, num_reactivate

# main class
class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        output_dim=None,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        diversity_weight = 0.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        reinmax = False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        learnable_codebook = False,
        in_place_codebook_optimizer: Callable[..., Optimizer] = None, # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0., # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
        if_force_sync = False,
        smart_re_K=0,
        continuous=False,
        reg=[1/4., 1/2.],
        reset_cluster_size=None,
        ema_entropy_ratio=0.7,
        frozen_embed=None,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        output_dim = output_dim or dim
        requires_out_projection = codebook_input_dim != output_dim
        self.project_out = nn.Linear(codebook_input_dim, output_dim) if requires_out_projection else nn.Identity()

        self.has_projections = requires_projection

        self.eps = eps
        self.reg = reg
        self.ema_entropy_ratio = ema_entropy_ratio
        self.diversity_weight = diversity_weight
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss # whether to use cross entropy loss to codebook as commitment loss
        # calculate reference entropy value
        a1 = codebook_size // codebook_dim
        ref = torch.tensor([0.0]*codebook_size)
        ref[:a1] = 1.0 * 10.0   # scaled positive logits
        ref[a1:] = 0.38 * 10.0   # scaled negative logits
        _, entropy_min_ref = calc_entropy(ref.unsqueeze(0))
        self.entropy_min_ref = entropy_min_ref.item()

        self.learnable_codebook = learnable_codebook
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'
        self.smart_re_K = smart_re_K
        self.sync_update_v = sync_update_v

        codebook_class = CosineSimCodebook

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through
        )
        
        if not exists(sync_codebook):
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            reset_cluster_size=reset_cluster_size,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update,
            if_force_sync = if_force_sync,
            smart_re_K=smart_re_K,
            frozen_embed=frozen_embed,
        )

        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        # continuous tricks
        self.register_buffer('continuous', torch.Tensor([continuous]))
        self.register_buffer('steps', torch.Tensor([0]))
        self.register_buffer('count', torch.zeros(1, self.codebook_size))

        self.frozen_embed = frozen_embed

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            # return rearrange(codes, '... h d -> ... (h d)')
            return codes
        
        indices, ps = pack_one(indices, 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes = unpack_one(codes, ps, 'b * d')
        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def forward(
        self,
        x,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False
    ):

        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        )

        self.steps += 1
        if self.steps > 2000 and self.continuous:
            self.continuous.data.copy_(torch.Tensor([False]))
            print("Starting quantizer mode...")

        # quantize
        if self.continuous:
            quantize = x
            embed_ind = torch.randint(0, self.codebook_size, size=quantize.shape[:2]).to(dtype=torch.int64, device=x.device)
            distances = torch.ones((1, quantize.shape[0], quantize.shape[1], self.codebook_size)).to(device=x.device)
            num_reactivate = 0
        else:
            quantize, embed_ind, distances, num_reactivate = self._codebook(x, **codebook_forward_kwargs)
        
        # one step in-place update

        if should_inplace_optimize and self.training and not freeze_codebook:

            if exists(mask):
                loss = F.mse_loss(quantize, x.detach(), reduction = 'none')

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c = loss.shape[0], h = loss.shape[1] // mask.shape[0])

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # quantize again

            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # determine code to use for commitment loss
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)            

            # straight through

            quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b = shape[0]),
                codes,
                ignore_index = -1
            )

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        if only_one:
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        # aggregate loss

        loss = torch.tensor([0.], device = device, requires_grad = self.training)
        log_dict = {
            "n_reactive": num_reactivate,
            "commit_loss": 0,
            "diversity_entropy": 0,
            "deterministic_entropy": 0,
            "perplexity": 0,
            "delta_embed": 0,
            "cosine_sim": einsum('h n d, h n d -> h n', quantize, x).mean().item()
        }
        if not self.continuous:
            log_dict["delta_embed"] = self._codebook.delta_embed.item()

        unpacked_onehot = F.one_hot(embed_ind.view(-1), num_classes=self.codebook_size).type(x.dtype)
        current_tokens = unpacked_onehot.sum(dim=0)
        self.count = self.count + current_tokens
        avg_probs = self.count.data / self.count.sum()
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp().item()
        
        log_dict['perplexity'] = perplexity

        if self.training and not (self.continuous):
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h = heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        # with variable lengthed sequences
                        commit_loss = F.mse_loss(commit_quantize, x, reduction = 'none')

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        # commit_loss = F.mse_loss(commit_quantize, x)
                        commit_loss = F.mse_loss(commit_quantize, x, reduction='sum') / len(x.flatten())

                loss = loss + commit_loss * self.commitment_weight
                log_dict["commit_loss"] = commit_loss.detach().item()
            

            all_distances = distances
            scaled_distances = all_distances * 10.0
            entropy_to_max, entropy_to_min = calc_entropy(
                scaled_distances.flatten(end_dim=-2), min_ref=self.entropy_min_ref
            )
            # codebook entropy
            if self.smart_re_K:
                # codebook_entropy = calc_codebook_entropy(scaled_distances)
                codebook_entropy, group_entropy = calc_ema_entropy(
                    scaled_distances, self._codebook.timestep_p_over_c[0], ratio_d=1.-self.ema_entropy_ratio
                )
                entropy = 0.5 * (codebook_entropy + group_entropy)
                # diversity_loss = -entropy_to_max
                group_perplexity = self._codebook.get_group_perplexity().mean()

                frac = group_perplexity / self.codebook_size
                reg = self.reg
                # reg = [0.64, 0.685]
                codebook_ent_weight = 0.5 if frac < reg[0] else max((0.5 - 0.5/(reg[1]-reg[0])*(frac-reg[0])), 0.0)
                log_dict['perplexity'] = group_perplexity.item()
                diversity_loss = -self.diversity_weight*codebook_ent_weight*entropy
            else:
                diversity_loss = -self.diversity_weight * entropy_to_max
            log_dict["diversity_entropy"] = codebook_entropy.detach().item()
            log_dict["deterministic_entropy"] = entropy_to_min.detach().item()
            loss = loss + diversity_loss
            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed
                # only calculate orthogonal loss for the activated codes for this batch
                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), 'orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet'
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')

        # if masking, only return quantized for where mask has True

        if exists(mask):
            quantize = torch.where(
                rearrange(mask, '... -> ... 1'),
                quantize,
                orig_input
            )
        return quantize, embed_ind, loss, log_dict