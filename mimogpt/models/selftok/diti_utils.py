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

import torch

class DiTi:
    def __init__(self, n_timesteps, K, stages, k_per_stage):
        if k_per_stage:
            k_per_stage = k_per_stage.split(",")
            k_per_stage = [int(k) for k in k_per_stage]
        else:
            k_per_stage = None

        if stages:
            stages = stages.split(",")
            stages = [int(k) for k in stages]
        else:
            stages = None
        self.stages = stages
        self.k_per_stage = k_per_stage

        self.t_to_idx = torch.zeros(n_timesteps).long()
        self.idx_to_max_t = torch.zeros(K).long()
        self.K = K
        if k_per_stage:
            assert stages is not None
            current_stage = 0
            sum_indices = 0
            for t in range(n_timesteps):
                if t == self.stages[current_stage]:
                    sum_indices += self.k_per_stage[current_stage]
                    current_stage += 1
                current_steps = float(self.stages[current_stage])
                current_steps = current_steps - self.stages[current_stage - 1] if current_stage > 0 else current_steps
                current_k = float(self.k_per_stage[current_stage])
                t_adj = t - self.stages[current_stage - 1] if current_stage > 0 else t
                idx = int(float(t_adj) / current_steps * current_k + sum_indices)
                self.t_to_idx[t] = idx
                self.idx_to_max_t[idx] = t
        else:
            for t in range(n_timesteps):
                idx = int(float(t) / (float(n_timesteps) / K))
                self.t_to_idx[t] = idx
                self.idx_to_max_t[idx] = t

    def get_key_timesteps(self):
        return [0] + (self.idx_to_max_t).tolist()

    def get_timestep_range(self, k):
        key_timesteps = self.get_key_timesteps()
        return key_timesteps[k], key_timesteps[k + 1]
    
    def get_position(self, k):
        return 1000 + (k * 8)
    
    def to_indices(self, t):
        device = t.device
        t = torch.floor(t).int().clamp(0, 999)
        return self.t_to_idx.to(device)[t].clamp(0, self.K - 1)

class Segment():
    def __init__(self, low, slope, base):
        self.low = low
        self.slope = slope
        self.base = base
    
    def process(self, x, y):
        xp = x - self.low
        y[xp >= 0] = (self.slope * xp).to(y.dtype)[xp >= 0] + self.base
        return y

class DiTi_cont():
    def __init__(self, n_timesteps, K, stages, k_per_stage):
        self.K = K
        assert k_per_stage
        k_per_stage = k_per_stage.split(",")
        self.k_per_stage = [int(k) for k in k_per_stage]
        assert stages
        stages = stages.split(",")
        self.stages = [int(k) for k in stages]
        n_stages = len(self.stages)
        self.stages = [0] + self.stages
        self.segments = []
        acc = 0
        for i in range(n_stages):
            self.segments.append(Segment(
                self.stages[i], float(self.k_per_stage[i]) / (self.stages[i+1]-self.stages[i]), acc
            ))
            acc += self.k_per_stage[i]

    def to_indices(self, t):
        ind = torch.zeros_like(t)
        for segment in self.segments:
            ind = segment.process(t, ind)
        return ind.to(torch.long).clamp(0, self.K - 1)
    
    def get_position(self, k):
        return 1000 + (k * 8)

class DiTi_normal():
    def __init__(self, n_timesteps, K, m=0.0, s=1.0):
        self.K = K
        self.m = m
        self.s = s
        
    def get_cdf(self, t):
        proj = torch.distributions.normal.Normal(self.m, self.s)
        t = torch.log(t / (1 - t))
        t = proj.cdf(t)
        return t

    def to_indices(self, t):
        ind = self.get_cdf(t)
        ind = torch.ceil(ind*self.K)
        return ind.to(torch.long).clamp(0, self.K - 1)
    
    def get_position(self, k):
        return 1000 + (k * 8)

if __name__ == "__main__":
    diti = DiTi(1000, 16, "100,600,1000", "2,10,4")
    print(diti.get_key_timesteps())
    print(diti.get_timestep_range(1))

    diti = DiTi(1000, 16, "", "")
    print(diti.get_key_timesteps())
    print(diti.get_timestep_range(1))
