"""
Point Prompt Training

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import numpy as np
import math

import torch
import torch.nn as nn
from pointcept.utils.comm import calc_t_emb
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria



# @MODELS.register_module("PPT-v1m1")
# class PointPromptTraining(nn.Module):
#     """
#     PointPromptTraining provides Data-driven Context and enables multi-dataset training with
#     Language-driven Categorical Alignment. PDNorm is supported by SpUNet-v1m3 to adapt the
#     backbone to a specific dataset with a given dataset condition and context.
#     """
#
#     def __init__(
#         self,
#         backbone=None,
#         criteria=None,
#         backbone_out_channels=96,
#         context_channels=256,
#         conditions=("Structured3D", "ScanNet", "S3DIS"),
#         template="[x]",
#         clip_model="ViT-B/16",
#         # fmt: off
#         class_name=(
#             "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
#             "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
#             "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
#             "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
#             "clutter", "otherstructure", "otherfurniture", "otherprop",
#         ),
#         valid_index=(
#             (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
#             (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
#             (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
#         ),
#         # fmt: on
#         backbone_mode=False,
#     ):
#         super().__init__()
#         assert len(conditions) == len(valid_index)
#         assert backbone.type in ["SpUNet-v1m3", "PT-v2m3", "PT-v3m1"]
#         self.backbone = MODELS.build(backbone)
#         self.criteria = build_criteria(criteria)
#         self.conditions = conditions
#         self.valid_index = valid_index
#         self.embedding_table = nn.Embedding(len(conditions), context_channels)
#         self.backbone_mode = backbone_mode
#         if not self.backbone_mode:
#             import clip
#
#             clip_model, _ = clip.load(
#                 clip_model, device="cpu", download_root="./.cache/clip"
#             )
#             clip_model.requires_grad_(False)
#             class_prompt = [template.replace("[x]", name) for name in class_name]
#             class_token = clip.tokenize(class_prompt)
#             class_embedding = clip_model.encode_text(class_token)
#             class_embedding = class_embedding / class_embedding.norm(
#                 dim=-1, keepdim=True
#             )
#             self.register_buffer("class_embedding", class_embedding)
#             self.proj_head = nn.Linear(
#                 backbone_out_channels, clip_model.text_projection.shape[1]
#             )
#             self.logit_scale = clip_model.logit_scale
#
#     def forward(self, data_dict):
#         condition = data_dict["condition"][0]
#         assert condition in self.conditions
#         context = self.embedding_table(
#             torch.tensor(
#                 [self.conditions.index(condition)], device=data_dict["coord"].device
#             )
#         )
#         data_dict["context"] = context
#         point = self.backbone(data_dict)
#         # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
#         # TODO: remove this part after make all backbone return Point only.
#         if isinstance(point, Point):
#             feat = point.feat
#         else:
#             feat = point
#         if self.backbone_mode:
#             # PPT serve as a multi-dataset backbone when enable backbone mode
#             return feat
#         feat = self.proj_head(feat)
#         feat = feat / feat.norm(dim=-1, keepdim=True)
#         sim = (
#             feat
#             @ self.class_embedding[
#                 self.valid_index[self.conditions.index(condition)], :
#             ].t()
#         )
#         logit_scale = self.log-=it_scale.exp()
#         seg_logits = logit_scale * sim
#         # train
#         if self.training:
#             loss = self.criteria(seg_logits, data_dict["segment"])
#             return dict(loss=loss)
#         # eval
#         elif "segment" in data_dict.keys():
#             loss = self.criteria(seg_logits, data_dict["segment"])
#             return dict(loss=loss, seg_logits=seg_logits)
#         # test
#         else:
#             return dict(seg_logits=seg_logits)

@MODELS.register_module("PPT-v1m1")
class PointPromptTraining(nn.Module):
    """
    PointPromptTraining provides Data-driven Context and enables multi-dataset training with
    Language-driven Categorical Alignment. PDNorm is supported by SpUNet-v1m3 to adapt the
    backbone to a specific dataset with a given dataset condition and context.
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        context_channels=256,
        conditions=("Structured3D", "ScanNet", "S3DIS"),
        template="[x]",
        clip_model="ViT-B/16",
        # fmt: off
        class_name=(
            "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
            "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
            "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
            "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
            "clutter", "otherstructure", "otherfurniture", "otherprop",
        ),
        valid_index=(
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
            (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
        ),
        # fmt: on
        backbone_mode=False,

        loss_type="EW",
        task_num=2,

        num_classes=20,
        T=1000,
        beta_start=0.0001,
        beta_end=0.02,
        noise_schedule="linear",
        T_dim=128,
        dm=False,
        dm_input="xt",
        dm_target="noise",
        dm_min_snr=None,
        condition=False,
        c_in_channels=6

    ):
        super().__init__()
        assert len(conditions) == len(valid_index)
        assert backbone.type in ["SpUNet-v1m3", "PT-v2m3", "PT-v3m1"]
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(cfg=criteria,loss_type=loss_type,task_num=task_num)
        self.conditions = conditions
        self.valid_index = valid_index
        self.embedding_table = nn.Embedding(len(conditions), context_channels)
        self.backbone_mode = backbone_mode
        if not self.backbone_mode:
            import clip

            clip_model, _ = clip.load(
                clip_model, device="cpu", download_root="./.cache/clip"
            )
            clip_model.requires_grad_(False)
            class_prompt = [template.replace("[x]", name) for name in class_name]
            class_token = clip.tokenize(class_prompt)
            class_embedding = clip_model.encode_text(class_token)
            class_embedding = class_embedding / class_embedding.norm(
                dim=-1, keepdim=True
            )
            self.register_buffer("class_embedding", class_embedding)
            self.logit_scale = clip_model.logit_scale

        self.num_classes = num_classes
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule = noise_schedule
        self.T_dim = T_dim
        self.condition = condition
        self.dm = dm
        self.dm_input = dm_input
        self.dm_target = dm_target
        self.dm_min_snr = dm_min_snr
        self.c_in_channels = c_in_channels

        if(self.dm):
            # ---- diffusion params ----
            self.eps = 1e-6
            self.Beta, self.Alpha ,self.Alpha_bar, self.Sigma, self.SNR= self.get_diffusion_hyperparams(
                noise_schedule=noise_schedule,
                T=self.T,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )
            # ---- diffusion params ----

            self.Beta = self.Beta.float().cuda()
            self.Alpha = self.Alpha.float().cuda()
            self.Alpha_bar = self.Alpha_bar.float().cuda()
            self.Sigma = self.Sigma.float().cuda()
            self.SNR = self.SNR.float().cuda() if dm_min_snr is None else torch.clamp(self.SNR.float().cuda(),max=dm_min_snr)

    def get_diffusion_hyperparams(
            self,
            noise_schedule,
            beta_start,
            beta_end,
            T
    ):
        """
        Compute diffusion process hyperparameters

        Parameters:
        T (int):                    number of diffusion steps
        beta_0 and beta_T (float):  beta schedule start/end value,
                                    where any beta_t in the middle is linearly interpolated

        Returns:
        a dictionary of diffusion hyperparameters including:
            T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
            These cpu tensors are changed to cuda tensors on each individual gpu
        """

        # Beta = torch.linspace(noise_schedule,beta_start, beta_end, T)
        Beta = self.get_diffusion_betas(
            type=noise_schedule,
            start=beta_start,
            stop=beta_end,
            T=T
        )
        # at = 1 - bt
        Alpha = 1 - Beta
        # at_
        Alpha_bar = Alpha + 0
        # 方差
        Beta_tilde = Beta + 0
        for t in range(1, T):
            # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
            Alpha_bar[t] *= Alpha_bar[t - 1]
            # \tilde{\beta}_t = (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t) * \beta_t
            Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
        # 标准差
        Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
        Sigma[0] = 0.0

        '''
            SNR = at ** 2 / sigma ** 2
            at = sqrt(at_), sigma = sqrt(1 - at_)
            q(xt|x0) = sqrt(at_) * x0 + sqrt(1 - at_) * noise
        '''
        SNR = Alpha_bar / (1 - Alpha_bar)

        return Beta, Alpha, Alpha_bar, Sigma, SNR

    def get_diffusion_betas(self, type='linear', start=0.0001, stop=0.02, T=1000):
        """Get betas from the hyperparameters."""
        if type == 'linear':
            # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
            # To be used with Gaussian diffusion models in continuous and discrete
            # state spaces.
            # To be used with transition_mat_type = 'gaussian'
            scale = 1000 / T
            beta_start = scale * start
            beta_end = scale * stop
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

        elif type == 'cosine':
            # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
            # To be used with transition_mat_type = 'uniform'.
            steps = T + 1
            s = 0.008
            t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)


        elif type == 'sigmoid':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
            # To be used with absorbing state models.
            # ensures that the probability of decaying to the absorbing state
            # increases linearly over time, and is 1 for t = T-1 (the final time).
            # To be used with transition_mat_type = 'absorbing'
            start = -3
            end = 3
            tau = 1
            steps = T + 1
            t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            v_start = torch.tensor(start / tau).sigmoid()
            v_end = torch.tensor(end / tau).sigmoid()
            alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)

        elif type == "laplace":
            mu = 0.0
            b = 0.5
            lmb = lambda t: mu - b * torch.sign(0.5 - t) * torch.log(1 - 2 * torch.abs(0.5 - t))

            snr_func = lambda t: torch.exp(lmb(t))
            alpha_func = lambda t: torch.sqrt(snr_func(t) / (1 + snr_func(t)))
            # sigma_func = lambda t: torch.sqrt(1 / (1 + snr_func(t)))

            timesteps = torch.linspace(0, 1, 1002)[1:-1]
            alphas_cumprod = []
            for t in timesteps:
                a = alpha_func(t) ** 2
                alphas_cumprod.append(a)
            alphas_cumprod = torch.cat(alphas_cumprod,dim=0)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise NotImplementedError(type)


    def continuous_p_ddim_sample(self, x_t, t, noise):

        if(self.dm_target == "noise"):
            # x0 = (xt - sqrt(1-at_) * noise) / sqrt(at_)
            c_x0 = (x_t - torch.sqrt(1 - self.Alpha_bar[t]) * noise) / torch.sqrt(self.Alpha_bar[t])
        elif(self.dm_target == "x0"):
            c_x0 = noise
            # noise = (xt - sqrt(1-at_) * x0) / sqrt(1-at_)
            noise = (x_t - torch.sqrt(self.Alpha_bar[t]) * c_x0) / torch.sqrt(1 - self.Alpha_bar[t])

        if(t[0] == 0):
            return c_x0

        # sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_)
        c_xt_1_1 = torch.sqrt(self.Alpha_bar[t-1]) * c_x0

        # sqrt(1 - at-1_) * noise
        c_xt_1_2 = torch.sqrt(1 - self.Alpha_bar[t-1]) * noise

        # xt-1 = sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_) + sqrt(1 - at-1_) * noise
        c_xt_1 = c_xt_1_1 + c_xt_1_2

        return c_xt_1

    def continuous_q_sample(self,x_0, t, noise=None):
        if(noise is None):
            # sampling from Gaussian distribution
            noise = torch.normal(0, 1, size=x_0.shape, dtype=torch.float32).cuda()
        # xt = sqrt(at_) * x0 + sqrt(1-at_) * noise
        x_t = torch.sqrt(self.Alpha_bar[t]) * x_0 + torch.sqrt(1 - self.Alpha_bar[t]) * noise
        return x_t

    def get_time_schedule(self, T=1000, step=5):
        times = np.linspace(-1, T - 1, num = step + 1, dtype=int)[::-1]
        return times

    def add_gaussian_noise(self, pts, sigma=0.1, clamp=0.03):
        # input: (b, 3, n)

        assert (clamp > 0)
        # jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp)
        jittered_data = sigma * torch.randn_like(pts).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def inference(self, input_dict, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)

        condition = input_dict["condition"][0]
        assert condition in self.conditions
        context = self.embedding_table(
            torch.tensor(
                [self.conditions.index(condition)], device=input_dict["coord"].device
            )
        )

        if(self.condition):
            ### ---- PT V3 + DM ---- ###
            c_point = {}
            c_point["coord"] = input_dict["coord"]
            c_point["grid_coord"] = input_dict["grid_coord"]
            c_point["offset"] = input_dict["offset"]
            c_point["condition"] = input_dict["condition"]
            c_point["context"] = context

            n_point = {}
            n_point["coord"] = input_dict["coord"]
            n_point["grid_coord"] = input_dict["grid_coord"]
            n_point["offset"] = input_dict["offset"]
            n_point["condition"] = input_dict["condition"]
            n_point["context"] = context

            # ---- initial input ---- #
            n_point["feat"] = input_dict["feat"]

            if (self.c_in_channels == 3):
                c_point['feat'] = c_target = input_dict["coord"]
            elif (self.c_in_channels == 6):
                c_point['feat'] = c_target = input_dict["feat"]

            t = 0
            if(self.dm and self.dm_input == "xt"):
                c_point['feat'] = torch.normal(0, 1, size=c_target.shape, dtype=torch.float32).cuda()
                t = self.T - 1
            # ---- initial input ---- #

            N = len(c_target)

            # ---- T steps ---- #
            ts = t * torch.ones((N, 1), dtype=torch.int64).cuda()
            if (self.T_dim != -1):
                c_point['t_emb'] = calc_t_emb(ts, t_emb_dim=self.T_dim).cuda()
            # ---- T steps ---- #

            # ---- pred c_epsilon and n_x0 ---- #
            c_point, n_point = self.backbone(c_point, n_point)
            # ---- pred c_epsilon and n_x0 ---- #
            ### ---- PT V3 + DM ---- ###
        else:
            ### ---- PT V3 ---- ###
            input_dict["context"] = context
            n_point = self.backbone(n_point=input_dict)
            ### ---- PT V3 ---- ###

        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(n_point, Point):
            feat = n_point.feat
        else:
            feat = n_point
        if self.backbone_mode:
            # PPT serve as a multi-dataset backbone when enable backbone mode
            return feat
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sim = (
            feat
            @ self.class_embedding[
                self.valid_index[self.conditions.index(condition)], :
            ].t()
        )
        logit_scale = self.logit_scale.exp()
        seg_logits = logit_scale * sim

        if(eval):
            point = {}
            point['n_pred'] = seg_logits
            point['n_target'] = input_dict['segment']
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=seg_logits)
        else:
            return dict(seg_logits=seg_logits)


    def forward(self, input_dict):

        point = {}

        condition = input_dict["condition"][0]
        assert condition in self.conditions
        context = self.embedding_table(
            torch.tensor(
                [self.conditions.index(condition)], device=input_dict["coord"].device
            )
        )

        if (self.condition):

            c_point = {}
            c_point["coord"] = input_dict["coord"]
            c_point["grid_coord"] = input_dict["grid_coord"]
            c_point["offset"] = input_dict["offset"]
            c_point["condition"] = input_dict["condition"]
            c_point["context"] = context

            n_point = {}
            n_point["coord"] = input_dict["coord"]
            n_point["grid_coord"] = input_dict["grid_coord"]
            n_point["offset"] = input_dict["offset"]
            n_point["condition"] = input_dict["condition"]
            n_point["context"] = context

            c_point = Point(c_point)
            n_point = Point(n_point)

            batch = n_point["batch"]
            B = len(torch.unique(batch))

            # ---- initial input ---- #
            n_point["feat"] = input_dict["feat"]
            if(self.c_in_channels == 3):
                c_point['feat'] = c_target = input_dict["coord"]
            elif(self.c_in_channels == 6):
                c_point['feat'] = c_target = input_dict["feat"]
            # ---- initial input ---- #

            # ---- continuous diffusion ---- #
            if(self.dm):

                # --- T_embeding ---- #
                ts = torch.randint(0, self.T, size=(B, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    c_point["t_emb"] = calc_t_emb(ts, self.T_dim)[batch, :]
                ts = ts[batch, :]
                # --- T_embeding ---- #

                # ---- add noise ---- #
                c_x0 = c_target
                c_noise = torch.normal(0, 1, size=c_x0.shape,dtype=torch.float32).cuda()
                c_xt = self.continuous_q_sample(c_x0, ts, c_noise)
                c_point['feat'] = c_xt
                # ---- add noise ---- #

                # ---- diffusion target ---- #
                if(self.dm_target == "noise"):
                    c_target = c_noise
                # ---- diffusion target ---- #

                # ---- SNR Loss Weight ----
                if (self.dm_min_snr is not None):
                    point["snr_loss_weight"] = self.SNR[ts]
                # ---- SNR Loss Weight ----
            # ---- continuous diffusion ---- #

            # ---- output ---- #
            c_point, n_point = self.backbone(c_point, n_point)
            # ---- output ---- #

            point['c_pred'] = c_point["feat"]
            point['c_target'] = c_target
        else:
            ### ---- PT V3 ---- ###
            input_dict["context"] = context
            n_point = self.backbone(n_point=input_dict)
            ### ---- PT V3 ---- ###

        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(n_point, Point):
            feat = n_point.feat
        else:
            feat = n_point
        if self.backbone_mode:
            # PPT serve as a multi-dataset backbone when enable backbone mode
            return feat
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sim = (
            feat
            @ self.class_embedding[
                self.valid_index[self.conditions.index(condition)], :
            ].t()
        )
        logit_scale = self.logit_scale.exp()
        seg_logits = logit_scale * sim

        point['n_pred'] = seg_logits
        point['n_target'] = input_dict['segment']
        point['loss_mode'] = "train"

        loss = self.criteria(point)
        return dict(loss=loss)