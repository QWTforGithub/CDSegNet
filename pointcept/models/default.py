import torch.nn as nn
import torch
import numpy as np
import math

from pointcept.utils.comm import calc_t_emb
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,

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

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(cfg=criteria,loss_type=loss_type,task_num=task_num)

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
            # t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            t = torch.linspace(start, stop, steps, dtype=torch.float64) / T
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

    def add_random_noise(self, pts, sigma=0.1, clamp=0.03):
        # input: (b, 3, n)

        assert (clamp > 0)
        #         jittered_data = torch.clamp(sigma * torch.rand_like(pts), -1 * clamp, clamp).cuda()
        jittered_data = sigma * torch.rand_like(pts).cuda()
        jittered_data = jittered_data + pts

        return jittered_data


    def add_laplace_noise(self, pts, sigma=0.1, clamp=0.03, loc=0.0, scale=1.0):
        # input: (b, 3, n)

        assert (clamp > 0)
        laplace_distribution = torch.distributions.Laplace(loc=loc, scale=scale)
        jittered_data = sigma * laplace_distribution.sample(pts.shape).cuda()
        # jittered_data = torch.clamp(sigma * laplace_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def add_possion_noise(self, pts, sigma=0.1, clamp=0.03, rate=3.0):
        # input: (b, 3, n)

        assert (clamp > 0)
        poisson_distribution = torch.distributions.Poisson(rate)
        jittered_data = sigma * poisson_distribution.sample(pts.shape).cuda()
        # jittered_data = torch.clamp(sigma * poisson_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def init_feature(self, input_dict):
        point = {}
        point["coord"] = input_dict["coord"]
        point["grid_coord"] = input_dict["grid_coord"]
        point["offset"] = input_dict["offset"]
        return point

    def inference_ddim(self, input_dict, T=1000, step=1, report=10, eval=True, mode="avg", noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)
            #input_dict["feat"] = self.add_random_noise(input_dict["feat"],sigma=noise_level)
            #input_dict["feat"] = self.add_laplace_noise(input_dict["feat"],sigma=noise_level)

        if(self.condition):
            ### ---- PT V3 + DM ---- ###
            c_point = self.init_feature(input_dict)
            n_point = self.init_feature(input_dict)

            # ---- initial input ---- #
            n_point["feat"] = input_dict["feat"]

            if(self.c_in_channels == n_point["feat"].shape[-1]):
                c_point['feat'] = c_target = input_dict["feat"]
            else:
                c_point['feat'] = c_target = input_dict["coord"]
            c_point['feat'] = torch.normal(0, 1, size=c_target.shape, dtype=torch.float32).cuda()
            # ---- initial input ---- #

            N = len(c_target)
            n_pred = torch.zeros(size=(N, self.num_classes), dtype=torch.float32).cuda()

            time_schedule = self.get_time_schedule(T, step)
            time_is = reversed(range(len(time_schedule)))

            for i, t in zip(time_is, time_schedule):

                if ((i + 1) % report == 0 or t <= 0):
                    print(f"  ---- current : [{i + 1 if t > 0 else 0}/{step}] steps ----")

                # ---- T steps ---- #
                ts = t * torch.ones((N, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    c_point['t_emb'] = calc_t_emb(ts, t_emb_dim=self.T_dim).cuda()
                # ---- T steps ---- #

                # ---- c_xt ---- #
                c_xt = c_point["feat"]
                # ---- c_xt ---- #

                # ---- pred c_epsilon and n_x0 ---- #
                c_point, n_point = self.backbone(c_point, n_point)
                # ---- pred c_epsilon and n_x0 ---- #

                # ---- c_xs ---- #
                c_epslon_ = c_point["feat"]
                c_xs = self.continuous_p_ddim_sample(
                    c_xt,
                    ts,
                    c_epslon_,
                ).float()
                c_point = self.init_feature(input_dict)
                c_point["feat"] = c_xs
                # ---- c_xs ---- #

                # ---- n_pred ---- #
                if(mode == "avg"):
                    n_pred += n_point["feat"]
                elif(mode == "final"):
                    n_pred = n_point["feat"]
                # ---- n_pred ---- #

                # ---- n_feature ---- #
                n_point = self.init_feature(input_dict)
                n_point["feat"] = input_dict["feat"]
                # ---- n_feature ---- #

                if (t <= 0):
                    break

            if(mode == "avg"):
                n_point["feat"] = n_pred / len(time_schedule)
            elif(mode == "final"):
                n_point["feat"] = n_pred
            ### ---- PT V3 + DM ---- ###
        else:
            ### ---- PT V3 ---- ###
            n_point = self.backbone(n_point=input_dict)
            ### ---- PT V3 ---- ###

        if(eval):
            point = {}
            point['n_pred'] = n_point["feat"]
            point['n_target'] = input_dict['segment']
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=n_point["feat"])
        else:
            return dict(seg_logits=n_point["feat"])

    def inference(self, input_dict, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)
            #input_dict["feat"] = self.add_random_noise(input_dict["feat"],sigma=noise_level)
            #input_dict["feat"] = self.add_laplace_noise(input_dict["feat"],sigma=noise_level)

        if(self.condition):
            ### ---- PT V3 + DM ---- ###
            c_point = self.init_feature(input_dict)
            n_point = self.init_feature(input_dict)

            # ---- initial input ---- #
            n_point["feat"] = input_dict["feat"]

            if(self.c_in_channels == n_point["feat"].shape[-1]):
                c_point['feat'] = c_target = input_dict["feat"]
            else:
                c_point['feat'] = c_target = input_dict["coord"]

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
            n_point = self.backbone(n_point=input_dict)
            ### ---- PT V3 ---- ###

        if(eval):
            point = {}
            point['n_pred'] = n_point["feat"]
            point['n_target'] = input_dict['segment']
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=n_point["feat"])
        else:
            return dict(seg_logits=n_point["feat"])

    def forward(self, input_dict):

        point = {}

        if(self.condition):
            ### ---- PT V3 + DM ---- ###
            c_point = self.init_feature(input_dict)
            n_point = self.init_feature(input_dict)

            c_point = Point(c_point)
            n_point = Point(n_point)

            batch = n_point["batch"]
            B = len(torch.unique(batch))

            # ---- initial input ---- #
            n_point["feat"] = input_dict["feat"]
            if(self.c_in_channels == n_point["feat"].shape[-1]):
                c_point['feat'] = c_target = input_dict["feat"]
            else:
                c_point['feat'] = c_target = input_dict["coord"]
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
            ### ---- PT V3 + DM ---- ###
        else:
            ### ---- PT V3 ---- ###
            n_point = Point(input_dict)
            n_point = self.backbone(n_point=n_point)
            ### ---- PT V3 ---- ###

        point['n_pred'] = n_point['feat']
        point['n_target'] = input_dict['segment']
        point['loss_mode'] = "train"

        loss = self.criteria(point)
        return dict(loss=loss)

# @MODELS.register_module()
# class DefaultSegmentorV2(nn.Module):
#     def __init__(
#         self,
#         num_classes,
#         backbone_out_channels,
#         backbone=None,
#         criteria=None,
#     ):
#         super().__init__()
#         self.seg_head = (
#             nn.Linear(backbone_out_channels, num_classes)
#             if num_classes > 0
#             else nn.Identity()
#         )
#         self.backbone = build_model(backbone)
#         self.criteria = build_criteria(criteria)
#
#     def forward(self, input_dict):
#         point = Point(input_dict)
#         point = self.backbone(point)
#         seg_logits = self.seg_head(point.feat)
#         # train
#         if self.training:
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return dict(loss=loss)
#         # eval
#         elif "segment" in input_dict.keys():
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return dict(loss=loss, seg_logits=seg_logits)
#         # test
#         else:
#             return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
