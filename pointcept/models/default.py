import torch.nn as nn
import torch
import numpy as np
import math
from scipy import special

from pointcept.utils.comm import calc_t_emb
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    '''
        GD + CN : Gaussion(Continous) Diffusion + Conditional Network
    '''
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

@MODELS.register_module()
class CCDMSegmentor(nn.Module):
    '''
        GD +GD : Gaussion(Continuous) Diffusion (GD) +  Gaussion(Continuous) Diffusion (GD)
    '''
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
            x0 = (x_t - torch.sqrt(1 - self.Alpha_bar[t]) * noise) / torch.sqrt(self.Alpha_bar[t])
        else:
            x0 = noise
            # noise = (xt - sqrt(1-at_) * x0) / sqrt(1-at_)
            noise = (x_t - torch.sqrt(self.Alpha_bar[t]) * x0) / torch.sqrt(1 - self.Alpha_bar[t])

        if(t[0] == 0):
            return x0

        # sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_)
        xs_1 = torch.sqrt(self.Alpha_bar[t-1]) * x0

        # sqrt(1 - at-1_) * noise
        xs_2 = torch.sqrt(1 - self.Alpha_bar[t-1]) * noise

        # xt-1 = sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_) + sqrt(1 - at-1_) * noise
        xs = xs_1 + xs_2

        return xs

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

    def feature_init(self, input_dict):
        point = {}
        point["coord"] = input_dict["coord"]
        point["grid_coord"] = input_dict["grid_coord"]
        point["offset"] = input_dict["offset"]
        return point


    def inference_ddim(self, input_dict, T=1000, step=1, report=20, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)

        if(self.condition):
            N = len(input_dict["feat"])

            c_point = self.feature_init(input_dict)
            n_point = self.feature_init(input_dict)

            # ---- initial input ---- #
            if (self.c_in_channels == 6):
                c_target = input_dict["feat"]
            else:
                c_target = input_dict["coord"]
            c_point['feat'] = torch.normal(0, 1, size=c_target.shape, dtype=torch.float32).cuda()
            n_point['feat'] = torch.normal(0, 1, size=(N, self.num_classes), dtype=torch.float32).cuda()
            # ---- initial input ---- #

            time_schedule = self.get_time_schedule(T, step)
            time_is = reversed(range(len(time_schedule)))
            for i, t in zip(time_is, time_schedule):

                if ((i + 1) % report == 0 or t <= 0):
                    print(f"  ---- current : [{i + 1 if t > 0 else 0}/{step}] steps ----")

                # ---- T steps ---- #
                t = t if t >= 0 else 0
                ts = t * torch.ones((N, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    c_point['t_emb'] = n_point['t_emb'] = calc_t_emb(ts, t_emb_dim=self.T_dim).cuda()
                # ---- T steps ---- #

                # ---- c_xt ---- #
                c_xt = c_point["feat"]
                # ---- c_xt ---- #

                # ---- n_xt ---- #
                n_xt = n_point["feat"]
                # ---- n_xt ---- #

                # ---- pred c_x0 and n_x0 ---- #
                c_point, n_point = self.backbone(c_point, n_point)
                # ---- pred c_x0 and n_x0 ---- #

                # ---- c_xs ---- #
                c_epslon_ = c_point["feat"]
                c_xs = self.continuous_p_ddim_sample(
                    c_xt,
                    ts,
                    c_epslon_,
                ).float()
                c_point = self.feature_init(input_dict)
                c_point["feat"] = c_xs
                # ---- n_xs ---- #

                # ---- n_xs ---- #
                n_epslon_ = n_point["feat"]
                n_xs = self.continuous_p_ddim_sample(
                    n_xt,
                    ts,
                    n_epslon_,
                ).float()
                n_point = self.feature_init(input_dict)
                n_point["feat"] = n_xs
                # ---- n_xs ---- #

                if (t <= 0):
                    break

        else:
            n_point = self.backbone(n_point=input_dict)

        if(eval):
            n_target = input_dict["segment"]
            if(self.condition and self.dm_target == "noise"):
                n_target = torch.log(torch.nn.functional.one_hot(n_target, self.num_classes) + self.eps)
            if("valid" in input_dict.keys()):
                n_point["feat"] = n_point["feat"][input_dict["valid"]]
                n_target = n_target[input_dict["valid"]]
                input_dict["segment"] = input_dict["segment"][input_dict["valid"]]

            point = {}
            point['n_pred'] = n_point["feat"]
            point['n_target'] = n_target
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=n_point["feat"])
        else:
            return dict(seg_logits=n_point["feat"])


    def forward(self, input_dict):

        point = {}

        n_target = input_dict["segment"]
        if(self.condition):
            point["valid"] = input_dict["valid"]

            ### ---- PT V3 + DM ---- ###
            c_point = {}
            c_point["coord"] = input_dict["coord"]
            c_point["grid_coord"] = input_dict["grid_coord"]
            c_point["offset"] = input_dict["offset"]

            n_point = {}
            n_point["coord"] = input_dict["coord"]
            n_point["grid_coord"] = input_dict["grid_coord"]
            n_point["offset"] = input_dict["offset"]

            c_point = Point(c_point)
            n_point = Point(n_point)

            batch = n_point["batch"]
            B = len(torch.unique(batch))

            # ---- initial input ---- #
            if(self.c_in_channels == 6):
                c_point['feat'] = c_target = input_dict["feat"]
            else:
                c_point['feat'] = c_target = input_dict["coord"]
            # ---- initial input ---- #

            # ---- continuous diffusion ---- #
            if(self.dm):

                # --- T_embeding ---- #
                ts = torch.randint(0, self.T, size=(B, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    n_point["t_emb"] = c_point["t_emb"] = calc_t_emb(ts, self.T_dim)[batch, :]
                ts = ts[batch, :]
                # --- T_embeding ---- #

                # ---- add noise ---- #
                c_x0 = c_target
                c_noise = torch.normal(0, 1, size=c_x0.shape, dtype=torch.float32).cuda()
                c_xt = self.continuous_q_sample(c_x0, ts, c_noise)
                c_point['feat'] = c_xt

                n_x0 = torch.log(torch.nn.functional.one_hot(n_target, self.num_classes) + self.eps)
                n_noise = torch.normal(0, 1, size=n_x0.shape,dtype=torch.float32).cuda()
                n_xt = self.continuous_q_sample(n_x0, ts, n_noise)
                n_point['feat'] = n_xt
                # ---- add noise ---- #

                # ---- diffusion target ---- #
                if(self.dm_target == "noise"):
                    c_target = c_noise
                    n_target = n_noise
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
        point['n_target'] = n_target
        point['loss_mode'] = "train"

        loss = self.criteria(point)
        return dict(loss=loss)

@MODELS.register_module()
class CDDMSegmentor(nn.Module):
    '''
        GD +CD : Gaussion(Continuous) Diffusion (GD) +  Categorical(Discrete) Diffusion (CD)
    '''
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
        c_in_channels=6,
        transfer_type="gaussian"
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
        self.transfer_type = transfer_type

        if(self.dm):
            # 1. 获取噪声时间表
            self.eps = 1e-6
            self.Beta, self.Alpha ,self.Alpha_bar, self.Sigma, self.SNR= self.get_diffusion_hyperparams(
                noise_schedule=noise_schedule,
                T=self.T,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )

            # 2. 获取转移矩阵Qt（每一步的）
            q_onestep_mats = []
            for beta in self.Beta.numpy():
                if self.transfer_type == "uniform":
                    mat = self.get_uniform_transition_mat(beta)
                    mat = torch.from_numpy(mat)
                    q_onestep_mats.append(mat)
                elif self.transfer_type == "gaussian":
                    mat = self.get_gaussian_transition_mat(beta)
                    mat = torch.from_numpy(mat)
                    q_onestep_mats.append(mat)
                else:
                    raise NotImplementedError
            q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

            # 3. 获取旋转矩阵的逆，Qt-1（每一步的），这里表达Qt本身是一个正交矩阵，QtT = Qt-1
            self.q_one_step_transposed = q_one_step_mats.transpose(1, 2).cuda()

            # 4. 获取累计旋转矩阵，Qt_
            q_mat_t = q_onestep_mats[0]
            q_mats = [q_mat_t]
            for idx in range(1, self.T):
                q_mat_t = q_mat_t @ q_onestep_mats[idx]  # 两个正交矩阵相乘结果仍然是正交矩阵
                q_mats.append(q_mat_t)
            self.q_mats = torch.stack(q_mats, dim=0).cuda()
            self.logit_type = "logit"

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
            x0 = (x_t - torch.sqrt(1 - self.Alpha_bar[t]) * noise) / torch.sqrt(self.Alpha_bar[t])
        else:
            x0 = noise
            # noise = (xt - sqrt(1-at_) * x0) / sqrt(1-at_)
            noise = (x_t - torch.sqrt(self.Alpha_bar[t]) * x0) / torch.sqrt(1 - self.Alpha_bar[t])

        if(t[0] == 0):
            return x0

        # sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_)
        xs_1 = torch.sqrt(self.Alpha_bar[t-1]) * x0

        # sqrt(1 - at-1_) * noise
        xs_2 = torch.sqrt(1 - self.Alpha_bar[t-1]) * noise

        # xt-1 = sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_) + sqrt(1 - at-1_) * noise
        xs = xs_1 + xs_2

        return xs

    def continuous_q_sample(self,x_0, t, noise=None):
        if(noise is None):
            # sampling from Gaussian distribution
            noise = torch.normal(0, 1, size=x_0.shape, dtype=torch.float32).cuda()
        # xt = sqrt(at_) * x0 + sqrt(1-at_) * noise
        x_t = torch.sqrt(self.Alpha_bar[t]) * x_0 + torch.sqrt(1 - self.Alpha_bar[t]) * noise
        return x_t

    def get_uniform_transition_mat(self, beta_t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).（xt会收敛于一个1/K的均匀分布）

        This method constructs a transition
        matrix Q with
        Q_{ij} = beta_t / num_pixel_vals       if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il} if i==j.
                 0                          else.
            1-(k-1)/k, i=j
        Qt=
            kbt,       i≠j
        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
        """

        # Assumes num_off_diags < num_pixel_vals
        transition_bands = self.num_classes - 1
        #beta_t = betas[t]

        mat = np.zeros((self.num_classes, self.num_classes), dtype=np.float32) # [[0,0],[0,0]]
        # [bt/k,]
        off_diag = np.full(shape=(transition_bands,), fill_value=beta_t / float(self.num_classes), dtype=np.float32)
        for k in range(1, transition_bands + 1):
            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)

        # mat = torch.ones(num_classes, num_classes) * beta / num_classes # [[bt/k,bt/k],[bt/k,bt/k]]
        # mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes) # [[1-(k-1)/k,bt/k],[bt/k,1-(k-1)/k]]

        return mat

    # 获取 高斯核变换矩阵 Qt
    def get_gaussian_transition_mat(self,  beta_t):
        r"""Computes transition matrix for q(x_t|x_{t-1})，计算x0到xt的转换矩阵（gaussian），然而，这种方式缺少随机过程

        This method constructs a transition matrix Q with
        decaying entries as a function of how far off diagonal the entry is.
        Normalization option 1:
        Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il}  if i==j.
                 0                          else.

        Normalization option 2:
        tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                         0                        else.

        Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (self.num_classes, self.num_classes).
        """
        transition_bands = self.num_classes - 1
        # beta_t, t
        #beta_t = betas[t]
        # [256,256]
        mat = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        # [256,], (0,255)
        values = np.linspace(start=0., stop=transition_bands, num=self.num_classes, endpoint=True, dtype=np.float32)
        values = values * 2. / (self.num_classes - 1.)  # values * 2 / 255 ??? (0,2)
        values = values[:transition_bands + 1]  # [256,]
        values = -values * values / beta_t  # -values * values / beta_t ????

        values = np.concatenate([values[:0:-1], values], axis=0)  # cat([255,],[256]) --> [511,]
        values = special.softmax(values, axis=0)  # softmax，归一化
        values = values[transition_bands:]  # 【511，】->[255,]
        for k in range(1, transition_bands + 1):
            # 用values的地k下标填补[self.self.num_classes - k,]矩阵
            off_diag = np.full(shape=(self.num_classes - k,), fill_value=values[k], dtype=np.float32)

            mat += np.diag(off_diag, k=k)  # 创建一个对角矩阵，以off_diag为元素， 【255,256】
            mat += np.diag(off_diag, k=-k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)  # [256.256]

        return mat

    # def _at(self, a, ts, t, x, offset):
    # xs = []
    # for ti,oi in zip(t,offset):
    #     xi = x[start:oi].transpose(0,1) # [1,N]
    #     ti = ti.reshape((1, *[1] * (x.dim() - 1))) # [1,1]
    #     xx = torch.squeeze(a[ti, xi,:])
    #     xs.append(xx)
    #     start = oi
    # logx1 = torch.cat(xs,dim=0)
    # xx = torch.equal(logx1,logx2)


    def _at(self, a, x, ts):
        # 根据x0作为索引，从Qt_中选取对应元素作为xt
        logx = a[ts,x,:]
        return logx

    def discrete_q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32: # [B,1,32,32,N]
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
            if(x_0_logits.dim() == 2):
                x_0_logits = torch.unsqueeze(x_0_logits,dim=1)

        assert x_0_logits.shape == x_t.shape + (self.num_classes,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # ---- xt-1_1 = xt * QtT ----
        # [T,num_class,num_class] * [N,C,num_class] => [N, C, num_class]
        fact1 = self._at(self.q_one_step_transposed, x_t, t)
        # ---- xt-1_1 = xt * QtT ----

        # ---- xt-1_2 = x0 * Qt-1_ ----
        # [N,C,num_class] * [N, num_class, num_class] => [N, C, num_class]
        fact2 = torch.einsum("ncl,nld->ncd", torch.softmax(x_0_logits, dim=-1), self.q_mats[t - 1].squeeze())
        # ---- xt-1_2 = x0 * Qt-1_ ----

        # ---- xt-1 = log(xt-1_1) + log(xt-1_2) ----
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        # ---- xt-1 = log(xt-1_1) + log(xt-1_2) ----

        # # [B,1,1]
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        # 如果当前索引为1，则从x0直接选取结果，否则从put中选取结果
        bc = torch.where(t_broadcast == 0, x_0_logits, out)

        return bc

    def discrete_q_sample(self, x_0, ts, noise=None):
        if(noise is None):
            # sampling from uniform distribution
            noise = torch.rand((*x_0.shape, self.num_classes), device=x_0.device)
        # q(xt|x0), xt = x0 * Qt_
        xt = self._at(self.q_mats, x_0, ts)
        # discrete sampling, argmax(gumbal_softmax(log(xt))), [N,1,C] -> [N,1]
        logits = torch.log(xt + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_t = torch.argmax(logits + gumbel_noise, dim=-1)
        return x_t

    def discrete_p_sample(self, xt, t, x0, noise=None):

        if(t[0] == 0):
            return x0

        pred_discrete_q_posterior_logits = self.discrete_q_posterior_logits(x0, xt, t)
        if(noise is None):
            noise = torch.rand((*xt.shape, self.num_classes)).cuda()
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((xt.shape[0], *[1] * (xt.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        xt_1 = torch.argmax(
            pred_discrete_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return xt_1

    def discrete_p_ddim_sample(self, t, x_0, noise=None):

        if(t[0] == 0):
            return x_0

        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32: # [B,1,32,32,N]
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
            if(x_0_logits.dim() == 2):
                x_0_logits = torch.unsqueeze(x_0_logits,dim=1)

        # ---- xt-1 = x0 * Qt-1_ ----
        # [N,C,num_class] * [N, num_class, num_class] => [N, C, num_class]
        pred_discrete_q_posterior_logits = torch.einsum("ncl,nld->ncd", torch.softmax(x_0_logits, dim=-1), self.q_mats[t - 1].squeeze())
        # ---- xt-1  = x0 * Qt-1_ ----

        if(noise is None):
            noise = torch.rand(pred_discrete_q_posterior_logits.shape).cuda()
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((x_0.shape[0], *[1] * (x_0.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        xt_1 = torch.argmax(
            pred_discrete_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return xt_1

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

    def feature_init(self, input_dict):
        point = {}
        point["coord"] = input_dict["coord"]
        point["grid_coord"] = input_dict["grid_coord"]
        point["offset"] = input_dict["offset"]
        return point


    def inference_ddim(self, input_dict, T=1000, step=10, report=20, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)

        if(self.condition):
            N = len(input_dict["feat"])

            c_point = self.feature_init(input_dict)
            n_point = self.feature_init(input_dict)

            # ---- initial input ---- #
            if (self.c_in_channels == 6):
                c_target = input_dict["feat"]
            else:
                c_target = input_dict["coord"]
            c_point['feat'] = torch.normal(0, 1, size=c_target.shape, dtype=torch.float32).cuda()
            n_point['feat'] = torch.randint(0, self.num_classes, size=(N,1)).cuda()
            # ---- initial input ---- #

            time_schedule = self.get_time_schedule(T, step)
            time_is = reversed(range(len(time_schedule)))
            for i, t in zip(time_is, time_schedule):

                if ((i + 1) % report == 0 or t <= 0):
                    print(f"  ---- current : [{i + 1 if t > 0 else 0}/{step}] steps ----")

                # ---- T steps ---- #
                t = t if t >= 0 else 0
                ts = t * torch.ones((N, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    c_point['t_emb'] = n_point['t_emb'] = calc_t_emb(ts, t_emb_dim=self.T_dim).cuda()
                # ---- T steps ---- #

                # ---- c_xt ---- #
                c_xt = c_point["feat"]
                # ---- c_xt ---- #

                # ---- n_xt ---- #
                n_xt = n_point["feat"]
                n_point["feat"] = (2 * n_xt.float() / self.num_classes) - 1.0
                # ---- n_xt ---- #

                # ---- pred c_x0 and n_x0 ---- #
                c_point, n_point = self.backbone(c_point, n_point)
                # ---- pred c_x0 and n_x0 ---- #

                # ---- c_xs ---- #
                c_epslon_ = c_point["feat"]
                c_xs = self.continuous_p_ddim_sample(
                    c_xt,
                    ts,
                    c_epslon_,
                ).float()
                c_point = self.feature_init(input_dict)
                c_point["feat"] = c_xs
                # ---- n_xs ---- #

                # ---- n_xs ---- #
                n_x0_ = n_point["feat"]
                n_xs = self.discrete_p_ddim_sample(
                    ts,
                    n_x0_,
                )
                n_point = self.feature_init(input_dict)
                n_point["feat"] = n_xs
                # ---- n_xs ---- #

                if (t <= 0):
                    break
        else:
            n_point = self.backbone(n_point=input_dict)

        if(eval):
            if("valid" in input_dict.keys()):
                n_point["feat"] = n_point["feat"][input_dict["valid"]]
                input_dict['segment'] = input_dict['segment'][input_dict["valid"]]
                n_xt = n_xt[input_dict["valid"]]
                ts = ts[input_dict["valid"]]

            point = {}

            # ---- poster distribution of discrete diffusion ----
            n_x0 = input_dict['segment'].view(len(input_dict['segment']), 1)  # [N,1]
            n_x0_ = n_point['feat']
            # 计算真实后验q(xt-1|xt,x0)，得到的也是logits的取值
            true_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0, n_xt, ts)
            # 计算预测后验p(xt-1|xt,x0~)，得到的也是logits的取值
            pred_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0_, n_xt, ts)

            point['n_true_q'] = true_discrete_q_posterior_logits.squeeze()
            point['n_pred_q'] = pred_discrete_q_posterior_logits.squeeze()
            # ---- poster distribution of discrete diffusion ----

            point['n_pred'] = n_point["feat"]
            point['n_target'] = input_dict['segment']
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=n_point["feat"])
        else:
            return dict(seg_logits=n_point["feat"])


    def forward(self, input_dict):

        point = {}

        n_target = input_dict["segment"]
        if(self.condition):
            point["valid"] = input_dict["valid"]

            ### ---- PT V3 + DM ---- ###
            c_point = {}
            c_point["coord"] = input_dict["coord"]
            c_point["grid_coord"] = input_dict["grid_coord"]
            c_point["offset"] = input_dict["offset"]

            n_point = {}
            n_point["coord"] = input_dict["coord"]
            n_point["grid_coord"] = input_dict["grid_coord"]
            n_point["offset"] = input_dict["offset"]

            c_point = Point(c_point)
            n_point = Point(n_point)

            batch = n_point["batch"]
            B = len(torch.unique(batch))

            # ---- initial input ---- #
            if(self.c_in_channels == 6):
                c_target = input_dict["feat"]
            else:
                c_target = input_dict["coord"]
            # ---- initial input ---- #

            # ---- discrete diffusion ---- #
            if(self.dm):

                # --- T_embeding ---- #
                ts = torch.randint(0, self.T, size=(B, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    c_point["t_emb"] = n_point["t_emb"] = calc_t_emb(ts, self.T_dim)[batch, :]
                ts = ts[batch, :]
                # --- T_embeding ---- #

                # ---- add noise ---- #
                c_x0 = c_target
                c_noise = torch.normal(0, 1, size=c_x0.shape, dtype=torch.float32).cuda()
                c_xt = self.continuous_q_sample(c_x0, ts, c_noise)
                c_point['feat'] = c_xt

                n_x0 = n_target.view(len(n_target), 1)  # [N,1]
                n_noise = torch.rand((*n_x0.shape, self.num_classes)).cuda()  # [N,C,num_class]
                n_xt = self.discrete_q_sample(n_x0, ts, n_noise)
                n_point['feat'] = (2 * n_xt.float() / self.num_classes) - 1.0
                # ---- add noise ---- #

                # ---- diffusion target ---- #
                if(self.dm_target == "noise"):
                    c_target = c_noise
                # ---- diffusion target ---- #

                # ---- SNR Loss Weight ----
                if (self.dm_min_snr is not None):
                    point["snr_loss_weight"] = self.SNR[ts]
                # ---- SNR Loss Weight ----
            # ---- discrete diffusion ---- #

            # ---- output ---- #
            c_point, n_point = self.backbone(c_point, n_point)
            # ---- output ---- #

            point['c_pred'] = c_point["feat"]
            point['c_target'] = c_target

            # ---- discrete diffusion ---- #
            if (self.dm):
                # ---- poster distribution of discrete diffusion ----
                n_x0_ = n_point['feat']
                # 计算真实后验q(xt-1|xt,x0)，得到的也是logits的取值 n_x0 : [N,1]
                true_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0, n_xt, ts)
                # 就算预测后验p(xt-1|xt,x0~)，得到的也是logits的取值 n_x0_ : [N,num_class]
                pred_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0_, n_xt, ts)

                point['n_true_q'] = true_discrete_q_posterior_logits.squeeze()
                point['n_pred_q'] = pred_discrete_q_posterior_logits.squeeze()
                # ---- poster distribution of discrete diffusion ----
            # ---- discrete diffusion ---- #

            ### ---- PT V3 + DM ---- ###
        else:
            ### ---- PT V3 ---- ###
            n_point = Point(input_dict)
            n_point = self.backbone(n_point=n_point)
            ### ---- PT V3 ---- ###

        point['n_pred'] = n_point['feat']
        point['n_target'] = n_target
        point['loss_mode'] = "train"

        loss = self.criteria(point)
        return dict(loss=loss)


@MODELS.register_module()
class DiscreteDMSegmentor(nn.Module):
    '''
        CN + CD : Conditional(No Dffusion Process) Network (CN) +  Categorical(Discrete) Diffusion (CD)
    '''
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
        c_in_channels=6,
        transfer_type="gaussian"
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
        self.transfer_type = transfer_type

        if(self.dm):
            # 1. 获取噪声时间表
            self.eps = 1e-6
            self.Beta, self.Alpha ,self.Alpha_bar, self.Sigma, self.SNR= self.get_diffusion_hyperparams(
                noise_schedule=noise_schedule,
                T=self.T,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )

            # 2. 获取转移矩阵Qt（每一步的）
            q_onestep_mats = []
            for beta in self.Beta:
                if self.transfer_type == "uniform":
                    mat = self.get_uniform_transition_mat(beta)
                    mat = torch.from_numpy(mat)
                    q_onestep_mats.append(mat)
                elif self.transfer_type == "gaussian":
                    mat = self.get_gaussian_transition_mat(beta)
                    mat = torch.from_numpy(mat)
                    q_onestep_mats.append(mat)
                else:
                    raise NotImplementedError
            q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

            # 3. 获取旋转矩阵的逆，Qt-1（每一步的），这里表达Qt本身是一个正交矩阵，QtT = Qt-1
            self.q_one_step_transposed = q_one_step_mats.transpose(1, 2).cuda()

            # 4. 获取累计旋转矩阵，Qt_
            q_mat_t = q_onestep_mats[0]
            q_mats = [q_mat_t]
            for idx in range(1, self.T):
                q_mat_t = q_mat_t @ q_onestep_mats[idx]  # 两个正交矩阵相乘结果仍然是正交矩阵
                q_mats.append(q_mat_t)
            self.q_mats = torch.stack(q_mats, dim=0).cuda()
            self.logit_type = "logit"

            # self.Beta = self.Beta.float().cuda()
            # self.Alpha = self.Alpha.float().cuda()
            # self.Alpha_bar = self.Alpha_bar.float().cuda()
            # self.Sigma = self.Sigma.float().cuda()
            # self.SNR = self.SNR.float().cuda() if dm_min_snr is None else torch.clamp(self.SNR.float().cuda(),max=dm_min_snr)


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

        return Beta.numpy(), Alpha.numpy(), Alpha_bar.numpy(), Sigma, SNR.numpy()

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

    def get_uniform_transition_mat(self, beta_t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).（xt会收敛于一个1/K的均匀分布）

        This method constructs a transition
        matrix Q with
        Q_{ij} = beta_t / num_pixel_vals       if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il} if i==j.
                 0                          else.
            1-(k-1)/k, i=j
        Qt=
            kbt,       i≠j
        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
        """

        # Assumes num_off_diags < num_pixel_vals
        transition_bands = self.num_classes - 1
        #beta_t = betas[t]

        mat = np.zeros((self.num_classes, self.num_classes), dtype=np.float32) # [[0,0],[0,0]]
        # [bt/k,]
        off_diag = np.full(shape=(transition_bands,), fill_value=beta_t / float(self.num_classes), dtype=np.float32)
        for k in range(1, transition_bands + 1):
            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)

        # mat = torch.ones(num_classes, num_classes) * beta / num_classes # [[bt/k,bt/k],[bt/k,bt/k]]
        # mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes) # [[1-(k-1)/k,bt/k],[bt/k,1-(k-1)/k]]

        return mat

    # 获取 高斯核变换矩阵 Qt
    def get_gaussian_transition_mat(self,  beta_t):
        r"""Computes transition matrix for q(x_t|x_{t-1})，计算x0到xt的转换矩阵（gaussian），然而，这种方式缺少随机过程

        This method constructs a transition matrix Q with
        decaying entries as a function of how far off diagonal the entry is.
        Normalization option 1:
        Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il}  if i==j.
                 0                          else.

        Normalization option 2:
        tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                         0                        else.

        Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (self.num_classes, self.num_classes).
        """
        transition_bands = self.num_classes - 1
        # beta_t, t
        #beta_t = betas[t]
        # [256,256]
        mat = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        # [256,], (0,255)
        values = np.linspace(start=0., stop=transition_bands, num=self.num_classes, endpoint=True, dtype=np.float32)
        values = values * 2. / (self.num_classes - 1.)  # values * 2 / 255 ??? (0,2)
        values = values[:transition_bands + 1]  # [256,]
        values = -values * values / beta_t  # -values * values / beta_t ????

        values = np.concatenate([values[:0:-1], values], axis=0)  # cat([255,],[256]) --> [511,]
        values = special.softmax(values, axis=0)  # softmax，归一化
        values = values[transition_bands:]  # 【511，】->[255,]
        for k in range(1, transition_bands + 1):
            # 用values的地k下标填补[self.self.num_classes - k,]矩阵
            off_diag = np.full(shape=(self.num_classes - k,), fill_value=values[k], dtype=np.float32)

            mat += np.diag(off_diag, k=k)  # 创建一个对角矩阵，以off_diag为元素， 【255,256】
            mat += np.diag(off_diag, k=-k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)  # [256.256]

        return mat

    # def _at(self, a, ts, t, x, offset):
    # xs = []
    # for ti,oi in zip(t,offset):
    #     xi = x[start:oi].transpose(0,1) # [1,N]
    #     ti = ti.reshape((1, *[1] * (x.dim() - 1))) # [1,1]
    #     xx = torch.squeeze(a[ti, xi,:])
    #     xs.append(xx)
    #     start = oi
    # logx1 = torch.cat(xs,dim=0)
    # xx = torch.equal(logx1,logx2)


    def _at(self, a, x, ts):
        # 根据x0作为索引，从Qt_中选取对应元素作为xt
        logx = a[ts,x,:]
        return logx

    def discrete_q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32: # [B,1,32,32,N]
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
            if(x_0_logits.dim() == 2):
                x_0_logits = torch.unsqueeze(x_0_logits,dim=1)

        assert x_0_logits.shape == x_t.shape + (self.num_classes,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # ---- xt-1_1 = xt * QtT ----
        # [T,num_class,num_class] * [N,C,num_class] => [N, C, num_class]
        fact1 = self._at(self.q_one_step_transposed, x_t, t)
        # ---- xt-1_1 = xt * QtT ----

        # ---- xt-1_2 = x0 * Qt-1_ ----
        # [N,C,num_class] * [N, num_class, num_class] => [N, C, num_class]
        fact2 = torch.einsum("ncl,nld->ncd", torch.softmax(x_0_logits, dim=-1), self.q_mats[t - 1].squeeze())
        # ---- xt-1_2 = x0 * Qt-1_ ----

        # ---- xt-1 = log(xt-1_1) + log(xt-1_2) ----
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        # ---- xt-1 = log(xt-1_1) + log(xt-1_2) ----

        # # [B,1,1]
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        # 如果当前索引为1，则从x0直接选取结果，否则从put中选取结果
        bc = torch.where(t_broadcast == 0, x_0_logits, out)

        return bc

    def discrete_q_sample(self, x_0, ts, noise=None):
        if(noise is None):
            # sampling from uniform distribution
            noise = torch.rand((*x_0.shape, self.num_classes), device=x_0.device)
        # q(xt|x0), xt = x0 * Qt_
        xt = self._at(self.q_mats, x_0, ts)
        # discrete sampling, argmax(gumbal_softmax(log(xt))), [N,1,C] -> [N,1]
        logits = torch.log(xt + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_t = torch.argmax(logits + gumbel_noise, dim=-1)
        return x_t

    def discrete_p_sample(self, xt, t, x0, noise=None):

        if(t[0] == 0):
            return x0

        pred_discrete_q_posterior_logits = self.discrete_q_posterior_logits(x0, xt, t)
        if(noise is None):
            noise = torch.rand((*xt.shape, self.num_classes)).cuda()
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((xt.shape[0], *[1] * (xt.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        xt_1 = torch.argmax(
            pred_discrete_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return xt_1

    def discrete_p_ddim_sample(self, t, x_0, noise=None):

        if(t[0] == 0):
            return x_0

        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32: # [B,1,32,32,N]
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
            if(x_0_logits.dim() == 2):
                x_0_logits = torch.unsqueeze(x_0_logits,dim=1)

        # ---- xt-1 = x0 * Qt-1_ ----
        # [N,C,num_class] * [N, num_class, num_class] => [N, C, num_class]
        pred_discrete_q_posterior_logits = torch.einsum("ncl,nld->ncd", torch.softmax(x_0_logits, dim=-1), self.q_mats[t - 1].squeeze())
        # ---- xt-1  = x0 * Qt-1_ ----

        if(noise is None):
            noise = torch.rand(pred_discrete_q_posterior_logits.shape).cuda()
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((x_0.shape[0], *[1] * (x_0.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        xt_1 = torch.argmax(
            pred_discrete_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return xt_1

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

    def feature_init(self, input_dict):
        point = {}
        point["coord"] = input_dict["coord"]
        point["grid_coord"] = input_dict["grid_coord"]
        point["offset"] = input_dict["offset"]
        return point


    def inference_ddim(self, input_dict, T=1000, step=1, report=20, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)

        if(self.condition):
            N = len(input_dict["feat"])

            c_point = self.feature_init(input_dict)
            n_point = self.feature_init(input_dict)

            # ---- initial input ---- #
            if (self.c_in_channels == 6):
                c_point['feat'] = c_feat = input_dict["feat"]
            else:
                c_point['feat'] = c_feat = input_dict["coord"]
            n_point['feat'] = torch.randint(0, self.num_classes, size=(N,1)).cuda()
            # ---- initial input ---- #

            time_schedule = self.get_time_schedule(T, step)
            time_is = reversed(range(len(time_schedule)))
            for i, t in zip(time_is, time_schedule):

                if ((i + 1) % report == 0 or t <= 0):
                    print(f"  ---- current : [{i + 1 if t > 0 else 0}/{step}] steps ----")

                # ---- T steps ---- #
                t = t if t >= 0 else 0
                ts = t * torch.ones((N, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    n_point['t_emb'] = calc_t_emb(ts, t_emb_dim=self.T_dim).cuda()
                # ---- T steps ---- #

                # ---- n_xt ---- #
                n_xt = n_point["feat"]
                n_point["feat"] = (2 * n_xt.float() / self.num_classes) - 1.0
                # ---- n_xt ---- #

                # ---- pred c_x0 and n_x0 ---- #
                c_point, n_point = self.backbone(c_point, n_point, c_decoder=False)
                # ---- pred c_x0 and n_x0 ---- #

                # ---- n_xs ---- #
                n_x0_ = n_point["feat"]
                n_xs = self.discrete_p_ddim_sample(
                    ts,
                    n_x0_,
                )
                n_point = self.feature_init(input_dict)
                n_point["feat"] = n_xs
                # ---- n_xs ---- #

                if (t <= 0):
                    break

                # ---- c_feat ---- #
                c_point = self.feature_init(input_dict)
                c_point["feat"] = c_feat
                # ---- c_feat ---- #

        else:
            n_point = self.backbone(n_point=input_dict)

        if(eval):
            if("valid" in input_dict.keys()):
                n_point["feat"] = n_point["feat"][input_dict["valid"]]
                input_dict['segment'] = input_dict['segment'][input_dict["valid"]]
                n_xt = n_xt[input_dict["valid"]]
                ts = ts[input_dict["valid"]]

            point = {}

            # ---- poster distribution of discrete diffusion ----
            n_x0 = input_dict['segment'].view(len(input_dict['segment']), 1)  # [N,1]
            n_x0_ = n_point['feat']
            # 计算真实后验q(xt-1|xt,x0)，得到的也是logits的取值
            true_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0, n_xt, ts)
            # 计算预测后验p(xt-1|xt,x0~)，得到的也是logits的取值
            pred_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0_, n_xt, ts)

            point['n_true_q'] = true_discrete_q_posterior_logits.squeeze()
            point['n_pred_q'] = pred_discrete_q_posterior_logits.squeeze()
            # ---- poster distribution of discrete diffusion ----

            point['n_pred'] = n_point["feat"]
            point['n_target'] = input_dict['segment']
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=n_point["feat"])
        else:
            return dict(seg_logits=n_point["feat"])


    def forward(self, input_dict):

        point = {}

        n_target = input_dict["segment"]
        if(self.condition):
            point["valid"] = input_dict["valid"]

            ### ---- PT V3 + DM ---- ###
            c_point = {}
            c_point["coord"] = input_dict["coord"]
            c_point["grid_coord"] = input_dict["grid_coord"]
            c_point["offset"] = input_dict["offset"]

            n_point = {}
            n_point["coord"] = input_dict["coord"]
            n_point["grid_coord"] = input_dict["grid_coord"]
            n_point["offset"] = input_dict["offset"]

            c_point = Point(c_point)
            n_point = Point(n_point)

            batch = n_point["batch"]
            B = len(torch.unique(batch))

            # ---- initial input ---- #
            if(self.c_in_channels == 6):
                c_point['feat'] = c_target = input_dict["feat"]
            else:
                c_point['feat'] = c_target = input_dict["coord"]
            # ---- initial input ---- #

            # ---- discrete diffusion ---- #
            if(self.dm):

                # --- T_embeding ---- #
                ts = torch.randint(0, self.T, size=(B, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    n_point["t_emb"] = calc_t_emb(ts, self.T_dim)[batch, :]
                ts = ts[batch, :]
                # --- T_embeding ---- #

                # ---- add noise ---- #
                n_x0 = n_target.view(len(n_target), 1)  # [N,1]
                n_noise = torch.rand((*n_x0.shape, self.num_classes)).cuda()  # [N,C,num_class]
                n_xt = self.discrete_q_sample(n_x0, ts, n_noise)
                n_point['feat'] = (2 * n_xt.float() / self.num_classes) - 1.0
                # ---- add noise ---- #

                # ---- diffusion target ---- #
                if(self.dm_target == "noise"):
                    n_target = n_noise
                # ---- diffusion target ---- #

                # ---- SNR Loss Weight ----
                if (self.dm_min_snr is not None):
                    point["snr_loss_weight"] = self.SNR[ts]
                # ---- SNR Loss Weight ----
            # ---- discrete diffusion ---- #

            # ---- output ---- #
            c_point, n_point = self.backbone(c_point, n_point)
            # ---- output ---- #

            point['c_pred'] = c_point["feat"]
            point['c_target'] = c_target

            # ---- discrete diffusion ---- #
            if (self.dm):
                # ---- poster distribution of discrete diffusion ----
                n_x0_ = n_point['feat']
                # 计算真实后验q(xt-1|xt,x0)，得到的也是logits的取值 n_x0 : [N,1]
                true_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0, n_xt, ts)
                # 就算预测后验p(xt-1|xt,x0~)，得到的也是logits的取值 n_x0_ : [N,num_class]
                pred_discrete_q_posterior_logits = self.discrete_q_posterior_logits(n_x0_, n_xt, ts)

                point['n_true_q'] = true_discrete_q_posterior_logits.squeeze()
                point['n_pred_q'] = pred_discrete_q_posterior_logits.squeeze()
                # ---- poster distribution of discrete diffusion ----
            # ---- discrete diffusion ---- #

            ### ---- PT V3 + DM ---- ###
        else:
            ### ---- PT V3 ---- ###
            n_point = Point(input_dict)
            n_point = self.backbone(n_point=n_point)
            ### ---- PT V3 ---- ###

        point['n_pred'] = n_point['feat']
        point['n_target'] = n_target
        point['loss_mode'] = "train"

        loss = self.criteria(point)
        return dict(loss=loss)

@MODELS.register_module()
class ContinuousDMSegmentor(nn.Module):
    '''
        CN + GD : Conditional(No Dffusion Process) Network (CN) +  Gaussion(Continous) Diffusion (CD)
    '''
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
            x0 = (x_t - torch.sqrt(1 - self.Alpha_bar[t]) * noise) / torch.sqrt(self.Alpha_bar[t])
        else:
            x0 = noise
            # noise = (xt - sqrt(1-at_) * x0) / sqrt(1-at_)
            noise = (x_t - torch.sqrt(self.Alpha_bar[t]) * x0) / torch.sqrt(1 - self.Alpha_bar[t])

        if(t[0] == 0):
            return x0

        # sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_)
        xs_1 = torch.sqrt(self.Alpha_bar[t-1]) * x0

        # sqrt(1 - at-1_) * noise
        xs_2 = torch.sqrt(1 - self.Alpha_bar[t-1]) * noise

        # xt-1 = sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_) + sqrt(1 - at-1_) * noise
        xs = xs_1 + xs_2

        return xs

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

    def feature_init(self, input_dict):
        point = {}
        point["coord"] = input_dict["coord"]
        point["grid_coord"] = input_dict["grid_coord"]
        point["offset"] = input_dict["offset"]
        return point


    def inference_ddim(self, input_dict, T=1000, step=1, report=20, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)

        if(self.condition):
            N = len(input_dict["feat"])

            c_point = self.feature_init(input_dict)
            n_point = self.feature_init(input_dict)

            # ---- initial input ---- #
            if (self.c_in_channels == 6):
                c_point['feat'] = c_feat = input_dict["feat"]
            else:
                c_point['feat'] = c_feat = input_dict["coord"]
            n_point['feat'] = torch.normal(0, 1, size=(N, self.num_classes), dtype=torch.float32).cuda()
            # ---- initial input ---- #

            time_schedule = self.get_time_schedule(T, step)
            time_is = reversed(range(len(time_schedule)))
            for i, t in zip(time_is, time_schedule):

                if ((i + 1) % report == 0 or t <= 0):
                    print(f"  ---- current : [{i + 1 if t > 0 else 0}/{step}] steps ----")

                # ---- T steps ---- #
                t = t if t >= 0 else 0
                ts = t * torch.ones((N, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    n_point['t_emb'] = calc_t_emb(ts, t_emb_dim=self.T_dim).cuda()
                # ---- T steps ---- #

                # ---- n_xt ---- #
                n_xt = n_point["feat"]
                # ---- n_xt ---- #

                # ---- pred c_x0 and n_x0 ---- #
                c_point, n_point = self.backbone(c_point, n_point)
                # ---- pred c_x0 and n_x0 ---- #

                # ---- n_xs ---- #
                n_epslon_ = n_point["feat"]
                n_xs = self.continuous_p_ddim_sample(
                    n_xt,
                    ts,
                    n_epslon_,
                ).float()
                n_point = self.feature_init(input_dict)
                n_point["feat"] = n_xs
                # ---- n_xs ---- #

                if (t <= 0):
                    break

                # ---- c_feat ---- #
                c_point = self.feature_init(input_dict)
                c_point["feat"] = c_feat
                # ---- c_feat ---- #

        else:
            n_point = self.backbone(n_point=input_dict)

        if(eval):
            n_target = input_dict["segment"]
            if(self.condition and self.dm_target == "noise"):
                n_target = torch.log(torch.nn.functional.one_hot(n_target, self.num_classes) + self.eps)
            if("valid" in input_dict.keys()):
                n_point["feat"] = n_point["feat"][input_dict["valid"]]
                n_target = n_target[input_dict["valid"]]
                input_dict["segment"] = input_dict["segment"][input_dict["valid"]]

            point = {}
            point['n_pred'] = n_point["feat"]
            point['n_target'] = n_target
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=n_point["feat"])
        else:
            return dict(seg_logits=n_point["feat"])

    def inference(self, input_dict, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)

        if(self.condition):
            ### ---- PT V3 + DM ---- ###
            c_point = {}
            c_point["coord"] = input_dict["coord"]
            c_point["grid_coord"] = input_dict["grid_coord"]
            c_point["offset"] = input_dict["offset"]

            n_point = {}
            n_point["coord"] = input_dict["coord"]
            n_point["grid_coord"] = input_dict["grid_coord"]
            n_point["offset"] = input_dict["offset"]

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
            c_point, n_point = self.backbone(c_point, n_point, c_decoder=False)
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

        n_target = input_dict["segment"]
        if(self.condition):
            point["valid"] = input_dict["valid"]

            ### ---- PT V3 + DM ---- ###
            c_point = {}
            c_point["coord"] = input_dict["coord"]
            c_point["grid_coord"] = input_dict["grid_coord"]
            c_point["offset"] = input_dict["offset"]

            n_point = {}
            n_point["coord"] = input_dict["coord"]
            n_point["grid_coord"] = input_dict["grid_coord"]
            n_point["offset"] = input_dict["offset"]

            c_point = Point(c_point)
            n_point = Point(n_point)

            batch = n_point["batch"]
            B = len(torch.unique(batch))

            # ---- initial input ---- #
            if(self.c_in_channels == 6):
                c_point['feat'] = c_target = input_dict["feat"]
            else:
                c_point['feat'] = c_target = input_dict["coord"]
            # ---- initial input ---- #

            # ---- continuous diffusion ---- #
            if(self.dm):

                # --- T_embeding ---- #
                ts = torch.randint(0, self.T, size=(B, 1), dtype=torch.int64).cuda()
                if (self.T_dim != -1):
                    n_point["t_emb"] = calc_t_emb(ts, self.T_dim)[batch, :]
                ts = ts[batch, :]
                # --- T_embeding ---- #

                # ---- add noise ---- #
                n_x0 = torch.log(torch.nn.functional.one_hot(n_target, self.num_classes) + self.eps)
                n_noise = torch.normal(0, 1, size=n_x0.shape,dtype=torch.float32).cuda()
                n_xt = self.continuous_q_sample(n_x0, ts, n_noise)
                n_point['feat'] = n_xt
                # ---- add noise ---- #

                # ---- diffusion target ---- #
                if(self.dm_target == "noise"):
                    n_target = n_noise
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
        point['n_target'] = n_target
        point['loss_mode'] = "train"

        loss = self.criteria(point)
        return dict(loss=loss)

@MODELS.register_module("DMSegmentor")
class DMSegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=20,
        T=1000,
        beta_start=0.0001,
        beta_end=0.02,
        transfer_type="gaussian",
        noise_schedule="linear",
        remove=True,
        T_dim=128
    ):
        super().__init__()

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.transfer_type = transfer_type
        self.noise_schedule = noise_schedule
        self.remove = remove
        self.T_dim = T_dim

        # ---- diffusion params ----
        # 1. 获取噪声时间表
        # self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.Beta, self.Alpha ,self.Alpha_bar, self.Sigma= self.get_diffusion_hyperparams(
            noise_schedule=noise_schedule,
            T=self.T,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )
        # ---- diffusion params ----

        # ---- discrate diffusion ----
        # 2. 获取转移矩阵Qt（每一步的）
        q_onestep_mats = []
        for beta in self.Beta:
            if self.transfer_type == "uniform":
                mat = self.get_uniform_transition_mat(beta)
                mat = torch.from_numpy(mat)
                q_onestep_mats.append(mat)
            elif self.transfer_type == "gaussian":
                mat = self.get_gaussian_transition_mat(beta)
                mat = torch.from_numpy(mat)
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        # 3. 获取旋转矩阵的逆，Qt-1（每一步的），这里表达Qt本身是一个正交矩阵，QtT = Qt-1
        self.q_one_step_transposed = q_one_step_mats.transpose(1, 2).cuda()

        # 4. 获取累计旋转矩阵，Qt_
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]  # 两个正交矩阵相乘结果仍然是正交矩阵
            q_mats.append(q_mat_t)
        self.q_mats = torch.stack(q_mats, dim=0).cuda()
        self.logit_type = "logit"
        # ---- discrate diffusion ----

        self.Beta = torch.from_numpy(self.Beta).cuda()
        self.Alpha = torch.from_numpy(self.Alpha).cuda()
        self.Alpha_bar = torch.from_numpy(self.Alpha_bar).cuda()
        self.Sigma = torch.from_numpy(self.Sigma).cuda()

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
        Sigma = np.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
        Sigma[0] = 0.0

        return Beta, Alpha, Alpha_bar, Sigma

    def get_diffusion_betas(self, type='linear', start=0.0001, stop=1.0, T=1000):
        """Get betas from the hyperparameters."""
        if type == 'linear':
            # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
            # To be used with Gaussian diffusion models in continuous and discrete
            # state spaces.
            # To be used with transition_mat_type = 'gaussian'
            return np.linspace(start, stop, T)
        elif type == 'cosine':
            # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
            # To be used with transition_mat_type = 'uniform'.
            steps = (np.arange(T + 1, dtype=np.float32) /T)
            alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
            betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
            return betas
        elif type == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
            # To be used with absorbing state models.
            # ensures that the probability of decaying to the absorbing state
            # increases linearly over time, and is 1 for t = T-1 (the final time).
            # To be used with transition_mat_type = 'absorbing'
            return 1. / np.linspace(T, 1., T)
        else:
            raise NotImplementedError(type)

    def get_uniform_transition_mat(self, beta_t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).（xt会收敛于一个1/K的均匀分布）

        This method constructs a transition
        matrix Q with
        Q_{ij} = beta_t / num_pixel_vals       if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il} if i==j.
                 0                          else.
            1-(k-1)/k, i=j
        Qt=
            kbt,       i≠j
        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
        """

        # Assumes num_off_diags < num_pixel_vals
        transition_bands = self.num_classes - 1
        #beta_t = betas[t]

        mat = np.zeros((self.num_classes, self.num_classes), dtype=np.float32) # [[0,0],[0,0]]
        # [bt/k,]
        off_diag = np.full(shape=(transition_bands,), fill_value=beta_t / float(self.num_classes), dtype=np.float32)
        for k in range(1, transition_bands + 1):
            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)

        # mat = torch.ones(num_classes, num_classes) * beta / num_classes # [[bt/k,bt/k],[bt/k,bt/k]]
        # mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes) # [[1-(k-1)/k,bt/k],[bt/k,1-(k-1)/k]]

        return mat

    # 获取 高斯核变换矩阵 Qt
    def get_gaussian_transition_mat(self,  beta_t):
        r"""Computes transition matrix for q(x_t|x_{t-1})，计算x0到xt的转换矩阵（gaussian），然而，这种方式缺少随机过程

        This method constructs a transition matrix Q with
        decaying entries as a function of how far off diagonal the entry is.
        Normalization option 1:
        Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il}  if i==j.
                 0                          else.

        Normalization option 2:
        tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                         0                        else.

        Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

        Args:
          t: timestep. integer scalar (or numpy array?)

        Returns:
          Q_t: transition matrix. shape = (self.num_classes, self.num_classes).
        """
        transition_bands = self.num_classes - 1
        # beta_t, t
        #beta_t = betas[t]
        # [256,256]
        mat = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        # [256,], (0,255)
        values = np.linspace(start=0., stop=transition_bands, num=self.num_classes, endpoint=True, dtype=np.float32)
        values = values * 2. / (self.num_classes - 1.)  # values * 2 / 255 ??? (0,2)
        values = values[:transition_bands + 1]  # [256,]
        values = -values * values / beta_t  # -values * values / beta_t ????

        values = np.concatenate([values[:0:-1], values], axis=0)  # cat([255,],[256]) --> [511,]
        values = special.softmax(values, axis=0)  # softmax，归一化
        values = values[transition_bands:]  # 【511，】->[255,]
        for k in range(1, transition_bands + 1):
            # 用values的地k下标填补[self.self.num_classes - k,]矩阵
            off_diag = np.full(shape=(self.num_classes - k,), fill_value=values[k], dtype=np.float32)

            mat += np.diag(off_diag, k=k)  # 创建一个对角矩阵，以off_diag为元素， 【255,256】
            mat += np.diag(off_diag, k=-k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)  # [256.256]

        return mat

    # def _at(self, a, ts, t, x, offset):
    # xs = []
    # for ti,oi in zip(t,offset):
    #     xi = x[start:oi].transpose(0,1) # [1,N]
    #     ti = ti.reshape((1, *[1] * (x.dim() - 1))) # [1,1]
    #     xx = torch.squeeze(a[ti, xi,:])
    #     xs.append(xx)
    #     start = oi
    # logx1 = torch.cat(xs,dim=0)
    # xx = torch.equal(logx1,logx2)


    def _at(self, a, x, ts):
        # 根据x0作为索引，从Qt_中选取对应元素作为xt
        logx = a[ts,x,:]
        return logx

    def discrete_q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32: # [B,1,32,32,N]
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
            if(x_0_logits.dim() == 2):
                x_0_logits = torch.unsqueeze(x_0_logits,dim=1)

        assert x_0_logits.shape == x_t.shape + (self.num_classes,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # ---- xt-1_1 = xt * QtT ----
        # [T,num_class,num_class] * [N,C,num_class] => [N, C, num_class]
        fact1 = self._at(self.q_one_step_transposed, x_t, t)
        # ---- xt-1_1 = xt * QtT ----

        # ---- xt-1_2 = x0 * Qt-1_ ----
        # [N,C,num_class] * [N, num_class, num_class] => [N, C, num_class]
        fact2 = torch.einsum("ncl,nld->ncd", torch.softmax(x_0_logits, dim=-1), self.q_mats[t - 1].squeeze())
        # ---- xt-1_2 = x0 * Qt-1_ ----

        # ---- xt-1 = log(xt-1_1) + log(xt-1_2) ----
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        # ---- xt-1 = log(xt-1_1) + log(xt-1_2) ----

        # # [B,1,1]
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        # 如果当前索引为1，则从x0直接选取结果，否则从put中选取结果
        bc = torch.where(t_broadcast == 0, x_0_logits, out)

        return bc

    def discrete_q_sample(self, x_0, ts, noise=None):
        if(noise is None):
            # sampling from uniform distribution
            noise = torch.rand((*x_0.shape, self.num_classes), device=x_0.device)
        # q(xt|x0), xt = x0 * Qt_
        xt = self._at(self.q_mats, x_0, ts)
        # discrete sampling, argmax(gumbal_softmax(log(xt))), [N,1,C] -> [N,1]
        logits = torch.log(xt + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_t = torch.argmax(logits + gumbel_noise, dim=-1)
        return x_t

    def discrete_p_sample(self, xt, t, x0, noise=None):

        if(t[0] == 0):
            return x0

        pred_discrete_q_posterior_logits = self.discrete_q_posterior_logits(x0, xt, t)
        if(noise is None):
            noise = torch.rand((*xt.shape, self.num_classes)).cuda()
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((xt.shape[0], *[1] * (xt.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        xt_1 = torch.argmax(
            pred_discrete_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return xt_1

    def discrete_p_ddim_sample(self, t, x_0, noise=None):

        if(t[0] == 0):
            return x_0

        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32: # [B,1,32,32,N]
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
            if(x_0_logits.dim() == 2):
                x_0_logits = torch.unsqueeze(x_0_logits,dim=1)

        # ---- xt-1 = x0 * Qt-1_ ----
        # [N,C,num_class] * [N, num_class, num_class] => [N, C, num_class]
        pred_discrete_q_posterior_logits = torch.einsum("ncl,nld->ncd", torch.softmax(x_0_logits, dim=-1), self.q_mats[t - 1].squeeze())
        # ---- xt-1  = x0 * Qt-1_ ----

        if(noise is None):
            noise = torch.rand(pred_discrete_q_posterior_logits.shape).cuda()
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((x_0.shape[0], *[1] * (x_0.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        xt_1 = torch.argmax(
            pred_discrete_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return xt_1

    def continuous_q_posterior_logits(self, x_t, t, noise, z=None, sigma=True):

        if(z is None):
            z = noise

        # xt-1=1/sqrt(at_) * (xt-(1-at)/(sqrt(1-at_)*noise))
        c_xt_1_1 = (1 / torch.sqrt(self.Alpha[t])) * (x_t - (1 - self.Alpha[t]) / torch.sqrt(1 - self.Alpha_bar[t]) * noise)
        # xt_1_2 = vart * z
        c_xt_1_2 = self.Sigma[t] * z if sigma else 0.0

        c_xt_1 = c_xt_1_1 + c_xt_1_2

        return c_xt_1

    def continuous_p_ddim_sample(self, x_t, t, noise):

        # x0 = (xt - sqrt(1-at_) * noise) / sqrt(at_)
        c_x0 = (x_t - torch.sqrt(1 - self.Alpha_bar[t]) * noise) / torch.sqrt(self.Alpha_bar[t])
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

    def continuous_p_sample(self, x_t, t, noise, z=None, sigma=True):

        return self.continuous_q_posterior_logits(x_t, t, noise, z, sigma)

    def get_time_schedule(self, T=1000, step=5):
        times = np.linspace(-1, T - 1, num = step + 1, dtype=int)[::-1]
        return times

    def inference_ddim(self, c_point, n_point, T=1000, step=100,report=10,eval=False):

        # ---- initial input ----
        c_target = c_point['coord']
        n_target = n_point['segment']
        n_feat = n_point["feat"]

        N = len(n_target)

        c_point['feat'] = torch.normal(0, 1, size=(N,3), dtype=torch.float32).cuda()
        n_pred = torch.zeros(size=(N,self.num_classes), dtype=torch.float32).cuda()
        # ---- initial input ----

        time_schedule = self.get_time_schedule(T,step)
        time_is = reversed(range(len(time_schedule)))
        for i, t in zip(time_is, time_schedule):

            if((i+1) % report == 0 or t <= 0):
                print(f"  ---- current : [{i+1 if t > 0 else 0}/{step}] steps ----")

            # ---- T steps ----
            t = t if t >=0 else 0
            ts = t * torch.ones((N,1),dtype=torch.int64).cuda()
            if (self.T_dim != -1):
                c_point['t_emb'] = calc_t_emb(ts,t_emb_dim=self.T_dim).cuda()
            # ---- T steps ----

            # ---- n_xt ----
            c_xt = c_point["feat"]
            # ---- n_xt ----

            # ---- pred c_epsilon and n_x0 ----
            c_point, n_point = self.backbone(c_point, n_point)
            # ---- pred c_epsilon and n_x0 ----

            # ---- c_xt-1 ----
            c_epslon_ = c_point["feat"]
            c_point["feat"] = self.continuous_p_ddim_sample(
                c_xt,
                ts,
                c_epslon_,
            ).float()
            # ---- c_xt-1 ----

            # ---- n_pred ----
            n_pred += n_point["feat"]
            # ---- n_pred ----

            # ---- n_feat ----
            n_point["feat"] = n_feat
            # ---- n_feat ----

            if(t <= 0):
                break

        n_point["feat"] = n_pred / len(time_schedule)

        if(eval):
            point = {}

            point['c_pred'] = c_point["feat"]
            point['c_target'] = c_target

            point['n_pred'] = n_point["feat"]
            point['n_target'] = n_target

            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=point['n_pred'], label=point["n_target"])
        else:
            return c_point, n_point

    def inference(self, c_point, n_point, T=1000, step=1000,report=100):

        # ---- initial input ----
        c_target = c_point['coord']
        n_target = n_point['segment']
        n_feat = n_point["feat"]

        N = len(n_target)

        c_point['feat'] = torch.normal(0, 1, size=(N,3), dtype=torch.float32).cuda()
        n_pred = torch.zeros(size=(N,self.num_classes), dtype=torch.float32).cuda()
        # ---- initial input ----

        time_schedule = self.get_time_schedule(T,step)

        for t in time_schedule:

            if((t + 1) % report == 0 or (t + 1) == 1):
                print(f"  ---- current : [{t+1}/{step}] steps ----")

            # ---- T steps ----
            t = t if t >= 0 else 0
            ts = t * torch.ones((N,1),dtype=torch.int64).cuda()
            if (self.T_dim != -1):
                c_point['t_emb'] = calc_t_emb(ts,t_emb_dim=self.T_dim).cuda()
            # ---- T steps ----

            # ---- n_xt ----
            c_xt = c_point["feat"]
            # ---- n_xt ----

            # ---- pred c_epsilon ----
            c_point, n_point = self.backbone(c_point, n_point)
            # ---- pred c_epsilon ----

            # ---- n_xt-1 ----
            c_epslon_ = n_point["feat"]
            c_point["feat"] = self.continuous_p_sample(
                c_xt,
                ts,
                c_epslon_,
                z=torch.normal(0, 1, size=(N,3)).cuda(),
                sigma=True if t>0 else False
            ).float()
            # ---- n_xt-1 ----

            # --- n_pred ----
            n_pred += n_point["feat"]
            # --- n_pred ----

            # ---- n_feat ----
            n_point["feat"] = n_feat
            # ---- n_feat ----

            if(t <= 0):
                break

        n_point["feat"] = n_pred / len(time_schedule)

        return c_point, n_point


    def forward(self, c_point, n_point):

        c_point = Point(c_point)
        n_point = Point(n_point)

        valid = n_point["valid"]
        batch = n_point["batch"]
        B = len(torch.unique(batch))
        ts = torch.randint(0,self.T,size=(B,1),dtype=torch.int64).cuda()
        c_target = c_point['coord']
        n_target = n_point['segment']

        # --- T_embeding ----
        if(self.T_dim != -1):
            c_point["t_emb"] = calc_t_emb(ts, self.T_dim)[batch, :]
        ts = ts[batch, :]
        # --- T_embeding ----

        # ---- continuous diffusio ----
        c_x0 = c_point['coord']
        c_noise = torch.normal(0, 1, size=c_x0.shape).cuda()

        # q(xt|x0)
        c_xt = self.continuous_q_sample(c_x0, ts, c_noise).float()
        # 将xt与条件合并
        # c_point['feat'] = torch.cat([c_x0, c_xt], dim=-1).float()
        c_point['feat'] = c_xt
        # ---- continuous diffusion ----

        # (N,6)  Encoder -> (N,64) Decoder,-> (N,20) seg head
        '''
            (N,6) -> 
            [(N,32) ->(N1,32)->(N2,64)->(N3,128)->(N4,256)->(N5,512)]
            [(N4,256)->(N3,128)->(N2,64)->(N1,64)->(N,64)]
            [(N,20)]
        '''
        # ---- output ----
        c_point, n_point = self.backbone(c_point,n_point)

        point = {}

        point['c_pred'] = c_point["feat"]
        point['c_target'] = c_target

        point['n_pred'] = n_point["feat"]
        point['n_target'] = n_target
        # ---- output

        if(not self.remove):
            point['c_pred'] = point['c_pred'][valid]
            point['c_target'] = point['c_target'][valid]

            point['n_pred'] = point['n_pred'][valid]
            point['n_target'] = point['n_target'][valid]

        loss = self.criteria(point)
        return dict(loss=loss)



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
