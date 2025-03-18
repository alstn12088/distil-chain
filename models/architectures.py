import torch
import numpy as np
from einops import rearrange
from torch import nn
import math


class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class FourierMLP(nn.Module):
    def __init__(
            self,
            in_shape=2,
            out_shape=2,
            num_layers=2,
            channels=128,
            zero_init=True,
    ):
        super().__init__()

        self.in_shape = (in_shape,)
        self.out_shape = (out_shape,)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64):
        super(TimeEncoding, self).__init__()

        pe = torch.arange(1, harmonics_dim + 1).float().unsqueeze(0) * 2 * math.pi
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.GELU()
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = (t * self.pe).sin()
        t_cos = (t * self.pe).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncoding(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncoding, self).__init__()

        self.x_model = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim),
            nn.GELU()
        )

    def forward(self, s):
        return self.x_model(s)


class JointPolicy(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 zero_init: bool = False):
        super(JointPolicy, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class FlowModel(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1):
        super(FlowModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class LangevinScalingModel(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, zero_init: bool = False):
        super(LangevinScalingModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.01)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64):
        super(TimeEncodingPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])

        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncodingPIS(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncodingPIS, self).__init__()

        self.x_model = nn.Linear(s_dim, s_emb_dim)

    def forward(self, s):
        return self.x_model(s)


class JointPolicyPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(JointPolicyPIS, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class FlowModelPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(FlowModelPIS, self).__init__()

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class LangevinScalingModelPIS(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 3,
                 zero_init: bool = False):
        super(LangevinScalingModelPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(2 * t_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.register_buffer('pe', pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, t):
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)


class DistilledSampler(nn.Module):
    """
    f_theta(x_t, t, s):
      - x_t: [B, s_dim]
      - t, s: [B, 1] with t < s
      Outputs:
      - x_{t->s}: [B, s_dim], predicted next state (residual version).

    This version individually encodes t and s, then combines those encodings
    (rather than encoding delta_t = s - t).
    """
    def __init__(
        self,
        s_dim=2,            # dimension of the state x_t
        harmonics_dim=128,  # used in TimeEncodingPIS
        hidden_dim=128,     # hidden dimensionality
        num_layers=2,       # number of layers in the JointPolicyPIS
        zero_init=True,     # whether to zero-init the last layer
        device='cuda'
    ):
        super().__init__()
        self.s_dim = s_dim
        self.device = device

        # 1) State encoding for x_t => embed into R^{hidden_dim}
        self.state_enc = StateEncodingPIS(
            s_dim=s_dim,
            hidden_dim=hidden_dim,
            s_emb_dim=hidden_dim
        )

        # 2) Time encoding 
        self.time_enc = TimeEncodingPIS(
            harmonics_dim=harmonics_dim,
            dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        # 3) Joint policy that fuses x_emb + combined time embedding
        self.joint_policy = JointPolicyPIS(
            s_dim=s_dim,
            s_emb_dim=hidden_dim,  # must match the embedding sizes above
            t_dim=hidden_dim,      # must match time_enc output
            hidden_dim=hidden_dim,
            out_dim=s_dim,         # produce a residual in R^{s_dim}
            num_layers=num_layers,
            zero_init=zero_init
        )

    def forward(self, x, t):
        """
        x: [B, s_dim]
        t: [B, 1] 
        returns: x_{t} = x + residual
        """
        # 1) Embed the state
        x_emb = self.state_enc(x)  # => shape [B, hidden_dim]
        
        # 2) Encode t and s individually
        t_emb = self.time_enc(t)     # => shape [B, hidden_dim]
        #s_emb = self.time_enc(s)     # => shape [B, hidden_dim]
        
        # 3) Combine time embeddings (e.g., sum)
        #    Then feed (x_emb, time_emb) into the joint policy
        #time_emb = t_emb + s_emb     # => shape [B, hidden_dim]
        time_emb = t_emb
        # 4) The joint policy expects two embeddings of the same shape
        #    So we interpret x_emb as "state embedding", time_emb as "time embedding"
        residual = self.joint_policy(x_emb, time_emb)  # => [B, s_dim]

        # 5) Return the next state = x_t + residual
        return x + residual
    
    def sample(self, batch_size, log_r=None):
        """
        Example sampling method that does a single step from (x0, t=0) to s=1.
        Adjust as needed for your actual use-case.
        """
        x_0 = torch.randn(batch_size, self.s_dim, device=self.device) * 3
        #t = torch.zeros(batch_size, 1, device=self.device)
        s = torch.ones(batch_size, 1, device=self.device)
        x_s = self.forward(x_0, s)
        return x_s
    
