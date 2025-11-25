#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RSSM CONFIG ---
EMBED_DIM = 1024         
STOCH_DIM = 32           
CLASS_DIM = 16           
ACTION_DIM = 6
INPUT_CHANNELS = 12 

class LayerNormGRUCell(nn.Module):
    """
    GRU Cell with Layer Normalization for stability.
    Standard in Dreamer-style architectures.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Concatenated weights for input (x) and hidden (h)
        # We process x and h separately to apply LayerNorm before combination
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)
        
    def forward(self, x, h):
        # Calculate gates
        # i_r, i_z, i_n = x @ W_ih
        # h_r, h_z, h_n = h @ W_hh
        
        ih = self.ln_ih(self.weight_ih(x))
        hh = self.ln_hh(self.weight_hh(h))
        
        # Split into reset, update, and new gates
        i_r, i_z, i_n = ih.chunk(3, dim=-1)
        h_r, h_z, h_n = hh.chunk(3, dim=-1)
        
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        
        h_next = (1 - update_gate) * new_gate + update_gate * h
        return h_next

class VisualEncoder(nn.Module):
    """ Encodes 64x64 image into a flat embedding vector """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, 4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 4x4
            nn.ReLU(),
            nn.Flatten() # 256 * 4 * 4 = 4096
        )
        self.out = nn.Linear(4096, 1024)
    
    def forward(self, x):
        y = self.net(x)
        return self.out(y)

class VisualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_in = nn.Linear(EMBED_DIM + STOCH_DIM * CLASS_DIM, 4096)
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, INPUT_CHANNELS, 4, stride=2, padding=1), # 64x64
            nn.Sigmoid()
        )

    def forward(self, h, z_flat):
        x = torch.cat([h, z_flat], dim=-1)
        x = self.net_in(x)
        x = x.view(-1, 256, 4, 4)
        return self.net(x)

class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = LayerNormGRUCell(ACTION_DIM + STOCH_DIM * CLASS_DIM, EMBED_DIM)
        
        # Posterior Network (Training)
        self.posterior_net = nn.Sequential(
            nn.Linear(EMBED_DIM + 1024, 512),
            nn.ELU(),
            nn.Linear(512, STOCH_DIM * CLASS_DIM)
        )
        
        # Prior Network (Dreaming)
        self.prior_net = nn.Sequential(
            nn.Linear(EMBED_DIM, 512),
            nn.ELU(),
            nn.Linear(512, STOCH_DIM * CLASS_DIM)
        )

    def get_feat(self, h, z_flat):
        return torch.cat([h, z_flat], dim=-1)

    def step(self, prev_h, prev_z_flat, action, embed=None):
        gru_input = torch.cat([prev_z_flat, action], dim=-1)
        h = self.gru(gru_input, prev_h)
        
        if embed is not None:
            post_in = torch.cat([h, embed], dim=-1)
            logits = self.posterior_net(post_in)
        else:
            logits = self.prior_net(h)
            
        logits = logits.view(-1, STOCH_DIM, CLASS_DIM)
        return h, logits

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisualEncoder()
        self.decoder = VisualDecoder()
        self.rssm = RSSM()
        
        feat_dim = EMBED_DIM + STOCH_DIM * CLASS_DIM
        self.reward_head = nn.Linear(feat_dim, 1)
        self.done_head = nn.Linear(feat_dim, 1)

    def get_stochastic_state(self, logits, temperature=1.0, hard=False):
        # Sample discrete latents
        z = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        z_flat = z.view(z.size(0), -1)
        return z, z_flat

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        feat_dim = EMBED_DIM + STOCH_DIM * CLASS_DIM
        
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, ACTION_DIM)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def get_action_logits(self, feature):
        return self.actor(feature)

    def get_action(self, feature, temperature=1.0):
        logits = self.actor(feature)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs

    def get_value(self, feature):
        return self.critic(feature)
