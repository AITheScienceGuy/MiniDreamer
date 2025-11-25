#!/usr/bin/env python

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.distributions import Categorical, OneHotCategorical
from torch.amp import GradScaler, autocast 

from utils import make_vector_env, get_device
from models import WorldModel, ActorCritic, ACTION_DIM, EMBED_DIM, STOCH_DIM, CLASS_DIM
from buffer import ReplayBuffer

# --- Hyperparameters ---
NUM_ENVS = 32                
TOTAL_ITERATIONS = 1600      
STEPS_PER_ITER = 4000        
WM_EPOCHS = 1               
AGENT_EPOCHS = 1           
BATCH_SIZE = 64              
SEQ_LEN = 50                 
BURN_IN_STEPS = 5  
IMAGINE_HORIZON = 15         
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_SCALE = 1e-3         

# Weighting
W_RECON = 1.0
W_REWARD = 1.0
W_DONE = 1.0

writer = SummaryWriter(log_dir="runs/SpaceInvaders")
global_step_wm = 0
global_step_agent = 0

def get_epsilon(iteration):
    return max(0.1, 1.0 - iteration / 500.0)

def get_kl_weight(iteration):
    if iteration < 50: return 0.0
    return min(0.15, 0.01 + (iteration - 50) * 0.05)

def log_visualizations(model, b_obs, b_act, global_step, device):
    model.eval()
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): 
            T_vis = 16
            seq_obs = b_obs[0, :T_vis].to(device)
            seq_act = b_act[0, :T_vis].to(device)
            
            embeds_flat = model.encoder(seq_obs) 
            embeds = embeds_flat.unsqueeze(0)
            h = torch.zeros(1, EMBED_DIM, device=device)
            z_flat = torch.zeros(1, STOCH_DIM*CLASS_DIM, device=device)
            
            recon_frames = []
            dream_frames = []
            
            # Posterior
            for t in range(T_vis):
                act = torch.zeros(1, ACTION_DIM, device=device) if t==0 else F.one_hot(seq_act[t-1:t], ACTION_DIM).float()
                h, logits = model.rssm.step(h, z_flat, act, embeds[:, t])
                z, z_flat = model.get_stochastic_state(logits, hard=True)
                rec = model.decoder(h, z_flat)
                recon_frames.append(rec)

            # Prior (Dream)
            h = torch.zeros(1, EMBED_DIM, device=device)
            z_flat = torch.zeros(1, STOCH_DIM*CLASS_DIM, device=device)
            warmup = 5
            
            for t in range(T_vis):
                act = torch.zeros(1, ACTION_DIM, device=device) if t==0 else F.one_hot(seq_act[t-1:t], ACTION_DIM).float()
                if t < warmup:
                    h, logits = model.rssm.step(h, z_flat, act, embeds[:, t])
                else:
                    h, logits = model.rssm.step(h, z_flat, act, embed=None)
                z, z_flat = model.get_stochastic_state(logits, hard=True)
                dream = model.decoder(h, z_flat)
                dream_frames.append(dream)

            real_vis = seq_obs[:, -3:].cpu().float() 
            recon_vis = torch.cat(recon_frames, dim=0)[:, -3:].cpu().float()
            dream_vis = torch.cat(dream_frames, dim=0)[:, -3:].cpu().float()
            
            combined = torch.cat([real_vis, recon_vis, dream_vis], dim=0)
            grid = torchvision.utils.make_grid(combined, nrow=T_vis, padding=2)
            writer.add_image('WM/Real_Recon_Dream', grid, global_step)
    model.train()

def collect_data(buffer, steps, wm=None, agent=None, epsilon=1.0, device="cpu"):
    print(f"Collecting {steps} frames (Eps: {epsilon:.2f})...")
    envs = make_vector_env(NUM_ENVS)
    obs, _ = envs.reset()
    
    prev_h = torch.zeros(NUM_ENVS, EMBED_DIM, device=device)
    prev_z_flat = torch.zeros(NUM_ENVS, STOCH_DIM * CLASS_DIM, device=device)
    
    if wm: wm.eval()
    if agent: agent.eval()
    
    use_amp = (device.type == "cuda")
    loops = steps // NUM_ENVS
    
    for _ in tqdm(range(loops), desc="Collecting Data", leave=False):
        if np.random.rand() < epsilon or agent is None:
            actions = envs.action_space.sample() 
        else:
            with torch.no_grad():
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): 
                    obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
                    embed = wm.encoder(obs_tensor)
                    
                    action_one_hot = F.one_hot(torch.tensor(prev_actions, device=device), ACTION_DIM).float() if 'prev_actions' in locals() else torch.zeros(NUM_ENVS, ACTION_DIM, device=device)
                    
                    h, logits = wm.rssm.step(prev_h, prev_z_flat, action_one_hot, embed)
                    z, z_flat = wm.get_stochastic_state(logits, hard=True)
                    feat = wm.rssm.get_feat(h, z_flat)
                    
                    action_probs = agent.get_action(feat, temperature=1.0)
                    dist = Categorical(probs=action_probs)
                    actions = dist.sample().cpu().numpy()
                    
                    prev_h = h
                    prev_z_flat = z_flat
        
        prev_actions = actions 
        next_obs, rewards, terms, truncs, _ = envs.step(actions)
        dones = np.logical_or(terms, truncs).astype(np.float32)
        
        if np.any(dones):
             mask = torch.tensor(1.0 - dones, device=device).unsqueeze(1)
             prev_h = prev_h * mask
             prev_z_flat = prev_z_flat * mask

        buffer.add_batch(obs, actions, rewards, dones)
        obs = next_obs
    envs.close()

def train_world_model(buffer, model, optimizer, scaler, device, epochs, kl_scale): 
    global global_step_wm
    model.train()
    steps_per_epoch = (len(buffer) // BATCH_SIZE) // 5 
    if steps_per_epoch < 1: steps_per_epoch = 1
    
    use_amp = (device.type == "cuda")
    
    for epoch in tqdm(range(epochs), desc="WM Epochs", leave=False):
        for _ in range(steps_per_epoch):
            batch = buffer.sample_sequence(BATCH_SIZE, recent_only_pct=0.5)
            if batch is None: break
            b_obs, b_act, b_rew, b_don = batch
            
            b_obs, b_act = b_obs.to(device), b_act.to(device)
            b_rew, b_don = b_rew.to(device), b_don.to(device)
            
            optimizer.zero_grad() 
            
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): 
                B, T, C, H, W = b_obs.shape
                flat_obs = b_obs.view(B*T, C, H, W)
                embeds = model.encoder(flat_obs).view(B, T, -1)
                
                prev_h = torch.zeros(B, EMBED_DIM, device=device)
                prev_z_flat = torch.zeros(B, STOCH_DIM * CLASS_DIM, device=device)
                
                loss_recon = 0; loss_kl = 0; loss_rew = 0; loss_done = 0
                
                for t in range(T):
                    act = torch.zeros(B, ACTION_DIM, device=device) if t==0 else F.one_hot(b_act[:, t-1], ACTION_DIM).float()
                    
                    h, post_logits = model.rssm.step(prev_h, prev_z_flat, act, embeds[:, t])
                    _, prior_logits = model.rssm.step(prev_h, prev_z_flat, act, embed=None)
                    z, z_flat = model.get_stochastic_state(post_logits)
                    
                    recon = model.decoder(h, z_flat)
                    
                    pixel_mask = (b_obs[:, t] > 0.01).float()
                    pixel_weights = 1.0 + 10.0 * pixel_mask
                    sq_err = (recon - b_obs[:, t]) ** 2
                    loss_recon += (sq_err * pixel_weights).mean()

                    q = OneHotCategorical(logits=post_logits)
                    p = OneHotCategorical(logits=prior_logits)
                    kl = torch.distributions.kl_divergence(q, p)
                    loss_kl += torch.maximum(kl, torch.tensor(1.0, device=device)).mean()
                    
                    feat = model.rssm.get_feat(h, z_flat)
                    pred_r = model.reward_head(feat)
                    pred_d = model.done_head(feat)
                    loss_rew += F.mse_loss(pred_r, b_rew[:, t])
                    loss_done += F.binary_cross_entropy_with_logits(pred_d, b_don[:, t])
                    
                    prev_h = h
                    prev_z_flat = z_flat
                
                loss_recon /= T; loss_kl /= T; loss_rew /= T; loss_done /= T
                loss = (W_RECON * loss_recon + kl_scale * loss_kl + W_REWARD * loss_rew + W_DONE * loss_done)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            scaler.step(optimizer)
            scaler.update()
            
            global_step_wm += 1
            if global_step_wm % 500 == 0:
                writer.add_scalar("WM/Total", loss.item(), global_step_wm)
                writer.add_scalar("WM/Recon", loss_recon.item(), global_step_wm)
                writer.add_scalar("WM/KL", loss_kl.item(), global_step_wm)
                log_visualizations(model, b_obs, b_act, global_step_wm, device)

    torch.save(model.state_dict(), "world_model.pth")

def compute_lambda_returns(rewards, values, dones, gamma=0.99, lambda_=0.95):
    returns = []
    last_lambda_ret = values[-1] 
    
    for t in reversed(range(len(rewards))):
        r_t = rewards[t]
        v_next = values[t+1]
        d_t = dones[t]
        
        one_step = r_t + gamma * (1.0 - d_t) * v_next
        lambda_ret = (1 - lambda_) * one_step + lambda_ * (r_t + gamma * (1.0 - d_t) * last_lambda_ret)
        returns.insert(0, lambda_ret)
        last_lambda_ret = lambda_ret
        
    return torch.stack(returns)

def train_policy_dreamer(buffer, wm, agent, optimizer, scaler, device, epochs): 
    global global_step_agent
    wm.eval()
    agent.train()
    wm.requires_grad_(False) 
    
    use_amp = (device.type == "cuda")

    for _ in tqdm(range(epochs), desc="Agent Epochs", leave=False):
        batch = buffer.sample_sequence(BATCH_SIZE, recent_only_pct=0.5)
        if batch is None: continue
        b_obs, b_act, _, _ = batch # Use real actions for burn-in
        
        # Move relevant parts to device
        context_obs = b_obs[:, :BURN_IN_STEPS].to(device)
        context_act = b_act[:, :BURN_IN_STEPS].to(device)
        
        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): 
            # --- CONTEXT BURN-IN (POSTERIOR) ---
            with torch.no_grad():
                # Encode context frames
                B, T_ctx, C, H, W = context_obs.shape
                flat_obs = context_obs.view(B*T_ctx, C, H, W)
                embeds = wm.encoder(flat_obs).view(B, T_ctx, -1)
                
                # Run Posterior on real data to get valid start state
                h = torch.zeros(B, EMBED_DIM, device=device)
                z_flat = torch.zeros(B, STOCH_DIM * CLASS_DIM, device=device)
                
                for t in range(T_ctx):
                    # Use real previous actions
                    act = torch.zeros(B, ACTION_DIM, device=device) if t==0 else F.one_hot(context_act[:, t-1], ACTION_DIM).float()
                    h, logits = wm.rssm.step(h, z_flat, act, embeds[:, t])
                    z, z_flat = wm.get_stochastic_state(logits, hard=True)
                
                # h and z_flat are now "warmed up" with T_ctx steps of reality
            
            # --- IMAGINATION LOOP (PRIOR) ---
            list_rewards, list_values, list_dones = [], [], []
            list_log_probs, list_entropies = [], []
            
            curr_h, curr_z_flat = h, z_flat
            
            # Value of the start state (from posterior)
            feat_start = wm.rssm.get_feat(curr_h, curr_z_flat).detach()
            list_values.append(agent.get_value(feat_start))
            
            for t in range(IMAGINE_HORIZON):
                feat = wm.rssm.get_feat(curr_h, curr_z_flat).detach() 
                
                # Agent Step
                logits_act = agent.get_action_logits(feat)
                dist = OneHotCategorical(logits=logits_act)
                action = dist.sample() 
                
                log_prob = dist.log_prob(action)
                list_log_probs.append(log_prob)
                list_entropies.append(dist.entropy())
                
                # World Model Step (Prior)
                curr_h, logits_next = wm.rssm.step(curr_h, curr_z_flat, action, embed=None)
                z_next, z_flat_next = wm.get_stochastic_state(logits_next, hard=True)
                
                feat_next = wm.rssm.get_feat(curr_h, z_flat_next)
                
                pred_rew = wm.reward_head(feat_next)
                pred_done = wm.done_head(feat_next) # We predict it but don't use it for discount yet
                pred_val = agent.get_value(feat_next)
                
                list_rewards.append(pred_rew)
                list_values.append(pred_val)
                # Use soft discount 0.0 (alive) for stability
                list_dones.append(torch.zeros_like(pred_done)) 
                
                curr_z_flat = z_flat_next
                
            rewards = torch.stack(list_rewards) 
            values = torch.stack(list_values)   
            dones = torch.stack(list_dones)     
            log_probs = torch.stack(list_log_probs) 
            entropies = torch.stack(list_entropies) 
            
            # Compute Targets
            lambda_targets = compute_lambda_returns(rewards, values, dones, GAMMA, LAMBDA).detach()
            
            # Critic Loss
            critic_loss = 0.5 * F.mse_loss(values[:-1], lambda_targets)
            
            # Actor Loss
            advantage = (lambda_targets - values[:-1]).detach()
            # Normalize Advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = advantage.clamp(-5.0, 5.0)
            
            actor_loss = - (log_probs * advantage.squeeze(-1)).mean() 
            entropy_loss = - entropies.mean()
            
            total_loss = actor_loss + critic_loss + (ENTROPY_SCALE * entropy_loss)
        
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 100.0)
        scaler.step(optimizer)
        scaler.update()
        
        global_step_agent += 1
        if global_step_agent % 100 == 0:
             writer.add_scalar("Agent/Actor", actor_loss.item(), global_step_agent)
             writer.add_scalar("Agent/Critic", critic_loss.item(), global_step_agent)
             writer.add_scalar("Agent/Value", values.mean().item(), global_step_agent)
             writer.add_scalar("Agent/Entropy", entropies.mean().item(), global_step_agent)

    wm.requires_grad_(True)
    torch.save(agent.state_dict(), "policy.pth")

def main():
    device = get_device()
    buffer = ReplayBuffer(capacity=10000, num_envs=NUM_ENVS, seq_len=SEQ_LEN)
    
    wm = WorldModel().to(device)
    agent = ActorCritic().to(device)
    
    opt_wm = optim.Adam(wm.parameters(), lr=3e-4)
    opt_agent = optim.Adam(agent.parameters(), lr=1e-4)
    
    use_cuda = (device.type == "cuda")
    scaler_wm = GradScaler(enabled=use_cuda)
    scaler_agent = GradScaler(enabled=use_cuda)
    
    collect_data(buffer, steps=100000 // NUM_ENVS, device=device)
    
    for i in range(TOTAL_ITERATIONS):
        print(f"Iter {i+1}/{TOTAL_ITERATIONS}")
        epsilon = get_epsilon(i)
        
        collect_data(buffer, STEPS_PER_ITER, wm, agent, epsilon, device)
        train_world_model(buffer, wm, opt_wm, scaler_wm, device, WM_EPOCHS, get_kl_weight(i))
        train_policy_dreamer(buffer, wm, agent, opt_agent, scaler_agent, device, AGENT_EPOCHS)

if __name__ == "__main__":
    main()
