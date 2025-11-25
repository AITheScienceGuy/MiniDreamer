#!/usr/bin/env python

import gymnasium as gym
import ale_py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models import WorldModel, ActorCritic, ACTION_DIM, EMBED_DIM, STOCH_DIM, CLASS_DIM
from utils import get_device, PreprocessAtari

def make_eval_env():
    gym.register_envs(ale_py)
    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                          screen_size=84, terminal_on_life_loss=False, 
                                          grayscale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4) 
    env = PreprocessAtari(env, size=(64, 64))
    return env

def run_benchmark(n_episodes=50):
    device = get_device()
    env = make_eval_env() 
    
    print("Loading models...")
    wm = WorldModel().to(device)
    try:
        wm.load_state_dict(torch.load("world_model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: world_model.pth not found. Train first!")
        return
    wm.eval()

    agent = ActorCritic().to(device)
    try:
        agent.load_state_dict(torch.load("policy.pth", map_location=device))
    except FileNotFoundError:
        print("Error: policy.pth not found. Train first!")
        return
    agent.eval()

    scores = []
    
    print(f"Running benchmark over {n_episodes} episodes...")
    for i in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        score = 0
        
        # Initialize RSSM state
        h = torch.zeros(1, EMBED_DIM, device=device)
        z_flat = torch.zeros(1, STOCH_DIM * CLASS_DIM, device=device)
        
        # Start with No-Op action
        action_idx = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            
            with torch.no_grad():
                # 1. Encode Observation
                embed = wm.encoder(obs_tensor)
                
                # 2. One-hot previous action
                action_one_hot = F.one_hot(torch.tensor([action_idx], device=device), ACTION_DIM).float()
                
                # 3. Step World Model (Recurrent Update)
                h, logits = wm.rssm.step(h, z_flat, action_one_hot, embed)
                z, z_flat = wm.get_stochastic_state(logits, hard=True)
                
                # 4. Construct Features for Agent
                feat = wm.rssm.get_feat(h, z_flat)
                
                # 5. Agent selects action (Argmax for evaluation usually, or low temp)
                action_probs = agent.get_action(feat, temperature=0.1)
                action_idx = torch.argmax(action_probs).item()
            
            obs, reward, term, trunc, _ = env.step(action_idx)
            done = term or trunc
            score += reward
            
        scores.append(score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\n--- Benchmark Results ({n_episodes} Episodes) ---")
    print(f"Average Score: {avg_score:.2f} +/- {std_score:.2f}")
    print(f"Min: {np.min(scores)} | Max: {np.max(scores)}")
    
    # Space Invaders Reference Scores (Approximate)
    if avg_score > 500:
        print("RESULT: Excellent! Likely learned shielding/leading shots.")
    elif avg_score > 250:
        print("RESULT: Good. Can hit aliens but dies eventually.")
    elif avg_score > 100:
        print("RESULT: Okay. Better than random, but basic.")
    else:
        print("RESULT: Poor. (Random agent ~150)")

if __name__ == "__main__":
    run_benchmark()
