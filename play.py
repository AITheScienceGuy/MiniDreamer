#!/usr/bin/env python

import gymnasium as gym
import torch
import numpy as np
import time
from models import WorldModel, ActorCritic
from utils import get_device, PreprocessAtari

def watch_agent():
    device = get_device()

    # Setup Environment EXACTLY like training
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", frameskip=1)
    
    env = gym.wrappers.AtariPreprocessing(
        env, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=84, 
        terminal_on_life_loss=False, 
        grayscale_obs=False
    )
    
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    # Use the wrapper from utils.py to handle resizing (4, 84, 84, 3) -> (12, 64, 64)
    env = PreprocessAtari(env, size=(64, 64))
    
    print("Loading models...")
    wm = WorldModel().to(device)
    wm.load_state_dict(torch.load("world_model.pth", map_location=device))
    wm.eval()

    agent = ActorCritic().to(device)
    agent.load_state_dict(torch.load("policy.pth", map_location=device))
    agent.eval()

    print("Starting Game...")
    
    for episode in range(5): 
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        # Initialize RSSM State
        h = torch.zeros(1, 1024, device=device)
        z_flat = torch.zeros(1, 1024, device=device) 
        
        # Initialize action for the first step (No-Op = 0)
        action = 0
        
        while not done:
            # Obs is already preprocessed to (12, 64, 64) by the wrappers
            # Just convert to tensor and add batch dimension
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            
            with torch.no_grad():
                # Encode Image
                embed = wm.encoder(obs_tensor)
                
                # Prepare Action (One-Hot)
                action_t = torch.tensor([action], device=device)
                action_one_hot = torch.nn.functional.one_hot(action_t, 6).float()
                
                # Step World Model
                h, logits = wm.rssm.step(h, z_flat, action_one_hot, embed=embed)
                z, z_flat = wm.get_stochastic_state(logits, hard=True)
                
                # Get Features for Agent
                feat = wm.rssm.get_feat(h, z_flat)
                
                # Agent Selects Action
                action_probs = agent.get_action(feat, temperature=0.1)
                action = torch.argmax(action_probs).item()
            
            # Step Environment
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
            
            # Slow down slightly to make it watchable
            time.sleep(0.03)

        print(f"Episode {episode+1}: Total Reward {total_reward}")

    env.close()

if __name__ == "__main__":
    watch_agent()
