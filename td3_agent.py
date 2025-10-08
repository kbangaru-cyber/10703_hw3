# td3_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from buffer import Buffer
from policies import Actor, Critic


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent that matches PPO's interface pattern.
    Simplified to remove unnecessary abstraction layers.
    """
    
    def __init__(self, env_info, lr=3e-4, gamma=0.99, tau=0.005, 
                 batch_size=128, update_every=1, buffer_size=100000, 
                 warmup_steps=5000, policy_noise=0.2, noise_clip=0.5, 
                 exploration_noise=0.1, delay = 2, device="cpu"):
        self.device = torch.device(device)
        
        # Environment info
        self.obs_dim = env_info["obs_dim"]
        self.act_dim = env_info["act_dim"]
        self.act_low = torch.as_tensor(env_info["act_low"], dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(env_info["act_high"], dtype=torch.float32, device=self.device)
        
        # TD3 hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = delay  # Standard TD3 delay
        
        # ================== Problem 2.1.1: TD3 initialization ==================
        ### BEGIN STUDENT SOLUTION - 2.1.1 ###
        self.actor    = Actor(self.obs_dim, self.act_dim, self.act_low, self.act_high).to(self.device)
        self.critic1  = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2  = Critic(self.obs_dim, self.act_dim).to(self.device)

        self.actor_tgt   = Actor(self.obs_dim, self.act_dim, self.act_low, self.act_high).to(self.device)
        self.critic1_tgt = Critic(self.obs_dim, self.act_dim).to(self.device)
        self.critic2_tgt = Critic(self.obs_dim, self.act_dim).to(self.device)

        # Hard copy params
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        self.actor.train(); self.critic1.train(); self.critic2.train()
        self.actor_tgt.eval(); self.critic1_tgt.eval(); self.critic2_tgt.eval()        
        ### END STUDENT SOLUTION  -  2.1.1 ###
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        self._buffer = Buffer(
            size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=device
        )
        
        # Training state
        self.total_steps = 0
        self._update_count = 0
    
    def _to_tensor_action(self, out):
        if torch.is_tensor(out):
            return out
    
        if hasattr(out, "mean"):
            m = out.mean
            try:
                m = m() if callable(m) else m
            except TypeError:
                pass
            if torch.is_tensor(m):
                return m
    
        for attr in ("loc", "mu"):
            if hasattr(out, attr) and torch.is_tensor(getattr(out, attr)):
                return getattr(out, attr)
    
        for holder in ("base", "base_dist", "_distribution", "dist"):
            if hasattr(out, holder):
                inner = getattr(out, holder)
                for attr in ("mean", "loc", "mu"):
                    if hasattr(inner, attr):
                        v = getattr(inner, attr)
                        v = v() if callable(v) else v
                        if torch.is_tensor(v):
                            return v
    
        if hasattr(out, "sample"):
            s = out.sample()
            if torch.is_tensor(s):
                return s
    
        raise TypeError(f"Actor output type {type(out)} could not be converted to a Tensor.")

    
    def act(self, obs):
        """Return action info dict matching PPO's interface"""
        
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # ---------------- Problem 2.2: Exploration noise at action time ----------------
            ### BEGIN STUDENT SOLUTION - 2.2 ###
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.actor(obs_t)
            mu = self._to_tensor_action(out)
            noise = torch.randn_like(mu) * self.exploration_noise
            action = torch.clamp(mu + noise, self.act_low, self.act_high)
            ### END STUDENT SOLUTION  -  2.2 ###
            
            return {
                "action": action.squeeze(0).cpu().numpy()
            }
    
    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        """
        Add transition to buffer and perform updates when ready.
        Matches PPO's step interface.
        """
        # Add to buffer using existing Buffer.add method
        obs_t = torch.as_tensor(transition["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(transition["next_obs"], dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(transition["action"], dtype=torch.float32, device=self.device)
        
        self._buffer.add(
            obs=obs_t,
            next_obs=next_obs_t,
            action=action_t,
            log_probs=0.0,  # Not used in TD3
            reward=float(transition["reward"]),
            done=float(transition["done"]),
            value=0.0,  # Not used in TD3
            advantage=0.0,  # Not used in TD3
            curr_return=0.0,  # Not used in TD3
            iteration=0  # Not used in TD3
        )
        
        self.total_steps += 1
        
        # ---------------- Problem 2.4: Exploration noise at action time ----------------
        ### BEGIN STUDENT SOLUTION - 2.4 ###
        if self.total_steps < self.warmup_steps:
            return {}

        if self.total_steps % self.update_every != 0:
            return {}

        ### END STUDENT SOLUTION - 2.4 ###
        
        # Perform TD3 updates
        return self._perform_update()
    
    def _perform_update(self) -> Dict[str, float]:
        """Perform TD3 updates and return stats"""
        all_stats = []
        
        # Perform updates based on update_every
        num_updates = max(1, self.update_every)
        
        for _ in range(num_updates):
            # Sample batch from buffer
            batch = self._buffer.sample(self.batch_size)
            stats = {}
            
            # ---------------- Problem 2.3: Delayed policy updates ----------------
            ### BEGIN STUDENT SOLUTION - 2.3 ###
            do_actor = (self._update_count % self.policy_delay == 0)
            stats = self._td3_update_step(batch, do_actor_update=do_actor)
            self._update_count += 1
            ### END STUDENT SOLUTION  -  2.3 ###

            all_stats.append(stats)
        
        # Average stats across updates
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}
    
    def _td3_update_step(self, batch, do_actor_update: bool):
        obs      = batch["obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"].unsqueeze(-1)
        next_obs = batch["next_obs"]
        dones    = batch["dones"].unsqueeze(-1)
    
        # ---- 2.1.2: target with policy smoothing (no optimizer step here) ----
        with torch.no_grad():
            out_tgt  = self.actor_tgt(next_obs)
            next_act = self._to_tensor_action(out_tgt)
            noise    = (torch.randn_like(next_act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = torch.clamp(next_act + noise, self.act_low, self.act_high)
    
            q1_tgt = self.critic1_tgt(next_obs, next_act)
            q2_tgt = self.critic2_tgt(next_obs, next_act)
            if q1_tgt.dim() == 1: q1_tgt = q1_tgt.unsqueeze(-1)
            if q2_tgt.dim() == 1: q2_tgt = q2_tgt.unsqueeze(-1)
            min_q  = torch.min(q1_tgt, q2_tgt)
            target_q = rewards + self.gamma * (1.0 - dones) * min_q
    
        # ---- 2.1.3: single critic update (one optimizer step total) ----
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        if current_q1.dim() == 1: current_q1 = current_q1.unsqueeze(-1)
        if current_q2.dim() == 1: current_q2 = current_q2.unsqueeze(-1)
    
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        critic_loss  = critic1_loss + critic2_loss
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
    
        # ---- 2.1.4: delayed actor update + Polyak ----
        if do_actor_update:
            out = self.actor(obs)
            pi  = self._to_tensor_action(out)
            actor_loss = -self.critic1(obs, pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
    
            self._soft_update(self.actor,   self.actor_tgt)
            self._soft_update(self.critic1, self.critic1_tgt)
            self._soft_update(self.critic2, self.critic2_tgt)
        else:
            actor_loss = torch.tensor(0.0, device=self.device)
    
        return {
            "actor_loss":  float(actor_loss.item()),
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
        }

    
    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters using Polyak averaging"""
        # ---------------- Problem 2.1.5: Polyak averaging ----------------
        ### BEGIN STUDENT SOLUTION - 2.1.5 ###
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
          tp.data.copy_( self.tau * lp.data + (1.0 - self.tau) * tp.data )  

        ### END STUDENT SOLUTION  -  2.1.5 ###
