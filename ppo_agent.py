# ppo_agent.py
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from buffer import Buffer
from policies import ActorCritic

class PPOAgent:
    def __init__(self, env_info, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 update_epochs=10, minibatch_size=64, rollout_steps=4096, device="cpu"):
        self.device = torch.device(device)
        policy = ActorCritic(
            env_info["obs_dim"],
            env_info["act_dim"],
            env_info["act_low"],
            env_info["act_high"],
            hidden=(64, 64),
        )
        self.actor = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        
        # PPO with KL penalty parameters
        self.beta = .5  # Initial KL penalty coefficient
        self.target_kl = 0.01  # Target KL divergence
        
        # Internal state for rollout collection
        self._curr_policy_rollout = []
        self._rollout_buffer = Buffer(
            size=rollout_steps*50,
            obs_dim=policy.obs_dim,
            act_dim=policy.act_dim,
            device=device
        )
        self._steps_collected_with_curr_policy = 0
        self._policy_iteration = 1
        self.use_kl_loss = False
        self.exp_16_mode = "none"
    
    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist, value = self.actor(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if log_prob.ndim > 1:
                log_prob = log_prob.sum(dim=-1)
            return {
                "action": action.squeeze(0).cpu().numpy(),
                "log_prob": float(log_prob.squeeze(0).item()),
                "value": float(value.squeeze(0).item())
            }


    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        """
        PPO-specific step: collect transitions until rollout is full, then update.
        
        transition should contain:
        - obs, action, reward, next_obs, done, truncated
        - log_prob, value (from act() call)
        """
        # Add to current rollout
        self._curr_policy_rollout.append(transition.copy())
        self._steps_collected_with_curr_policy += 1
        stop = transition['done'] or transition['truncated']
        # ---------------- Problem 1.3.1: PPO Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.3.1 ###
        ret = {}

        if stop:
            advantages, returns = self._compute_gae(self._curr_policy_rollout)
            batch = self._prepare_batch(advantages, returns)
            self._rollout_buffer.add_batch(batch)
            self._curr_policy_rollout = []

        if self._steps_collected_with_curr_policy >= self.rollout_steps:
            buf = self._rollout_buffer.sample(filter={"iteration": [self._policy_iteration]})
            # print("DEBUG PPO: buffered steps this iter =", buf["obs"].shape[0])
            buf_n = buf["obs"].shape[0]
            
            if buf_n >= self.rollout_steps:
                stats = self._perform_update()
                if stats:
                    ret = stats
                self._policy_iteration += 1
                self._steps_collected_with_curr_policy = 0

        ### END STUDENT SOLUTION - 1.3.1 ###

        return ret  # Leave this as an empty dictionary if no update is performed

    def _perform_update(self) -> Dict[str, float]:
        """Perform PPO update using collected rollout"""
        all_stats = []

        # To log metrics correctly, make sure you have the following lines in this function
        # loss, stats = self._ppo_loss(minibatch)
        # all_stats.append(stats)
        
        # ---------------- Problem 1.3.2: PPO Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.3.2 ###
        batch = self._rollout_buffer.sample(filter={"iteration": [self._policy_iteration]})
        if batch["obs"].shape[0] == 0:
            return {} 
        adv = batch["advantages"]
        batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        mode = getattr(self, "exp_16_mode", "none")

        if mode == "none":
          B = batch["obs"].shape[0]
          mb = self.minibatch_size

          for epoch in range(self.update_epochs):
              perm = torch.randperm(B, device=self.device)
              for start in range(0, B, mb):
                  idx = perm[start:start+mb]
                  minibatch = {
                      "obs":        batch["obs"][idx],
                      "actions":    batch["actions"][idx],
                      "log_probs":  batch["log_probs"][idx],
                      "advantages": batch["advantages"][idx],
                      "returns":    batch["returns"][idx],
                  }

                  # Compute loss & stats, then optimize
                  loss, stats = self._ppo_loss(minibatch)
                  self.optimizer.zero_grad(set_to_none=True)
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                  self.optimizer.step()

                  all_stats.append(stats)

        ### EXPERIMENT 1.6 CODE ###
        
        if mode == "full":
            all_batch = self._rollout_buffer.sample()
            if all_batch["obs"].shape[0] > 0:
                adv_all = all_batch["advantages"]
                all_batch["advantages"] = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

                B_all = all_batch["obs"].shape[0]
                mb = self.minibatch_size
                for _ in range(self.update_epochs):
                    perm = torch.randperm(B_all, device=self.device)
                    for start in range(0, B_all, mb):
                        idx = perm[start:start+mb]
                        minibatch = {
                            "obs":        all_batch["obs"][idx],
                            "actions":    all_batch["actions"][idx],
                            "log_probs":  all_batch["log_probs"][idx],
                            "advantages": all_batch["advantages"][idx],
                            "returns":    all_batch["returns"][idx],
                        }
                        loss, stats = self._ppo_loss(minibatch)
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        all_stats.append(stats)

        elif mode == "half":
            curr_batch = self._rollout_buffer.sample(filter={"iteration": [self._policy_iteration]})
            all_batch  = self._rollout_buffer.sample()
            B_curr = curr_batch["obs"].shape[0]
            B_all  = all_batch["obs"].shape[0]
            mb = self.minibatch_size
            epochs = self.update_epochs

            curr_adv = curr_batch["advantages"]
            curr_batch["advantages"] = (curr_adv - curr_adv.mean()) / (curr_adv.std() + 1e-8)
            all_adv = all_batch["advantages"]
            all_batch["advantages"] = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

            for _ in range(epochs):
                perm_curr = torch.randperm(B_curr, device=self.device)
                perm_all  = torch.randperm(B_all,  device=self.device)
                p1 = p2 = 0
                while p1 < B_curr and p2 < B_all:
                    h = mb // 2
                    idx_curr = perm_curr[p1:p1+h];  p1 += h
                    idx_all  = perm_all[p2:p2+(mb-h)];  p2 += (mb - h)

                    if idx_curr.numel() < h:
                        perm_curr = torch.randperm(B_curr, device=self.device); p1 = 0
                        idx_curr = perm_curr[p1:p1+h]; p1 += h
                    if idx_all.numel() < (mb - h):
                        perm_all = torch.randperm(B_all, device=self.device); p2 = 0
                        idx_all = perm_all[p2:p2+(mb - h)]; p2 += (mb - h)

                    minibatch = {}
                    for key in ["obs","actions","log_probs","advantages","returns"]:
                        minibatch[key] = torch.cat([curr_batch[key][idx_curr], all_batch[key][idx_all]], dim=0)

                    loss, stats = self._ppo_loss(minibatch)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    all_stats.append(stats)

          ### EXPERIMENT 1.6 CODE END ###
    
        ### END STUDENT SOLUTION - 1.3.2 ###
        
        # ---------------- Problem 1.4.2: KL Divergence Beta Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.4.2 ###

        if all_stats:
          kl = float(np.mean([s["kl"] for s in all_stats]))
          if kl > 1.5 * self.target_kl:
              self.beta *= 1.5
          elif kl < self.target_kl / 1.5:
              self.beta /= 1.5
          self.beta = float(np.clip(self.beta, 1e-4, 100.0))

        ### END STUDENT SOLUTION - 1.4.2 ###
        
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}
        
    def _compute_gae(self, rollout) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rollout)
        rewards = np.array([t["reward"] for t in rollout], dtype=np.float32)
        values = np.array([t["value"] for t in rollout], dtype=np.float32)
        dones = np.array([t["done"] for t in rollout], dtype=np.bool_)
        truncated_flags = np.array([t.get("truncated", False) for t in rollout], dtype=np.bool_)

        # Bootstrap value for the step after the last element
        next_obs = rollout[-1]["next_obs"]
        with torch.no_grad():
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, final_v = self.actor(obs_t)
            final_v = float(final_v.squeeze(0).item())

        advantages = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        gamma = self.gamma
        lam = self.gae_lambda

        for t in reversed(range(T)):
            if t == T - 1:
                # if last step was truncated (time-limit), we can bootstrap using final_v
                if truncated_flags[t] and (not dones[t]):
                    next_nonterminal = 1.0
                    next_value = final_v
                else:
                    # if last step was terminal (done==True) we don't bootstrap
                    next_nonterminal = 0.0
                    next_value = 0.0
            else:
                next_nonterminal = 0.0 if dones[t+1] else 1.0
                next_value = values[t+1]

            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns
    
    def _ppo_loss(self, batch):
        """Standard PPO loss computation"""
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        dist, values = self.actor(obs)
        log_probs = dist.log_prob(actions)
        if log_probs.ndim > 1:
            log_probs = log_probs.sum(dim=-1)

        # ---------------- Problem 1.4.2: KL Divergence Policy Loss ----------------
        ### BEGIN STUDENT SOLUTION - 1.4.2 ###
        if self.use_kl_loss:
          ratio = torch.exp(log_probs - old_log_probs)
          approx_kl_term = (old_log_probs - log_probs)
          approx_kl = approx_kl_term.mean()
          unclipped_obj = ratio * advantages
          policy_loss = -(unclipped_obj.mean() - self.beta * approx_kl)

        ### END STUDENT SOLUTION - 1.4.2 ###
        
        # ---------------- Problem 1.1.1: PPO Clipped Surrogate Objective Loss ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.1 ###
        else:
          ratio = torch.exp(log_probs - old_log_probs)

          clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
          surrogate1 = ratio * advantages
          surrogate2 = clipped_ratio * advantages
          policy_loss = -torch.min(surrogate1, surrogate2).mean()

        ### END STUDENT SOLUTION - 1.1.1 ###
        
        
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)

        entropy_loss = 0 # Placeholder
        value_loss = 0 # Placeholder

        # ---------------- Problem 1.1.2: PPO Total Loss (Include Entropy Bonus and Value Loss) ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.2 ###

        value_loss = torch.mean((values.squeeze(-1) - returns)**2) if values.ndim > 1 else torch.mean((values - returns)**2)
        entropy_loss = -entropy.mean()
        total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        ### END STUDENT SOLUTION - 1.1.2 ###

        # Stats
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
        
        return total_loss, {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(-entropy_loss.item()),
            "kl": float(approx_kl.item()),
            "clipfrac": float(clipfrac.item()),
        }
        
    def _prepare_batch(self, advantages, returns):
        """Collate the current rollout into a batch for the buffer"""
        obs = torch.stack([torch.as_tensor(t["obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        next_obs = torch.stack([torch.as_tensor(t["next_obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        actions = torch.stack([torch.as_tensor(t["action"], dtype=torch.float32) for t in self._curr_policy_rollout])
        log_probs = torch.tensor([t["log_prob"] for t in self._curr_policy_rollout], dtype=torch.float32)
        values = torch.tensor([t["value"] for t in self._curr_policy_rollout], dtype=torch.float32)
        rewards = torch.tensor([t["reward"] for t in self._curr_policy_rollout], dtype=torch.float32)
        
        return {
            "obs": obs.to(self.device),
            "next_obs": next_obs.to(self.device),
            "actions": actions.to(self.device),
            "log_probs": log_probs.to(self.device),
            "rewards": rewards.to(self.device),
            "values": values.to(self.device),
            "dones": torch.tensor([t["done"] for t in self._curr_policy_rollout], dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            "iteration": torch.full((len(self._curr_policy_rollout),), self._policy_iteration, dtype=torch.int32, device=self.device)
        }