"""A4.20.1 stat tracker copy for the AvgKNN correction-clipping experiment.

Call with exact rollout group IDs, not decoded prompt strings. Different
rollout groups may have identical prompts and must remain independent.
"""

import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std, adv_clip_max, temperature, energy_max,
                 energy_min, reward_std_epsilon, reward_fn=None):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        self.reward_fn = reward_fn
        self.adv_clip_max = adv_clip_max
        self.energy_max = energy_max
        self.energy_min = energy_min
        self.temperature = temperature
        self.reward_std_epsilon = reward_std_epsilon

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False, hard_gating=False, diagnostics=None):
        """Update statistics, optionally copying per-sample intermediates for logging.

        ``diagnostics`` never changes the returned training values. Its arrays
        retain the input reward shape and order; advantage_clipped is captured
        before temperature scaling, and energy_before_clip after rescaling.
        """
        if hard_gating:
            raise ValueError("AvgKNN correction clipping uses K * softmax - 1, not hard gating")
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)
        energies = np.zeros_like(rewards) if exp else None
        energies_stds = np.zeros_like(rewards) if exp else None
        if diagnostics is not None:
            diagnostics["advantage_clipped"] = np.zeros_like(rewards)
            if exp:
                diagnostics["softmax_prob"] = np.zeros_like(rewards)
                diagnostics["energy_before_clip"] = np.zeros_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + self.reward_std_epsilon
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + self.reward_std_epsilon
            prompt_adv = (prompt_rewards - mean) / std
            advantages[prompts == prompt] = prompt_adv
            if exp:
                k = len(prompt_rewards)
                assert k >= 2, "Group size must be at least 2 for softmax energy"
                prompt_adv_clipped = (
                    np.clip(prompt_adv, -self.adv_clip_max, self.adv_clip_max)
                    if self.adv_clip_max is not None else prompt_adv
                )
                if diagnostics is not None:
                    diagnostics["advantage_clipped"][prompts == prompt] = prompt_adv_clipped
                prompt_adv_clipped = prompt_adv_clipped / self.temperature
                exp_prompt_adv = np.exp(prompt_adv_clipped - np.max(prompt_adv_clipped, axis=0, keepdims=True))
                probs = exp_prompt_adv / np.sum(exp_prompt_adv, axis=0, keepdims=True)
                raw_energies = probs - (1.0 / k)

                energy_stds = np.std(raw_energies, axis=0, keepdims=True)
                energies_reranged = raw_energies * k

                if diagnostics is not None:
                    diagnostics["softmax_prob"][prompts == prompt] = probs
                    diagnostics["energy_before_clip"][prompts == prompt] = energies_reranged
                energies_clipped = np.clip(energies_reranged, self.energy_min, self.energy_max)
                energies[prompts == prompt] = energies_clipped
                energies_stds[prompts == prompt] = energy_stds
            elif diagnostics is not None:
                # This tracker does not clip advantages outside energy mode.
                diagnostics["advantage_clipped"][prompts == prompt] = prompt_adv
        if diagnostics is not None:
            # Copies keep later logging/serialization independent of training arrays.
            diagnostics["advantage_raw"] = advantages.copy()
            if exp:
                diagnostics["energy_used"] = energies.copy()
        if exp:
            return advantages, energies, energies_stds
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


