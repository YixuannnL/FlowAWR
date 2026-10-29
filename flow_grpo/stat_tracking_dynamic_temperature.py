import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std=False, adv_clip_max=None, reward_fn=None, temperature=1.0, energy_max=3.0, energy_min=-3.0, dynamic_temperature=None):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        self.reward_fn = reward_fn
        self.adv_clip_max = adv_clip_max
        self.energy_max = energy_max
        self.energy_min = energy_min
        self.temperature = temperature
        self.dynamic_temperature = None
        if dynamic_temperature is not None:
            self.dynamic_temperature = dict(dynamic_temperature)
            p = self.dynamic_temperature
            if not (0 <= p["reference_start"] <= p["reference_end"] < p["activation_step"]
                    and 1 <= p["window_size"] <= p["activation_step"] + 1
                    and 0 < p["update_rate"] <= 1 and 0 <= p["tolerance"] < 1
                    and 0 < p["min_temperature"] <= 1 and temperature == 1.0):
                raise ValueError("Invalid dynamic temperature parameters")
        self.temperature_reference_sum = 0.0
        self.temperature_reference_count = 0
        self.temperature_recent = deque(maxlen=(self.dynamic_temperature or {}).get("window_size"))
        self.temperature_last_step = -1
        self.temperature_metrics = {}

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False, hard_gating=False, diagnostics=None, global_step=None):
        """Update statistics, optionally copying per-sample intermediates for logging.

        ``diagnostics`` never changes the returned training values. Its arrays
        retain the input reward shape and order; advantage_clipped is captured
        before temperature scaling, and energy_before_clip after rescaling.
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        if self.dynamic_temperature is not None:
            if not self.global_std or not exp or hard_gating or self.stats:
                raise ValueError("Dynamic temperature requires global_std, energy mode, no hard gating, and cleared prompt stats")
            if global_step != self.temperature_last_step + 1 or not rewards.size or not np.isfinite(rewards).all():
                raise ValueError("Expected the next global_step with finite rewards; restore temperature state when resuming")
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
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            prompt_adv = (prompt_rewards - mean) / std
            advantages[prompts == prompt] = prompt_adv
        if self.dynamic_temperature is not None:
            self._update_temperature(advantages, global_step)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            prompt_adv = advantages[prompts == prompt]
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
                if hard_gating:
                    gating_energy_stds = np.maximum(energy_stds + 1e-4, 1e-2)
                    energies_reranged = raw_energies / gating_energy_stds
                else:
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
            for key, value in self.temperature_metrics.items():
                diagnostics[key] = np.full_like(rewards, value)
        if exp:
            return advantages, energies, energies_stds
        return advantages

    def _update_temperature(self, advantages, global_step):
        # One observation per gathered rollout. The window includes this rollout;
        # S is measured before advantage clipping and before temperature scaling.
        p = self.dynamic_temperature
        advantages = np.asarray(advantages, dtype=np.float64)
        if (not isinstance(global_step, (int, np.integer)) or isinstance(global_step, bool)
                or global_step != self.temperature_last_step + 1
                or not advantages.size or not np.isfinite(advantages).all()):
            raise ValueError("Expected the next global_step and finite raw advantages")
        scale = float(np.sqrt(np.mean(advantages ** 2)))
        if not np.isfinite(scale):
            raise ValueError("Nonfinite advantage scale")
        reference_size = p["reference_end"] - p["reference_start"] + 1
        if global_step >= p["activation_step"] and (
                self.temperature_reference_count != reference_size or self.temperature_reference_sum <= 0):
            raise ValueError("Dynamic temperature requires a complete positive reference from the configured early steps")
        if p["reference_start"] <= global_step <= p["reference_end"]:
            self.temperature_reference_sum += scale
            self.temperature_reference_count += 1
        self.temperature_recent.append(scale)
        recent = float(np.mean(self.temperature_recent))
        reference = (self.temperature_reference_sum / reference_size
                     if self.temperature_reference_count == reference_size else None)
        old_temperature = self.temperature
        q = recent / (old_temperature * reference) if reference is not None and reference > 0 else None
        if global_step < p["activation_step"]:
            self.temperature = 1.0
        elif not 1 - p["tolerance"] <= q <= 1 + p["tolerance"]:
            self.temperature = float(np.clip(
                (1 - p["update_rate"]) * old_temperature + p["update_rate"] * recent / reference,
                p["min_temperature"], 1.0))
        self.temperature_last_step = int(global_step)
        self.temperature_metrics = dict(temperature=self.temperature, temperature_scale=scale,
            temperature_recent_scale=recent, temperature_adjusted=int(self.temperature != old_temperature))
        if reference is not None:
            self.temperature_metrics["temperature_reference_scale"] = reference
        if q is not None:
            self.temperature_metrics["temperature_effective_ratio"] = q
        return self.temperature

    def temperature_state_dict(self):
        # Prompt stats are cleared every rollout; only these temperature fields persist.
        return dict(parameters=dict(self.dynamic_temperature) if self.dynamic_temperature is not None else None,
            temperature=self.temperature, reference_sum=self.temperature_reference_sum,
            reference_count=self.temperature_reference_count, recent=list(self.temperature_recent),
            last_step=self.temperature_last_step)

    def load_temperature_state_dict(self, state):
        if self.dynamic_temperature is None or state["parameters"] != self.dynamic_temperature:
            raise ValueError("Temperature checkpoint parameters do not match this experiment")
        p = self.dynamic_temperature
        temperature, reference_sum = float(state["temperature"]), float(state["reference_sum"])
        last_step, count = state["last_step"], state["reference_count"]
        recent = list(state["recent"])
        if (not isinstance(last_step, int) or last_step < -1
                or count != max(0, min(last_step, p["reference_end"]) - p["reference_start"] + 1)
                or len(recent) != min(last_step + 1, p["window_size"])
                or not np.isfinite([temperature, reference_sum, *recent]).all()
                or not p["min_temperature"] <= temperature <= 1 or reference_sum < 0
                or any(value < 0 for value in recent)
                or (count == 0 and reference_sum != 0)
                or (last_step < p["activation_step"] and temperature != 1)
                or (last_step >= p["activation_step"] and reference_sum <= 0)):
            raise ValueError("Invalid or incomplete temperature checkpoint state")
        self.temperature = temperature
        self.temperature_reference_sum, self.temperature_reference_count = reference_sum, count
        self.temperature_recent = deque(recent, maxlen=p["window_size"])
        self.temperature_last_step = last_step
        self.temperature_metrics = {}

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


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
