import numpy as np
from collections import deque


def preserve_energy_sign(delta, energy):
    """Project clipped coefficients onto the sign of the original Kp - 1.

    Zero delta stays zero. Preserve the input precision and do not recenter:
    restoring zero sum here could turn an originally positive sample negative.
    """
    delta = np.asarray(delta)
    energy = np.asarray(energy)
    if delta.shape != energy.shape:
        raise ValueError("delta and energy must have identical shapes")
    return np.where(
        delta > 0, np.maximum(energy, 0),
        np.where(delta < 0, np.minimum(energy, 0), np.zeros_like(energy)),
    )


def gaussian_rarity(features, bandwidth):
    """Return (rarity, log KDE density, density / mean density) for one group.

    Features are fixed Euclidean coordinates, including any normalization done
    by the feature extractor. The Gaussian KDE excludes self matches and uses
    every member of the group, independent of its reward. All sums run in log
    space; the common Gaussian normalizer cancels in the density ratio.
    """
    features = np.asarray(features, dtype=np.float64)
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("rarity bandwidth must be finite and positive")
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("rarity features must have shape [K >= 2, dimension >= 1]")
    if not np.isfinite(features).all():
        raise ValueError("rarity features must be finite")

    k, dimension = features.shape
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        differences = (features[:, None, :] - features[None, :, :]) / bandwidth
        log_kernel = -0.5 * np.einsum("ijm,ijm->ij", differences, differences)
    if not np.isfinite(log_kernel).all():
        raise ValueError("feature distances / bandwidth exceed float64 range")
    np.fill_diagonal(log_kernel, -np.inf)

    # Shift all kernels first: even when every off-diagonal kernel would
    # underflow, the relative similarities and their group mean remain usable.
    kernel_offset = np.max(log_kernel)
    log_kernel -= kernel_offset
    row_max = np.max(log_kernel, axis=1)
    log_mass = row_max + np.log(np.exp(log_kernel - row_max[:, None]).sum(axis=1))
    mass_offset = np.max(log_mass)
    log_mean_mass = mass_offset + np.log(np.exp(log_mass - mass_offset).mean())
    density_ratio = np.exp(log_mass - log_mean_mass)
    # In exact arithmetic s < 1. Avoid rounding an extremely rare sample to 1
    # (which would entirely erase a negative correction).
    rarity = np.clip(1.0 - density_ratio, 0.0, np.nextafter(1.0, 0.0))
    log_density = (
        log_mass + kernel_offset - np.log(k - 1)
        - 0.5 * dimension * (np.log(2.0 * np.pi) + 2.0 * np.log(bandwidth))
    )
    return rarity, log_density, density_ratio


class PerPromptStatTracker:
    def __init__(self, global_std=False, adv_clip_max=None, reward_fn=None, temperature=1.0, energy_max=3.0, energy_min=-3.0, rarity_bandwidth=0.14):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        self.reward_fn = reward_fn
        self.adv_clip_max = adv_clip_max
        self.energy_max = energy_max
        self.energy_min = energy_min
        self.temperature = temperature
        self.rarity_bandwidth = rarity_bandwidth

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False, hard_gating=False, diagnostics=None, features=None):
        """Update statistics, optionally copying per-sample intermediates for logging.

        ``diagnostics`` never changes the returned training values. Its arrays
        retain the input reward shape and order; advantage_clipped is captured
        before temperature scaling, and energy_before_clip after rescaling.
        """
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
                diagnostics["energy_before_sign_preserve"] = np.zeros_like(rewards)
                diagnostics["sign_preserve_applied"] = np.zeros_like(rewards)
            if features is not None:
                for key in ("rarity_score", "rarity_log_density", "rarity_density_ratio", "rarity_group_size"):
                    diagnostics[key] = np.zeros((len(prompts), 1), dtype=np.float64)
                diagnostics["rarity_energy_before_clip"] = np.zeros_like(rewards)
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

                # Capture the reward coefficient before rarity changes its sign.
                original_delta = energies_reranged.copy()
                if diagnostics is not None:
                    diagnostics["softmax_prob"][prompts == prompt] = probs
                    diagnostics["energy_before_clip"][prompts == prompt] = energies_reranged
                if features is not None:
                    rarity, log_density, density_ratio = gaussian_rarity(
                        features[prompts == prompt], self.rarity_bandwidth
                    )
                    rarity_expanded = rarity.reshape((-1,) + (1,) * (rewards.ndim - 1))
                    rarity_increment = rarity_expanded * np.abs(energies_reranged)
                    # Center within this prompt group, per timestep, before clipping.
                    rarity_increment -= rarity_increment.mean(axis=0, keepdims=True)
                    energies_reranged = energies_reranged + rarity_increment
                    if diagnostics is not None:
                        diagnostics["rarity_score"][prompts == prompt, 0] = rarity
                        diagnostics["rarity_log_density"][prompts == prompt, 0] = log_density
                        diagnostics["rarity_density_ratio"][prompts == prompt, 0] = density_ratio
                        diagnostics["rarity_group_size"][prompts == prompt, 0] = k
                        diagnostics["rarity_energy_before_clip"][prompts == prompt] = energies_reranged
                energies_clipped = np.clip(energies_reranged, self.energy_min, self.energy_max)
                energies_sign_preserved = preserve_energy_sign(original_delta, energies_clipped)
                if diagnostics is not None:
                    diagnostics["energy_before_sign_preserve"][prompts == prompt] = energies_clipped
                    diagnostics["sign_preserve_applied"][prompts == prompt] = (
                        energies_sign_preserved != energies_clipped
                    )
                energies[prompts == prompt] = energies_sign_preserved
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
