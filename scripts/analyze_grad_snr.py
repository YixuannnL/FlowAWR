"""CPU analysis for the offline gradient-SNR probe (``scripts/probe_grad_snr_sd3.py``).

Reads ``sketch_rank*.npz`` + ``prompts_rank*.json`` + ``probe_meta.json`` from a
probe output directory, re-weights the per-sample basis gradients under a list
of weighting schemes, and reports for every scheme

* split-half cosine: prompt groups are split at random into two halves, the
  two half-batch gradients are formed, and their cosine is measured. Higher
  means the direction is more consistent across independent data, i.e. a
  higher signal-to-noise ratio. Reported for the raw weights and for
  group-centred weights (e - mean_group(e)), the latter removes the plain
  "pull towards my own samples" SFT component that any non-zero-sum weighting
  carries and that is not reward signal;
* cosine of the scheme's full gradient to the baseline (FlowAWR as trained);
* weight statistics: positive fraction, positive/negative mass per group,
  group sum, clip fractions, percentiles.

Only numpy is required. Typical use::

    python scripts/analyze_grad_snr.py --probe_dir probe_runs/ckpt900 --splits 300

Outputs ``<probe_dir>/analysis/summary.csv``, ``summary.json`` and a markdown
table on stdout. Nothing else is written.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

# flow_grpo/stat_tracking.py only needs numpy, so the real tracker can be used
# to verify the re-implementation below when the repo root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from flow_grpo.stat_tracking import PerPromptStatTracker
except Exception:  # pragma: no cover - optional
    PerPromptStatTracker = None


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------
def awr_energies(
    rewards,
    groups,
    rounds,
    *,
    std_mode="global",
    std_floor="none",
    temperature=1.0,
    adv_clip=5.0,
    energy_clip=3.0,
    pre_scale=1.0,
    post_scale=1.0,
    positive_only=False,
    hard_gating=False,
):
    """Re-implementation of PerPromptStatTracker.update(exp=True) plus the probe knobs.

    Statistics are computed per round (training clears the tracker every epoch).
    ``pre_scale`` multiplies the energy before the [-clip, clip] clip (so the clip
    becomes active), ``post_scale`` after it (pure rescale).
    Returns energies, advantages (unclipped), softmax probabilities and a dict of
    clip bookkeeping arrays.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    e = np.zeros_like(rewards)
    adv_out = np.zeros_like(rewards)
    prob_out = np.zeros_like(rewards)
    clip_hi = np.zeros(rewards.shape, dtype=bool)
    clip_lo = np.zeros(rewards.shape, dtype=bool)
    floor_hit = np.zeros(rewards.shape, dtype=bool)
    for rnd in np.unique(rounds):
        rm = rounds == rnd
        rr = rewards[rm]
        gg = groups[rm]
        idx_round = np.where(rm)[0]
        gstd = np.std(rr) + 1e-4
        if std_floor == "median":
            within = np.array([np.std(rr[gg == g]) for g in np.unique(gg)])
            floor = float(np.median(within))
        else:
            floor = 0.0
        for g in np.unique(gg):
            sel = np.where(gg == g)[0]
            x = rr[sel]
            k = len(x)
            mean = np.mean(x)
            if std_mode == "global":
                std = gstd
            elif std_mode == "prompt":
                s = np.std(x)
                if s < floor:
                    floor_hit[idx_round[sel]] = True
                    s = floor
                std = s + 1e-4
            else:
                raise ValueError(std_mode)
            adv = (x - mean) / std
            adv_out[idx_round[sel]] = adv
            a = np.clip(adv, -adv_clip, adv_clip) if adv_clip is not None else adv
            a = a / temperature
            z = np.exp(a - np.max(a))
            p = z / np.sum(z)
            prob_out[idx_round[sel]] = p
            raw = p - 1.0 / k
            if hard_gating:
                gate = max(np.std(raw) + 1e-4, 1e-2)
                en = raw / gate
            else:
                en = raw * k
            en = en * pre_scale
            clip_hi[idx_round[sel]] = en > energy_clip
            clip_lo[idx_round[sel]] = en < -energy_clip
            en = np.clip(en, -energy_clip, energy_clip)
            if positive_only:
                en = np.maximum(en, 0.0)
            e[idx_round[sel]] = en * post_scale
    return e, adv_out, prob_out, {"clip_hi": clip_hi, "clip_lo": clip_lo, "floor_hit": floor_hit}


def scheme_weights(spec, rewards, groups, rounds, rng, baseline_e=None):
    kind = spec.get("kind", "awr")
    kw = {k: v for k, v in spec.items() if k not in ("name", "kind", "note")}
    if kind == "awr":
        return awr_energies(rewards, groups, rounds, **kw)
    if kind in ("linear", "sign", "best_of_k", "top_frac"):
        # advantages with the requested std mode; weights derived from them
        e_dummy, adv, prob, info = awr_energies(
            rewards, groups, rounds,
            std_mode=kw.get("std_mode", "global"), std_floor=kw.get("std_floor", "none"),
            adv_clip=kw.get("adv_clip", 5.0),
        )
        if kind == "linear":
            e = np.clip(adv, -kw.get("adv_clip", 5.0), kw.get("adv_clip", 5.0))
        elif kind == "sign":
            e = np.sign(adv)
        else:
            e = np.zeros_like(adv)
            frac = 1.0 if kind == "best_of_k" else kw.get("frac", 0.25)
            for rnd in np.unique(rounds):
                for g in np.unique(groups[rounds == rnd]):
                    sel = np.where((rounds == rnd) & (groups == g))[0]
                    n_top = 1 if kind == "best_of_k" else max(1, int(round(frac * len(sel))))
                    top = sel[np.argsort(-adv[sel], kind="stable")[:n_top]]
                    e[top] = 1.0
        if kw.get("positive_only", False):
            e = np.maximum(e, 0.0)
        return e, adv, prob, info
    if kind == "shuffled":
        assert baseline_e is not None
        e = baseline_e.copy()
        for rnd in np.unique(rounds):
            for g in np.unique(groups[rounds == rnd]):
                sel = np.where((rounds == rnd) & (groups == g))[0]
                e[sel] = e[rng.permutation(sel)]
        info = {"clip_hi": np.zeros(len(e), bool), "clip_lo": np.zeros(len(e), bool), "floor_hit": np.zeros(len(e), bool)}
        return e, np.zeros_like(e), np.zeros_like(e), info
    raise ValueError(kind)


def default_schemes():
    S = []
    S.append(dict(name="awr_global_t1 (baseline)", kind="awr", std_mode="global", temperature=1.0))
    S.append(dict(name="awr_global_t1_x10_postclip (scale control)", kind="awr", std_mode="global", temperature=1.0, post_scale=10.0))
    S.append(dict(name="awr_global_t1_x10_preclip (clip at 0.3)", kind="awr", std_mode="global", temperature=1.0, pre_scale=10.0))
    for tau in (0.5, 0.3, 0.2):
        S.append(dict(name=f"awr_global_t{tau}", kind="awr", std_mode="global", temperature=tau))
    S.append(dict(name="awr_prompt_std_t1", kind="awr", std_mode="prompt", temperature=1.0))
    S.append(dict(name="awr_prompt_std_floor_t1", kind="awr", std_mode="prompt", std_floor="median", temperature=1.0))
    S.append(dict(name="awr_prompt_std_floor_t0.5", kind="awr", std_mode="prompt", std_floor="median", temperature=0.5))
    S.append(dict(name="awr_global_t1_pos", kind="awr", std_mode="global", temperature=1.0, positive_only=True))
    S.append(dict(name="awr_prompt_std_floor_t1_pos", kind="awr", std_mode="prompt", std_floor="median", temperature=1.0, positive_only=True))
    S.append(dict(name="linear_adv_global", kind="linear", std_mode="global"))
    S.append(dict(name="linear_adv_prompt_floor", kind="linear", std_mode="prompt", std_floor="median"))
    S.append(dict(name="sign_adv", kind="sign", std_mode="global"))
    S.append(dict(name="top25pct_one_hot", kind="top_frac", frac=0.25, std_mode="global"))
    S.append(dict(name="best_of_k_one_hot", kind="best_of_k", std_mode="global"))
    S.append(dict(name="shuffled_baseline (noise floor)", kind="shuffled"))
    return S


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_probe(probe_dir):
    files = sorted(glob.glob(os.path.join(probe_dir, "sketch_rank*.npz")))
    if not files:
        raise FileNotFoundError(f"no sketch_rank*.npz under {probe_dir}")
    parts = [np.load(f) for f in files]
    data = {}
    for key in parts[0].files:
        if key.startswith("exact_"):
            continue
        data[key] = np.concatenate([p[key] for p in parts], axis=0)
    exact = []
    for p in parts:
        if "exact_sketch" in p.files:
            exact.append(dict(sketch=p["exact_sketch"], sqnorm=float(p["exact_sqnorm"]), finite=bool(p["exact_finite"]),
                              rank=int(p["rank"][0]), n=len(p["rank"])))
    prompts = {}
    for f in sorted(glob.glob(os.path.join(probe_dir, "prompts_rank*.json"))):
        rk = int(os.path.basename(f)[len("prompts_rank"):-len(".json")])
        with open(f, encoding="utf-8") as fh:
            for item in json.load(fh):
                prompts[(rk, item["local_index"])] = item["prompt"]
    meta_path = os.path.join(probe_dir, "probe_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return data, exact, prompts, meta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def group_gradients(S, e, group_index, n_groups):
    """G[g] = sum_{i in g} e_i S_i   (n_groups x K)."""
    W = np.zeros((n_groups, len(e)), dtype=np.float32)
    W[group_index, np.arange(len(e))] = e.astype(np.float32)
    return W @ S


def split_half_cosines(G, masks):
    """masks: (R, n_groups) in {0,1}. Returns cos between the two half sums per row."""
    A = masks @ G
    B = (1.0 - masks) @ G
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    return num / den


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def weight_stats(e, group_index, n_groups, info, energy_clip):
    gsum = np.bincount(group_index, weights=e, minlength=n_groups)
    pos_mass = np.bincount(group_index, weights=np.maximum(e, 0), minlength=n_groups)
    neg_mass = np.bincount(group_index, weights=np.minimum(e, 0), minlength=n_groups)
    return dict(
        pos_frac=float(np.mean(e > 0)),
        zero_frac=float(np.mean(e == 0)),
        pos_mass_per_group=float(np.mean(pos_mass)),
        neg_mass_per_group=float(np.mean(neg_mass)),
        group_sum_mean=float(np.mean(gsum)),
        e_std=float(np.std(e)),
        e_min=float(np.min(e)),
        e_p01=float(np.percentile(e, 1)),
        e_p50=float(np.percentile(e, 50)),
        e_p99=float(np.percentile(e, 99)),
        e_max=float(np.max(e)),
        clip_hi_frac=float(np.mean(info["clip_hi"])),
        clip_lo_frac=float(np.mean(info["clip_lo"])),
        floor_hit_frac=float(np.mean(info["floor_hit"])),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--splits", type=int, default=300, help="number of random half splits of the prompt groups")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--energy_clip", type=float, default=3.0)
    ap.add_argument("--adv_clip", type=float, default=5.0)
    ap.add_argument("--schemes_json", default="", help="optional JSON list of scheme specs replacing the defaults")
    ap.add_argument("--report_dir", default="", help="default <probe_dir>/analysis")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data, exact, prompts, meta = load_probe(args.probe_dir)
    report_dir = args.report_dir or os.path.join(args.probe_dir, "analysis")
    os.makedirs(report_dir, exist_ok=True)

    S = data["sketch"].astype(np.float32)
    finite = data["finite"].astype(bool)
    rewards = data["reward_avg"].astype(np.float64)
    rounds = data["round"].astype(np.int64)
    prompt_idx = data["prompt_idx"].astype(np.int64)
    n, K = S.shape
    print(f"loaded {n} samples, sketch K={K}, rounds={sorted(set(rounds.tolist()))}, finite={int(finite.sum())}/{n}")
    if not finite.all():
        print(f"WARNING: {int((~finite).sum())} samples had non-finite gradients; their weights are set to 0 in every scheme")

    # groups = (round, prompt_idx) -> contiguous ids
    keys = rounds * 10_000_000 + prompt_idx
    uniq, group_index = np.unique(keys, return_inverse=True)
    n_groups = len(uniq)
    sizes = np.bincount(group_index)
    print(f"{n_groups} prompt groups, sizes min/max = {sizes.min()}/{sizes.max()}")
    if sizes.min() != sizes.max():
        print("WARNING: unequal group sizes (some rank's samples missing?) - results still computed")

    # sketch quality: sketched norm^2 vs exact gradient norm^2 per sample
    sk_sq = np.sum(S.astype(np.float64) ** 2, axis=1)
    ratio = sk_sq[finite] / np.maximum(data["grad_sqnorm"][finite], 1e-30)
    print(f"sketch check: ||sketch||^2 / ||grad||^2 mean={ratio.mean():.4f} std={ratio.std():.4f} (ideal 1 +- ~sqrt(2/K)*... small)")

    # baseline weights and tracker cross-check
    base_spec = dict(kind="awr", std_mode="global", temperature=1.0, adv_clip=args.adv_clip, energy_clip=args.energy_clip)
    e_base, adv_base, _, _ = awr_energies(rewards, group_index, rounds, **{k: v for k, v in base_spec.items() if k != "kind"})
    if "energy_baseline" in data and np.isfinite(data["energy_baseline"]).all():
        d = np.max(np.abs(e_base - data["energy_baseline"]))
        print(f"tracker check: max |e_reimpl - e_tracker(training code)| = {d:.3e}  (grouping here is by prompt index, training groups by decoded text)")
    if PerPromptStatTracker is not None and len(np.unique(rounds)) >= 1:
        rnd0 = rounds == rounds.min()
        trk = PerPromptStatTracker(True, adv_clip_max=args.adv_clip, temperature=1.0)
        _, e_trk, _ = trk.update([str(g) for g in group_index[rnd0]], rewards[rnd0], exp=True)
        print(f"tracker check (same grouping, round {rounds.min()}): max diff = {np.max(np.abs(e_trk - e_base[rnd0])):.3e}")

    # exact-check: per-rank accumulated real-loss gradient vs sum_i e_i sketch_i
    if exact:
        tot_recon = np.zeros(K, dtype=np.float64)
        tot_exact = np.zeros(K, dtype=np.float64)
        for ex in exact:
            m = data["rank"] == ex["rank"]
            recon = (data["energy_baseline"][m][:, None] * S[m]).sum(axis=0)
            c = cosine(recon, ex["sketch"])
            nr = np.linalg.norm(recon) / (np.linalg.norm(ex["sketch"]) + 1e-30)
            print(f"exact check rank {ex['rank']:2d}: cos(sum e_i b_i, real-loss grad) = {c:.6f}, norm ratio = {nr:.4f}, finite={ex['finite']}")
            tot_recon += recon
            tot_exact += ex["sketch"]
        print(f"exact check all ranks: cos = {cosine(tot_recon, tot_exact):.6f}")

    # masks for split halves (shared by all schemes -> paired comparison)
    masks = np.zeros((args.splits, n_groups), dtype=np.float32)
    half = n_groups // 2
    for r in range(args.splits):
        masks[r, rng.permutation(n_groups)[:half]] = 1.0

    schemes = default_schemes()
    if args.schemes_json:
        schemes = json.load(open(args.schemes_json))
    for s in schemes:
        if s.get("kind", "awr") in ("awr", "linear", "sign", "best_of_k", "top_frac"):
            s.setdefault("adv_clip", args.adv_clip)
        if s.get("kind", "awr") == "awr":
            s.setdefault("energy_clip", args.energy_clip)

    G_base = None
    rows = []
    for spec in schemes:
        e, adv, prob, info = scheme_weights(spec, rewards, group_index, rounds, rng, baseline_e=e_base)
        e = np.where(finite, e, 0.0)
        G = group_gradients(S, e, group_index, n_groups)
        if G_base is None:
            G_base = G
        e_c = e - np.bincount(group_index, weights=e, minlength=n_groups)[group_index] / sizes[group_index]
        Gc = group_gradients(S, e_c, group_index, n_groups)
        cos_raw = split_half_cosines(G, masks)
        cos_cen = split_half_cosines(Gc, masks)
        full = G.sum(axis=0)
        full_base = G_base.sum(axis=0)
        st = weight_stats(e, group_index, n_groups, info, spec.get("energy_clip", args.energy_clip))
        row = dict(
            scheme=spec["name"],
            cos_half_raw=float(cos_raw.mean()), cos_half_raw_std=float(cos_raw.std()),
            cos_half_centered=float(cos_cen.mean()), cos_half_centered_std=float(cos_cen.std()),
            cos_to_baseline=cosine(full, full_base),
            grad_norm_rel_baseline=float(np.linalg.norm(full) / (np.linalg.norm(full_base) + 1e-30)),
            **st,
        )
        rows.append(row)

    # --- report ---
    cols = ["scheme", "cos_half_raw", "cos_half_raw_std", "cos_half_centered", "cos_half_centered_std",
            "cos_to_baseline", "grad_norm_rel_baseline", "pos_frac", "zero_frac", "pos_mass_per_group",
            "neg_mass_per_group", "group_sum_mean", "clip_hi_frac", "clip_lo_frac", "floor_hit_frac",
            "e_std", "e_min", "e_p01", "e_p50", "e_p99", "e_max"]
    with open(os.path.join(report_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, n_samples=int(n), n_groups=int(n_groups), splits=args.splits, rows=rows), f, indent=2)

    print()
    print(f"split-half cosine over {args.splits} random halves of {n_groups} groups (higher = more consistent direction)")
    hdr = f"{'scheme':48s} {'cos raw':>9s} {'±':>6s} {'cos cent':>9s} {'±':>6s} {'cos→base':>9s} {'|g|/|g0|':>9s} {'pos%':>6s} {'Σe/grp':>8s} {'clip+%':>7s} {'clip-%':>7s} {'e_max':>7s} {'e_min':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['scheme'][:48]:48s} {r['cos_half_raw']:9.4f} {r['cos_half_raw_std']:6.3f} {r['cos_half_centered']:9.4f} {r['cos_half_centered_std']:6.3f} "
              f"{r['cos_to_baseline']:9.4f} {r['grad_norm_rel_baseline']:9.3f} {100*r['pos_frac']:6.1f} {r['group_sum_mean']:8.3f} "
              f"{100*r['clip_hi_frac']:7.2f} {100*r['clip_lo_frac']:7.2f} {r['e_max']:7.3f} {r['e_min']:7.3f}")
    print(f"\nwritten: {os.path.join(report_dir, 'summary.csv')} and summary.json")


if __name__ == "__main__":
    main()
