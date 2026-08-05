"""Table 1 statistics for the LP rebuttal: per-game mean +/- std over seeds, plus
human-normalized mean/median with across-seed spread.

Reads data/*.json (per-seed final scores). Run:
    python analysis/table1_stats.py
"""
import json
import statistics as st
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Canonical Atari100k random scores (paper Table 1 "Random" column).
RANDOM = {
    "Alien": 228, "Amidar": 6, "Assault": 222, "Asterix": 210, "BankHeist": 14,
    "BattleZone": 2360, "Boxing": 0, "Breakout": 2, "ChopperCommand": 811,
    "CrazyClimber": 10780, "DemonAttack": 152, "Freeway": 0, "Frostbite": 65,
    "Gopher": 258, "Hero": 1027, "Jamesbond": 29, "Kangaroo": 52, "Krull": 1598,
    "KungFuMaster": 258, "MsPacman": 307, "Pong": -21, "PrivateEye": 25,
    "Qbert": 164, "RoadRunner": 12, "Seaquest": 68, "UpNDown": 533,
}

METHODS = {
    "LP-DreamerV3": "DELULU-DV3-FINAL.json",
    "DreamerV3": "DREAMERV3.json",
    "LP-STORM": "DELULU-STORM-FINAL.json",
    "STORM": "STORM_atari.json",
}


def nkey(k):
    return "".join(c for c in k.lower() if c.isalnum())


def _is_seed_dict(v):
    return isinstance(v, dict) and all(str(s).lstrip("-").isdigit() for s in v)


def load_scores(fname):
    """Return {normkey: [seed scores]}. Handles per-seed dicts and lists;
    skips games stored as training curves (e.g. STORM_atari 'GIT-STORM' format)."""
    raw = json.load(open(DATA / fname))
    out = {}
    for k, v in raw.items():
        if _is_seed_dict(v):
            out[nkey(k)] = [v[s] for s in sorted(v, key=int)]
        elif isinstance(v, list):
            out[nkey(k)] = list(v)
        # else: unsupported nested/curve format -> skip this game
    return out


RAND = {nkey(k): v for k, v in RANDOM.items()}
HUMAN = {nkey(k): v[0] for k, v in json.load(open(DATA / "HUMAN.json")).items()}
PER = {m: load_scores(f) for m, f in METHODS.items()}
GAMES = sorted(RAND)


def stdev(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


print("=== Per-game score: mean +/- std (sample std over seeds) ===")
print("game".ljust(15) + "".join(m.ljust(24) for m in METHODS))
for g in GAMES:
    row = g.ljust(15)
    for m in METHODS:
        vals = PER[m].get(g)
        row += (f"{st.mean(vals):.0f} +/- {stdev(vals):.0f}" if vals else "-").ljust(24)
    print(row)

print("\n=== Human-normalized aggregate (%), mean +/- std across seeds ===")
for m in METHODS:
    sc = PER[m]
    games_m = [g for g in GAMES if g in sc and g in HUMAN]
    if not games_m:
        print(f"{m.ljust(15)} (no parseable per-seed scores; skipped)")
        continue
    n_seeds = min(len(sc[g]) for g in games_m)
    means, medians = [], []
    for i in range(n_seeds):
        norm = [100 * (sc[g][i] - RAND[g]) / (HUMAN[g] - RAND[g]) for g in games_m]
        means.append(st.mean(norm))
        medians.append(st.median(norm))
    print(f"{m.ljust(15)} games={len(games_m)} seeds={n_seeds} | "
          f"Normed Mean {st.mean(means):.1f} +/- {stdev(means):.1f} | "
          f"Normed Median {st.mean(medians):.1f} +/- {stdev(medians):.1f}")

# Focused 5-game variance table for the rebuttal (our runs only; baselines are reported scores).
SELECTED = [
    ("kungfumaster", "Kung Fu Master"), ("mspacman", "Ms Pacman"),
    ("frostbite", "Frostbite"), ("roadrunner", "Road Runner"), ("boxing", "Boxing"),
]
print("\n=== Rebuttal variance table: 5 games, our runs, mean +/- std (5 seeds) ===")
print("Game".ljust(16) + "LP-DreamerV3".ljust(22) + "LP-STORM".ljust(22))
for key, name in SELECTED:
    lp = PER["LP-DreamerV3"].get(key)
    ls = PER["LP-STORM"].get(key)
    row = name.ljust(16)
    row += (f"{st.mean(lp):.0f} +/- {stdev(lp):.0f}" if lp else "-").ljust(22)
    row += (f"{st.mean(ls):.0f} +/- {stdev(ls):.0f}" if ls else "-").ljust(22)
    print(row)
