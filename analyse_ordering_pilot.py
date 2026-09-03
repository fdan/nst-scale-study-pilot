"""
Analysis for the texture-scale ordering pilot.

Reads the pilot_*.json files exported by the pilot page and reports:

  1. Per-sweep rank correlation between each arrangement and the true order.
  2. Per-adjacent-pair transposition rates — where in the sweep errors fall.
  3. Tile size used, so you can check whether errors track small viewing size.

Usage:  python3 analyse_ordering_pilot.py path/to/results/
"""

import json, sys, glob, os
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr


def load(folder):
    trials, sessions = [], []
    files = sorted(glob.glob(os.path.join(folder, "pilot_*.json")))
    if not files:
        sys.exit(f"No pilot_*.json files found in {folder}")
    for f in files:
        s = json.load(open(f))
        pid = s.get("participant") or os.path.basename(f)
        sessions.append((pid, s))
        for t in s["trials"]:
            t["participant"] = pid
            t["assigned"] = s.get("assignedSets", [])
            trials.append(t)
    print(f"{len(files)} participants, {len(trials)} orderings\n")

    # flag anyone who ignored the desktop-only instruction
    suspect = []
    for pid, s in sessions:
        d = s.get("device")
        if d is None:
            suspect.append((pid, "no device data (older version of the page)"))
        elif not d.get("likelyDesktop"):
            why = [k for k in ("coarsePointer", "noHover",
                               "uaMobileString", "uaMobileHint") if d.get(k)]
            vp = s.get("viewport", {})
            suspect.append((pid, f"{', '.join(why) or 'unknown'}; "
                                 f"viewport {vp.get('w')}x{vp.get('h')}"))
    if suspect:
        print("NOT A DESKTOP — consider excluding:")
        for pid, why in suspect:
            print(f"  {pid}: {why}")
        print()
    return trials


def orientation(order, n):
    """Participants may sort coarse-to-fine. Score against whichever
    direction fits better, and report how often that happened."""
    truth = list(range(1, n + 1))
    asc = spearmanr(order, truth).statistic
    return ("descending", -asc) if asc < 0 else ("ascending", asc)


def main(folder):
    trials = load(folder)
    n = len(trials[0]["order"])

    by_set = defaultdict(list)
    flipped = 0
    for t in trials:
        direction, rho = orientation(t["order"], n)
        if direction == "descending":
            flipped += 1
            t["order"] = t["order"][::-1]
        t["rho"] = rho
        by_set[t["set"]].append(t)

    # ---- 1. per-sweep agreement
    print("PER-SWEEP RANK CORRELATION (1.0 = perfect)")
    print(f"{'sweep':<8} {'n':>3} {'mean':>6} {'min':>6} {'exact':>6}")
    for s in sorted(by_set):
        ts = by_set[s]
        rhos = [t["rho"] for t in ts]
        exact = sum(1 for t in ts if t["order"] == list(range(1, n + 1)))
        print(f"{s:<8} {len(ts):>3} {np.mean(rhos):>6.2f} {min(rhos):>6.2f} "
              f"{exact:>4}/{len(ts):<2}")

    all_rho = [t["rho"] for t in trials]
    print(f"\noverall mean rho {np.mean(all_rho):.3f}  "
          f"({flipped}/{len(trials)} sorted coarse-to-fine, rescored)")

    # random assignment is lumpy at small n — flag sweeps that need a top-up
    counts = {s: len(ts) for s, ts in by_set.items()}
    thin = sorted(s for s, c in counts.items() if c < 4)
    if thin:
        print(f"UNDER-SAMPLED (fewer than 4 observers): {', '.join(thin)}")
    missing = [s for s in
               sorted({s for t in trials for s in t.get("assigned", [])} | set(counts))
               if s not in counts]
    if missing:
        print(f"NOT SEEN AT ALL: {', '.join(missing)}")

    # ---- 2. where the errors are
    # For each true adjacent pair (k, k+1), how often was it out of order?
    print("\nADJACENT-PAIR TRANSPOSITIONS (position in the true sweep)")
    print("higher = that pair of steps was harder to tell apart\n")
    header = "sweep    " + "  ".join(f"{k}-{k+1}" for k in range(1, n))
    print(header)
    grand = np.zeros(n - 1)
    for s in sorted(by_set):
        ts = by_set[s]
        bad = np.zeros(n - 1)
        for t in ts:
            pos = {step: i for i, step in enumerate(t["order"])}
            for k in range(1, n):
                if pos[k] > pos[k + 1]:
                    bad[k - 1] += 1
        rate = bad / len(ts)
        grand += bad
        cells = "  ".join(f"{r:>3.0%}" for r in rate)
        print(f"{s:<8} {cells}")
    print(f"{'ALL':<8} " + "  ".join(f"{r:>3.0%}" for r in grand / len(trials)))

    # ---- 3. did viewing size matter?
    sizes = [t.get("tileWidthPx") for t in trials if t.get("tileWidthPx")]
    if sizes:
        print(f"\nVIEWING SIZE: median {int(np.median(sizes))}px, "
              f"range {min(sizes)}–{max(sizes)}px")
        if len(set(sizes)) > 1:
            rhos = [t["rho"] for t in trials if t.get("tileWidthPx")]
            r = spearmanr(sizes, rhos).statistic
            print(f"  correlation between tile size and accuracy: {r:+.2f}")
            print("  (clearly positive means small viewing hurt performance)")
        else:
            print("  everyone used the same size — no comparison possible")

    # ---- 4. how far each step travelled
    # The adjacent-pair table flags whether a pair is out of order, not how
    # badly. A step displaced by two or three positions is a stronger signal
    # than a neighbouring swap.
    print("\nMEAN DISPLACEMENT PER STEP (positions away from where it belongs)")
    disp = np.zeros(n)
    for t in trials:
        for true_step, placed_at in ((st, i) for i, st in enumerate(t["order"])):
            disp[true_step - 1] += abs(placed_at - (true_step - 1))
    disp /= len(trials)
    print("step     " + "  ".join(f"{k:>4}" for k in range(1, n + 1)))
    print("mean     " + "  ".join(f"{d:>4.2f}" for d in disp))

    # ---- timing
    times = [t["ms"] / 1000 for t in trials if t.get("ms")]
    if times:
        print(f"\nTIME PER SWEEP: median {np.median(times):.0f}s, "
              f"range {min(times):.0f}–{max(times):.0f}s")
        per_p = len(trials) / len({t["participant"] for t in trials})
        print(f"  ≈ {np.median(times) * per_p / 60:.1f} min per session "
              f"at {per_p:.0f} sweeps each")

    # ---- verdict
    print("\n" + "-" * 58)
    worst = grand.max() / len(trials)
    worst_pair = int(grand.argmax()) + 1
    print(f"Worst adjacent pair: steps {worst_pair}-{worst_pair+1}, "
          f"transposed {worst:.0%} of the time.")
    if worst < 0.15:
        print("All adjacent steps look reliably discriminable at this spacing.")
    elif worst < 0.30:
        print("Marginal at that position — inspect it before committing.")
    else:
        print("That pair is not reliably discriminable. Widen the spacing "
              "or drop a step.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
