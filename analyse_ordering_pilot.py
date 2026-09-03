"""
Analysis for the texture-scale ordering pilot.

Reads the pilot_*.json files exported by the pilot page and reports:

  1. Per-sweep rank correlation between each arrangement and the true order.
  2. Per-adjacent-pair transposition rates — where in the sweep errors fall.
  3. Sweeps flagged as ambiguous, with the notes participants left.
  4. Tile size used, so you can check whether errors track small viewing size.

Usage:  python3 analyse_ordering_pilot.py path/to/results/
"""

import json, sys, glob, os
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr


def load(folder):
    trials = []
    files = sorted(glob.glob(os.path.join(folder, "pilot_*.json")))
    if not files:
        sys.exit(f"No pilot_*.json files found in {folder}")
    for f in files:
        s = json.load(open(f))
        pid = s.get("participant") or os.path.basename(f)
        for t in s["trials"]:
            t["participant"] = pid
            trials.append(t)
    print(f"{len(files)} participants, {len(trials)} orderings\n")
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
    print(f"{'sweep':<8} {'n':>3} {'mean':>6} {'min':>6} {'exact':>6} {'ambig':>6}")
    for s in sorted(by_set):
        ts = by_set[s]
        rhos = [t["rho"] for t in ts]
        exact = sum(1 for t in ts if t["order"] == list(range(1, n + 1)))
        amb = sum(1 for t in ts if t.get("ambiguous"))
        print(f"{s:<8} {len(ts):>3} {np.mean(rhos):>6.2f} {min(rhos):>6.2f} "
              f"{exact:>4}/{len(ts):<2} {amb:>4}/{len(ts):<2}")

    all_rho = [t["rho"] for t in trials]
    print(f"\noverall mean rho {np.mean(all_rho):.3f}  "
          f"({flipped}/{len(trials)} sorted coarse-to-fine, rescored)")

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

    # ---- 3. ambiguity notes
    notes = [(t["set"], t["participant"], t["note"])
             for t in trials if t.get("ambiguous") and t.get("note")]
    if notes:
        print("\nAMBIGUITY NOTES")
        for s, p, note in notes:
            print(f"  {s} ({p}): {note}")

    # ---- 4. did viewing size matter?
    sizes = [t.get("tileWidthPx") for t in trials if t.get("tileWidthPx")]
    if sizes:
        rhos = [t["rho"] for t in trials if t.get("tileWidthPx")]
        r = spearmanr(sizes, rhos).statistic
        print(f"\nVIEWING SIZE: median {int(np.median(sizes))}px, "
              f"range {min(sizes)}–{max(sizes)}px")
        print(f"  correlation between tile size and accuracy: {r:+.2f}")
        print("  (a clear positive value means small viewing hurt performance,")
        print("   so the size control is doing real work)")

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
