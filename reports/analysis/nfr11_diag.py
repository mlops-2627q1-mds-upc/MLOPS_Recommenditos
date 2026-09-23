"""Why does the control window flag too, and what separates ES from it?

Varies (a) how the control window is drawn (seller-grouped, as the project's split
protocol prescribes, vs i.i.d. rows), and (b) the reference size, and compares the
effect sizes (KS statistic, Cramer's V) of ES windows against control windows.
"""

import sys

import numpy as np

sys.argv = [sys.argv[0], "cars.csv"]
from nfr11_check import FEATURES, NUM, P_VAL, SEED, chi2_p, ks_p, load

WINDOW = 1000
TRIALS = 100
THR = P_VAL / len(FEATURES)


def scores(ref, win):
    """Per-feature (p, effect size). Effect: KS statistic, or Cramer's V for chi-square."""
    out = {}
    for f in FEATURES:
        if f in NUM:
            p, stat = ks_p(ref[f].to_numpy(dtype=float), win[f].to_numpy(dtype=float))
            out[f] = (p, stat)
        else:
            p, chi2 = chi2_p(ref[f], win[f])
            n = len(ref) + len(win)
            out[f] = (p, float(np.sqrt(chi2 / n)))
    return out


def rate(ref, pool, rng, trials=TRIALS):
    flags, flags_nc, eff = [], [], {f: [] for f in FEATURES}
    for _ in range(trials):
        win = pool.iloc[rng.choice(len(pool), size=WINDOW, replace=False)]
        s = scores(ref, win)
        drifted = [f for f, (p, _) in s.items() if p < THR]
        flags.append(bool(drifted))
        flags_nc.append(any(f != "country_code" for f in drifted))
        for f, (_, e) in s.items():
            eff[f].append(e)
    return (
        float(np.mean(flags)),
        float(np.mean(flags_nc)),
        {f: float(np.mean(v)) for f, v in eff.items()},
    )


def main():
    df, _ = load()
    es = df[df["country_code"] == "ES"]
    rest = df[df["country_code"] != "ES"]
    makes = rest["make"].value_counts()
    supported = set(makes[makes >= 300].index)
    es = es[es["make"].isin(supported)]
    rest = rest[rest["make"].isin(supported)]

    rng = np.random.default_rng(SEED)
    groups = rest["group"].unique()
    rng.shuffle(groups)
    ref_groups = set(groups[: int(0.8 * len(groups))])
    ref_full = rest[rest["group"].isin(ref_groups)]
    ctl_grouped = rest[~rest["group"].isin(ref_groups)]

    # i.i.d. control: same 80/20 proportions, but split by row, ignoring the seller
    perm = np.random.default_rng(SEED).permutation(len(rest))
    cut = int(0.8 * len(rest))
    ref_iid = rest.iloc[perm[:cut]]
    ctl_iid = rest.iloc[perm[cut:]]

    print(f"features={len(FEATURES)}  bonferroni threshold={THR:.5f}  window={WINDOW}\n")
    print(f"{'reference':<34}{'ES':>8}{'ES w/o country':>16}{'control':>10}")
    for label, ref, ctl in [
        ("seller-grouped split (project plan)", ref_full, ctl_grouped),
        ("i.i.d. row split", ref_iid, ctl_iid),
    ]:
        for n_ref in [2000, 10000, len(ref)]:
            r = ref.sample(n=min(n_ref, len(ref)), random_state=SEED)
            es_rate, es_nc, es_eff = rate(r, es, np.random.default_rng(SEED + 1))
            ct_rate, _, ct_eff = rate(r, ctl, np.random.default_rng(SEED + 2))
            print(f"{label[:20]:<22}n={len(r):<10}{es_rate:>8.2f}{es_nc:>16.2f}{ct_rate:>10.2f}")
            if n_ref == 10000 and label.startswith("seller"):
                keep = es_eff
                keep_ct = ct_eff

    print(
        "\nMean effect size per feature at n_ref=10000, seller-grouped "
        "(KS statistic for numerical, Cramer's V for categorical):"
    )
    print(f"{'feature':<30}{'ES':>8}{'control':>10}{'ratio':>8}")
    for f in sorted(FEATURES, key=lambda f: -keep[f]):
        ratio = keep[f] / keep_ct[f] if keep_ct[f] > 0 else float("inf")
        print(f"{f:<30}{keep[f]:>8.3f}{keep_ct[f]:>10.3f}{ratio:>8.1f}")

    print("\nSeparation check: how many features exceed an effect-size floor?")
    for floor in [0.05, 0.10, 0.15, 0.20]:
        es_n = sum(1 for f in FEATURES if keep[f] >= floor and f != "country_code")
        ct_n = sum(1 for f in FEATURES if keep_ct[f] >= floor and f != "country_code")
        print(f"  floor={floor:.2f}  ES(excl. country)={es_n:>2}  control={ct_n:>2}")


if __name__ == "__main__":
    main()
