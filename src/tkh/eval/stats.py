"""Small statistics helpers shared by the faithfulness and blind-rating
evaluations: Wilson intervals for rates, a one-sided binomial test against
a chance rate, and Cohen's kappa with a bootstrap CI."""
import numpy as np
from scipy import stats as scipy_stats


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n. Unlike the
    normal approximation it stays inside [0, 1] and behaves at k=0 or k=n,
    which matters at the sample sizes here (29 to 58 glosses per snapshot)."""
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def rate_with_ci(k, n):
    return {"k": int(k), "n": int(n), "rate": (k / n if n else None), "ci95": wilson_ci(k, n)}


def binomial_p_greater(k, n, p0):
    """One-sided p-value that the true rate exceeds p0."""
    if n == 0:
        return None
    return float(scipy_stats.binomtest(int(k), int(n), p0, alternative="greater").pvalue)


def cohen_kappa(a, b):
    """Cohen's kappa between two equal-length label sequences."""
    a, b = list(a), list(b)
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def cohen_kappa_ci(a, b, rng, n_boot=2000):
    """Kappa with a percentile bootstrap CI over items."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    n = len(a)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        k = cohen_kappa(a[idx], b[idx])
        if k is not None:
            boot.append(k)
    return {"kappa": cohen_kappa(a, b), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": int(n)}
