#!/usr/bin/env python3
import os
import re
import argparse
import locale
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Locale & plotting configuration
# -----------------------------------------------------------------------------

try:
    locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
except locale.Error:
    print("German locale not available. Using default formatting.")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# -----------------------------------------------------------------------------
# Field reader
# -----------------------------------------------------------------------------

def read_field(time_dir: str, field: str):
    """Read OpenFOAM scalar field (uniform or nonuniform)."""
    try:
        path = Path(time_dir) / field
        if not path.exists():
            return None

        text = path.read_text()

        uniform = re.search(r"internalField\s+uniform\s+([0-9.eE+-]+)", text)
        if uniform:
            return float(uniform.group(1)) * np.ones(1000)  # dummy length

        nonuniform = re.search(
            r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)[^0-9]*((?:\s*[0-9.eE+-]+)+)",
            text
        )
        if nonuniform:
            values = list(map(float, re.findall(r"[0-9.eE+-]+", nonuniform.group(2))))
            return np.array(values)

    except Exception as e:
        print(f"Error reading {path}: {e}")

    return None


# -----------------------------------------------------------------------------
# Mixing metrics
# -----------------------------------------------------------------------------

def compute_metrics(H2, O2, N2, time_str):
    eps = 1e-10
    total = H2 + O2 + N2 + eps
    f = H2 / total

    mean_f = np.mean(f)
    std_f = np.std(f)

    kmv = std_f**2
    mixing_index = 1 - std_f / np.sqrt(mean_f * (1 - mean_f)) if 0 < mean_f < 1 else 0
    segregation_index = std_f / mean_f if mean_f > eps else 0

    air = O2 + N2
    lam = np.full_like(H2, np.inf)
    mask = H2 > eps
    lam[mask] = (air[mask] / H2[mask]) / 34.3

    return {
        "time": float(time_str) * 1000.0,
        "kmv": kmv,
        "mixing_index": mixing_index,
        "segregation_index": segregation_index,
        "rich": np.mean(lam < 0.95) * 100,
        "stoich": np.mean((lam >= 0.95) & (lam <= 1.05)) * 100,
        "lean": np.mean(lam > 1.05) * 100,
    }


# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------

def plot_metric(times, data, labels, ylabel, title, filename, outdir):
    plt.figure(figsize=(12, 7))
    for label, values in zip(labels, data):
        plt.plot(times, values, linewidth=2.5, label=label)

    plt.xlabel("Time (ms)", fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.legend(title="Injection mass", bbox_to_anchor=(1.25, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(outdir / filename, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenFOAM post-processing: mixing metrics & lambda zones"
    )

    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Base directory containing mass-variation cases"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="Results",
        help="Output directory name (created inside case-dir)"
    )

    args = parser.parse_args()

    base_dir = args.case_dir.resolve()
    output_dir = base_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    all_results = {}

    for case in cases:
        os.chdir(case)
        times = []

        for d in os.listdir():
            try:
                times.append((d, float(d)))
            except ValueError:
                continue

        times.sort(key=lambda x: x[1])
        results = []

        for t, _ in times:
            H2 = read_field(t, "H2")
            O2 = read_field(t, "O2")
            N2 = read_field(t, "N2")

            if H2 is None or O2 is None or N2 is None:
                continue

            n = min(len(H2), len(O2), len(N2))
            results.append(compute_metrics(H2[:n], O2[:n], N2[:n], t))

        if results:
            all_results[case.name] = results

    if not all_results:
        print("No valid results found.")
        return

    times = [r["time"] for r in next(iter(all_results.values()))]
    labels = list(all_results.keys())

    plot_metric(
        times,
        [[r["kmv"] for r in res] for res in all_results.values()],
        labels,
        r"$\sigma_f^2$",
        "Injection mass influence on H₂ mixing variance",
        "KMV_mass.png",
        output_dir,
    )

    print(f"\nPost-processing completed. Results in: {output_dir}")


if __name__ == "__main__":
    main()
