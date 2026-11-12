"""Summarize saved covariance comparisons without rerunning the detector fits.

Produces a six-data-set CSV/JSON overview, Holm-adjusted block p-values, a
comparison plot, and an updated section in experimental_data/README.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent
CASES = (
    ("doi_10.48550_arXiv.2606.18791", "10mT_S3", "10 mT"),
    ("doi_10.48550_arXiv.2606.18791", "30mT_S3", "30 mT"),
    ("doi_10.48550_arXiv.2606.18791", "100mT_S3", "100 mT"),
    ("doi_10.48550_arXiv.2606.18791", "1T_S3", "1 T"),
    ("doi_10.1103_5yc2-pv4y", "igc_nanoporous_Fe_0p1T", "PRL 0.1 T"),
    ("doi_10.1103_5yc2-pv4y", "igc_nanoporous_Fe_9T_15m_5K", "PRL 9 T"),
)
BEGIN = "<!-- covariance-comparison:start -->"
END = "<!-- covariance-comparison:end -->"


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if not np.all(np.isfinite(pvalues) & (pvalues >= 0) & (pvalues <= 1)):
        raise ValueError("Holm adjustment requires valid p-values for the complete block family")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.minimum(
        1.0, np.maximum.accumulate((pvalues.size - np.arange(pvalues.size)) * pvalues[order])
    )
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent_text(value):
    return "≈0" if abs(value) < 1e-8 else f"{value:+.3f}"


def main():
    output = DATA / "analysis_results" / "covariance_comparison"
    output.mkdir(parents=True, exist_ok=True)
    summaries, holm_rows, report_rows = [], [], []
    infos = []
    for collection, stem, label in CASES:
        folder = DATA / collection / "analysis_results" / "covariance_comparison"
        info = json.loads((folder / f"{stem}_summary.json").read_text())
        with (folder / f"{stem}_blocks.csv").open() as stream:
            blocks = list(csv.DictReader(stream))
        assert len(blocks) == 51
        adjusted = {
            method: holm([float(block[f"{method}_p"]) for block in blocks])
            for method in ("legacy_normal", "corrected")
        }
        names = [f"{'cos' if block['branch'] == 'cosine' else 'sin'} {block['order']}"
                 for block in blocks]
        old_holm = [name for name, p in zip(names, adjusted["legacy_normal"]) if p < 0.05]
        new_holm = [name for name, p in zip(names, adjusted["corrected"]) if p < 0.05]
        for i, block in enumerate(blocks):
            holm_rows.append({
                "dataset": stem, "branch": block["branch"], "order": block["order"],
                "legacy_p": float(block["legacy_normal_p"]),
                "corrected_p": float(block["corrected_p"]),
                "legacy_holm_p": float(adjusted["legacy_normal"][i]),
                "corrected_holm_p": float(adjusted["corrected"][i]),
            })
        metrics = info["ratios_new_over_baseline"]["legacy_normal"]
        row = {
            "dataset": stem,
            "samples": info["samples"],
            "fractional_weight_samples": info["partial_weight_samples"],
        }
        for metric in ("coefficient_sigma", "power_sigma", "spectral_sigma", "T"):
            for quantile in ("min", "p05", "median", "p95", "max"):
                row[f"{metric}_{quantile}_delta_percent"] = 100 * (metrics[metric][quantile] - 1)
        row.update({
            "nominal_significant_old": info["significant_counts"]["legacy_normal"],
            "nominal_significant_new": info["significant_counts"]["corrected"],
            "nominal_changed_blocks": "; ".join(item["block"] for item in info["changed_decisions"]),
            "holm_significant_old": len(old_holm), "holm_significant_new": len(new_holm),
            "holm_changed_blocks": "; ".join(sorted(set(old_holm) ^ set(new_holm))),
            "weighted_design_condition": info["weighted_design_condition"],
            "legacy_vs_qr_covariance_relative_difference":
                info["relative_legacy_vs_precision_qr_covariance_difference"],
            "rank_changes": len(info["changed_ranks"]),
        })
        summaries.append(row)
        infos.append(info)
        detail = f"{collection}/analysis_results/covariance_comparison/{stem}_comparison.png"
        report_rows.append(
            f"| [{label}]({detail}) | "
            f"{percent_text(row['power_sigma_median_delta_percent'])} | "
            f"{percent_text(row['spectral_sigma_median_delta_percent'])} | "
            f"{percent_text(row['T_median_delta_percent'])} | "
            f"{row['nominal_significant_old']} → {row['nominal_significant_new']} | "
            f"{len(old_holm)} → {len(new_holm)} |"
        )
        info["comparison_holm"] = {
            "alpha": 0.05, "family_size": 51,
            "significant_old": old_holm, "significant_new": new_holm,
        }
        print(label, "nominal", row["nominal_significant_old"], "->",
              row["nominal_significant_new"], "Holm", old_holm, "->", new_holm)

    # Confirm that all six results used exactly the same analysis implementation.
    assert all(info["source_sha256"] == infos[0]["source_sha256"] for info in infos)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "holm_blocks.csv", holm_rows)
    (output / "summary.json").write_text(json.dumps({
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Ratios at identical coefficients and fixed selection; nominal alpha=0.05",
        "data_sets": infos,
    }, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True, constrained_layout=True)
    labels = [label for _, _, label in CASES]
    y = np.arange(len(CASES))
    for axis, metric, title in zip(
        axes, ("power_sigma", "spectral_sigma", "T"),
        ("Angular-power error bars", "Radial-spectrum band widths", "Angular-block T statistics"),
    ):
        for i, row in enumerate(summaries):
            mid = row[f"{metric}_median_delta_percent"]
            low = row[f"{metric}_min_delta_percent"]
            high = row[f"{metric}_max_delta_percent"]
            axis.errorbar(
                mid, y[i], xerr=[[max(0, mid-low)], [max(0, high-mid)]],
                fmt="o", capsize=3, color="#1764ab" if i < 4 else "0.45",
            )
        axis.axvline(0, color="black", lw=0.8)
        axis.set(xlabel="100 × (corrected / legacy − 1) [%]", title=title)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Covariance correction: dots = medians, bars = full ranges; same fitted signals")
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)

    changes = [f"{info['dataset']}: {item['block']} "
               f"(p={item['old_p']:.5g} → {item['new_p']:.5g})"
               for info in infos for item in info["changed_decisions"]]
    holm_changes = [row["dataset"] for row in summaries if row["holm_changed_blocks"]]
    maximum_band_reduction = -min(
        row["spectral_sigma_min_delta_percent"] for row in summaries[:4]
    )
    maximum_control_difference = max(
        info["relative_legacy_vs_precision_qr_covariance_difference"] for info in infos
    )
    block = "\n".join([
        BEGIN, "## Covariance-correction comparison", "",
        "All six standard analyses were rerun after the absolute-noise covariance",
        "correction. Their existing q ranges, mode limits (25, 25), sample selection,",
        "area weights, and centering settings were retained. The standard PNGs above",
        "were regenerated. The comparison evaluates both covariances at exactly the",
        "same fitted coefficients; intensities, raw powers, and radial spectra are fixed.", "",
        "Reproduce both the detector analyses and the lightweight summary:", "",
        "~~~sh", "PYTHONPATH=src python experimental_data/compare_covariance.py",
        "python experimental_data/summarize_covariance.py", "~~~", "",
        "Both scripts use one BLAS worker and a noninteractive plotting backend.",
        "The analysis also checks memory pressure before each data set on macOS.", "",
        "The following entries are median percentage changes. Power errors cover",
        "51 angular blocks; spectral bands cover all 51 branches/orders on the",
        "same 300-point q grid. Rejection counts are formal per-data-set calculations",
        "under the assumed noise model at alpha=0.05, not classifications of",
        "physically identified contributions.", "",
        "| Data set | Δ σ(P_n) [%] | Δ σ(I_n) [%] | Δ T_n [%] | Nominal model rejections, old → new | Holm model rejections, old → new |",
        "|---|---:|---:|---:|---:|---:|", *report_rows, "",
        "Nominal decision changes: " + ("; ".join(changes) if changes else "none") + ".",
        "Holm correction uses the complete family of 51 existing block tests per",
        "data set. Its significant block sets " +
        ("change for " + ", ".join(holm_changes) if holm_changes else "remain unchanged") +
        "; no reduced-model selection or refit is introduced.", "",
        f"The strongest local reduction of a spectral band is {maximum_band_reduction:.2f}%.",
        "These larger effects occur near annulus boundaries; the median over the",
        "whole radial grid is much smaller. The two PRL benchmarks have unit sample",
        "weights and remain unchanged to numerical precision.", "",
        "Three covariances are distinguished: legacy_normal uses the previous",
        "inverse normal matrix; precision_qr evaluates the same precision-model",
        "formula with QR factors; corrected propagates the actual pixel errors.",
        "The maximum relative Frobenius difference between the first two is",
        f"{maximum_control_difference:.3g}; numerical rank changes are absent in all six cases.",
        "Thus the observed changes are attributable to geometric weights rather",
        "than the switch to a more stable matrix factorization.", "",
        "The comparisons retain the package's diagonal-noise, fixed-design model.",
        "They do not propagate uncertainty of fitted centers or reduction-induced",
        "pixel correlations. The pre-existing PRL coordinate-unit caveat still applies.", "",
        "The p-values and Holm counts are supplementary, model-conditional",
        "calculations. Their noise assumptions have not been established by",
        "the synthetic calibration tests for these experimental maps. They do",
        "not measure the frequency of experimental artifacts or physical",
        "misidentifications. The standard plots do not highlight Holm decisions.",
        "For a parity-even observable, odd powers can instead provide a",
        "descriptive empirical reference level; this is not assigned a formal",
        "significance threshold, and physical odd contributions must first",
        "be excluded.", "",
        "Detailed per-block CSV files, metadata JSON files, and comparison PNGs",
        "are in each collection's analysis_results/covariance_comparison directory.",
        "The heat maps use a symmetric scale at the 99th percentile of absolute",
        "changes; larger values are color-saturated. Full numerical ranges are",
        "retained in the tables and CSV/JSON summaries.", "",
        "- [Summary CSV](analysis_results/covariance_comparison/summary.csv)",
        "- [Combined metadata and results](analysis_results/covariance_comparison/summary.json)",
        "- [Raw and Holm-adjusted p-values](analysis_results/covariance_comparison/holm_blocks.csv)", "",
        "![Covariance comparison](analysis_results/covariance_comparison/overview.png)", END,
    ])
    readme = DATA / "README.md"
    text = readme.read_text()
    if BEGIN in text:
        before, rest = text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    readme.write_text(text)


if __name__ == "__main__":
    main()

