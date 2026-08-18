import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm as tqdm
from constants import *
from data_utils import *
from figure_utils import *
from scipy import stats

mpl.style.use("assets/stylesheet.mplstyle")


def plot_pseudobulk_correlation():
    pseudobulk_correlation_with_ccle = load_file(
        "pseudobulk_correlation_with_ccle.csv", local_dir=PROCESSED_DIR
    )

    all_cl = pseudobulk_correlation_with_ccle["scCellLine"].unique().tolist()
    ordered_cl = (
        pseudobulk_correlation_with_ccle[pseudobulk_correlation_with_ccle["Identity"]]
        .sort_values("TopVarCorrelation", ascending=False)["scCellLine"]
        .tolist()
    )
    cl_order = ordered_cl + sorted(list(set(all_cl) - set(ordered_cl)))
    cl_order = cl_order[:-1]  # removes cell lines missing expression data

    plt.figure(figsize=(50 * mm, 120 * mm))
    plt.subplots_adjust(left=0.3, top=0.95, bottom=0.07)
    sns.boxplot(
        pseudobulk_correlation_with_ccle[~pseudobulk_correlation_with_ccle["Identity"]],
        y="scCellLine",
        x="TopVarCorrelation",
        order=cl_order,
        hue="OncotreeLineage",
        palette=lineage_palette,
        showfliers=False,
        legend=False,
    )
    sns.scatterplot(
        pseudobulk_correlation_with_ccle[pseudobulk_correlation_with_ccle["Identity"]],
        y="scCellLine",
        x="TopVarCorrelation",
        c="tab:red",
        marker="X",
    )
    plt.ylabel("Cell Line")
    plt.xlabel("Pearson's $r$")
    plt.savefig(os.path.join(FIGURE_DIR, "control_pseudobulk_ccle_correlation.png"))


def plot_depletion_vs_gene_effect_paneled():
    depletion_table = load_file(
        "depletion_vs_crispr_table.csv", local_dir=PROCESSED_DIR
    )

    cl_to_correlation = {}
    for cl in depletion_table["cell_line"].unique().tolist():
        cl_subset = depletion_table[depletion_table["cell_line"] == cl]
        cl_to_correlation[cl] = stats.pearsonr(
            cl_subset.dropna()["depletion"], cl_subset.dropna().gene_effect
        ).statistic
    cl_to_correlation = pd.Series(cl_to_correlation)

    fig, axs = plt.subplots(
        4,
        4,
        figsize=(120 * mm, 120 * mm),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.25},
    )
    plt.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.09)
    axs = axs.flatten()

    for i, cl in enumerate(
        cl_to_correlation.sort_values(ascending=False).index.tolist()
    ):
        sns.scatterplot(
            depletion_table[depletion_table["cell_line"] == cl],
            x="gene_effect",
            y="depletion",
            hue="target_class",
            palette=gene_class_palette,
            alpha=0.8,
            legend=False,
            s=8,
            ax=axs[i],
        )
        sns.regplot(
            depletion_table[depletion_table["cell_line"] == cl],
            x="gene_effect",
            y="depletion",
            scatter=False,
            line_kws=dict(color="black", linestyle="solid", linewidth=1),
            ax=axs[i],
        )
        axs[i].text(
            0.05,
            0.9,
            f"r = {cl_to_correlation[cl]:.02f}",
            transform=axs[i].transAxes,
            size=ANNOT_SIZE,
        )
        axs[i].axhline(0, linestyle="solid", color="black", zorder=0, linewidth=0.5)
        axs[i].axvline(0, linestyle="solid", color="black", zorder=0, linewidth=0.5)
        axs[i].set_title(cl, size=LABEL_SIZE)
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")

    fig.supxlabel("Dependency score (Chronos)", size=TITLE_SIZE)
    fig.supylabel("Relative cell depletion", size=TITLE_SIZE)

    plt.savefig(os.path.join(FIGURE_DIR, "cell_recovery_vs_viability_paneled.png"))


def plot_upregulated_targets(fdr_threshold=0.05):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    cell_lines = cl_metadata["cell_line"].tolist()

    target_up_detected = pd.DataFrame(
        {
            cl: pd.Series(
                np.diag(
                    ((sceptre_fdr < fdr_threshold) & (sceptre_zscore > 0))
                    .loc[:, pd.IndexSlice[cl, :]]
                    .droplevel(level=0, axis=1)
                    .reindex(
                        index=ko_metadata["knockout"].tolist(),
                        columns=ko_metadata["knockout"].tolist(),
                    )
                ),
                index=ko_metadata["knockout"].tolist(),
            ).fillna(False)
            for cl in cell_lines
        }
    )

    target_up_df = (
        target_up_detected.sum(axis=1)
        .loc[lambda x: x > 0]
        .sort_values(ascending=False)
        .rename("n_lines")
        .rename_axis("knockout")
        .reset_index()
    )
    target_up_df = target_up_df.merge(ko_metadata[["knockout", "target_class"]])
    plt.figure(figsize=(30 * mm, 45 * mm))
    plt.subplots_adjust(left=0.3, right=0.98, top=0.95, bottom=0.18)
    sns.barplot(
        target_up_df,
        x="knockout",
        y="n_lines",
        hue="target_class",
        palette=gene_class_palette,
        legend=False,
    )
    plt.xlabel("sgRNA Target")
    plt.ylabel("Cell lines with upregulation")
    plt.savefig(os.path.join(FIGURE_DIR, "upregulated_targets.png"))


def plot_supplemental_cell_quality_covariates():
    cell_quality_covariates = load_file(
        "cell_quality_covariates.csv", local_dir=PROCESSED_DIR
    )
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    cell_quality_covariates = cell_quality_covariates.merge(
        cl_metadata.set_index("cell_line")["OncotreeLineage"],
        left_on="cell_line",
        right_index=True,
    )

    xlabels = ["avg_umis_per_ko", "n_cells_per_ko", "avg_genes_per_ko"]
    xprettys = [
        "Mean UMIs per cell per knockout",
        "Mean cells per knockout",
        "Mean genes per cell per knockout",
    ]
    ylabel = "n_self_downregulated"
    ypretty = "sgRNA targets downregulated"

    for i, xlab in enumerate(xlabels):
        plt.figure(figsize=(45 * mm, 45 * mm))
        plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
        sns.scatterplot(
            cell_quality_covariates,
            x=xlab,
            y=ylabel,
            hue="OncotreeLineage",
            palette=lineage_palette,
            edgecolor="black",
            legend=False,
        )
        sns.regplot(
            cell_quality_covariates,
            x=xlab,
            y=ylabel,
            scatter=False,
            line_kws={"color": "black"},
        )
        corr_result = stats.pearsonr(
            cell_quality_covariates[xlab], cell_quality_covariates[ylabel]
        )
        print(f"{xprettys[i]} correlation:", corr_result)
        plt.text(
            0.04,
            0.93,
            f"r = {corr_result.statistic:.02f}",
            transform=plt.gca().transAxes,
        )
        plt.xlabel(xprettys[i])
        plt.ylabel(ypretty)
        plt.savefig(os.path.join(FIGURE_DIR, f"targets_down_vs_{xlab}.png"))


def main():
    # figure 3a
    plot_pseudobulk_correlation()

    # figure 3b
    plot_depletion_vs_gene_effect_paneled()

    # figure 3c
    plot_upregulated_targets(fdr_threshold=0.05)

    # figure 3d-f
    plot_supplemental_cell_quality_covariates()


if __name__ == "__main__":
    main()
