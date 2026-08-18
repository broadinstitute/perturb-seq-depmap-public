import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from constants import *
from data_utils import *
from figure_utils import *
from matplotlib.lines import Line2D
from scipy import stats

mpl.style.use("assets/stylesheet.mplstyle")


# figure 3a
def plot_depletion_vs_gene_effect():
    cells_per_ko_table = load_file(
        "depletion_vs_crispr_table.csv", local_dir=PROCESSED_DIR
    )

    plt.figure(figsize=(75 * mm, 60 * mm))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    sns.scatterplot(
        cells_per_ko_table.sample(frac=1),  # randomized layering
        x="gene_effect",
        y="depletion",
        hue="target_class",
        palette=gene_class_palette,
        alpha=0.8,
        legend=False,
        s=8,
    )

    ordered_regression = cells_per_ko_table.sort_values("gene_effect", ascending=True)
    plt.plot(
        ordered_regression["gene_effect"],
        ordered_regression["y_pred"],
        c="black",
        linestyle="dashed",
    )

    handles = [
        Line2D(
            [],
            [],
            color=col,
            marker="o",
            linestyle="None",
            markersize=4,
            markeredgecolor="white",
            label=gc,
        )
        for gc, col in gene_class_palette.items()
    ]
    legend = plt.legend(
        title="Target class",
        handles=handles,
        handleheight=1,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1.5,
        loc="upper left",
        ncol=1,
        fontsize=ANNOT_SIZE,
    )

    corr_result = stats.pearsonr(
        cells_per_ko_table.dropna()["depletion"],
        cells_per_ko_table.dropna()["gene_effect"],
    )
    print("correlation:", corr_result)
    plt.text(
        0.1, 0.62, f"r = {corr_result.statistic:.02f}", transform=plt.gca().transAxes
    )
    plt.xlabel("Gene effect (Chronos)")
    plt.ylabel("Cell recovery relative to controls")
    plt.savefig(os.path.join(FIGURE_DIR, "cell_recovery_vs_viability.png"))


# figure 3b
def plot_myc_target_validation():
    myc_df = load_file("myc_validation_enrichment_table.csv", local_dir=PROCESSED_DIR)

    cl_order = (
        myc_df.sort_values("median_z_myc_targets", ascending=True)["cell_line"]
        .unique()
        .tolist()
    )
    hue_norm = (0, 10)
    cmap = "Blues"

    fig, axs = plt.subplots(1, 2, figsize=(85 * mm, 60 * mm), width_ratios=[39, 1])

    sns.boxplot(
        myc_df,
        x="cell_line",
        y="z",
        order=cl_order,
        hue="p_transform",
        palette=cmap,
        hue_norm=hue_norm,
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
        medianprops={"linewidth": 1, "color": "tab:red"},
        showfliers=False,
        legend=False,
        ax=axs[0],
    )
    axs[0].set_ylabel("MYC target z-score after MYC knockout")
    axs[0].set_xlabel("Cell line")
    axs[0].set_xticks(axs[0].get_xticks(), cl_order, rotation=90)
    axs[0].axhline(0, color="black", linestyle="dashed", alpha=0.5, zorder=-1)

    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.25)
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(hue_norm[0], hue_norm[1]), cmap=cmap
        ),
        cax=axs[1],
        orientation="vertical",
        label=r"-$\log_{10}$(p-value) of MYC target enrichment",
    )
    plt.savefig(os.path.join(FIGURE_DIR, "myc_ko_target_zscores.png"))


# figure 3c
def plot_mtpap_validation():
    melt_df = load_file("mtpap_validation_table.csv", local_dir=PROCESSED_DIR)

    gene_order = (
        melt_df.groupby("response_id")["z_orig"].median().sort_values().index.tolist()
    )

    fig, axs = plt.subplots(1, 2, figsize=(85 * mm, 60 * mm), width_ratios=[39, 1])
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.25)
    sns.boxplot(
        melt_df,
        x="response_id",
        y="z_orig",
        order=gene_order,
        # hue='mean_control_mt_expr', hue_norm=(5, 9), palette='viridis',
        showfliers=False,
        fill=False,
        color="black",
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
        medianprops={"linewidth": 1, "color": "tab:red"},
        ax=axs[0],
    )
    sns.stripplot(
        melt_df,
        x="response_id",
        y="z_orig",
        order=gene_order,
        hue="mt_expr",
        hue_norm=(4, 9),
        palette="viridis",
        edgecolor="black",
        linewidth=0.25,
        s=3,
        legend=False,
        ax=axs[0],
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(4, 9), cmap="viridis"),
        cax=axs[1],
        orientation="vertical",
        label="Average expression in controls\n($\\log$(TPM + 1))",
    )

    axs[0].axhline(0, color="black", linestyle="dashed", alpha=0.5)
    axs[0].set_xticks(
        axs[0].get_xticks(),
        [t.get_text() for t in axs[0].get_xticklabels()],
        rotation=90,
    )
    axs[0].set_xlabel("Gene")
    axs[0].set_ylabel("Z-scored expression (relative to controls)")
    plt.savefig(os.path.join(FIGURE_DIR, "mtpap_ko_zscores.png"))


# figure 3d
def plot_cell_quality_covariates():
    cell_quality_covariates = load_file(
        "cell_quality_covariates.csv", local_dir=PROCESSED_DIR
    )
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    cell_quality_covariates = cell_quality_covariates.merge(
        cl_metadata.set_index("cell_line")["OncotreeLineage"],
        left_on="cell_line",
        right_index=True,
    )

    plt.figure(figsize=(73 * mm, 60 * mm))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    sns.scatterplot(
        cell_quality_covariates,
        x="total_umis_per_ko",
        y="n_self_downregulated",
        hue="OncotreeLineage",
        palette=lineage_palette,
        edgecolor="black",
        legend=False,
    )
    sns.regplot(
        cell_quality_covariates,
        x="total_umis_per_ko",
        y="n_self_downregulated",
        scatter=False,
        line_kws={"color": "black"},
    )

    corr_result = stats.pearsonr(
        cell_quality_covariates["total_umis_per_ko"],
        cell_quality_covariates["n_self_downregulated"],
    )
    print("correlation:", corr_result)
    plt.text(
        0.04, 0.93, f"r = {corr_result.statistic:.02f}", transform=plt.gca().transAxes
    )
    plt.xlabel("Mean total UMIs in targeted cells")
    plt.ylabel("Number of gRNA targets downregulated")
    plt.savefig(os.path.join(FIGURE_DIR, "targets_down_vs_total_umis.png"))


# figure 3e
def plot_downsample_curve():
    pct_sig_df = load_file("deep_downsample_curve.csv", local_dir=PROCESSED_DIR)

    targets = pct_sig_df["target"].unique()
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))[: len(targets)]
    palette = {item: tab10_colors[i] for i, item in enumerate(targets)}

    ko_handles = [
        Line2D([0], [0], color=color, lw=1, label=label)
        for label, color in palette.items()
    ]
    cl_handles = [
        Line2D([0], [0], color="k", lw=1, label="KMRC20"),
        Line2D([0], [0], color="k", lw=1, ls="--", label="UMRC3"),
    ]
    ref_handles = [
        Line2D([0], [0], color="k", lw=1, ls=":", label="Replogle\net al. 2022"),
        Line2D([0], [0], color="red", lw=1, ls=":", label="5e6"),
    ]
    handles = (
        [Line2D([0], [0], color="w", label="target")]
        + ko_handles
        + [Line2D([0], [0], color="w", label="cell line")]
        + cl_handles
        + [Line2D([0], [0], color="w", label="reference")]
        + ref_handles
    )

    plt.figure(figsize=(100 * mm, 60 * mm))
    ax = plt.gca()

    sns.lineplot(
        pct_sig_df,
        x="mean_umi_per_ko",
        y="pct_sig",
        hue="target",
        style="cell_line",
        errorbar=None,
        palette=palette,
        ax=ax,
    )
    ax.set_ylabel("Percent significant expected genes\ncompared to deepest sample")
    ax.set_xlabel("Mean UMIs per knockout")
    ax.axvline(2.2e6, color="black", ls=":")
    ax.axvline(5e6, color="red", ls=":")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5))

    plt.subplots_adjust(left=0.15, right=0.7, bottom=0.2)
    plt.savefig(os.path.join(FIGURE_DIR, "rescreen_downsample.png"))


def main():
    # figure 3a
    plot_depletion_vs_gene_effect()

    # figure 3b
    plot_myc_target_validation()

    # figure 3c
    plot_mtpap_validation()

    # figure 3d
    plot_cell_quality_covariates()

    # figure 3e
    plot_downsample_curve()


if __name__ == "__main__":
    main()
