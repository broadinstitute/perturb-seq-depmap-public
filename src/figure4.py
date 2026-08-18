import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import *
from data_utils import *
from figure_utils import *
from scipy import stats
from taigapy import create_taiga_client_v3

tc = create_taiga_client_v3()

mpl.style.use("assets/stylesheet.mplstyle")


# figure 4a
def plot_transcriptional_change_vs_crispr():
    transcriptional_change_df = load_file(
        "transcriptional_change_table.csv", local_dir=PROCESSED_DIR
    )
    transcriptional_change_df = transcriptional_change_df.dropna(
        subset=["deviation_from_basal", "gene_effect"]
    )

    plt.figure(figsize=(60 * mm, 60 * mm))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    sns.scatterplot(
        transcriptional_change_df,
        x="gene_effect",
        y="deviation_from_basal",
        hue="target_class",
        palette=gene_class_palette,
        s=8,
        alpha=0.8,
    )
    sns.regplot(
        transcriptional_change_df,
        x="gene_effect",
        y="deviation_from_basal",
        scatter=False,
        color="tab:gray",
    )

    corr_result = stats.pearsonr(
        transcriptional_change_df["gene_effect"],
        transcriptional_change_df["deviation_from_basal"],
    )
    print("correlation:", corr_result)
    plt.text(
        0.03, 0.94, f"r = {corr_result.statistic:.02f}", transform=plt.gca().transAxes
    )
    plt.xlabel("Gene effect (Chronos)")
    plt.ylabel(r"Deviation from basal state ($\sqrt{1 - r^2}$)")
    plt.legend(
        title="Target class",
        handleheight=1,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1.5,
        loc="upper right",
        ncol=1,
        fontsize=ANNOT_SIZE,
    )
    plt.savefig(os.path.join(FIGURE_DIR, "transcriptional_change_vs_gene_effect.png"))


# figure 4b
def plot_top_dependency_diff_expr_genes():
    top_dependency_diff_expr_gene_df = load_file(
        "dependency_top_diff_expr_gene_table.csv", local_dir=PROCESSED_DIR
    )
    expr_order = (
        top_dependency_diff_expr_gene_df.set_index("response_id")["response_id_order"]
        .drop_duplicates()
        .sort_values()
        .index.tolist()
    )
    ko_order = (
        top_dependency_diff_expr_gene_df.set_index("grna_target")["target_order"]
        .drop_duplicates()
        .sort_values()
        .index.tolist()
    )

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(110 * mm, 55 * mm),
        width_ratios=[49, 1],
        gridspec_kw={"wspace": 0.05},
    )
    plt.subplots_adjust(left=0.21, right=0.92, top=0.95, bottom=0.3)
    categorical_scatterplot(
        top_dependency_diff_expr_gene_df,
        xlabel="grna_target",
        ylabel="response_id",
        xorder=ko_order,
        yorder=expr_order,
        is_x_cat=True,
        is_y_cat=True,
        xtickrot=90,
        dodge_shift=(0, 0),
        jitter=(0, 0),
        hue="mean_dep_z",
        palette="RdBu_r",
        hue_norm=(-5, 5),
        size="frac_dependent",
        sizes=(8, 24),
        ax=axs[0],
    )
    axs[0].set_xlabel("Knockout")
    axs[0].set_ylabel("Response gene")

    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-5, vmax=5)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=axs[1],
        orientation="vertical",
        label="Mean z-score",
    )

    axs[0].legend(
        title="Fraction of dependent\nlines significant",
        handles=axs[0].get_legend_handles_labels()[0][-6:],
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(-0.31, -0.46),
    )
    plt.savefig(os.path.join(FIGURE_DIR, "top_differential_genes.png"))


# figure 4c
def plot_dependency_hallmark_geneset_scores():
    geneset_zscore_matrix = load_file(
        "geneset_mean_zscore_matrix.csv",
        local_dir=PROCESSED_DIR,
        header=[0, 1],
        index_col=[0, 1],
    )
    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)

    selected_genesets = [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_MYC_TARGETS_V2",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_P53_PATHWAY",
        "HALLMARK_APOPTOSIS",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    ]

    dependent_geneset_z_matrix = geneset_zscore_matrix.loc[
        pd.IndexSlice["Hallmark", selected_genesets],
        crispr_table[crispr_table["is_dependent"]]
        .apply(lambda x: (x["cell_line"], x["guide"]), axis=1)
        .values.tolist(),
    ].droplevel(0, axis=0)

    dependent_geneset_mean_z_matrix = (
        dependent_geneset_z_matrix.T.groupby(level=1).mean().T
    )

    ko_order = (
        dependent_geneset_mean_z_matrix.loc[
            [
                "HALLMARK_E2F_TARGETS",
                "HALLMARK_G2M_CHECKPOINT",
                "HALLMARK_MYC_TARGETS_V1",
                "HALLMARK_MYC_TARGETS_V2",
            ],
            # ['HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT'],
            :,
        ]
        .mean()
        .sort_values()
        .index.tolist()
    )
    dependent_geneset_mean_z_matrix.index = dependent_geneset_mean_z_matrix.index.map(
        lambda x: clean_geneset_name(x, nth=10, remove_prefix=True)
    )

    plt.figure(figsize=(170 * mm, 50 * mm))
    plt.subplots_adjust(left=0.2, right=1.05, top=0.95, bottom=0.3)
    sns.heatmap(
        dependent_geneset_mean_z_matrix.loc[:, ko_order],
        vmin=-2,
        vmax=2,
        cmap="RdBu_r",
        cbar_kws={"label": "Mean z-score"},
    )
    plt.xlabel("Knockout")
    plt.ylabel("Term")
    plt.savefig(os.path.join(FIGURE_DIR, "dependency_pathway_heatmap.png"))


# figure 4d
def plot_mistimed_perturbation_scatter():
    timepoint_pred_table = load_file(
        "timepoint_prediction_table.csv", local_dir=PROCESSED_DIR
    )

    normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5)
    colormap = mpl.cm.PiYG

    plt.figure(figsize=(85 * mm, 60 * mm))

    sns.scatterplot(
        timepoint_pred_table,
        x="depletion",
        y="y_pred",
        hue="CellCycleDependentMeanZ",
        palette=colormap,
        hue_norm=normalize,
        s=8,
        linewidth=0.25,
        edgecolor="gray",
        legend=False,
    )

    plt.axline((0, 0), slope=1, linestyle="dashed", color="black", zorder=-1)
    plt.xlabel("Observed cell recovery")
    plt.ylabel("Expected cell recovery")

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.fill_between(
        [xmin, ymin, ymax],
        y1=ymax,
        y2=[ymin, ymin, ymax],
        zorder=-1,
        facecolor="tab:gray",
        alpha=0.1,
    )
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "mistimed_scatter.png"))


# figure 4e
def plot_mistimed_perturbation_bar():
    timepoint_pred_table = load_file(
        "timepoint_prediction_table.csv", local_dir=PROCESSED_DIR
    )
    mistimed_bar_df = load_file(
        "mistimed_perturbation_table.csv", local_dir=PROCESSED_DIR, index_col=0
    )

    normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5)
    colormap = mpl.cm.PiYG

    fig, axs = plt.subplots(1, 2, figsize=(85 * mm, 60 * mm), width_ratios=[39, 1])

    depletion_ratio_label = "observed_vs_expected_cell_ratio_total"
    mistimed_bar_df[depletion_ratio_label] = mistimed_bar_df[depletion_ratio_label] - 1
    sns.barplot(
        mistimed_bar_df.reset_index(),
        y="assigned_ko",
        x=depletion_ratio_label,
        hue="mean_cc",
        order=mistimed_bar_df["mean_cc"].sort_values(ascending=False).index.tolist(),
        palette=colormap,
        hue_norm=normalize,
        linewidth=0.25,
        edgecolor="black",
        width=0.75,
        legend=False,
        ax=axs[0],
    )

    axs[0].set_ylabel("Target")
    axs[0].set_xlabel("Total observed / expected cells recovered")
    axs[0].set_xticks(
        axs[0].get_xticks(), [f"{x + 1:.1f}" for x in axs[0].get_xticks()]
    )

    scalarmappaple = mpl.cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(timepoint_pred_table["CellCycleDependentMeanZ"])
    plt.colorbar(scalarmappaple, cax=axs[1])
    axs[1].set_ylabel("Mean z-score of cell cycle genes")
    axs[1].yaxis.set_label_position("left")
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "mistimed_bar.png"))


def main():
    # figure 4a
    plot_transcriptional_change_vs_crispr()

    # figure 4b
    plot_top_dependency_diff_expr_genes()

    # figure 4c
    plot_dependency_hallmark_geneset_scores()

    # figure 4d
    plot_mistimed_perturbation_scatter()

    # figure 4e
    plot_mistimed_perturbation_bar()


if __name__ == "__main__":
    main()
