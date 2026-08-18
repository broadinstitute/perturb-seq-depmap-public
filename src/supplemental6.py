import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm
from adjustText import adjust_text
from constants import *
from data_utils import *
from figure_utils import *

mpl.style.use("assets/stylesheet.mplstyle")


# sup 6a
def plot_rna_processing_correlation_heatmap():
    full_corr_matrix = load_file(
        "full_corr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=[0, 1]
    )

    def plot_symmetric_correlation_heatmap(
        corr_matrix,
        cell_line_subset=slice(None),
        gene_subset=slice(None),
        figsize=(7, 6),
        row_label_type="cell_line",
        col_label_type="gene",
        divider_linewidth=0.25,
        **plt_kwargs,
    ):
        # preprocess inputs
        if row_label_type not in ["cell_line", "gene"]:
            row_label_type = "cell_line"
        if col_label_type not in ["cell_line", "gene"]:
            col_label_type = "gene"

        # subset the correlation matrix with the corresponding information for each level
        subset = corr_matrix.loc[
            (cell_line_subset, gene_subset), (cell_line_subset, gene_subset)
        ]

        # plot the subset
        plt.figure(figsize=figsize)
        ax = sns.heatmap(subset, center=0, vmin=-1, vmax=1, cmap="RdBu_r", **plt_kwargs)

        # identify which level to pull based on the annotation type
        if row_label_type == "cell_line":
            row_label_order = [x[0] for x in subset.index.tolist()]
        elif row_label_type == "gene":
            row_label_order = [x[1] for x in subset.index.tolist()]

        if col_label_type == "cell_line":
            col_label_order = [x[0] for x in subset.index.tolist()]
        elif col_label_type == "gene":
            col_label_order = [x[1] for x in subset.index.tolist()]

        # identify the midpoint indices, the corresponding labels, and the breakpoints for each label
        row_idxs, row_labels, row_breakpoints = identify_label_centers(row_label_order)
        col_idxs, col_labels, col_breakpoints = identify_label_centers(col_label_order)

        # if the breakpoints are more separable than just the heatmap grid, plot the dividing lines
        if len(row_breakpoints) < subset.shape[0]:
            for rb in row_breakpoints[1:-1]:
                plt.axhline(rb, color="black", linewidth=divider_linewidth)
                plt.axvline(rb, color="black", linewidth=divider_linewidth)
        if len(col_breakpoints) < subset.shape[1]:
            for cb in col_breakpoints[1:-1]:
                plt.axhline(cb, color="black", linewidth=divider_linewidth)
                plt.axvline(cb, color="black", linewidth=divider_linewidth)
            xtickrotation = 0
        else:
            xtickrotation = 90

        ticks_info = {
            "x_idx": col_idxs + 0.5,
            "x_label": col_labels,
            "y_idx": row_idxs + 0.5,
            "y_label": row_labels,
        }
        return ticks_info

    ti = plot_symmetric_correlation_heatmap(
        full_corr_matrix,
        gene_subset=["SMG6", "XRN1", "ADAR"],
        cbar_kws={"label": "Pearson r"},
        figsize=(80 * mm, 65 * mm),
        row_label_type="gene",
        divider_linewidth=1,
    )
    plt.xlabel("Perturbation")
    plt.ylabel("Perturbation")
    plt.gca().set_xticks(ti["y_idx"], ti["y_label"], rotation=0)
    plt.gca().set_yticks(ti["x_idx"], ti["x_label"], rotation=0)
    plt.subplots_adjust(left=0.15, right=0.94)
    plt.savefig(os.path.join(FIGURE_DIR, "rna_processing_heatmap.png"))


# sup 6b
def plot_rna_processing_scatterplot():
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)

    snornas = all_genesets[
        all_genesets["term"] == "Small nucleolar RNA non-coding host genes"
    ]["gene"].tolist()
    print(
        (
            sceptre_zscore.groupby(level=1, axis=1)
            .mean()
            .loc[:, ["SMG6", "XRN1", "ADAR"]]
            .reindex(index=snornas)
            > 5
        )
        .sum()
        .rename("# snoRNAs with z-score > 5")
    )

    genes_to_show = (
        (sceptre_fdr < 0.05).sum(axis=1).loc[lambda x: x >= 5].index.tolist()
    )
    df_to_plot = (
        sceptre_zscore.groupby(level=1, axis=1)
        .mean()
        .loc[:, ["SMG6", "XRN1", "ADAR"]]
        .loc[genes_to_show, :]
    )

    xlabel = "SMG6"
    ylabel = "XRN1"

    fig, axs = plt.subplots(1, 2, figsize=(90 * mm, 65 * mm), width_ratios=[39, 1])
    sns.scatterplot(
        df_to_plot,
        x="SMG6",
        y="XRN1",
        hue="ADAR",
        hue_norm=(-5, 5),
        palette="RdBu_r",
        linewidth=0.25,
        edgecolor="black",
        s=8,
        ax=axs[0],
        legend=False,
        zorder=500,
    )
    axs[0].axhline(0, color="black", linestyle="solid", alpha=0.5, zorder=0)
    axs[0].axvline(0, color="black", linestyle="solid", alpha=0.5, zorder=0)

    annots = df_to_plot.loc[
        list(
            set(
                df_to_plot["SMG6"].sort_values(ascending=False).head(10).index.tolist()
            ).union(
                df_to_plot["XRN1"].sort_values(ascending=False).head(10).index.tolist()
            )
        )
    ]
    texts = [
        axs[0].text(
            r[xlabel], r[ylabel], i, ha="center", va="center", fontsize=ANNOT_SIZE
        )
        for i, r in annots.iterrows()
    ]
    adjust_text(
        texts,
        ax=axs[0],
        expand=(1.7, 1.6),
        force_static=(0.4, 0.2),
        arrowprops=annot_arrow_props,
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-5, 5), cmap="RdBu_r"),
        cax=axs[1],
        orientation="vertical",
        label="Mean z-score after ADAR knockout",
    )
    axs[0].set_xlabel("Mean z-score after SMG6 knockout")
    axs[0].set_ylabel("Mean z-score after XRN1 knockout")

    plt.subplots_adjust(left=0.15, right=0.88)
    plt.savefig(os.path.join(FIGURE_DIR, "rna_processing_scatter.png"))


# sup 6c
def p53_pathway_by_cell_line_property_scatter():
    geneset_zscore_matrix = load_file(
        "geneset_mean_zscore_matrix.csv",
        local_dir=PROCESSED_DIR,
        header=[0, 1],
        index_col=[0, 1],
    )
    cdf_full = load_file(
        "p53_cell_line_properties.csv", local_dir=PROCESSED_DIR, index_col=[0, 1]
    )

    p53_table = (
        geneset_zscore_matrix.loc[
            ("Hallmark", "HALLMARK_P53_PATHWAY"), pd.IndexSlice[:, ["MDM2", "UBE3A"]]
        ]
        .rename("p53_mean_z")
        .rename_axis(["cell_line", "grna_target"])
        .reset_index()
        .pivot(index="cell_line", columns="grna_target", values="p53_mean_z")
        .merge(
            cdf_full.reset_index().drop_duplicates(subset=["cell_line"])[
                ["cell_line", "OncotreeLineage", "TP53_damaging"]
            ],
            left_index=True,
            right_on="cell_line",
        )
    )
    p53_table["TP53 status"] = p53_table["TP53_damaging"].replace(
        {0: "WT", 1: "Mut.", 2: "Mut."}
    )
    p53_table["Lineage"] = p53_table["OncotreeLineage"]

    plt.figure(figsize=(70 * mm, 60 * mm))
    sns.scatterplot(
        p53_table,
        x="MDM2",
        y="UBE3A",
        hue="Lineage",
        palette=lineage_palette,
        style="TP53 status",
        markers={"WT": "o", "Mut.": "X"},
        linewidth=0.5,
        edgecolor="black",
    )

    annots = p53_table[
        p53_table["cell_line"].isin(["C4I", "SKGII", "UMRC3", "SLR23"])
    ].set_index("cell_line")
    manually_annotate(plt.gca(), "SKGII", annots, "MDM2", "UBE3A", (0.4, -0.1))
    manually_annotate(plt.gca(), "C4I", annots, "MDM2", "UBE3A", (0.4, -0.1))
    manually_annotate(plt.gca(), "UMRC3", annots, "MDM2", "UBE3A", (-0.4, 0.2))
    manually_annotate(plt.gca(), "SLR23", annots, "MDM2", "UBE3A", (-0.4, 0.1))

    h, _ = plt.gca().get_legend_handles_labels()
    plt.legend(
        title="TP53 status", handles=h[-2:], loc="upper right", bbox_to_anchor=(1, 1)
    )
    plt.xlabel("P53 pathway z-score after MDM2 knockout")
    plt.ylabel("P53 pathway z-score after UBE3A knockout")
    plt.axvline(0, linestyle="solid", color="black", zorder=0, alpha=0.5)
    plt.axhline(0, linestyle="solid", color="black", zorder=0, alpha=0.5)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "p53_cell_line_property_pathway_scatter.png"))


# sup 6d
def plot_tfrc_dependency_gapdh_expression_scatter():
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)

    tfrc_table = (
        pd.concat(
            [
                sceptre_zscore.loc["GAPDH", pd.IndexSlice[:, "TFRC"]].rename("GAPDH_z"),
                sceptre_fdr.loc["GAPDH", pd.IndexSlice[:, "TFRC"]].rename("GAPDH_FDR"),
                crispr_table[crispr_table["guide"] == "TFRC"].set_index(
                    ["cell_line", "guide"]
                )["gene_effect"],
            ],
            axis=1,
        )
        .rename_axis(["cell_line", "assigned_ko"])
        .reset_index()
    )
    tfrc_table = tfrc_table.merge(cl_metadata[["cell_line", "OncotreeLineage"]])
    print(tfrc_table)

    plt.figure(figsize=(100 * mm, 60 * mm))

    sns.scatterplot(
        tfrc_table,
        x="gene_effect",
        y="GAPDH_z",
        hue="OncotreeLineage",
        palette=lineage_palette,
        linewidth=0.5,
        edgecolor="black",
        # legend=False
    )

    corr_result = scipy.stats.pearsonr(tfrc_table["gene_effect"], tfrc_table["GAPDH_z"])
    print("correlation:", corr_result)

    sns.regplot(
        tfrc_table, x="gene_effect", y="GAPDH_z", scatter=False, color="tab:gray"
    )

    plt.text(
        0.05,
        0.05,
        f"r = {corr_result.statistic:.02f}",
        transform=plt.gca().transAxes,
        fontsize=ANNOT_SIZE,
    )

    plt.xlabel("TFRC gene effect")
    plt.ylabel("GAPDH z-score after TFRC knockout")

    plt.legend(title="Lineage", loc="upper left", bbox_to_anchor=(1.03, 1))

    plt.subplots_adjust(left=0.12, right=0.68, top=0.9, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "tfrc_gapdh_scatter.png"))


def main():
    # sup 6a
    plot_rna_processing_correlation_heatmap()

    # sup 6b
    plot_rna_processing_scatterplot()

    # sup 6c
    p53_pathway_by_cell_line_property_scatter()

    # sup 6d
    plot_tfrc_dependency_gapdh_expression_scatter()


if __name__ == "__main__":
    main()
