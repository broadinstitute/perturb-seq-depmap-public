import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm as tqdm
from constants import *
from data_utils import *
from figure_utils import *

mpl.style.use("assets/stylesheet.mplstyle")


# figure 1b
def plot_cell_line_panel_composition():
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    annotations = cl_metadata.groupby("OncotreeLineage").apply(
        lambda x: ", ".join(sorted(list(x["cell_line"])))
    )
    lineage_counts = (
        cl_metadata.value_counts("OncotreeLineage").rename("n_cell_lines").reset_index()
    )

    plt.figure(figsize=(60 * mm, 50 * mm))
    plt.subplots_adjust(left=0.45, right=0.95, bottom=0.15)
    ax = sns.barplot(
        lineage_counts,
        x="n_cell_lines",
        y="OncotreeLineage",
        hue="OncotreeLineage",
        palette=lineage_palette,
    )
    plt.ylabel("Oncotree lineage")
    plt.xlabel("Number of cell lines")
    # plt.title('Cell Line Panel')
    plt.xticks([0, 1, 2, 3], [0, 1, 2, 3])

    for i, r in lineage_counts.iterrows():
        ax.annotate(
            annotations.loc[r["OncotreeLineage"]],
            xy=(0.05, i),
            ha="left",
            va="center",
            size=ANNOT_SIZE,
        )

    plt.savefig(os.path.join(FIGURE_DIR, "lineage_breakdown.png"))


# figure 1c
def plot_library_composition():
    ko_metadata = load_file("knockout_metadata.csv", local_dir=None)
    class_count = ko_metadata.groupby("target_class").count()
    class_count_dict = class_count["knockout"].to_dict()
    depmap_gene_effects = load_file(
        "CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0
    )
    depmap_gene_effects.columns = [
        x.split(" (")[0] for x in depmap_gene_effects.columns
    ]
    ko_gene_effects_longform = (
        depmap_gene_effects.reindex(columns=ko_metadata["knockout"].tolist())
        .dropna(how="all", axis=1)
        .melt(ignore_index=False, var_name="knockout", value_name="gene_effect")
        .rename_axis("arxspan_id")
        .reset_index()
        .merge(ko_metadata)
    )

    group_order = [
        "Common Essential",
        "High Variance",
        "Selective",
        "Olfactory Control",
    ]

    fig, ax_map = prepare_gridspec(
        ax_map=pd.DataFrame([[True, True]]),
        figsize=(90 * mm, 50 * mm),
        height_ratios=[1],
        width_ratios=[1.1, 1],
        wspace=0.05,
        inner_gs_dict={
            (0, 0): {"nrows": 1, "ncols": 1},
            (0, 1): {"nrows": 4, "ncols": 1, "hspace": 0},
        },
    )

    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.15)
    sns.barplot(
        class_count.reset_index(),
        x="knockout",
        y="target_class",
        order=group_order,
        hue="target_class",
        palette=gene_class_palette,
        ax=ax_map[0][0][(0, 0)],
    )

    rects = ax_map[0][0][(0, 0)].patches
    for rect, label in zip(rects, list(class_count_dict.keys())):
        ax_map[0][0][(0, 0)].text(
            rect.get_x() + rect.get_width() - 1,
            rect.get_y() + (rect.get_height() / 2),
            class_count_dict[label],
            ha="right",
            va="center",
            fontdict={"color": "white", "size": ANNOT_SIZE},
        )

    for i, grp in enumerate(group_order):
        grp_subset = ko_gene_effects_longform[
            ko_gene_effects_longform["target_class"] == grp
        ]
        for gene_id in grp_subset["cds_gene_id"].unique().tolist():
            gene_subset = grp_subset[grp_subset["cds_gene_id"] == gene_id]
            sns.kdeplot(
                gene_subset,
                x="gene_effect",
                c=gene_class_palette[grp],
                ax=ax_map[0][1][(i, 0)],
                linewidth=0.5,
            )
        ax_map[0][1][(i, 0)].set_xlim(-4.75, 0.75)
        ax_map[0][1][(i, 0)].set_ylabel("")
        ax_map[0][1][(i, 0)].set_yticks([])
        ax_map[0][1][(i, 0)].axvline(
            0, linestyle="dashed", color="black", linewidth=0.5
        )

    ax_map[0][0][(0, 0)].set_xlabel("Knockouts")
    ax_map[0][0][(0, 0)].set_ylabel("Target class")
    ax_map[0][1][(len(group_order) - 1, 0)].set_xlabel("Gene dependency (Chronos)")

    plt.savefig(os.path.join(FIGURE_DIR, "guide_library_breakdown.png"))


def main():
    # figure 1b
    plot_cell_line_panel_composition()

    # figure 1c
    plot_library_composition()


if __name__ == "__main__":
    main()
