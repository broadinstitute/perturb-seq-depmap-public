import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm
from constants import *
from data_utils import *
from figure_utils import *

mpl.style.use("assets/stylesheet.mplstyle")


def plot_all_by_all_correlation_heatmap():
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

    sorted_full_corr_matrix = (
        full_corr_matrix.dropna(how="all", axis=0)
        .dropna(how="all", axis=1)
        .sort_index(level=[1, 0], axis=0)
        .sort_index(level=[1, 0], axis=1)
    )
    low_cell_line_knockouts = (
        sorted_full_corr_matrix.groupby(level=[1])
        .count()
        .iloc[:, 0]
        .loc[lambda x: x < 10]
        .index.tolist()
    )
    ti = plot_symmetric_correlation_heatmap(
        sorted_full_corr_matrix,
        row_label_type="gene",
        col_label_type="gene",
        figsize=(170 * mm, 160 * mm),
        xticklabels=False,
        cbar_kws={"label": "Pearson's r"},
    )
    plt.xlabel("Perturbation")
    plt.ylabel("Perturbation")
    plt.gca().set_yticks(
        ti["y_idx"],
        [
            lb + (15 * " ") if lb in low_cell_line_knockouts else lb
            for lb in ti["y_label"]
        ],
        rotation=0,
    )
    for y, lb in zip(ti["y_idx"], ti["y_label"]):
        if lb in low_cell_line_knockouts:
            plt.text(-85, y - 3, (6 * "_"), transform=plt.gca().transData)
    plt.subplots_adjust(left=0.12, right=1.02, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(FIGURE_DIR, "all_by_all_corr_heatmap.png"))


def main():
    # figure s5
    plot_all_by_all_correlation_heatmap()


if __name__ == "__main__":
    main()
