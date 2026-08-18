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
from matplotlib.patches import Rectangle

mpl.style.use("assets/stylesheet.mplstyle")


# 1a
def plot_target_predictability_by_class():
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    predictability_data = load_file(
        "Chronos_Combined_predictability_results.csv", local_dir=DOWNLOADED_DIR
    )

    predictability_df = (
        predictability_data[
            predictability_data["gene"].isin(ko_metadata["cds_gene_id"].tolist())
            & predictability_data["best"]
        ]
        .loc[:, ["gene", "model", "pearson", "best"]]
        .merge(ko_metadata, left_on="gene", right_on="cds_gene_id")
    )

    plt.figure(figsize=(90 * mm, 55 * mm))

    sns.boxplot(
        predictability_df,
        x="pearson",
        y="target_class",
        hue="target_class",
        palette=gene_class_palette,
        showfliers=False,
        fill=False,
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
        medianprops={"linewidth": 1},
        zorder=-1,
        legend=False,
    )

    sns.stripplot(
        predictability_df,
        x="pearson",
        y="target_class",
        hue="target_class",
        palette=gene_class_palette,
        edgecolor="black",
        linewidth=0.5,
        zorder=0,
        legend=False,
    )

    plt.ylabel("Target dependency class")
    plt.xlabel("Accuracy of best DepMap 25Q2 predictive model")
    plt.subplots_adjust(left=0.25, bottom=0.15, top=0.95, right=0.95)
    plt.savefig(os.path.join(FIGURE_DIR, "predictability_by_target_class.png"))


# 1b
def discretize_matrix_by_interval(value_matrix, interval_dict):
    discretized_matrix = value_matrix.copy()
    discretized_matrix.loc[:, :] = np.nan
    replacements = dict()
    i = 0
    for lb, interval in interval_dict.items():
        left_boolean = (
            value_matrix >= interval.left
            if interval.closed_left
            else value_matrix > interval.left
        )
        right_boolean = (
            value_matrix <= interval.right
            if interval.closed_left
            else value_matrix < interval.right
        )
        in_interval = (left_boolean & right_boolean).astype(bool)
        discretized_matrix[in_interval] = i
        replacements[i] = lb
        i += 1
    return discretized_matrix.replace(replacements)


def plot_oncoprint():
    from matplotlib.legend_handler import HandlerPatch
    # custom legend elements

    # styling for smaller figures (linewidths are reduced)

    ADJUST_X = 0.7
    ADJUST_Y = 0.7
    LINEWIDTH = 0.5
    SLASHWIDTH = 0.5

    class ForwardSlashRectangle(Rectangle):
        def __init__(
            self,
            xy=(0, 0),
            width=0,
            height=0,
            *,
            angle=0.0,
            rotation_point="xy",
            **kwargs,
        ):
            super().__init__(
                xy, width, height, angle=angle, rotation_point=rotation_point, **kwargs
            )

    class DotRectangle(Rectangle):
        def __init__(
            self,
            xy=(0, 0),
            width=0,
            height=0,
            *,
            angle=0.0,
            rotation_point="xy",
            **kwargs,
        ):
            super().__init__(
                xy, width, height, angle=angle, rotation_point=rotation_point, **kwargs
            )

    class ForwardSlashHandler(HandlerPatch):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            lower_left = handlebox.xdescent, handlebox.ydescent - 1
            width, height = handlebox.width, handlebox.height
            p = plt.Rectangle(
                xy=lower_left,
                width=width,
                height=height,
                facecolor="white",
                edgecolor="black",
                linewidth=LINEWIDTH,
            )
            handlebox.add_artist(p)
            l = plt.Line2D(
                xdata=[0 + ADJUST_X, width - handlebox.xdescent],
                ydata=[0 + ADJUST_Y - 1, height - handlebox.ydescent + ADJUST_Y - 1],
                linewidth=SLASHWIDTH,
                color="black",
            )
            handlebox.add_artist(l)
            return p

    class DotRectangleHandler(HandlerPatch):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            lower_left = handlebox.xdescent, handlebox.ydescent - 1
            width, height = handlebox.width, handlebox.height
            center = (
                0.5 * (width - handlebox.xdescent + ADJUST_X) - 0.2,
                0.5 * (height - handlebox.ydescent + ADJUST_Y) - 0.5,
            )
            p = plt.Rectangle(
                xy=lower_left,
                width=width,
                height=height,
                facecolor="white",
                edgecolor="black",
                linewidth=LINEWIDTH,
            )
            handlebox.add_artist(p)
            c = plt.Circle(xy=center, radius=0.25 * width, facecolor="black")
            handlebox.add_artist(c)
            return p

    custom_handler_map = {
        ForwardSlashRectangle: ForwardSlashHandler(),
        DotRectangle: DotRectangleHandler(),
    }

    # load data
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    ko_metadata = load_file("knockout_metadata.csv")
    genes_to_check = ko_metadata["cds_gene_id"].tolist() + ["TP53 (7157)"]
    damaging = (
        load_file(
            "OmicsSomaticMutationsMatrixDamaging", local_dir=DOWNLOADED_DIR, index_col=0
        )
        .reindex(index=cl_metadata["arxspan_id"].tolist(), columns=genes_to_check)
        .dropna(how="all")
    )
    hotspot = (
        load_file(
            "OmicsSomaticMutationsMatrixHotspot", local_dir=DOWNLOADED_DIR, index_col=0
        )
        .reindex(index=cl_metadata["arxspan_id"].tolist(), columns=genes_to_check)
        .dropna(how="all")
    )
    cn = (
        load_file("OmicsCNGene")
        .reindex(index=cl_metadata["arxspan_id"].tolist(), columns=genes_to_check)
        .dropna(how="all")
    )

    # identify relevant mutations to show
    cn_discrete = discretize_matrix_by_interval(
        cn,
        interval_dict={
            "Amplification": pd.Interval(left=3, right=np.inf, closed="neither"),
            "Deletion": pd.Interval(left=-np.inf, right=0.25, closed="neither"),
            "Neutral Copy": pd.Interval(left=0.25, right=3, closed="both"),
        },
    )

    has_damaging = (damaging > 0).sum() > 0
    has_hotspot = (hotspot > 0).sum() > 0
    has_cn_alt = (cn_discrete.isin(["Amplification", "Deletion"])).sum() > 0
    mutations = (
        (has_damaging | has_hotspot | has_cn_alt)
        .reindex(genes_to_check)
        .dropna()
        .loc[lambda x: x == True]
        .index.tolist()
    )

    # create a longform annotation table
    dam_mut_aligned = damaging.reindex(columns=mutations).fillna(0) != 0
    hot_mut_aligned = hotspot.reindex(columns=mutations).fillna(0) != 0
    cn_aligned = cn_discrete.reindex(columns=mutations).fillna("Neutral Copy")
    oncoprint_summary = (
        dam_mut_aligned.melt(value_name="damaging", ignore_index=False)
        .reset_index()
        .merge(
            hot_mut_aligned.melt(value_name="hotspot", ignore_index=False).reset_index()
        )
        .merge(cn_aligned.melt(value_name="cn_alt", ignore_index=False).reset_index())
        .rename({"index": "arxspan_id", "variable": "gene"}, axis=1)
    )
    oncoprint_summary["altered"] = (
        oncoprint_summary["damaging"]
        | oncoprint_summary["hotspot"]
        | oncoprint_summary["cn_alt"].isin(["Amplification", "Deletion"])
    )
    oncoprint_summary["gene_symbol"] = oncoprint_summary["gene"].apply(
        lambda x: x.split(" (")[0]
    )

    # order genes
    gene_frequency = (
        oncoprint_summary.groupby("gene_symbol")["altered"]
        .mean()
        .reset_index()
        .sort_values(["altered", "gene_symbol"], ascending=[False, True])
        .set_index("gene_symbol")["altered"]
        .rename("frequency")
    )
    gene_rank = gene_frequency.rank(ascending=False, method="first").rename(
        "gene_order"
    )
    oncoprint_summary = oncoprint_summary.merge(
        gene_frequency, left_on="gene_symbol", right_index=True
    ).merge(gene_rank, left_on="gene_symbol", right_index=True)

    # order cell lines
    cell_line_order_matrix = oncoprint_summary.pivot_table(
        index="arxspan_id", columns="gene_symbol", values="altered"
    ).merge(
        cl_metadata.set_index("arxspan_id")[["cell_line", "OncotreeLineage"]],
        left_index=True,
        right_index=True,
    )
    cell_line_order = cell_line_order_matrix.sort_values(
        gene_rank.sort_values().index.tolist() + ["OncotreeLineage"],
        ascending=[False for _ in range(gene_rank.shape[0] + 1)],
        axis=0,
    ).index.tolist()
    cell_line_rank = pd.Series(
        np.arange(1, len(cell_line_order) + 1), index=cell_line_order
    ).rename("cell_line_order")
    oncoprint_summary = oncoprint_summary.merge(
        cl_metadata.set_index("arxspan_id")[["cell_line", "OncotreeLineage"]],
        left_on="arxspan_id",
        right_index=True,
    ).merge(cell_line_rank, left_on="arxspan_id", right_index=True)

    # prepare cn annotations
    cn_alteration_map = {"Amplification": 2, "Deletion": 1, "Neutral Copy": 0}
    oncoprint_palette = {0: "white", 1: plt.cm.coolwarm(0), 2: plt.cm.coolwarm(256)}
    oncoprint_summary["alt_value"] = oncoprint_summary["cn_alt"].replace(
        cn_alteration_map
    )

    # prepare marker positions
    oncoprint_summary["right"] = oncoprint_summary["cell_line_order"]
    oncoprint_summary["ha_center"] = oncoprint_summary["cell_line_order"] - 0.5
    oncoprint_summary["left"] = oncoprint_summary["cell_line_order"] - 1
    oncoprint_summary["top"] = oncoprint_summary["gene_order"]
    oncoprint_summary["va_center"] = oncoprint_summary["gene_order"] - 0.5
    oncoprint_summary["bottom"] = oncoprint_summary["gene_order"] - 1

    # extract annotations in sorted order
    cell_line_labels = cell_line_rank.index.tolist()
    gene_labels = gene_rank.index.tolist()
    frequency_labels = (
        oncoprint_summary.drop_duplicates(subset=["gene_symbol"])
        .set_index("gene_symbol")["frequency"]
        .reindex(gene_labels)
        .apply(lambda x: f"{x * 100:.1f}%")
    )
    lineage_colors = (
        oncoprint_summary.drop_duplicates(subset=["arxspan_id"])
        .set_index("arxspan_id")["OncotreeLineage"]
        .replace(lineage_palette)
        .reindex(index=cell_line_labels)
    )
    cell_line_names = (
        oncoprint_summary.drop_duplicates(subset=["arxspan_id"])
        .set_index("arxspan_id")["cell_line"]
        .reindex(index=cell_line_labels)
    )
    sorted_heatmap = oncoprint_summary.pivot(
        index="gene_symbol", columns="arxspan_id", values="alt_value"
    ).reindex(index=gene_labels, columns=cell_line_labels)

    # plot the base heatmap
    plt.figure(figsize=(80 * mm, 50 * mm))
    ax = sns.heatmap(
        sorted_heatmap,
        cmap=list(oncoprint_palette.values()),
        cbar=False,
        linewidth=0.5,
        linecolor="black",
        xticklabels=False,
        yticklabels=True,
    )
    plt.xlabel("")
    plt.ylabel("")

    # column patches
    for i, color in enumerate(lineage_colors.tolist()):
        ax.add_patch(
            plt.Rectangle(
                xy=(i, (0 - 0.5) - 0.25),
                width=1,
                height=0.5,
                facecolor=color,
                lw=0.25,
                edgecolor="black",
                transform=ax.transData,
                clip_on=False,
            )
        )
    ax.text(
        -0.2,
        0 - 0.5,
        "Lineage",
        transform=ax.transData,
        size=LABEL_SIZE,
        ha="right",
        va="center",
    )

    # column names
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks(
        np.arange(0, 16) + 0.25, cell_line_names.tolist(), rotation=60, ha="left"
    )
    ax.tick_params(axis="x", top=False, pad=8)

    # row frequencies
    ax_right = ax.secondary_yaxis("right")
    row_positions = ax.get_yticks()
    ax_right.set_yticks(row_positions)
    ax_right.set_yticklabels(frequency_labels)

    # markers
    for _, r in oncoprint_summary[oncoprint_summary["damaging"]].iterrows():
        plt.plot(
            [r["left"], r["right"]], [r["top"], r["bottom"]], color="black", linewidth=1
        )
    for _, r in oncoprint_summary[oncoprint_summary["hotspot"]].iterrows():
        ax.add_patch(plt.Circle((r["ha_center"], r["va_center"]), 0.2, color="black"))

    # legend
    alteration_handles = [
        Patch(
            facecolor=plt.cm.coolwarm(256), edgecolor="black", linewidth=0.5, alpha=0.7
        ),
        ForwardSlashRectangle(linewidth=0.25),
        DotRectangle(linewidth=0.25),
    ]
    alteration_labels = ["Amplification", "Damaging mutation", "Hotspot mutation"]
    ax.legend(
        handles=alteration_handles,
        labels=alteration_labels,
        ncols=3,
        handlelength=1,
        handleheight=1,
        handletextpad=0.3,
        columnspacing=1,
        loc="lower left",
        bbox_to_anchor=(0.08, -0.2),
        fontsize=5,
        handler_map=custom_handler_map,
    )

    # save
    plt.subplots_adjust(left=0.14, bottom=0.15, top=0.75, right=0.91)
    plt.savefig(os.path.join(FIGURE_DIR, "oncoprint.png"))


# 1c
def convert_categoricals_by_cmap(
    series, palette=dict(), null_value=0, null_color="tab:grey", palette_order=None
):
    """
    Replaces the entries in a series with an enumeration (integers), and additionally returns the mapping of values to integers and the ordered colormap
    """
    from matplotlib.colors import ListedColormap

    all_values_in_series = series.unique().tolist()
    if palette_order is not None:
        ordered_values = [x for x in palette_order if x in all_values_in_series]
        ordered_values = ordered_values + sorted(
            list(set(all_values_in_series) - set(ordered_values))
        )
    else:
        ordered_values = all_values_in_series
    reverse_ordered_values = ordered_values[::-1]

    category_to_enum = dict()
    enum_to_color = dict()

    enum = null_value - 1
    for k in reverse_ordered_values:
        if k in palette:
            category_to_enum[k] = enum
            enum_to_color[enum] = palette[k]
            enum -= 1
        else:
            category_to_enum[k] = null_value
            enum_to_color[null_value] = null_color
    return (
        series.replace(category_to_enum),
        category_to_enum,
        ListedColormap([enum_to_color[k] for k in sorted(enum_to_color.keys())]),
    )


def plot_dependency_heatmap():
    achilles_scores = load_file(
        "CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0
    )
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    controls = ko_metadata.query(
        'target_class == "Olfactory Control"'
    ).knockout.tolist()

    # reformat viability scores into long form and subset to the cell lines in the experiment
    achilles_scores.columns = [x.split(" (")[0] for x in achilles_scores.columns]
    crispr_df = (
        achilles_scores.reindex(
            index=cl_metadata.arxspan_id, columns=ko_metadata.knockout
        )
        .melt(var_name="guide", value_name="gene_effect", ignore_index=False)
        .reset_index()
        .replace(cl_metadata.set_index("arxspan_id").cell_line.to_dict())
        .rename({"arxspan_id": "cell_line"}, axis=1)
    )

    dependency_intervals = {
        "Strong": pd.Interval(left=-np.inf, right=-1, closed="right"),
        "Moderate": pd.Interval(left=-1, right=-0.7, closed="right"),
        "Weak": pd.Interval(left=-0.7, right=-0.5, closed="right"),
        "Nondependent": pd.Interval(left=-0.5, right=np.inf, closed="neither"),
    }

    # crispr_df['dependency_strength_class'] = discretize_series_by_interval(crispr_df['gene_effect'], interval_dict=dependency_intervals)
    # dependency_pairs = crispr_df[crispr_df['dependency_strength_class'] == "Strong"].apply(lambda x: (x['cell_line'], x['guide']), axis=1).tolist()
    dependency_matrix = crispr_df.pivot(
        index="guide", columns="cell_line", values="gene_effect"
    )
    # dependency_strength_matrix = crispr_df.pivot(index='guide', columns='cell_line', values='dependency_strength_class')
    crispr_df = crispr_df.merge(
        ko_metadata[["knockout", "target_class"]], left_on="guide", right_on="knockout"
    )

    cell_line_order = dependency_matrix.mean(axis=0).sort_values().index.tolist()
    # knockout_order = dependency_matrix.mean(axis=1).drop(controls).sort_values().index.tolist() + dependency_matrix.loc[controls].mean(axis=1).sort_values().index.tolist()
    knockout_order = (
        crispr_df.groupby("guide")
        .agg({"target_class": "first", "gene_effect": "mean"})
        .replace(
            {
                "Common Essential": 0,
                "High Variance": 1,
                "Selective": 2,
                "Olfactory Control": 3,
            }
        )
        .sort_values(["target_class", "gene_effect"])
        .index.tolist()
    )

    col_annot_height = 0.5
    col_annot_padding = 0.25
    row_annot_width = 0.5
    row_annot_padding = 0.25
    cmap = "Purples_r"

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(170 * mm, 45 * mm),
        width_ratios=[1, 199],
        gridspec_kw={"wspace": 0.01},
    )
    plt.subplots_adjust(left=0.1, right=1.1, bottom=0.3)

    lin_heatmap = cl_metadata.set_index("cell_line").loc[
        cell_line_order, ["OncotreeLineage"]
    ]
    lin_heatmap["OncotreeLineage"], lin_heatmap_enum, lin_heatmap_cmap = (
        convert_categoricals_by_cmap(lin_heatmap["OncotreeLineage"], lineage_palette)
    )
    sns.heatmap(
        lin_heatmap,
        cmap=lin_heatmap_cmap,
        cbar=False,
        yticklabels=True,
        xticklabels=False,
        ax=axs[0],
        linewidth=0.25,
        linecolor="gray",
    )
    axs[0].set_ylabel("Cell line")

    sns.heatmap(
        dependency_matrix.loc[knockout_order, cell_line_order].T,
        vmin=-1,
        vmax=0,
        cmap=cmap,
        linewidth=0.25,
        linecolor="gray",
        xticklabels=True,
        cbar_kws={"pad": 0.03},
        ax=axs[1],
    )
    axs[1].set_xlabel("Knockout")
    axs[1].set_xticklabels(knockout_order, fontdict={"size": ANNOT_SIZE - 0.5})
    axs[1].set_ylabel("")
    axs[1].set_yticks([])

    col_colors = (
        ko_metadata.set_index("knockout")
        .loc[knockout_order, "target_class"]
        .replace(gene_class_palette)
    )
    for i, color in enumerate(col_colors):
        axs[1].add_patch(
            plt.Rectangle(
                xy=(i, -col_annot_height - col_annot_padding),
                width=1,
                height=col_annot_height,
                facecolor=color,
                lw=0.25,
                edgecolor="gray",
                transform=axs[1].transData,
                clip_on=False,
            )
        )

    cbar = axs[1].collections[0].colorbar
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Gene effect")

    cbar.set_label("Gene effect", fontdict={"size": LABEL_SIZE})

    plt.savefig(os.path.join(FIGURE_DIR, "dependency_strength_heatmap_continuous.png"))


# legend
def create_legend():
    plt.figure(figsize=(80 * mm, 10 * mm))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    handles = [
        Patch(facecolor=col, edgecolor="tab:gray", label=lin, linewidth=0.5)
        for lin, col in lineage_palette.items()
        if lin != "Other"
    ]
    legend = plt.legend(
        handles=handles,
        handleheight=1,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1.5,
        loc="lower left",
        ncol=4,
        fontsize=ANNOT_SIZE,
    )

    plt.gca().spines[["left", "right", "top", "bottom"]].set_visible(False)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(FIGURE_DIR, "supplemental_fig1_lineage_legend.png"))


def main():
    # figure 1a
    plot_target_predictability_by_class()

    # figure 1b
    plot_oncoprint()

    # figure 1c
    plot_dependency_heatmap()

    # legend
    create_legend()


if __name__ == "__main__":
    main()
