import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from constants import *
from data_utils import *
from figure_utils import *
from matplotlib.patches import Patch

mpl.style.use("assets/stylesheet.mplstyle")


# sup a
def plot_scyl1_cluster_heatmap():
    def identify_label_centers(labels):
        label_series = pd.Series(labels).rename("label")
        breakpoints = np.array(
            [0]
            + (
                np.where(label_series.iloc[:-1].values != label_series.iloc[1:].values)[
                    0
                ]
                + 1
            ).tolist()
            + [len(label_series)]
        )
        midpoints = (breakpoints[:-1] + (breakpoints[1:] - 1)) / 2
        midpoint_labels = label_series.iloc[breakpoints[:-1]].values.tolist()
        return midpoints, midpoint_labels, breakpoints

    def plot_symmetric_correlation_heatmap(
        corr_matrix,
        cell_line_subset=slice(None),
        gene_subset=slice(None),
        figsize=(70 * mm, 70 * mm),
        row_label_type="cell_line",
        col_label_type="gene",
        divider_linewidth=1,
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

        ax.set_xticks(col_idxs + 0.5, col_labels, rotation=xtickrotation)
        ax.set_yticks(row_idxs + 0.5, row_labels)

        return ax

    full_corr_matrix = load_file(
        "full_corr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=[0, 1]
    )
    geneset_zscore = load_file(
        "geneset_mean_zscore_matrix.csv",
        local_dir=PROCESSED_DIR,
        index_col=[0, 1],
        header=[0, 1],
    )
    cl_order = (
        geneset_zscore.loc[
            ("Hallmark", "HALLMARK_UNFOLDED_PROTEIN_RESPONSE"),
            pd.IndexSlice[:, "SCYL1"],
        ]
        .droplevel(1)
        .sort_values(ascending=False)
        .index.tolist()
    )
    ax = plot_symmetric_correlation_heatmap(
        full_corr_matrix.loc[pd.IndexSlice[cl_order, :], pd.IndexSlice[cl_order, :]],
        gene_subset=["SCYL1", "SEC23IP", "IER3IP1", "SLC39A9", "GET4"],
        cbar_kws={"label": "Pearson's r"},
        figsize=(80 * mm, 60 * mm),
        row_label_type="gene",
        divider_linewidth=0.75,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.subplots_adjust(left=0.2, right=0.9)
    plt.savefig(os.path.join(FIGURE_DIR, "upr_inducing_heatmap.png"))


# sup b
def plot_upr_gene_waterfalls():
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
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

    upr_crispr_longform = ko_gene_effects_longform[
        ko_gene_effects_longform["knockout"].isin(
            ["SCYL1", "SEC23IP", "IER3IP1", "SLC39A9", "GET4"]
        )
    ].copy()
    upr_crispr_longform["rank"] = upr_crispr_longform.groupby("knockout")[
        "gene_effect"
    ].rank()
    upr_crispr_longform["included"] = upr_crispr_longform["arxspan_id"].isin(
        cl_metadata["arxspan_id"].unique().tolist()
    )

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(60 * mm, 60 * mm),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.1},
    )
    axs = axs.flatten()
    for i, g in zip(
        [0, 1, 3, 4, 5], ["SCYL1", "SEC23IP", "IER3IP1", "SLC39A9", "GET4"]
    ):
        sns.scatterplot(
            upr_crispr_longform[upr_crispr_longform["knockout"] == g].sort_values(
                "included"
            ),
            x="rank",
            y="gene_effect",
            hue="included",
            palette={False: "tab:gray", True: "tab:red"},
            alpha=0.8,
            s=8,
            ax=axs[i],
            legend=False,
        )
        axs[i].text(0.08, 0.88, g, transform=axs[i].transAxes)
        axs[i].set_ylabel("")
        axs[i].set_xlabel("")
        axs[i].axhline(0, linestyle="dashed", color="black", zorder=0, linewidth=0.5)
    axs[2].set_visible(False)
    fig.supylabel("Gene effect (Chronos)", x=0.03, y=0.57)
    fig.supxlabel("Cell line rank", y=0.03, x=0.57)
    axs[1].legend(
        title="In perturb-seq",
        handles=[
            Line2D(
                [],
                [],
                color="tab:red",
                marker="o",
                linestyle="None",
                markersize=5,
                markeredgewidth=0,
                label="True",
            ),
            Line2D(
                [],
                [],
                color="tab:gray",
                marker="o",
                linestyle="None",
                markersize=5,
                markeredgewidth=0,
                label="False",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(1, 0.75),
        fontsize=ANNOT_SIZE,
    )
    plt.subplots_adjust(left=0.16, right=0.95, bottom=0.16, top=0.95)
    plt.savefig(os.path.join(FIGURE_DIR, "upr_gene_waterfalls.png"))


# sup c
def plot_shared_response_scatter():
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    models = list(cl_metadata["cell_line"])
    model_to_cl = cl_metadata.set_index("arxspan_id")["cell_line"]
    cl_to_model = cl_metadata.set_index("cell_line")["arxspan_id"]
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    cl_gene_effect = gene_effect.loc[cl_to_model[models]].rename(model_to_cl)
    ps100_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )

    cluster_kos = ["SCYL1", "SEC23IP", "SLC39A9", "GET4", "IER3IP1"]
    shared_response = (
        ps100_zscore.T.loc[:, cluster_kos, :]
        .loc[
            :,
            lambda x: (
                ((x.groupby(level=0).max() > 3).mean() > 0.5)
                | ((x.groupby(level=0).min() < -3).mean() > 0.5)
            ),
        ]
        .fillna(0)
    )
    print("shared response genes: ", " ".join(shared_response.columns))

    tab10_no_red = sns.color_palette("tab10")[0:3] + sns.color_palette("tab10")[4:6]
    color_lut2 = dict(zip(cluster_kos, tab10_no_red))

    plt.figure(figsize=(60 * mm, 60 * mm))
    sns.scatterplot(
        x=cl_gene_effect.stack()
        .rename_axis([None, None])
        .reindex(shared_response.index),
        y=shared_response.mean(axis=1).rename_axis([None, None]),
        hue=shared_response.mean(axis=1).index.get_level_values(1),
        palette=color_lut2,
        s=20,
        alpha=0.8,
    )
    sns.regplot(
        x=cl_gene_effect.stack()
        .rename_axis([None, None])
        .reindex(shared_response.index),
        y=shared_response.mean(axis=1).rename_axis([None, None]),
        scatter=False,
        ci=None,
        line_kws={"color": "gray", "alpha": 0.5, "zorder": -1},
    )
    plt.gca().axhline(0, color="gray", alpha=0.5, zorder=-1, ls="--")
    plt.gca().axvline(0, color="gray", alpha=0.5, zorder=-1, ls="--")

    plt.legend(
        title="Target class",
        handleheight=1,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1.5,
        loc="lower left",
        ncol=1,
        fontsize=ANNOT_SIZE,
    )
    plt.xlabel("Gene effect (Chronos)")
    plt.ylabel("Mean shared response z-score")
    plt.subplots_adjust(left=0.16, right=0.95, bottom=0.16, top=0.95)

    plt.savefig(os.path.join(FIGURE_DIR, "shared_response_scatter.png"))


# sup d
def plot_cl_response_heatmap():
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    models = list(cl_metadata["cell_line"])
    model_to_cl = cl_metadata.set_index("arxspan_id")["cell_line"]
    cl_to_model = cl_metadata.set_index("cell_line")["arxspan_id"]
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    cl_gene_effect = gene_effect.loc[cl_to_model[models]].rename(model_to_cl)
    ps100_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )

    ier3ip1_dep_min = (
        ps100_zscore.T.loc[:, "IER3IP1", :]
        .loc[cl_gene_effect["IER3IP1"] < -0.75]
        .fillna(0)
        .min()
    )
    other_ko_max = ps100_zscore.T.loc[
        :, ["SCYL1", "SEC23IP", "SLC39A9", "GET4"], :
    ].max()
    diff_min_max = (ier3ip1_dep_min - other_ko_max).loc[lambda x: x > 0]
    to_label = diff_min_max.loc[lambda x: x > 2].index

    cl_ier3ip1_dep = cl_gene_effect["IER3IP1"]
    norm = plt.Normalize(vmin=cl_ier3ip1_dep.min(), vmax=cl_ier3ip1_dep.max())
    colors = [plt.cm.gray(norm(v)) for v in cl_ier3ip1_dep]

    cg = sns.clustermap(
        ps100_zscore.T.loc[:, "IER3IP1", :][to_label].fillna(0),
        cmap="RdBu_r",
        center=0,
        annot=False,
        figsize=(45 * mm, 60 * mm),
        cbar_pos=None,
        dendrogram_ratio=(0.05, 0.1),
        row_colors=pd.Series(colors, index=cl_ier3ip1_dep.index, name="IER3IP1 effect"),
    )
    cg.ax_heatmap.set_xlabel("Cell line")
    cg.ax_heatmap.set_ylabel("Response gene")
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90)

    plt.subplots_adjust(bottom=0.3, right=0.7)
    plt.savefig(os.path.join(FIGURE_DIR, "shared_response_heatmap.png"))

    # separate cbar
    heatmap = cg.ax_heatmap.collections[0]
    fig_cbar = plt.figure(figsize=(10 * mm, 50 * mm))
    ax_cbar = fig_cbar.add_axes([0.1, 0.1, 0.2, 0.8])
    fig_cbar.colorbar(heatmap, cax=ax_cbar).set_label("Z-score")

    plt.savefig(os.path.join(FIGURE_DIR, "shared_response_heatmap_cbar.png"))


# sup e
def plot_nrf2_z_scatter():
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    models = list(cl_metadata["cell_line"])
    model_to_cl = cl_metadata.set_index("arxspan_id")["cell_line"]
    cl_to_model = cl_metadata.set_index("cell_line")["arxspan_id"]
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    cl_gene_effect = gene_effect.loc[cl_to_model[models]].rename(model_to_cl)
    geneset_zscore_matrix = load_file(
        "geneset_mean_zscore_matrix.csv",
        local_dir=PROCESSED_DIR,
        header=[0, 1],
        index_col=[0, 1],
    )

    term = "REACTOME_NFE2L2_REGULATING_ANTI_OXIDANT_DETOXIFICATION_ENZYMES"
    ko_gene = "IER3IP1"
    term_z = (
        geneset_zscore_matrix.xs(term, axis=0, level=1)
        .stack()
        .droplevel(0)
        .loc[ko_gene]
    )
    ko_effect = cl_gene_effect[ko_gene]
    lineages = cl_metadata.set_index("cell_line")["OncotreeLineage"]

    plt.figure(figsize=(50 * mm, 50 * mm))
    sns.scatterplot(
        x=ko_effect,
        y=term_z,
        hue=lineages,
        palette=lineage_palette,
        legend=False,
        linewidth=0.5,
        edgecolor="black",
    )
    plt.gca().axhline(0, color="gray", alpha=0.5, zorder=-1, ls="--")
    plt.gca().axvline(0, color="gray", alpha=0.5, zorder=-1, ls="--")
    plt.ylabel(f"Reactome NFE2L2 response\nz-score in {ko_gene} knockout")
    plt.xlabel(f"{ko_gene} gene effect")

    plt.subplots_adjust(left=0.3, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "nrf2_z_scatter.png"))


# sup f
def plot_nrf2_bulk_scatter():
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    geneset_table = load_file("all_genesets.csv", local_dir=DATA_DIR)
    group_columns = ["collection", "term"]
    geneset_groups = geneset_table.groupby(group_columns).apply(
        lambda x: list(x["gene"])
    )
    depmap_bulk = load_file(
        "OmicsExpressionProteinCodingGenesTPMLogp1",
        local_dir=DOWNLOADED_DIR,
        index_col=0,
    )
    depmap_bulk.columns = depmap_bulk.columns.to_series().apply(
        lambda x: x.split(" ")[0]
    )

    shared_lines = depmap_bulk.index.intersection(gene_effect.index)
    term = "REACTOME_NFE2L2_REGULATING_ANTI_OXIDANT_DETOXIFICATION_ENZYMES"
    ko_gene = "IER3IP1"
    term_genes = geneset_groups["Reactome"][term]
    avg_term_expr = depmap_bulk.loc[shared_lines, term_genes].mean(axis=1)
    df = pd.DataFrame(avg_term_expr, columns=["avg_term_expr"])
    df["in_pilot"] = df.index.isin(cl_metadata["arxspan_id"])
    df["gene_effect"] = gene_effect.loc[shared_lines, ko_gene]
    df = df.sort_values("in_pilot")
    df["hue"] = df.index.map(
        cl_metadata.set_index("arxspan_id")["OncotreeLineage"]
    ).fillna("not in perturb-seq")

    plt.figure(figsize=(50 * mm, 50 * mm))
    palette = lineage_palette.copy()
    palette["not in perturb-seq"] = "gray"
    sns.scatterplot(
        df.query('hue == "not in perturb-seq"'),
        x="gene_effect",
        y="avg_term_expr",
        hue="hue",
        s=8,
        alpha=0.5,
        palette=palette,
    )
    sns.scatterplot(
        df.query('hue != "not in perturb-seq"'),
        x="gene_effect",
        y="avg_term_expr",
        hue="hue",
        s=8,
        palette=palette,
        linewidth=0.5,
        edgecolor="black",
    )
    plt.gca().axvline(0, color="gray", alpha=0.5, zorder=-1, ls="--")
    plt.ylabel("Reactome NFE2L2 response\naverage expression")
    plt.xlabel(f"{ko_gene} gene effect")
    plt.gca().get_legend().remove()
    plt.subplots_adjust(left=0.3, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "nrf2_bulk_scatter.png"))


def plot_clustermap_common_responses():
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    models = list(cl_metadata["cell_line"])
    model_to_cl = cl_metadata.set_index("arxspan_id")["cell_line"]
    cl_to_model = cl_metadata.set_index("cell_line")["arxspan_id"]
    ps100_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    cl_gene_effect = gene_effect.loc[cl_to_model[models]].rename(model_to_cl)
    dep_in_ps100 = (
        cl_gene_effect.loc[models]
        .reindex(columns=ps100_zscore.columns.get_level_values(1).unique())
        .stack()
        .loc[lambda x: x < -0.75]
    )

    cluster_kos = ["SCYL1", "SEC23IP", "SLC39A9", "GET4", "IER3IP1"]
    shared_response = (
        ps100_zscore.T.loc[:, cluster_kos, :]
        .loc[
            :,
            lambda x: (
                ((x.groupby(level=0).max() > 3).mean() > 0.5)
                | ((x.groupby(level=0).min() < -3).mean() > 0.5)
            ),
        ]
        .fillna(0)
    )
    print("shared response genes: ", " ".join(shared_response.columns))

    color_lut = {True: "tab:red", False: "lightgray"}
    col_colors = (
        shared_response.index.to_series()
        .isin(dep_in_ps100.index)
        .rename("Dependent")
        .map(color_lut)
    )
    tab10_no_red = sns.color_palette("tab10")[0:3] + sns.color_palette("tab10")[4:6]
    color_lut2 = dict(zip(cluster_kos, tab10_no_red))
    col_colors2 = (
        shared_response.index.to_series()
        .apply(lambda x: x[1])
        .rename("KO")
        .map(color_lut2)
    )

    cg = sns.clustermap(
        shared_response,
        cmap="RdBu_r",
        center=0,
        row_colors=[col_colors, col_colors2],
        method="ward",
        figsize=(100 * mm, 60 * mm),
        dendrogram_ratio=(0.05, 0.05),
        yticklabels=False,
        cbar_pos=None,
    )
    cg.ax_heatmap.set_xlabel("Response gene")
    cg.ax_heatmap.set_ylabel("")
    cg.ax_row_colors.set_xticks(
        [0.5, 1.5], labels=["Dependent", "Knockout"], rotation=45, ha="right"
    )
    offset_x_labels(plt.gcf(), cg.ax_row_colors, 15, 0)

    handles = [Patch(facecolor="white", edgecolor="white", label="Knockout")]
    for ko, color in color_lut2.items():
        handles = handles + [Patch(facecolor=color, edgecolor="white", label=ko)]
    handles = handles + [
        Patch(facecolor="white", edgecolor="white", label="Dependent"),
        Patch(facecolor="tab:red", edgecolor="white", label="True"),
        Patch(facecolor="lightgray", edgecolor="white", label="False"),
    ]

    cg.ax_heatmap.legend(
        handles=handles,
        ncol=6,
        handlelength=1,
        handleheight=1,
        handletextpad=0.5,
        columnspacing=1,
        loc="lower left",
        bbox_to_anchor=(0, -0.7),
        prop={"size": ANNOT_SIZE},
    )
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.4)

    plt.savefig(os.path.join(FIGURE_DIR, "IER3IP1_clustermap.png"))

    # separate cbar
    heatmap = cg.ax_heatmap.collections[0]
    fig_cbar = plt.figure(figsize=(10 * mm, 60 * mm))
    ax_cbar = fig_cbar.add_axes([0.1, 0.1, 0.2, 0.8])
    fig_cbar.colorbar(heatmap, cax=ax_cbar).set_label("Z-score")

    plt.savefig(os.path.join(FIGURE_DIR, "IER3IP1_clustermap_cbar.png"))


def main():
    # sup a
    plot_scyl1_cluster_heatmap()

    # sup b
    plot_upr_gene_waterfalls()

    # sup c
    plot_clustermap_common_responses()

    # sup d
    plot_shared_response_scatter()

    # sup e
    plot_cl_response_heatmap()

    # sup f
    plot_nrf2_z_scatter()

    # sup g
    plot_nrf2_bulk_scatter()


if __name__ == "__main__":
    main()
