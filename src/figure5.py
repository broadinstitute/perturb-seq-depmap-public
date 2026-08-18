import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from constants import *
from data_utils import *
from figure_utils import *
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from taigapy import create_taiga_client_v3

tc = create_taiga_client_v3()

mpl.style.use("assets/stylesheet.mplstyle")


def plot_mock_correlation_heatmap(genelist=["MTOR", "XRN1", "SMG6", "MDM2"]):
    def plot_diagonal_correlation_heatmap(
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
        ax = sns.heatmap(
            subset,
            center=0,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            mask=np.tril(subset, k=-1),
            **plt_kwargs,
        )

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
        ax.set_xticks(col_idxs + 0.5, col_labels, rotation=xtickrotation)
        return ticks_info

    full_corr_matrix = load_file(
        "full_corr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=[0, 1]
    )
    partial_corr_matrix = full_corr_matrix.loc[
        pd.IndexSlice[:, genelist], pd.IndexSlice[:, genelist]
    ]
    partial_corr_matrix = (
        partial_corr_matrix.sort_index(level=[1, 0], axis=0)
        .sort_index(level=[1, 0], axis=1)
        .dropna(how="all")
        .dropna(how="all", axis=1)
    )

    ti = plot_diagonal_correlation_heatmap(
        partial_corr_matrix,
        row_label_type="gene",
        col_label_type="gene",
        figsize=(40 * mm, 40 * mm),
        divider_linewidth=0.75,
        xticklabels=False,
        yticklabels=False,
        cbar=False,  # , cbar_kws={'label': "Pearson's r"}
    )
    plt.xlabel("")
    plt.xticks(ti["x_idx"], ["A", "B", "C", "D"])
    plt.ylabel("Perturbation")
    plt.savefig(os.path.join(FIGURE_DIR, "mock_corr_heatmap.png"))


def plot_mock_correlation_network(genelist=["MTOR", "XRN1", "SMG6", "MDM2"]):
    import random

    import igraph as ig

    ko_pair_summary = load_file(
        "perturbation_pair_correlation_summary.csv", local_dir=PROCESSED_DIR
    ).set_index(["grna_target1", "grna_target2"])
    ko_pair_summary["all_cross_lines"] = ko_pair_summary["all_cross_lines_medium"]

    mock_subset = ko_pair_summary.loc[
        pd.IndexSlice[genelist, genelist], :
    ].reset_index()

    g = ig.Graph.DataFrame(mock_subset, use_vids=False, directed=False)

    random.seed(1)

    vertices = g.get_vertex_dataframe().join(
        pd.DataFrame(g.layout_fruchterman_reingold().coords, columns=["x", "y"])
    )
    edges = (
        g.get_edge_dataframe()
        .merge(vertices, left_on="source", right_index=True)
        .merge(
            vertices,
            left_on="target",
            right_index=True,
            suffixes=["_source", "_target"],
        )
    )
    vertices.loc[:, "color"] = "black"
    vertices.loc[vertices["name"] == "MDM2", "color"] = "lightgray"

    fig, ax = plt.subplots(figsize=(40 * mm, 40 * mm))
    sns.scatterplot(
        vertices,
        x="x",
        y="y",
        linewidth=0.5,
        edgecolor="black",
        c="black",
        hue="color",
        palette={"black": "tab:blue", "lightgray": "lightgray"},
        ax=ax,
        legend=False,
    )

    self_loops = edges[edges["source"] == edges["target"]]
    nonself_edges = edges[edges["source"] != edges["target"]]

    source_to_angles = {v: [] for v in vertices.index.tolist()}

    for i, r in nonself_edges.iterrows():
        if r["all_cross_lines"]:
            c = "red"
        else:
            c = "black"
        xy_source = np.array([r["x_source"], r["y_source"]])
        xy_target = np.array([r["x_target"], r["y_target"]])
        xy_delta = xy_target - xy_source
        source_to_angles[r["source"]].append(get_angle_from_xaxis(xy_delta))
        source_to_angles[r["target"]].append(get_angle_from_xaxis(-xy_delta))

        if r["n_pairs_high_corr"] < 2:
            alpha = 0.1
        else:
            alpha = 1

        ax.plot(
            [r["x_source"], r["x_target"]],
            [r["y_source"], r["y_target"]],
            color=c,
            zorder=-1,
            alpha=alpha,
        )

    source_to_best_angle = dict()
    for s in source_to_angles:
        angle_list = sorted(source_to_angles[s])
        if len(angle_list) > 0:
            extended_angle_list = angle_list + [(2 * np.pi) + angle_list[0]]
            angle_pos = np.argmax(np.diff(extended_angle_list))
            best_angle = (
                extended_angle_list[angle_pos] + extended_angle_list[angle_pos + 1]
            ) / 2
        else:
            best_angle = 0
        source_to_best_angle[s] = best_angle

    for i, r in self_loops.iterrows():
        c = "black"
        radius = 0.1
        if r["n_pairs_high_corr"] < 10:
            alpha = 0.1
        else:
            alpha = 1
        plot_loop(
            (r["x_source"], r["y_source"]),
            r=(0.1, 0.1),
            offset_angle=source_to_best_angle[r["source"]],
            ax=ax,
            color=c,
            zorder=-1,
            alpha=alpha,
        )

    ax.set_axis_off()
    plt.savefig(os.path.join(FIGURE_DIR, "mock_graph_filtration.png"))


def plot_strong_correlation_network():
    import random

    import igraph as ig

    ko_pair_summary = load_file(
        "perturbation_pair_correlation_summary.csv", local_dir=PROCESSED_DIR
    )
    ko_pair_summary["all_cross_lines"] = ko_pair_summary["all_cross_lines_medium"]
    graph_subset_df = ko_pair_summary[ko_pair_summary["n_pairs_high_corr"] >= 4]
    graph_subset_df.loc[
        graph_subset_df["grna_target1"] == graph_subset_df["grna_target2"],
        "all_cross_lines",
    ] = False
    # graph_subset_df['all_cross_lines'] = graph_subset_df['all_cross_lines'].fillna(True)

    g = ig.Graph.DataFrame(graph_subset_df, use_vids=False, directed=False)

    connected_components = g.connected_components("weak").membership
    cmap = dict(
        zip(
            np.unique(connected_components),
            sns.color_palette(n_colors=len(np.unique(connected_components))),
        )
    )

    random.seed(1)

    vertices = g.get_vertex_dataframe().join(
        pd.DataFrame(g.layout_fruchterman_reingold().coords, columns=["x", "y"])
    )
    vertices.loc[vertices["name"].isin(["MTOR", "PIK3CA", "MYC", "MAD2L1"]), "x"] = (
        vertices.loc[vertices["name"].isin(["MTOR", "PIK3CA", "MYC", "MAD2L1"]), "x"]
        - 1
    )
    vertices.loc[vertices["name"].isin(["GET4", "SEC23IP", "MYC", "MAD2L1"]), "x"] = (
        vertices.loc[vertices["name"].isin(["GET4", "SEC23IP", "MYC", "MAD2L1"]), "x"]
        - 0.5
    )
    vertices.loc[vertices["name"].isin(["MYC", "MAD2L1"]), "y"] = (
        vertices.loc[vertices["name"].isin(["MYC", "MAD2L1"]), "y"] + 1.75
    )
    vertices.loc[vertices["name"].isin(["GET4", "SEC23IP", "MDM2", "UBE3A"]), "y"] = (
        vertices.loc[vertices["name"].isin(["GET4", "SEC23IP", "MDM2", "UBE3A"]), "y"]
        + 0.5
    )
    edges = (
        g.get_edge_dataframe()
        .merge(vertices, left_on="source", right_index=True)
        .merge(
            vertices,
            left_on="target",
            right_index=True,
            suffixes=["_source", "_target"],
        )
    )
    edges["edgeweight"] = scale_weights(
        edges["n_pairs_high_corr"], vmin=5, vmax=25, lims=(0.5, 1.5)
    )

    fig, ax = plt.subplots(figsize=(80 * mm, 50 * mm))
    sns.scatterplot(
        vertices, x="x", y="y", linewidth=0.5, edgecolor="black", c="tab:blue", ax=ax
    )

    self_loops = edges[edges["source"] == edges["target"]]
    nonself_edges = edges[edges["source"] != edges["target"]]

    source_to_angles = {v: [] for v in vertices.index.tolist()}

    for i, r in nonself_edges.iterrows():
        if r["all_cross_lines"] & r["is_bimodal"]:
            c = "red"
        else:
            c = "black"
        xy_source = np.array([r["x_source"], r["y_source"]])
        xy_target = np.array([r["x_target"], r["y_target"]])
        xy_delta = xy_target - xy_source
        source_to_angles[r["source"]].append(get_angle_from_xaxis(xy_delta))
        source_to_angles[r["target"]].append(get_angle_from_xaxis(-xy_delta))

        ax.plot(
            [r["x_source"], r["x_target"]],
            [r["y_source"], r["y_target"]],
            color=c,
            linewidth=r["edgeweight"],
            zorder=-1,
        )

    source_to_best_angle = dict()
    for s in source_to_angles:
        angle_list = sorted(source_to_angles[s])
        if len(angle_list) > 0:
            extended_angle_list = angle_list + [(2 * np.pi) + angle_list[0]]
            angle_pos = np.argmax(np.diff(extended_angle_list))
            best_angle = (
                extended_angle_list[angle_pos] + extended_angle_list[angle_pos + 1]
            ) / 2
        else:
            best_angle = 0
        source_to_best_angle[s] = best_angle

    for i, r in self_loops.iterrows():
        c = "black"
        radius = 0.1
        plot_loop(
            (r["x_source"], r["y_source"]),
            r=(0.1, 0.2),
            offset_angle=source_to_best_angle[r["source"]],
            ax=ax,
            color=c,
            linewidth=r["edgeweight"],
            zorder=-1,
        )

    for i, r in vertices.iterrows():
        if r["name"] in ["POLR2D", "MTPAP", "PSMA1", "BRD4"]:
            yoff = -0.4
        elif r["name"] in ["ADAR"]:
            yoff = -0.1
        elif r["name"] in ["MYC"]:
            yoff = 0.2
        elif r["name"] in ["SEC23IP", "GET4"]:
            yoff = 0.4
        else:
            yoff = 0.05
        if r["name"] in ["MDM2", "UBE3A", "MAD2L1", "MYC"]:
            ha = "right"
            xoff = -0.1
        elif r["name"] in ["BRD4", "GET4"]:
            ha = "center"
            xoff = 0
        else:
            ha = "center"
            xoff = 0.35

        color = "black"
        manually_annotate(
            ax,
            r["name"],
            vertices.set_index("name"),
            xcol="x",
            ycol="y",
            xyoffset=(xoff, yoff),
            ha=ha,
            color=color,
            arrowprops=None,
        )

    ax.set_axis_off()
    leg = plt.legend(
        title="Number of cell line\npairs with r > 0.5",
        handles=[
            # Patch(color='white', label='Number of cell line\npairs with r > 0.5'),
            Line2D([], [], linestyle="solid", color="black", linewidth=0.5, label="5"),
            Line2D([], [], linestyle="solid", color="black", linewidth=1.0, label="15"),
            Line2D([], [], linestyle="solid", color="black", linewidth=1.5, label="25"),
            # Patch(color='white', label='Disjoint cell\nline pairs'),
            # Line2D([], [], linestyle='solid', color='red', linewidth=1.0, label='True'),
            # Line2D([], [], linestyle='solid', color='black', linewidth=1.0, label='False'),
            Line2D(
                [],
                [],
                linestyle="solid",
                color="red",
                linewidth=1.0,
                label="Disjoint cell line sets",
            ),
        ],
        fontsize=ANNOT_SIZE,
    )
    plt.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(FIGURE_DIR, "strong_correlation_graph_filtration.png"))


def plot_rna_perturbation_geneset_enrichment(fdr_threshold=0.05):
    individual_rna_ko_hgeom_results = load_file(
        "rna_perturbation_deg_geneset_enrichment.csv", local_dir=PROCESSED_DIR
    )
    individual_rna_ko_hgeom_results = individual_rna_ko_hgeom_results[
        individual_rna_ko_hgeom_results["Knockout"].isin(["XRN1", "SMG6"])
    ]
    top_genesets = (
        individual_rna_ko_hgeom_results[
            (individual_rna_ko_hgeom_results["FDR"] < fdr_threshold)
        ]
        .value_counts("term")
        .head(5)
        .index.tolist()
    )

    rna_hgeom_df_to_plot = individual_rna_ko_hgeom_results[
        individual_rna_ko_hgeom_results["term"].isin(top_genesets)
    ].copy()
    rna_hgeom_df_to_plot["fdr_transform"] = -np.log10(rna_hgeom_df_to_plot["FDR"])
    rna_hgeom_df_to_plot["clean_term"] = rna_hgeom_df_to_plot["term"].apply(
        lambda x: clean_geneset_name(x, remove_prefix=False)
    )

    fig = plt.figure(figsize=(60 * mm, 60 * mm))
    ax = plt.gca()

    sns.boxplot(
        rna_hgeom_df_to_plot,
        x="fdr_transform",
        y="clean_term",
        hue="Knockout",
        hue_order=["SMG6", "XRN1"],
        showfliers=False,
        fill=False,
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
        medianprops={"linewidth": 1},
        zorder=-1,
        ax=ax,
        legend=False,
    )

    _, ax = categorical_scatterplot(
        rna_hgeom_df_to_plot.rename({"n_overlap": "N genes"}, axis=1),
        xlabel="fdr_transform",
        ylabel="clean_term",
        hue="Knockout",
        dodge_shift=(-0.2, 0.2),
        hue_order=["SMG6", "XRN1"],
        jitter=(0.1, 0.1),
        size="N genes",
        sizes=(4, 48),
        linewidth=0.25,
        edgecolor="black",
        ax=ax,
        # legend=False
    )
    plt.ylabel("")
    plt.xlabel("$-\\log_{10}$(Fisher's exact q-value)")
    plt.subplots_adjust(left=0.35, right=0.95, bottom=0.15)
    plt.axvline(
        -np.log10(fdr_threshold),
        linestyle="dashed",
        color="tab:red",
        zorder=-1,
        linewidth=0.5,
    )
    plt.savefig(os.path.join(FIGURE_DIR, "rna_processing_enrichment.png"))


def plot_tfrc_top_genes(fdr_threshold=0.05):
    def create_top_gene_boxplot(
        all_results_df,
        grna_target,
        top_n,
        min_lines=8,
        ax=None,
        figsize=(60 * mm, 40 * mm),
    ):
        result_subset = all_results_df[
            (all_results_df["grna_target"] == grna_target)
            & (all_results_df["response_id_n_cell_lines"] >= min_lines)
            & (all_results_df["response_id_n_significant_cell_lines"] >= 1)
        ].copy()
        result_subset["-log10(q-value)"] = -np.log10(result_subset["FDR"])
        top_genes = (
            result_subset.groupby("response_id")["zscore"]
            .median()
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        result_subset = result_subset[result_subset["response_id"].isin(top_genes)]
        result_subset["-gene_effect"] = -result_subset["gene_effect"]
        sig_sbst = result_subset[result_subset["FDR"] < 0.05]
        insig_sbst = result_subset[result_subset["FDR"] >= 0.05]

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        if len(top_genes) > 0:
            sns.boxplot(
                result_subset,
                x="zscore",
                y="response_id",
                order=top_genes,
                fill=False,
                showfliers=False,
                boxprops={"linewidth": 0.5},
                whiskerprops={"linewidth": 0.5},
                capprops={"linewidth": 0.5},
                medianprops={"linewidth": 1, "color": "tab:red"},
                color="tab:gray",
                ax=ax,
                zorder=0,
            )
            if sig_sbst.shape[0] > 0:
                categorical_scatterplot(
                    sig_sbst,
                    xlabel="zscore",
                    ylabel="response_id",
                    yorder=top_genes,
                    hue="lineage",
                    palette=lineage_palette,
                    dodge_shift=(0, 0),
                    jitter=(0, 0.2),
                    size="-gene_effect",
                    sizes=(4, 32),
                    size_norm=(0, 1.5),
                    ax=ax,
                    linewidth=0.5,
                    edgecolor="black",
                    legend=False,
                )
            if insig_sbst.shape[0] > 0:
                categorical_scatterplot(
                    insig_sbst,
                    xlabel="zscore",
                    ylabel="response_id",
                    yorder=top_genes,
                    hue="lineage",
                    palette=lineage_palette,
                    dodge_shift=(0, 0),
                    jitter=(0, 0.2),
                    size="-gene_effect",
                    sizes=(4, 32),
                    size_norm=(0, 1.5),
                    ax=ax,
                    linewidth=0.5,
                    edgecolor="black",
                    legend=False,
                    alpha=0.3,
                )
            ax.set_ylabel("Response gene")
            ax.set_xlabel("Z-scored expression")
            ax.axvline(
                0, linestyle="solid", zorder=0, color="black", alpha=0.8, linewidth=0.75
            )

    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)
    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)

    z_long = (
        sceptre_zscore.loc[:, pd.IndexSlice[:, "TFRC"]]
        .melt(ignore_index=False, value_name="zscore")
        .reset_index()
    )
    fdr_long = (
        sceptre_fdr.loc[:, pd.IndexSlice[:, "TFRC"]]
        .melt(ignore_index=False, value_name="FDR")
        .reset_index()
    )

    longform_sceptre_results = pd.concat([z_long, fdr_long["FDR"]], axis=1).dropna(
        subset=["zscore"]
    )
    longform_sceptre_results = longform_sceptre_results.merge(
        crispr_table.rename({"guide": "grna_target"}, axis=1)[
            ["cell_line", "grna_target", "gene_effect"]
        ]
    )
    longform_sceptre_results["abs_z"] = longform_sceptre_results["zscore"].abs()
    rank_by_perturbation = (
        longform_sceptre_results[longform_sceptre_results["FDR"] < 0.05]
        .groupby(["cell_line", "grna_target"])["abs_z"]
        .rank(ascending=False)
    )
    longform_sceptre_results["significant_rank"] = rank_by_perturbation
    longform_sceptre_results = longform_sceptre_results.merge(
        longform_sceptre_results.value_counts(["response_id", "grna_target"])
        .rename("response_id_n_cell_lines")
        .reset_index()
    )
    longform_sceptre_results["lineage"] = longform_sceptre_results["cell_line"].map(
        cl_metadata.set_index("cell_line")["OncotreeLineage"]
    )
    longform_sceptre_results = longform_sceptre_results.merge(
        longform_sceptre_results[longform_sceptre_results["FDR"] < fdr_threshold]
        .value_counts(["grna_target", "response_id"])
        .rename("response_id_n_significant_cell_lines")
        .reset_index(),
        how="left",
    )

    create_top_gene_boxplot(
        longform_sceptre_results,
        grna_target="TFRC",
        top_n=10,
        figsize=(80 * mm, 45 * mm),
    )
    ax = plt.gca()
    for ytl in ax.get_yticklabels():
        if ytl.get_text() in ["PGK1", "PGAM1", "ALDOA", "LDHA"]:
            ytl.set_color("tab:red")
    plt.legend(
        title="TFRC gene effect",
        handles=[
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=5,
                markeredgewidth=0,
                label="-1.5",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=4.25,
                markeredgewidth=0,
                label="-1",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=3.5,
                markeredgewidth=0,
                label="-0.5",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=2.75,
                markeredgewidth=0,
                label="0",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(1, 0),
    )
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "TFRC_top_zscores.png"))


def prepare_mdm2_ube3a_heatmap_tables(out_dir=PROCESSED_DIR):
    somatic_damaging_mutations = load_file(
        "OmicsSomaticMutationsMatrixDamaging", local_dir=DOWNLOADED_DIR, index_col=0
    )
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)

    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)

    pseudobulk_by_ko = load_file(
        "pseudobulk_sum_by_ko.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    pseudobulk_tpm = np.log1p((pseudobulk_by_ko / pseudobulk_by_ko.sum()) * 1e6)

    cdkn2a_expr_after_ko = pseudobulk_tpm.loc[
        "CDKN2A", pd.IndexSlice[:, ["MDM2", "UBE3A"]]
    ]

    # heatmap
    hm = sceptre_zscore.loc[
        (
            (~sceptre_zscore.loc[:, pd.IndexSlice[:, ["MDM2", "UBE3A"]]].isna()).sum(
                axis=1
            )
            >= 25
        )
        & (
            (sceptre_fdr.loc[:, pd.IndexSlice[:, ["MDM2", "UBE3A"]]] < 0.05).sum(axis=1)
            >= 2
        ),
        pd.IndexSlice[:, ["MDM2", "UBE3A"]],
    ]

    # gene metadata
    gdf = pd.DataFrame(index=hm.index)
    gdf["P53 Pathway"] = gdf.index.isin(
        all_genesets[all_genesets["term"] == "HALLMARK_P53_PATHWAY"]["gene"].tolist()
    )
    gdf["P53 Pathway"] = gdf["P53 Pathway"].replace(
        {True: "tab:green", False: "lightgray"}
    )

    # cell metadata
    cdf = pd.DataFrame(
        [x[0] for x in hm.columns], index=hm.columns, columns=["cell_line"]
    ).rename_axis(["cl", "guide"])
    cdf["grna_target"] = [x[1] for x in hm.columns]
    cdf = cdf.merge(cl_metadata)
    cdf["Knockout"] = cdf["grna_target"].replace(
        {"MDM2": "tab:purple", "UBE3A": "tab:olive"}
    )
    cdf = cdf.merge(crispr_table.rename({"guide": "grna_target"}, axis=1), how="left")
    cdf = cdf.merge(
        cdf.loc[cdf["grna_target"] == "MDM2", :]
        .set_index("cell_line")
        .loc[:, "gene_effect"]
        .rename("MDM2 gene effect"),
        left_on="cell_line",
        right_index=True,
    )
    cdf = cdf.merge(
        cdf.loc[cdf["grna_target"] == "UBE3A", :]
        .set_index("cell_line")
        .loc[:, "gene_effect"]
        .rename("UBE3A gene effect"),
        left_on="cell_line",
        right_index=True,
    )
    mdm2_dep_cmap = sns.light_palette("tab:purple", reverse=True, as_cmap=True)
    ube3a_dep_cmap = sns.light_palette("tab:olive", reverse=True, as_cmap=True)
    # dep_cmap = sns.color_palette('Purples_r', as_cmap=True)
    dep_norm = Normalize(vmin=-1, vmax=0, clip=True)
    # cdf['Dependency'] = cdf['dependency_strength_class'].replace({'Nondependent': 'white', 'Weak': '#dadaeb', 'Moderate': '#9e9ac8', 'Strong':'#6950a3'})
    # cdf['Dependency'] = cdf['gene_effect'].apply(lambda x: to_hex(dep_cmap(dep_norm(x))))
    cdf["MDM2 dependency"] = cdf["MDM2 gene effect"].apply(
        lambda x: to_hex(mdm2_dep_cmap(dep_norm(x)))
    )
    cdf["UBE3A dependency"] = cdf["UBE3A gene effect"].apply(
        lambda x: to_hex(ube3a_dep_cmap(dep_norm(x)))
    )
    cdf["Lineage"] = cdf["OncotreeLineage"].replace(lineage_palette)
    cdf = cdf.merge(
        somatic_damaging_mutations.reindex(
            index=cl_metadata.arxspan_id, columns=["TP53 (7157)"]
        )["TP53 (7157)"]
        .rename("TP53_damaging")
        .reset_index()
    )
    cdf["TP53 status"] = cdf["TP53_damaging"].replace(
        {0: "lightgray", 1: "tab:red", 2: "tab:red"}
    )
    cdf = cdf.merge(
        cdkn2a_expr_after_ko.rename_axis(["cell_line", "grna_target"])
        .rename("CDKN2A")
        .reset_index()
    )
    cdkn2a_cmap = sns.light_palette("tab:orange", reverse=False, as_cmap=True)
    cdkn2a_norm = Normalize(vmin=0, vmax=7, clip=True)
    cdf["CDKN2A expr."] = cdf["CDKN2A"].apply(
        lambda x: to_hex(cdkn2a_cmap(cdkn2a_norm(x)))
    )
    cell_lines_to_keep = cdf[cdf["TP53_damaging"] == 0]["cell_line"].unique().tolist()
    e6_expression_subset = load_file(
        "OmicsExpressionTPMLogp1Virus_e6_subset.csv",
        local_dir=DOWNLOADED_DIR,
        index_col=0,
    )["Viral E6"]
    cdf = cdf.merge(
        e6_expression_subset, left_on="arxspan_id", right_index=True, how="left"
    ).fillna(0)
    e6_cmap = sns.light_palette("tab:pink", reverse=False, as_cmap=True)
    e6_norm = Normalize(vmin=0, vmax=6, clip=True)
    cdf["HPV E6 expr."] = cdf["Viral E6"].apply(lambda x: to_hex(e6_cmap(e6_norm(x))))
    cdf = cdf.set_index(["cell_line", "grna_target"])
    cdf_full = cdf.copy()

    if out_dir is not None:
        cdf_full.to_csv(os.path.join(out_dir, "p53_cell_line_properties.csv"))

    cdf = cdf.loc[pd.IndexSlice[cell_lines_to_keep, :]]

    cl_order = ["SKGII", "C4I", "MG63", "U343", "SLR23", "UMRC3"]
    pb_order = [(cl, "MDM2") for cl in cl_order if cl != "U343"] + [
        (cl, "UBE3A") for cl in cl_order
    ]
    cdf = cdf.loc[pb_order, :]

    return hm, gdf, cdf


def plot_mdm2_ube3a_heatmap(p53_heatmap, gene_df, cell_df):
    hm = p53_heatmap
    gdf = gene_df
    cdf = cell_df

    cl_order = ["SKGII", "C4I", "MG63", "U343", "SLR23", "UMRC3"]
    pb_order = [(cl, "MDM2") for cl in cl_order if cl != "U343"] + [
        (cl, "UBE3A") for cl in cl_order
    ]
    cdf = cdf.loc[pb_order, :]

    cg = sns.clustermap(
        hm.loc[:, cdf.index.tolist()].fillna(0).T,
        cmap="RdBu_r",
        vmin=-10,
        center=0,
        vmax=10,
        method="ward",
        figsize=(100 * mm, 60 * mm),
        col_cluster=True,
        row_cluster=False,
        col_colors=gdf[
            "P53 Pathway"
        ],  # col_colors=cdf[['Target', 'Lineage', 'TP53 status', 'Dependency']],
        row_colors=cdf[
            [
                "Knockout",
                "MDM2 dependency",
                "Lineage",
                "HPV E6 expr.",
                "UBE3A dependency",
                "CDKN2A expr.",
            ]
        ],
        dendrogram_ratio=(0, 0.05),
        colors_ratio=(0.03, 0.02),
        # cbar_pos=(0.2, 0.12, 0.6, 0.05), cbar_kws={'label': 'Z-score', 'orientation': 'horizontal'},
        xticklabels=False,
        yticklabels=False,
    )
    cg.ax_heatmap.set_xlabel("Response gene")
    cg.ax_heatmap.set_yticks(
        np.arange(0, len(pb_order)) + 0.5, [lbl[0] for lbl in pb_order]
    )
    cg.ax_heatmap.set_ylabel("")
    cg.ax_row_colors.set_xticks(
        np.arange(0, 6) + 0.5,
        labels=[t.get_text() for t in cg.ax_row_colors.get_xticklabels()],
        rotation=45,
        ha="right",
    )
    offset_x_labels(plt.gcf(), cg.ax_row_colors, 15, 0)
    cg.ax_row_colors.set_yticks([2.5, 8], labels=["MDM2", "UBE3A"])
    cg.cax.set_axis_off()

    handles = (
        [
            #     Patch(facecolor='white', edgecolor='white', label='Target'),
            #     Patch(facecolor='tab:purple', edgecolor='white', label='MDM2'),
            #     Patch(facecolor='tab:olive', edgecolor='white', label='UBE3A')
            # ] + [
            Patch(facecolor="white", edgecolor="white", label="Lineage")
        ]
        + [
            Patch(facecolor=c, edgecolor="white", label=lin)
            for lin, c in lineage_palette.items()
            if lin != "Other"
        ]
        + [
            Patch(facecolor="white", edgecolor="white", label="In P53 pathway"),
            Patch(facecolor="tab:green", edgecolor="white", label="True"),
            Patch(facecolor="lightgray", edgecolor="white", label="False"),
        ]
    )  # + [
    #     Patch(facecolor='white', edgecolor='white', label='Gene effect'),
    #     Patch(facecolor=to_hex(dep_cmap(dep_norm(-1))), edgecolor='white', label='-1'),
    #     Patch(facecolor=to_hex(dep_cmap(dep_norm(-0.5))), edgecolor='white', label='-0.5'),
    #     Patch(facecolor=to_hex(dep_cmap(dep_norm(1))), edgecolor='black', label='0')
    # ]
    cg.ax_heatmap.legend(
        handles=handles,
        ncol=3,
        handlelength=1,
        handleheight=1,
        handletextpad=0.5,
        columnspacing=1,
        loc="lower left",
        bbox_to_anchor=(0.35, -0.9),
        prop={"size": ANNOT_SIZE},
    )

    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.5)

    plt.savefig(os.path.join(FIGURE_DIR, "p53_clustermap.png"))


def plot_p53_viral_expression():
    cdf_full = load_file(
        "p53_cell_line_properties.csv", local_dir=PROCESSED_DIR, index_col=[0, 1]
    )

    cdf_full["TP53 status"] = cdf_full["TP53_damaging"].replace(
        {0: "WT", 1: "Mut.", 2: "Mut."}
    )

    plt.figure(figsize=(90 * mm, 45 * mm))
    sns.scatterplot(
        cdf_full.loc[pd.IndexSlice[:, "UBE3A"], :].sort_values(
            "TP53 status", ascending=False
        ),
        x="Viral E6",
        y="CDKN2A",
        hue="OncotreeLineage",
        palette=lineage_palette,
        style="TP53 status",
        markers={"WT": "o", "Mut.": "X"},
        linewidth=0.5,
        edgecolor="black",
    )
    plt.legend(loc="lower left", bbox_to_anchor=(1.1, -0.15), fontsize=ANNOT_SIZE)
    plt.subplots_adjust(left=0.15, right=0.6, top=0.9, bottom=0.15)
    plt.xlabel("HPV E6 expression")
    plt.ylabel("CDKN2A expression\nafter UBE3A knockout $\\log_{2}$(TPM + 1)")
    plt.savefig(os.path.join(FIGURE_DIR, "ube3a_viral_e6_cdkn2a_expr.png"))


def main():
    # figure 5a
    plot_mock_correlation_heatmap()
    plot_mock_correlation_network()

    # figure 5b
    plot_strong_correlation_network()

    # figure 5c
    plot_rna_perturbation_geneset_enrichment()

    # figure 5d
    p53_heatmap, gene_df, cell_df = prepare_mdm2_ube3a_heatmap_tables(
        out_dir=PROCESSED_DIR
    )
    plot_mdm2_ube3a_heatmap(p53_heatmap, gene_df, cell_df)

    # figure 5e
    plot_p53_viral_expression()

    # figure 5f
    plot_tfrc_top_genes(fdr_threshold=0.05)


if __name__ == "__main__":
    main()
