import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools
import scipy.sparse
import seaborn as sns
from constants import *
from data_utils import *
from figure_utils import *
from matplotlib.lines import Line2D
from taigapy import create_taiga_client_v3

tc = create_taiga_client_v3()

mpl.style.use("assets/stylesheet.mplstyle")


# figure 6a
def plot_broad_correlation_network():
    ko_pair_summary = load_file(
        "perturbation_pair_correlation_summary.csv", local_dir=PROCESSED_DIR
    )
    graph_subset_df = ko_pair_summary[ko_pair_summary["n_pairs_medium_corr"] >= 8]

    import igraph as ig

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

    vertices.loc[vertices["name"].isin(["CDK4", "MTOR", "PIK3CA"]), "y"] = (
        vertices.loc[vertices["name"].isin(["CDK4", "MTOR", "PIK3CA"]), "y"] + 0.75
    )
    vertices.loc[vertices["name"].isin(["CDK4", "MTOR", "PIK3CA"]), "x"] = (
        vertices.loc[vertices["name"].isin(["CDK4", "MTOR", "PIK3CA"]), "x"] + 1
    )

    vertices.loc[
        vertices["name"].isin(["IER3IP1", "SEC23IP", "SCYL1", "SLC39A9", "GET4"]), "y"
    ] = (
        vertices.loc[
            vertices["name"].isin(["IER3IP1", "SEC23IP", "SCYL1", "SLC39A9", "GET4"]),
            "y",
        ]
        + 0.75
    )

    vertices.loc[
        vertices["name"].isin(["IER3IP1", "SEC23IP", "SCYL1", "SLC39A9", "GET4"]), "x"
    ] = (
        vertices.loc[
            vertices["name"].isin(["IER3IP1", "SEC23IP", "SCYL1", "SLC39A9", "GET4"]),
            "x",
        ]
        + 0.25
    )

    # flip
    vertices["x"] = -vertices["x"]
    vertices["y"] = -vertices["y"]

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
        edges["n_pairs_high_corr"], vmin=10, vmax=50, lims=(0.5, 1.5)
    )

    fig, ax = plt.subplots(figsize=(75 * mm, 60 * mm))
    sns.scatterplot(
        vertices, x="x", y="y", linewidth=0.5, edgecolor="black", c="tab:gray", ax=ax
    )
    sns.scatterplot(
        vertices[
            vertices["name"].isin(["IER3IP1", "SEC23IP", "SCYL1", "SLC39A9", "GET4"])
        ],
        x="x",
        y="y",
        linewidth=0.5,
        edgecolor="black",
        c="tab:blue",
        ax=ax,
    )
    sns.scatterplot(
        vertices[vertices["name"].isin(["IER3IP1"])],
        x="x",
        y="y",
        linewidth=0.5,
        edgecolor="black",
        c="lightskyblue",
        ax=ax,
    )

    self_loops = edges[edges["source"] == edges["target"]]
    nonself_edges = edges[edges["source"] != edges["target"]]

    source_to_angles = {v: [] for v in vertices.index.tolist()}

    for i, r in nonself_edges.iterrows():
        if r["all_cross_lines_medium"]:
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
        plot_loop(
            (r["x_source"] - 0.197, r["y_source"]),
            r=(0.2, 0.2),
            ax=ax,
            color=c,
            linewidth=r["edgeweight"],
            zorder=-1,
        )

    ax.set_axis_off()
    leg = plt.legend(
        title="Number of cell line\npairs with r > 0.35",
        handles=[
            # Patch(color='white', label='Number of cell line\npairs with r > 0.35'),
            Line2D([], [], linestyle="solid", color="black", linewidth=0.5, label="10"),
            Line2D([], [], linestyle="solid", color="black", linewidth=1.0, label="30"),
            Line2D([], [], linestyle="solid", color="black", linewidth=1.5, label="50"),
        ],
        fontsize=ANNOT_SIZE,
        loc="lower left",
        bbox_to_anchor=(-0, 0.7),
    )

    # labels
    vertices["xoff"] = 0.25
    vertices["yoff"] = -0.25
    custom_pos = {"TRAIP": (0.25, 0), "SLC39A10": (-1.5, 0), "ATP5PO": (0, -0.55)}

    for g in ["NRBP1", "CAB39", "PSMA1", "PRKRA", "XRN1"]:
        custom_pos[g] = (-1.25, 0)  # left side
    for g in ["SLC39A9", "MAD2L1", "POLR2D"]:
        custom_pos[g] = (-0.5, -0.55)  # under
    for g in ["KIF11", "MDM2", "BRD4", "GET4"]:
        custom_pos[g] = (-0.5, 0.3)  # above
    for g, pos in custom_pos.items():
        (
            vertices.loc[vertices["name"] == g, "xoff"],
            vertices.loc[vertices["name"] == g, "yoff"],
        ) = pos

    for i in range(vertices.shape[0]):
        alpha = 0
        if vertices["name"][i] in ["MYC", "POLR2D", "MAD2L1"]:
            alpha = 0.75
        props = dict(
            facecolor="white", alpha=alpha, edgecolor="none", boxstyle="square,pad=0"
        )
        plt.text(
            x=vertices["x"][i] + vertices["xoff"][i],
            y=vertices["y"][i] + vertices["yoff"][i],
            s=vertices["name"][i],
            fontdict=dict(color="black", size=ANNOT_SIZE, bbox=props),
        )

    plt.subplots_adjust(left=0.02, right=0.9, top=0.99, bottom=0.05)
    plt.savefig(os.path.join(FIGURE_DIR, "broad_weak_correlation_graph_filtration.png"))


# figure 6b
def plot_shared_response_gsea_barplot():
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    models = list(cl_metadata["cell_line"])
    model_to_cl = cl_metadata.set_index("arxspan_id")["cell_line"]
    cl_to_model = cl_metadata.set_index("cell_line")["arxspan_id"]
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    cl_gene_effect = gene_effect.loc[cl_to_model[models]].rename(model_to_cl)
    dep_in_ps100 = (
        cl_gene_effect.loc[models]
        .reindex(columns=sceptre_zscore.columns.get_level_values(1).unique())
        .stack()
        .loc[lambda x: x < -0.75]
    )

    cluster_kos = ["SCYL1", "SEC23IP", "SLC39A9", "GET4", "IER3IP1"]
    shared_response = (
        sceptre_zscore.T.loc[:, cluster_kos, :]
        .loc[
            :,
            lambda x: (
                ((x.groupby(level=0).max() > 3).mean() > 0.5)
                | ((x.groupby(level=0).min() < -3).mean() > 0.5)
            ),
        ]
        .fillna(0)
    )
    shared_response_genes = shared_response.columns
    print("shared response genes: ", " ".join(shared_response_genes))

    geneset_collection = all_genesets[
        all_genesets["collection"].isin(["Reactome"])
        & (all_genesets["original_set_size"] >= 25)
        & (all_genesets["original_set_size"] <= 200)
    ]
    shared_response_enrichment = run_multiple_hypergeometric(
        query_genes=shared_response_genes,
        geneset_table=geneset_collection,
        all_genes=sceptre_zscore.index.tolist(),
    )

    shared_response_enrichment["fdr_transform"] = -np.log10(
        shared_response_enrichment["FDR"]
    )
    shared_response_enrichment["clean_term"] = shared_response_enrichment["term"].apply(
        divide_geneset_name_into_two_lines
    )

    plt.figure(figsize=(60 * mm, 60 * mm))
    sns.barplot(shared_response_enrichment.head(10), x="fdr_transform", y="clean_term")
    plt.xlabel("$-\\log_{10}$(q-value)")
    plt.ylabel("")
    plt.yticks(fontsize=ANNOT_SIZE - 0.5)
    plt.axvline(-np.log10(0.05), color="tab:red", linestyle="dashed", linewidth=1)
    plt.subplots_adjust(left=0.65, right=0.99, bottom=0.15, top=0.95)
    plt.savefig(os.path.join(FIGURE_DIR, "shared_response_gsea_barplot.png"))


# figure 6c
def plot_ranked_z_difference():
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    models = list(cl_metadata["cell_line"])
    model_to_cl = cl_metadata.set_index("arxspan_id")["cell_line"]
    cl_to_model = cl_metadata.set_index("cell_line")["arxspan_id"]
    ps100_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    cl_gene_effect = gene_effect.loc[cl_to_model[models]].rename(model_to_cl)

    # mean - mean
    plt.figure(figsize=(40 * mm, 60 * mm))
    ier3ip1_dep_mean = (
        ps100_zscore.T.loc[:, "IER3IP1", :]
        .loc[cl_gene_effect["IER3IP1"] < -0.75]
        .fillna(0)
        .mean()
    )
    other_dep = {}
    for ko in ["SCYL1", "SEC23IP", "SLC39A9", "GET4"]:
        other_dep[ko] = (
            ps100_zscore.T.loc[:, ko, :]
            .loc[cl_gene_effect[ko] < -0.75]
            .fillna(0)
            .mean()
        )
    other_dep_mean = pd.DataFrame(other_dep).T.mean()
    diff_means = (ier3ip1_dep_mean - other_dep_mean).loc[lambda x: x > 0]
    to_label = diff_means.loc[lambda x: x > 5.75].index

    sns.scatterplot(
        x=diff_means.rank(ascending=False), y=diff_means, s=8, alpha=0.8, color="gray"
    )
    sns.scatterplot(
        x=diff_means.rank(ascending=False).loc[["TXNRD1", "SLC7A11"]],
        y=diff_means.loc[["TXNRD1", "SLC7A11"]],
        s=8,
        color="tab:blue",
    )
    sns.scatterplot(
        x=diff_means.rank(ascending=False).loc[["HYOU1"]],
        y=diff_means.loc[["HYOU1"]],
        s=16,
        color="orange",
    )

    to_label = dict(to_label.to_series())
    # to_label['HYOU1'] = 'HYOU1 (11q23.3)'
    texts = [
        plt.text(
            diff_means.rank(ascending=False).loc[i] + 500,
            diff_means.loc[i],
            label,
            ha="left",
            va="center",
        )
        for i, label in to_label.items()
    ]

    plt.ylabel("Difference in mean IER3IP1 z-score")
    plt.xlabel("Rank")
    plt.subplots_adjust(left=0.25, right=0.8, bottom=0.15, top=0.95)

    plt.savefig(os.path.join(FIGURE_DIR, "IER3IP1_ranked_scatter.png"))


# figure 6e
def plot_IER3IP1_volcano():
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    profile_to_model_id = (
        load_file("OmicsProfiles", local_dir=DOWNLOADED_DIR)
        .query("is_default_entry")
        .loc[lambda x: x["DataType"] == "wgs"]
        .set_index("ProfileID")["ModelID"]
    )
    cn_seg = load_file("OmicsCNSegmentsProfileWGS", local_dir=DOWNLOADED_DIR)
    cn_seg["ModelID"] = cn_seg["ProfileID"].apply(
        lambda x: profile_to_model_id[x] if x in profile_to_model_id.index else pd.NA
    )
    cn_seg = cn_seg.rename(
        columns={
            "Chromosome": "CONTIG",
            "Start": "START",
            "End": "END",
            "SegmentMean": "SEGMENT_COPY_NUMBER",
        }
    )
    cn_seg = cn_seg[
        ["CONTIG", "START", "END", "Status", "SEGMENT_COPY_NUMBER", "ModelID"]
    ]

    cytoband = load_file("cytoband", local_dir=DOWNLOADED_DIR)
    a = pybedtools.BedTool(cytoband)
    # OmicsCNSegmentsWGS
    b = pybedtools.BedTool.from_dataframe(cn_seg)
    # get intersection, outputting the overlap
    cytoband_seg_overlap = a.intersect(b, wo=True).to_dataframe(
        names=[
            "chrom_cytoband",
            "start_cytoband",
            "end_cytoband",
            "name_cytoband",
            "stain_cytoband",
            "chrom",
            "start",
            "end",
            "state",
            "segment_cn",
            "ModelID",
            "overlap",
        ]
    )
    # label the cytobands
    cytoband_seg_overlap["cytoband"] = (
        cytoband_seg_overlap["chrom_cytoband"].str.replace("chr", "")
        + cytoband_seg_overlap["name_cytoband"]
    )
    # weight the segment cn by the length of overlap
    cytoband_seg_overlap["weighted_cn"] = (
        cytoband_seg_overlap["segment_cn"] * cytoband_seg_overlap["overlap"]
    )
    # normalize by the total overlap
    cytoband_cn = (
        cytoband_seg_overlap.groupby(["ModelID", "cytoband"])
        .agg({"weighted_cn": "sum", "overlap": "sum"})
        .reset_index()
    )
    cytoband_cn["weighted_avg_cn"] = cytoband_cn["weighted_cn"] / cytoband_cn["overlap"]
    cytoband_cn = cytoband_cn.set_index(["ModelID", "cytoband"])[
        "weighted_avg_cn"
    ].unstack()

    plt.figure(figsize=(45 * mm, 60 * mm))
    gene_dep = gene_effect < -0.75

    cytoband_cn_renamed, gene_effect_ov = np.log2(cytoband_cn + 1).align(
        gene_dep["IER3IP1"].dropna(), join="inner", axis=0
    )
    # cyto_cn_dep = cytoband_cn_renamed.loc[gene_effect_ov > 0.75].dropna(how="all")
    # cyto_cn_not_dep = cytoband_cn_renamed.loc[gene_effect_ov < 0.75].dropna(how="all")

    pearson_r, pearson_p = scipy.stats.pearsonr(
        cytoband_cn_renamed, gene_effect.loc[cytoband_cn_renamed.index, ["IER3IP1"]]
    )
    cor_df = pd.DataFrame(
        pearson_r, columns=["pearson_r"], index=cytoband_cn_renamed.columns
    )
    cor_df["pearson_p"] = pearson_p
    cor_df = cor_df.dropna()
    cor_df["FDR"] = scipy.stats.false_discovery_control(cor_df["pearson_p"])

    sns.scatterplot(
        x=cor_df["pearson_r"],
        y=-np.log10(cor_df["FDR"]),
        label="other",
        color="grey",
        s=8,
        alpha=0.8,
    )
    sns.scatterplot(
        x=cor_df.loc[cor_df.index.str.startswith("18q21"), "pearson_r"],
        y=-np.log10(cor_df["FDR"]),
        label="18q21",
        color="red",
        s=8,
        alpha=0.8,
    )
    sns.scatterplot(
        x=cor_df.loc[
            cor_df.index.str.startswith("18q")
            & (~cor_df.index.str.startswith("18q21")),
            "pearson_r",
        ],
        y=-np.log10(cor_df["FDR"]),
        label="other 18q",
        color="blue",
        s=8,
        alpha=0.8,
    )
    sns.scatterplot(
        x=cor_df.loc[
            cor_df.index.str.startswith("11q23") | cor_df.index.str.startswith("11q24"),
            "pearson_r",
        ],
        y=-np.log10(cor_df["FDR"]),
        label="11q23-4",
        color="orange",
        s=8,
        alpha=0.8,
    )

    label_bands = ["18q21.1", "18q21.32", "11q23.3"]
    texts = [
        plt.text(
            cor_df.loc[i, "pearson_r"] - 0.06,
            -np.log10(cor_df.loc[i, "FDR"]),
            i,
            ha="center",
            va="center",
        )
        for i in label_bands
    ]

    plt.legend(
        title="",
        handleheight=1,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1.5,
        loc="lower left",
        ncol=1,
        fontsize=ANNOT_SIZE,
    )

    plt.axhline(-np.log10(0.1), color="lightgray", ls="dashed")
    plt.xlabel("CN versus IER3IP1 gene effect Pearson's r")
    plt.ylabel("$-\\log_{10}$(FDR)")
    plt.subplots_adjust(left=0.2, right=0.85)

    plt.savefig(os.path.join(FIGURE_DIR, "IER3IP1_volcano.png"))


# figure 6f
def plot_IER3IP1_CN_box():
    gene_effect = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    gene_effect.columns = gene_effect.columns.str.split().str[0]
    profile_to_model_id = (
        load_file("OmicsProfiles", local_dir=DOWNLOADED_DIR)
        .query("is_default_entry")
        .loc[lambda x: x["DataType"] == "wgs"]
        .set_index("ProfileID")["ModelID"]
    )
    cn_seg = load_file("OmicsCNSegmentsProfileWGS", local_dir=DOWNLOADED_DIR)
    cn_seg["ModelID"] = cn_seg["ProfileID"].apply(
        lambda x: profile_to_model_id[x] if x in profile_to_model_id.index else pd.NA
    )
    cn_seg = cn_seg.rename(
        columns={
            "Chromosome": "CONTIG",
            "Start": "START",
            "End": "END",
            "SegmentMean": "SEGMENT_COPY_NUMBER",
        }
    )
    cn_seg = cn_seg[
        ["CONTIG", "START", "END", "Status", "SEGMENT_COPY_NUMBER", "ModelID"]
    ]

    cytoband = load_file("cytoband", local_dir=DOWNLOADED_DIR)
    a = pybedtools.BedTool(cytoband)
    # OmicsCNSegmentsWGS
    b = pybedtools.BedTool.from_dataframe(cn_seg)
    # get intersection, outputting the overlap
    cytoband_seg_overlap = a.intersect(b, wo=True).to_dataframe(
        names=[
            "chrom_cytoband",
            "start_cytoband",
            "end_cytoband",
            "name_cytoband",
            "stain_cytoband",
            "chrom",
            "start",
            "end",
            "state",
            "segment_cn",
            "ModelID",
            "overlap",
        ]
    )
    # label the cytobands
    cytoband_seg_overlap["cytoband"] = (
        cytoband_seg_overlap["chrom_cytoband"].str.replace("chr", "")
        + cytoband_seg_overlap["name_cytoband"]
    )
    # weight the segment cn by the length of overlap
    cytoband_seg_overlap["weighted_cn"] = (
        cytoband_seg_overlap["segment_cn"] * cytoband_seg_overlap["overlap"]
    )
    # normalize by the total overlap
    cytoband_cn = (
        cytoband_seg_overlap.groupby(["ModelID", "cytoband"])
        .agg({"weighted_cn": "sum", "overlap": "sum"})
        .reset_index()
    )
    cytoband_cn["weighted_avg_cn"] = cytoband_cn["weighted_cn"] / cytoband_cn["overlap"]
    cytoband_cn = cytoband_cn.set_index(["ModelID", "cytoband"])[
        "weighted_avg_cn"
    ].unstack()

    plt.figure(figsize=(80 * mm, 60 * mm))
    gene_dep = gene_effect < -0.75

    cytoband_cn_renamed, gene_effect_ov = np.log2(cytoband_cn + 1).align(
        gene_dep["IER3IP1"].dropna(), join="inner", axis=0
    )

    features = (
        (
            cytoband_cn_renamed.loc[
                :, lambda x: x.columns.str.startswith("18q21")
            ].mean(axis=1)
            < 0.75
        )
        .replace({True: "18q21 loss", False: None})
        .to_frame("18q21")
    )
    features["11q23-4"] = (
        cytoband_cn_renamed.loc[:, lambda x: x.columns.str.contains("^11q2(3|4)")].mean(
            axis=1
        )
        < 0.75
    ).replace({True: "11q23-4 loss", False: None})
    features["CN status"] = features.apply(
        lambda x: " and\n".join(x.dropna()) if x.notnull().sum() > 0 else "neither",
        axis=1,
    )
    features["CN status"] = np.where(
        features["CN status"].str.contains("and|neither"),
        features["CN status"],
        features["CN status"] + " only",
    )

    plt.figure(figsize=(50 * mm, 60 * mm))

    order = features["CN status"].value_counts().index
    sns.boxplot(
        x=gene_effect["IER3IP1"], y=features["CN status"], showfliers=False, order=order
    )
    sns.stripplot(
        x=gene_effect["IER3IP1"],
        y=features["CN status"],
        palette="dark:k",
        order=order,
        size=1.5,
        alpha=0.8,
    )
    plt.xlabel("IER3IP1 gene effect")

    plt.subplots_adjust(left=0.4, right=0.99)
    plt.savefig(os.path.join(FIGURE_DIR, "IER3IP1_box.png"))


def main():
    # figure 6a
    plot_broad_correlation_network()

    # figure 6b
    plot_shared_response_gsea_barplot()

    # figure 6c
    plot_ranked_z_difference()

    # figure 6e
    plot_IER3IP1_volcano()

    # figure 6f
    plot_IER3IP1_CN_box()


if __name__ == "__main__":
    main()
