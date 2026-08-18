import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from constants import *
from data_utils import *
from figure_utils import *
from matplotlib.lines import Line2D
from taigapy import create_taiga_client_v3

tc = create_taiga_client_v3()

mpl.style.use("assets/stylesheet.mplstyle")


# figure 2a
def plot_truncation_example():
    truncation_example_longform = load_file(
        "single_cell_arm_loss_example.csv", local_dir=PROCESSED_DIR
    )
    target = truncation_example_longform[
        (truncation_example_longform["Cell subset"] == "Normal KO")
    ]["assigned_ko"].iloc[0]
    gene_locations = load_file("gene_locations.csv", local_dir=DATA_DIR)
    chrarm, target_pos = gene_locations[gene_locations["Gene name"] == target][
        ["ChrArm", "Midgene estimate"]
    ].iloc[0]

    plt.figure(figsize=(90 * mm, 60 * mm))
    sns.lineplot(
        truncation_example_longform,
        y="z_expr_rolled",
        x="Midgene estimate",
        hue="Cell subset",
        palette=truncation_cell_group_palette,
        hue_order=[
            "Normal KO",
            "Arm loss KO",
            "Arm gain KO",
            "Other control KO",
            "Unperturbed",
        ],
        drawstyle="steps-pre",
        linewidth=1,
    )
    plt.xlabel(f"Gene Coordinate on {chrarm} (bp)", fontsize=LABEL_SIZE)
    plt.ylabel("Z-scored expression", fontsize=LABEL_SIZE)
    if "p" in chrarm:
        plt.text(
            0.03,
            0.93,
            r"Distal $\longleftarrow$",
            transform=plt.gca().transAxes,
            ha="left",
        )
        plt.text(
            0.97,
            0.93,
            r"$\longrightarrow$ Proximal",
            transform=plt.gca().transAxes,
            ha="right",
        )
    elif "q" in chrarm:
        plt.text(
            0.03,
            0.93,
            r"Proximal $\longleftarrow$",
            transform=plt.gca().transAxes,
            ha="left",
        )
        plt.text(
            0.97,
            0.93,
            r"$\longrightarrow$ Distal",
            transform=plt.gca().transAxes,
            ha="right",
        )

    plt.axhline(0, linestyle="solid", c="tab:gray", zorder=0)
    plt.axvline(target_pos, linestyle="dashed", c="black")
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(
        title="Cell Group",
        handles=h
        + [
            Line2D(
                [],
                [],
                color="black",
                linestyle="dashed",
                marker=None,
                label=f"{target} position",
            )
        ],
        ncol=2,
        columnspacing=1,
        fontsize=ANNOT_SIZE,
        loc="lower left",
    )
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.9)
    plt.savefig(os.path.join(FIGURE_DIR, "arm_loss_visualization.png"))


# figure 2b
def plot_arm_truncation_correction():
    gene_locs = load_file("gene_locations.csv", local_dir=DATA_DIR)
    zscore_changes_by_arm = load_file(
        "single_cell_ko_arm_zscores.csv",
        local_dir=PROCESSED_DIR,
        index_col=[0, 1, 2],
        header=[0],
    )
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    controls = sorted(
        ko_metadata[ko_metadata["target_class"] == "Olfactory Control"]["knockout"]
        .unique()
        .tolist()
    )

    arm_zscore_longform = (
        zscore_changes_by_arm.loc[pd.IndexSlice[:, controls, "all"]]
        .groupby(level=["assigned_ko"])
        .mean()
        .melt(ignore_index=False)
        .reset_index()
        .merge(
            zscore_changes_by_arm.loc[pd.IndexSlice[:, controls, "normal"]]
            .groupby(level=["assigned_ko"])
            .mean()
            .melt(ignore_index=False)
            .reset_index(),
            left_on=["assigned_ko", "variable"],
            right_on=["assigned_ko", "variable"],
        )
        .rename({"value_x": "all", "value_y": "normal", "variable": "arm"}, axis=1)
    )
    arm_zscore_longform = arm_zscore_longform.merge(
        gene_locs[["Gene name", "ChrArm"]], left_on="assigned_ko", right_on="Gene name"
    )
    arm_zscore_longform["Targeted arm"] = (
        arm_zscore_longform["ChrArm"] == arm_zscore_longform["arm"]
    ).astype(str)

    plt.figure(figsize=(80 * mm, 60 * mm))

    sns.scatterplot(
        arm_zscore_longform,
        x="all",
        y="normal",
        hue="Targeted arm",
        palette={"True": "tab:red", "False": "tab:gray"},
        hue_order=["True", "False"],
        s=16,
    )
    plt.xlabel("Arm-level expression change (unfiltered)")
    plt.ylabel("Arm-level expression change (filtered)")

    annots = arm_zscore_longform[
        (arm_zscore_longform["normal"].abs() > 0.025)
        & (arm_zscore_longform["all"].abs() > 0.025)
    ]
    texts = []
    for i, r in annots.iterrows():
        if r["normal"] < 0:
            ha, va = "center", "center"
            sign = 1
        else:
            ha, va = "center", "center"
            sign = -1
        texts.append(
            plt.text(
                r["all"] + sign * 0.008,
                r["normal"] + sign * 0.002,
                "{} in\n{} KO".format(r["arm"], r["assigned_ko"]),
                ha=ha,
                va=va,
            )
        )

    plt.axhline(0, color="black", linestyle="solid", zorder=0)
    plt.axvline(0, color="black", linestyle="solid", zorder=0)
    plt.axline((0, 0), slope=1, color="black", linestyle="dotted", zorder=0)

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

    plt.savefig(os.path.join(FIGURE_DIR, "arm_loss_correction_scatter.png"))


# figure 2c
def plot_arm_loss_predictors():
    arm_alteration_summary_table = load_file(
        "arm_alteration_covariates.csv", local_dir=PROCESSED_DIR
    )
    arm_alteration_correlations = load_file(
        "arm_alteration_correlations.csv", local_dir=PROCESSED_DIR
    )

    fig, axs = plt.subplots(1, 2, figsize=(90 * mm, 70 * mm), width_ratios=[39, 1])
    plt.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.15)

    arm_alteration_subset = arm_alteration_correlations[
        arm_alteration_correlations["Outcome"] == "Arm truncation frequency"
    ]
    arm_alteration_subset = arm_alteration_subset.merge(
        arm_alteration_summary_table.groupby("cell_line")["Total cells per KO"].mean(),
        left_on="Cell line",
        right_index=True,
    )
    predictor_order = (
        arm_alteration_subset.drop_duplicates(subset=["Outcome", "Predictor"])
        .sort_values("Aggregate correlation with outcome")["Predictor"]
        .tolist()
    )

    sns.barplot(
        arm_alteration_subset.drop_duplicates(subset=["Outcome", "Predictor"]),
        x="Aggregate correlation with outcome",
        y="Predictor",
        order=predictor_order,
        fill=False,
        color="black",
        linewidth=1,
        ax=axs[0],
    )

    sns.stripplot(
        arm_alteration_subset,
        x="Correlation with outcome",
        y="Predictor",
        order=predictor_order,
        hue="Total cells per KO",
        hue_norm=(25, 175),
        palette="viridis",
        legend=False,
        edgecolor="black",
        linewidth=0.5,
        ax=axs[0],
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(25, 175), cmap="viridis"),
        cax=axs[1],
        orientation="vertical",
        label="Average number of cells per knockout",
    )

    axs[0].set_xlabel("Correlation with arm truncation frequency (Pearson's r)")
    axs[0].set_ylabel("Feature")

    plt.savefig(os.path.join(FIGURE_DIR, "gene_wise_arm_loss_predictors.png"))


# figure 2d
def rescreen_dna_genes_lfc():
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    geneset_collection = all_genesets[
        all_genesets["collection"].isin(["Reactome"])
        & (all_genesets["original_set_size"] >= 25)
        & (all_genesets["original_set_size"] <= 200)
    ]
    deep_ctrl_results = load_file("deep_ctrl_results.csv", local_dir=PROCESSED_DIR)

    damage_terms = [x for x in geneset_collection["term"].unique() if "DNA_DAMAGE" in x]
    repair_terms = [x for x in geneset_collection["term"].unique() if "DNA_REPAIR" in x]
    dna_terms = damage_terms + repair_terms
    dna_genes = geneset_collection[geneset_collection["term"].isin(dna_terms)][
        "gene"
    ].unique()

    dna_lfc_df = pd.DataFrame()
    for cl in ["KMRC20", "UMRC3"]:
        ctrl_results = deep_ctrl_results.query(f'cell_line == "{cl}"')
        df = ctrl_results[ctrl_results["response_id"].isin(dna_genes)].copy()
        df["cell_line"] = cl
        dna_lfc_df = pd.concat([dna_lfc_df, df])

    controls = ["OR13C7", "OR9Q1", "OR13H1", "OR10J3"]
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))[: len(controls)]
    ctrl_palette = {item: tab10_colors[i] for i, item in enumerate(controls)}

    plt.figure(figsize=(80 * mm, 60 * mm))
    ax = sns.boxplot(
        dna_lfc_df,
        x="cell_line",
        y="log_2_fold_change",
        hue="grna_target",
        fliersize=0,
        palette=ctrl_palette,
    )
    sns.stripplot(
        dna_lfc_df,
        x="cell_line",
        y="log_2_fold_change",
        hue="grna_target",
        dodge=True,
        s=1,
        alpha=0.5,
        palette=ctrl_palette,
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("LFC of DNA damage\nand repair genes")
    ax.get_legend().remove()
    plt.legend(title="", loc="upper left")
    plt.gca().legend(
        loc="center left", bbox_to_anchor=(-0.1, 1.1), ncols=len(ctrl_palette)
    )
    plt.subplots_adjust(left=0.15, right=0.95, top=0.8)
    plt.savefig(os.path.join(FIGURE_DIR, "rescreen_dna_genes.png"))


# figure 2e
def plot_and_print_single_cell_high_variance_stats(out_dir=PROCESSED_DIR):
    from generate import select_high_variance_genes

    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    (
        control_single_cell_gene_stats,
        median_single_cell_stats,
        high_var_common_genes,
        high_expr_excluded,
    ) = select_high_variance_genes(
        load_file("single_cell_control_gene_stats.csv", local_dir=PROCESSED_DIR)
    )
    selected = (
        control_single_cell_gene_stats[control_single_cell_gene_stats["selected"]][
            "gene"
        ]
        .unique()
        .tolist()
    )
    high_variance_correlate_enrichment = load_file(
        "single_cell_high_variance_correlate_geneset_enrichment.csv",
        local_dir=PROCESSED_DIR,
    )
    selected_gene_order = (
        high_variance_correlate_enrichment.groupby("high_variance_gene")["FDR"]
        .min()
        .sort_values()
        .index.tolist()
    )

    mitochondrial_genes = all_genesets[
        (all_genesets["collection"] == "HGNC")
        & (all_genesets["term"].str.contains("Mitochondrially encoded"))
    ]["gene"].tolist()
    ribosomal_genes = all_genesets[
        (all_genesets["collection"] == "HGNC")
        & (all_genesets["term"].str.contains("(R|r)ibosom"))
    ]["gene"].tolist()
    housekeeping_genes = all_genesets[
        all_genesets["term"] == "HSIAO_HOUSEKEEPING_GENES"
    ]["gene"].tolist()

    high_variance_stats_df = high_var_common_genes.copy()
    high_variance_stats_df["highly_expressed"] = ~high_variance_stats_df.index.isin(
        high_expr_excluded.index.tolist()
    )
    high_variance_stats_df["mitochondrial"] = high_variance_stats_df.index.isin(
        mitochondrial_genes
    )
    high_variance_stats_df["ribosomal"] = high_variance_stats_df.index.isin(
        ribosomal_genes
    )
    high_variance_stats_df["housekeeping"] = high_variance_stats_df.index.isin(
        housekeeping_genes
    )
    high_variance_stats_df["selected"] = high_variance_stats_df.index.isin(selected)

    print("Highly variable genes:", high_variance_stats_df.shape[0])
    print("Highly expressed genes:", high_variance_stats_df["highly_expressed"].sum())
    print("Lowly expressed genes:", (~high_variance_stats_df["highly_expressed"]).sum())
    print("Mitochondrial genes:", high_variance_stats_df["mitochondrial"].sum())
    print("Ribosomal genes:", high_variance_stats_df["ribosomal"].sum())
    print("Housekeeping genes:", high_variance_stats_df["housekeeping"].sum())

    xlabel = "lines_present"
    ylabel = "index_of_dispersion"

    selected_gene_order = (
        high_variance_correlate_enrichment.groupby("high_variance_gene")["FDR"]
        .min()
        .sort_values()
        .index.tolist()
    )
    high_variance_stats_df["hue"] = (
        high_variance_stats_df.index.to_series()
        .map({x: x for x in selected_gene_order})
        .fillna("other")
    )
    palette = sns.color_palette("tab10", 5)
    palette.append("0.7")

    plt.figure(figsize=(50 * mm, 50 * mm))
    plt.subplots_adjust(left=0.22, right=0.95, top=0.95, bottom=0.15)
    ax = sns.stripplot(
        high_variance_stats_df[~high_variance_stats_df["highly_expressed"]],
        y=ylabel,
        x=xlabel,
        hue="hue",
        hue_order=selected_gene_order + ["other"],
        palette=palette,
        s=5,
        alpha=0.8,
        legend=False,
    )

    x_map = {
        label: i
        for i, label in enumerate(high_variance_stats_df[xlabel].sort_values().unique())
    }
    annots = high_variance_stats_df[~high_variance_stats_df["highly_expressed"]].loc[
        selected_gene_order
    ]
    texts = [
        ax.text(
            x_map[r[xlabel]],
            r[ylabel],
            i,
            ha="center",
            va="center",
            fontsize=ANNOT_SIZE,
        )
        for i, r in annots.iterrows()
    ]
    plt.ylabel("Median index of dispersion\n(variance / mean)")
    plt.xlabel("Number of cell lines")
    adjust_text(
        texts,
        ax=ax,
        expand=(2, 2),
        force_static=(0.3, 0.3),
        arrowprops=annot_arrow_props,
    )
    plt.margins(x=0.1)
    plt.savefig(os.path.join(FIGURE_DIR, "single_cell_high_variance_stats.png"))

    if out_dir is not None:
        high_variance_stats_df.to_csv(
            os.path.join(out_dir, "fig2e_table.csv"), index=True
        )

    return high_variance_stats_df


# figure 2f
def plot_and_print_single_cell_example_high_variance(g):
    top_variance_expression = load_file(
        "single_cell_high_variance_raw_expression.csv", local_dir=PROCESSED_DIR
    )

    top_var_gene_expr_subset = top_variance_expression.dropna(subset=[g])
    non_zero_frac = (
        top_var_gene_expr_subset.assign(nonzero=lambda x: x[g] > 0)
        .groupby("cell_line")["nonzero"]
        .mean()
        .rename_axis("")
    )
    print("Nonzero fraction of cells:", non_zero_frac)
    print(
        "total:",
        top_var_gene_expr_subset.assign(nonzero=lambda x: x[g] > 0)["nonzero"].mean(),
    )

    df = top_var_gene_expr_subset.fillna(0)
    df["log_counts"] = np.log2(df[g] + 1)
    non_zero_df = df[df[g] != 0]
    non_zero_df["non_zero_frac"] = non_zero_df["cell_line"].map(non_zero_frac)
    cl_show = non_zero_df["cell_line"].unique()[
        non_zero_df["S100A8"].groupby(non_zero_df["cell_line"]).sum() > 1
    ]
    non_zero_df = non_zero_df[non_zero_df["cell_line"].isin(cl_show)]

    fig, axs = plt.subplots(1, 2, figsize=(50 * mm, 50 * mm), width_ratios=[10, 1])
    cmap = plt.get_cmap("viridis_r")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    palette = non_zero_frac.map(lambda x: cmap(norm(x))).to_dict()

    ax = axs.flat[0]
    sns.violinplot(
        non_zero_df,
        x="log_counts",
        y="cell_line",
        inner=None,
        hue="cell_line",
        palette=palette,
        linewidth=0,
        ax=ax,
    )
    sns.stripplot(
        non_zero_df,
        x="log_counts",
        y="cell_line",
        jitter=True,
        size=1,
        color="k",
        alpha=0.8,
        ax=ax,
    )
    ax.set_xlabel(f"Non-zero {g} expression $\\log_{2}$(TPM+1)")
    ax.set_ylabel("")

    cbar = fig.colorbar(sm, cax=axs.flat[1])
    cbar.set_label(f"Fraction of cells expressing {g}")

    plt.subplots_adjust(left=0.2, right=0.85, top=0.95, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "single_cell_high_variance_example.png"))

    return top_variance_expression


# figure 2g
def plot_high_variance_correlate_geneset_enrichment(fdr_threshold=0.05):
    from data_utils import clean_geneset_name

    high_variance_correlate_enrichment = load_file(
        "single_cell_high_variance_correlate_geneset_enrichment.csv",
        local_dir=PROCESSED_DIR,
    )
    selected_genesets = (
        high_variance_correlate_enrichment[
            high_variance_correlate_enrichment["FDR"] < fdr_threshold
        ]["term"]
        .value_counts()
        .head(5)
        .index.tolist()
    )
    selected_gene_order = (
        high_variance_correlate_enrichment.groupby("high_variance_gene")["FDR"]
        .min()
        .sort_values()
        .index.tolist()
    )

    enrichment_to_plot = high_variance_correlate_enrichment[
        high_variance_correlate_enrichment["term"].isin(selected_genesets)
    ].copy()
    enrichment_to_plot["fdr_transform"] = -np.log10(enrichment_to_plot["FDR"])
    enrichment_to_plot["clean_term"] = enrichment_to_plot["term"].apply(
        lambda x: clean_geneset_name(x, remove_prefix=True, nth=1)
    )
    enrichment_to_plot = enrichment_to_plot.replace(
        {"TNFA\nSIGNALING\nVIA\nNFKB": "TNFA SIGNALING\nVIA NFKB"}
    )

    fig = plt.figure(figsize=(60 * mm, 50 * mm))
    ax = plt.gca()

    sns.boxplot(
        enrichment_to_plot,
        x="fdr_transform",
        y="clean_term",
        hue="high_variance_gene",
        hue_order=selected_gene_order,
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
        enrichment_to_plot.rename({"n_overlap": "N genes"}, axis=1),
        xlabel="fdr_transform",
        ylabel="clean_term",
        hue="high_variance_gene",
        dodge_shift=(-0.3, 0.3),
        hue_order=selected_gene_order,
        jitter=(0.05, 0.05),
        s=4,
        # size='N genes', sizes=(4, 48),
        linewidth=0.25,
        edgecolor="black",
        ax=ax,
        alpha=0.5,
        # legend=False
    )
    plt.ylabel("")
    plt.xlabel("$-\\log_{10}$(Fisher's exact q-value)")
    plt.legend(title="High variance gene", loc="lower right")
    plt.subplots_adjust(left=0.28, right=0.95, bottom=0.15, top=0.95)
    plt.axvline(
        -np.log10(fdr_threshold),
        linestyle="dashed",
        color="tab:red",
        zorder=-1,
        linewidth=0.5,
    )
    plt.savefig(
        os.path.join(FIGURE_DIR, "single_cell_high_variance_correlate_genesets.png")
    )


def main():

    # figure 2a
    plot_truncation_example()

    # figure 2b
    plot_arm_truncation_correction()

    # figure 2c
    plot_arm_loss_predictors()

    # figure 2d
    rescreen_dna_genes_lfc()

    # figure 2e
    high_variance_stats_df = plot_and_print_single_cell_high_variance_stats(
        out_dir=PROCESSED_DIR
    )

    # figure 2f
    top_variance_expression = plot_and_print_single_cell_example_high_variance("S100A8")

    # figure 2g
    plot_high_variance_correlate_geneset_enrichment(fdr_threshold=0.05)


if __name__ == "__main__":
    main()
