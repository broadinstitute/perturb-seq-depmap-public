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

mpl.style.use("assets/stylesheet.mplstyle")


# figure s4a
def plot_top_dependency_diff_expr_correlations():
    top_dependency_diff_expr_gene_df = load_file(
        "dependency_top_diff_expr_gene_table.csv", local_dir=PROCESSED_DIR
    )
    expr_order = (
        top_dependency_diff_expr_gene_df.set_index("response_id")["response_id_order"]
        .drop_duplicates()
        .sort_values()
        .index.tolist()
    )
    top_deg_corrs = fast_cor(
        top_dependency_diff_expr_gene_df.pivot(
            index="grna_target", columns="response_id", values="mean_dep_z"
        )
    ).loc[expr_order, expr_order]
    up_corrs = top_deg_corrs.loc[expr_order[:10], expr_order[:10]]
    down_corrs = top_deg_corrs.loc[expr_order[10:], expr_order[10:]]
    up_corrs_flat = up_corrs.values[(np.triu(np.ones_like(up_corrs), k=1) == 1)]
    down_corrs_flat = down_corrs.values[(np.triu(np.ones_like(down_corrs), k=1) == 1)]
    print("upreg median correlation:", np.quantile(up_corrs_flat, q=0.25))
    print("downreg median correlation:", np.quantile(down_corrs_flat, q=0.25))

    up_cm = sns.clustermap(up_corrs, method="ward")
    plt.clf()
    up_order = up_cm.data2d.index.tolist()
    down_cm = sns.clustermap(down_corrs, method="ward")
    plt.clf()
    down_order = down_cm.data2d.index.tolist()
    gene_order = up_order + down_order

    plt.figure(figsize=(85 * mm, 70 * mm))
    sns.heatmap(
        top_deg_corrs.loc[gene_order, gene_order],
        xticklabels=False,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson's r"},
    )
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)
    plt.xticks([])
    plt.xlabel("")
    plt.ylabel("")

    plt.savefig(os.path.join(FIGURE_DIR, "top_differential_genes_correlations.png"))


# figure s4b
def plot_common_diff_expr_gene_enrichment(
    n_top_genes=50, min_targets=5, fdr_threshold=0.05
):
    top_diff_expr_genes = load_file(
        "dependency_top_diff_expr_gene_summary.csv", local_dir=PROCESSED_DIR
    )
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)

    up_gsea_results = run_multiple_hypergeometric(
        query_genes=top_diff_expr_genes[
            top_diff_expr_genes["n_targets_up"] >= min_targets
        ]
        .sort_values("mean_zscore", ascending=False)
        .head(n_top_genes)["response_id"]
        .tolist(),
        geneset_table=all_genesets[
            all_genesets["collection"].isin(["Hallmark", "Reactome"])
        ],
        all_genes=top_diff_expr_genes["response_id"].tolist(),
    )
    up_gsea_results["clean_term"] = up_gsea_results["term"].apply(
        lambda x: clean_geneset_name(x, nth=2, remove_prefix=True)
    )
    up_gsea_results["fdr_transform"] = -np.log10(up_gsea_results["FDR"])

    down_gsea_results = run_multiple_hypergeometric(
        query_genes=top_diff_expr_genes[
            top_diff_expr_genes["n_targets_down"] >= min_targets
        ]
        .sort_values("mean_zscore", ascending=True)
        .head(n_top_genes)["response_id"]
        .tolist(),
        geneset_table=all_genesets[
            all_genesets["collection"].isin(["Hallmark", "Reactome"])
        ],
        all_genes=top_diff_expr_genes["response_id"].tolist(),
    )
    down_gsea_results["clean_term"] = down_gsea_results["term"].apply(
        lambda x: clean_geneset_name(x, nth=2, remove_prefix=True)
    )
    down_gsea_results["fdr_transform"] = -np.log10(down_gsea_results["FDR"])

    fig, axs = plt.subplots(1, 2, figsize=(85 * mm, 60 * mm))
    plt.subplots_adjust(left=0.35, right=0.8, top=0.85, bottom=0.18)

    sns.barplot(down_gsea_results.head(5), x="fdr_transform", y="clean_term", ax=axs[0])
    axs[0].invert_xaxis()
    axs[0].set_xlabel("$-\\log_{10}$(q-value)")
    axs[0].set_ylabel("")

    sns.barplot(up_gsea_results.head(5), x="fdr_transform", y="clean_term", ax=axs[1])

    axs[1].yaxis.set_ticks_position("right")
    axs[1].set_xlabel("$-\\log_{10}$(q-value)")
    axs[1].set_ylabel("")

    axs[0].set_title("Downregulated")
    axs[1].set_title("Upregulated")

    axs[0].axvline(-np.log10(fdr_threshold), linestyle="dashed", color="tab:red")
    axs[1].axvline(-np.log10(fdr_threshold), linestyle="dashed", color="tab:red")
    plt.savefig(os.path.join(FIGURE_DIR, "common_differential_genes_gsea.png"))


# figure s4c
def plot_dependent_hallmark_geneset_correlation():
    dependent_hallmark_mean_z = load_file(
        "dependency_hallmark_mean_z_matrix.csv", local_dir=PROCESSED_DIR
    )
    dependent_hallmark_mean_z = dependent_hallmark_mean_z.set_index("term")

    plt.figure(figsize=(85 * mm, 50 * mm))
    sns.heatmap(
        fast_cor(dependent_hallmark_mean_z.T),
        xticklabels=False,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson's r"},
    )
    plt.subplots_adjust(left=0.38, right=0.95, top=0.95, bottom=0.05)
    plt.xticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(os.path.join(FIGURE_DIR, "hallmark_dependency_z_correlations.png"))


# figure s4d
def plot_mistimed_gene_expression_heatmap(
    targets_to_expand=[
        "PITRM1",
        "NHLRC2",
        "MTPAP",
        "PRKRA",
        "RNF31",
        "SUZ12",
        "SLC25A3",
    ],
    fdr_threshold=0.05,
    n_top_genes=None,
    out_dir=PROCESSED_DIR,
):
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)

    dependencies_to_expand = (
        crispr_table[
            crispr_table["is_dependent"] & crispr_table["guide"].isin(targets_to_expand)
        ]
        .set_index(["cell_line", "guide"])
        .index.tolist()
    )

    significant_genes = (
        sceptre_fdr.loc[:, dependencies_to_expand] < fdr_threshold
    ).apply(lambda x: x.loc[lambda y: y == True].index.tolist())
    gene_universes = sceptre_fdr.apply(lambda x: x.dropna().index.tolist())
    gsea_enrichments = (
        pd.concat(
            {
                x: run_multiple_hypergeometric(
                    query_genes=significant_genes.loc[x],
                    geneset_table=all_genesets[
                        (all_genesets["collection"] == "GO:BP")
                        & (all_genesets["original_set_size"] >= 25)
                    ],
                    all_genes=gene_universes.loc[x],
                    report_genes=True,
                )
                for x in dependencies_to_expand
                if len(significant_genes.loc[x]) >= 2
            }
        )
        .droplevel(2, axis=0)
        .rename_axis(["cell_line", "assigned_ko"])
        .reset_index()
    )
    if out_dir is not None:
        gsea_enrichments.to_csv(
            os.path.join(out_dir, "mistimed_perturbation_enrichment_table.csv"),
            index=False,
        )

    genesets_to_highlight = (
        gsea_enrichments[gsea_enrichments["FDR"] < fdr_threshold]
        .sort_values("FDR")
        .groupby("assigned_ko")
        .head(1)["term"]
        .unique()
        .tolist()
    )
    # manual palette assignment
    geneset_to_color = {
        "GOBP_OXIDATIVE_PHOSPHORYLATION": "tab:blue",
        "GOBP_CELLULAR_RESPONSE_TO_MOLECULE_OF_BACTERIAL_ORIGIN": "tab:olive",
        "GOBP_ATP_SYNTHESIS_COUPLED_ELECTRON_TRANSPORT": "tab:blue",
        "GOBP_REGULATION_OF_PROGRAMMED_CELL_DEATH": "tab:red",
    }
    genesets_to_gene = {
        gs: all_genesets[all_genesets["term"] == gs]["gene"].tolist()
        for gs in genesets_to_highlight
    }
    print("significant genesets: ", genesets_to_highlight)
    print("palette: ", geneset_to_color)

    expr_genes_to_show_df = pd.DataFrame(
        {
            "perturbations_significant": (
                sceptre_fdr.loc[:, dependencies_to_expand] < fdr_threshold
            ).sum(axis=1),
            "significant_abs_z": (
                sceptre_zscore[sceptre_fdr < fdr_threshold].loc[
                    :, dependencies_to_expand
                ]
            )
            .abs()
            .max(axis=1),
        }
    )
    expr_genes_to_show = (
        expr_genes_to_show_df[expr_genes_to_show_df["perturbations_significant"] >= 1]
        .sort_values("significant_abs_z", ascending=False)
        .head(n_top_genes)
        .index.tolist()
    )

    mtx = sceptre_zscore.loc[expr_genes_to_show, dependencies_to_expand].fillna(0)
    cg = sns.clustermap(mtx, cmap="RdBu_r", vmin=-5, center=0, vmax=5, method="ward")
    plt.clf()

    expr_order = cg.data2d.index.tolist()
    ko_to_cls = dict()
    ko_to_heatmap = dict()
    for ko in targets_to_expand:
        ko_to_heatmap[ko] = cg.data2d.loc[:, pd.IndexSlice[:, ko]]
        ko_to_cls[ko] = ko_to_heatmap[ko].droplevel(level=1, axis=1).columns.tolist()

    fig, axs = plt.subplots(
        1,
        len(ko_to_heatmap),
        figsize=(55 * mm, 230 * mm),
        width_ratios=[mtx.shape[1] for mtx in ko_to_heatmap.values()],
        gridspec_kw={"wspace": 0},
    )

    for i, ko in enumerate(targets_to_expand):
        cl_order = ko_to_cls[ko]
        sns.heatmap(
            ko_to_heatmap[ko],
            cmap="RdBu_r",
            vmin=-5,
            center=0,
            vmax=5,
            xticklabels=True,
            yticklabels=False,
            ax=axs[i],
            cbar=False,
        )
        axs[i].set_ylabel("")
        axs[i].set_xlabel("")
        axs[i].set_xticks(
            np.arange(len(cl_order)) + 0.5, cl_order, rotation=90, fontsize=ANNOT_SIZE
        )
        twiny = axs[i].twiny()
        twiny.set_xticks([0.5], [ko], rotation=90, fontsize=ANNOT_SIZE)

    axs[-1].yaxis.set_ticks_position("right")
    axs[-1].set_yticks(
        np.arange(len(expr_order)) + 0.5, expr_order, fontsize=ANNOT_SIZE
    )
    axs[0].set_ylabel("Response gene")
    plt.text(
        0.45,
        0.01,
        "Cell line",
        fontsize=LABEL_SIZE,
        ha="center",
        transform=fig.transFigure,
    )
    plt.text(
        0.45,
        0.98,
        "Knockout",
        fontsize=LABEL_SIZE,
        ha="center",
        transform=fig.transFigure,
    )

    new_yticklabels = []
    for tl in axs[-1].get_yticklabels():
        txt = tl.get_text()
        colors = []
        for gsname, gs in genesets_to_gene.items():
            if txt in gs:
                colors.append(geneset_to_color[gsname])
                if gsname == "GOBP_ATP_SYNTHESIS_COUPLED_ELECTRON_TRANSPORT":
                    txt = txt + "*"
        tl.set_color(blend_colors(colors))
        new_yticklabels.append(txt)
    axs[-1].set_yticklabels(new_yticklabels)

    plt.subplots_adjust(left=0.08, right=0.8, top=0.935, bottom=0.09)
    plt.savefig(os.path.join(FIGURE_DIR, "mistimed_significant_gene_heatmap.png"))

    fig, axs = plt.subplots(1, 1, figsize=(10 * mm, 55 * mm))
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-5, vmax=5)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=axs,
        orientation="vertical",
        label="Z-score expression after knockout",
    )
    plt.subplots_adjust(left=0.05, right=0.25, top=0.95, bottom=0.07)
    plt.savefig(
        os.path.join(FIGURE_DIR, "mistimed_significant_gene_heatmap_colorbar.png")
    )


def main():
    # figure s4a
    plot_top_dependency_diff_expr_correlations()

    # figure s4b
    plot_common_diff_expr_gene_enrichment(
        n_top_genes=50, min_targets=5, fdr_threshold=0.05
    )

    # figure s4c
    plot_dependent_hallmark_geneset_correlation()

    # figure s4d & legend
    plot_mistimed_gene_expression_heatmap(
        targets_to_expand=[
            "PITRM1",
            "NHLRC2",
            "MTPAP",
            "PRKRA",
            "RNF31",
            "SUZ12",
            "SLC25A3",
        ],
        fdr_threshold=0.05,
        n_top_genes=None,
        out_dir=PROCESSED_DIR,
    )


if __name__ == "__main__":
    main()
