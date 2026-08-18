import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from constants import *
from data_utils import *
from figure_utils import *
from tqdm import tqdm

mpl.style.use("assets/stylesheet.mplstyle")


def plot_cellwise_qc_step_summary():
    all_cells_table = load_file("all_cells_table.csv", local_dir=PROCESSED_DIR)

    plt.figure(figsize=(110 * mm, 60 * mm))

    qc_summary_df = pd.DataFrame(
        {
            "Initial": all_cells_table.value_counts("cell_line"),
            "gRNA assignment": all_cells_table[
                (all_cells_table["grna_max_umi"] >= 5)
                & (
                    all_cells_table["grna_max_umi"] / all_cells_table["grna_n_umis"]
                    >= 0.8
                )
            ].value_counts("cell_line"),
            "UMIs": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["grna_max_umi"] >= 5)
                        & ((x["grna_max_umi"] / x["grna_n_umis"]) >= 0.8)
                        & (x["response_n_umis"] >= x["response_n_umis"].quantile(0.01))
                        & (x["response_n_umis"] <= x["response_n_umis"].quantile(0.99))
                    ]
                )
            ),
            "Genes": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["grna_max_umi"] >= 5)
                        & ((x["grna_max_umi"] / x["grna_n_umis"]) >= 0.8)
                        & (x["response_n_umis"] >= x["response_n_umis"].quantile(0.01))
                        & (x["response_n_umis"] <= x["response_n_umis"].quantile(0.99))
                        & (
                            x["response_n_nonzero"]
                            >= x["response_n_nonzero"].quantile(0.01)
                        )
                        & (
                            x["response_n_nonzero"]
                            <= x["response_n_nonzero"].quantile(0.99)
                        )
                    ]
                )
            ),
            "Mitochondrial fraction": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["grna_max_umi"] >= 5)
                        & ((x["grna_max_umi"] / x["grna_n_umis"]) >= 0.8)
                        & (x["response_n_umis"] >= x["response_n_umis"].quantile(0.01))
                        & (x["response_n_umis"] <= x["response_n_umis"].quantile(0.99))
                        & (
                            x["response_n_nonzero"]
                            >= x["response_n_nonzero"].quantile(0.01)
                        )
                        & (
                            x["response_n_nonzero"]
                            <= x["response_n_nonzero"].quantile(0.99)
                        )
                        & (x["response_p_mito"] <= 0.25)
                    ]
                )
            ),
            "Cas9 aberration": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["grna_max_umi"] >= 5)
                        & ((x["grna_max_umi"] / x["grna_n_umis"]) >= 0.8)
                        & (x["response_n_umis"] >= x["response_n_umis"].quantile(0.01))
                        & (x["response_n_umis"] <= x["response_n_umis"].quantile(0.99))
                        & (
                            x["response_n_nonzero"]
                            >= x["response_n_nonzero"].quantile(0.01)
                        )
                        & (
                            x["response_n_nonzero"]
                            <= x["response_n_nonzero"].quantile(0.99)
                        )
                        & (x["response_p_mito"] <= 0.25)
                        & (~x["arm_trunc"] & ~x["arm_gain"])
                    ]
                )
            ),
        }
    )

    print("Cas9 aberration frequency")
    print(
        1
        - (
            qc_summary_df["Cas9 aberration"] / qc_summary_df["Mitochondrial fraction"]
        ).sort_values()
    )

    sns.barplot(
        qc_summary_df.melt(
            ignore_index=False, var_name="QC step", value_name="n_cells"
        ).reset_index(),
        x="cell_line",
        y="n_cells",
        order=qc_summary_df.sort_values(
            "Cas9 aberration", ascending=False
        ).index.tolist(),
        hue="QC step",
        hue_order=[
            "Initial",
            "gRNA assignment",
            "UMIs",
            "Genes",
            "Mitochondrial fraction",
            "Cas9 aberration",
        ],
        palette="cividis_r",
    )
    plt.ylabel("Total cells remaining")
    plt.xlabel("Cell line")
    plt.xticks(rotation=90)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.25)
    plt.savefig(os.path.join(FIGURE_DIR, "qc_cell_triage.png"))


def plot_and_print_arm_loss_frequency_vs_tp53():
    arm_alteration_covariates = load_file(
        "arm_alteration_covariates.csv", local_dir=PROCESSED_DIR
    )
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    tp53_damaging = (
        (
            load_file(
                "OmicsSomaticMutationsMatrixDamaging",
                local_dir=DOWNLOADED_DIR,
                index_col=0,
            ).reindex(index=cl_metadata["arxspan_id"].tolist())["TP53 (7157)"]
            > 0
        )
        .rename("TP53 status")
        .replace({True: "Mutant", False: "WT"})
    )
    arm_alteration_covariates = arm_alteration_covariates.merge(
        cl_metadata.merge(tp53_damaging, left_on="arxspan_id", right_index=True)[
            ["cell_line", "TP53 status"]
        ]
    )

    print("Median truncation frequency by TP53 status:")
    print(
        arm_alteration_covariates.groupby(["cell_line", "TP53 status"])[
            "Arm truncation frequency"
        ]
        .median()
        .groupby(level=1)
        .median()
    )

    mwu_result = scipy.stats.mannwhitneyu(
        arm_alteration_covariates[(arm_alteration_covariates["TP53 status"] == "WT")]
        .groupby("cell_line")["Arm truncation frequency"]
        .median()
        .tolist(),
        arm_alteration_covariates[
            (arm_alteration_covariates["TP53 status"] == "Mutant")
        ]
        .groupby("cell_line")["Arm truncation frequency"]
        .median()
        .tolist(),
    )
    print("Mann-Whitney result on medians: TP53 mut. vs. WT")
    print(mwu_result)

    fig = plt.figure(figsize=(50 * mm, 60 * mm))
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.15)
    sns.boxplot(
        arm_alteration_covariates,
        x="Arm truncation frequency",
        y="cell_line",
        order=arm_alteration_covariates.groupby("cell_line")["Arm truncation frequency"]
        .median()
        .sort_values()
        .index.tolist(),
        hue="TP53 status",
        showfliers=False,
    )
    plt.xlabel("Fraction of cells with arm loss")
    plt.ylabel("Cell line")
    plt.savefig(os.path.join(FIGURE_DIR, "arm_loss_frequency_over_cell_lines.png"))


def plot_arm_loss_vector_consistency(min_cells=50):
    arm_alteration_covariates = load_file(
        "arm_alteration_covariates.csv", local_dir=PROCESSED_DIR
    )
    vector_comparison_table = arm_alteration_covariates[
        arm_alteration_covariates["Total cells per KO"] >= min_cells
    ]

    plt.figure(figsize=(60 * mm, 60 * mm))

    xlabel = "v1_arm_trunc_freq"
    ylabel = "v2_arm_trunc_freq"
    sns.scatterplot(vector_comparison_table, x=xlabel, y=ylabel, s=8)
    sns.regplot(
        vector_comparison_table,
        x=xlabel,
        y=ylabel,
        scatter=False,
        line_kws={"color": "black", "alpha": 0.5},
    )
    plt.axhline(0, color="black", linestyle="solid", zorder=0)
    plt.axvline(0, color="black", linestyle="solid", zorder=0)

    corr_result = scipy.stats.pearsonr(
        vector_comparison_table[xlabel], vector_comparison_table[ylabel]
    )
    plt.text(
        0.1, 0.9, f"r = {corr_result.statistic:.02f}", transform=plt.gca().transAxes
    )

    print(corr_result)

    plt.xlabel("Vector 1 arm truncation frequency")
    plt.ylabel("Vector 2 arm truncation frequency")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "vector_arm_loss_agreement.png"))


def plot_and_print_top_arm_loss_predictor_scatterplot():
    arm_alteration_covariates = load_file(
        "arm_alteration_covariates.csv", local_dir=PROCESSED_DIR
    )
    arm_alteration_correlations = load_file(
        "arm_alteration_correlations.csv", local_dir=PROCESSED_DIR
    )
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    cl_to_lineage = cl_metadata.set_index("cell_line")["OncotreeLineage"].to_dict()

    top_cell_line, top_predictor = (
        arm_alteration_correlations[
            arm_alteration_correlations["Outcome"] == "Arm truncation frequency"
        ]
        .sort_values("Correlation with outcome")
        .iloc[0, :]
        .loc[["Cell line", "Predictor"]]
    )

    cl_subset = arm_alteration_covariates[
        arm_alteration_covariates["cell_line"] == top_cell_line
    ]
    cl_color = lineage_palette[cl_to_lineage[top_cell_line]]

    plt.figure(figsize=(60 * mm, 60 * mm))

    sns.scatterplot(
        cl_subset,
        x=top_predictor,
        y="Arm truncation frequency",
        color=cl_color,
        edgecolor="black",
        label=top_cell_line,
        zorder=0,
    )
    sns.scatterplot(
        arm_alteration_covariates,
        x=top_predictor,
        y="Arm truncation frequency",
        color="tab:gray",
        alpha=0.5,
        label="Other",
        zorder=-1,
    )

    sns.regplot(
        arm_alteration_covariates,
        x=top_predictor,
        y="Arm truncation frequency",
        color="tab:gray",
        line_kws={"color": "black"},
        scatter=False,
    )
    sns.regplot(
        cl_subset,
        x=top_predictor,
        y="Arm truncation frequency",
        color=cl_color,
        scatter=False,
    )
    plt.ylim(-0.05, 0.55)
    plt.ylabel("Arm truncation frequency")
    plt.xlabel(top_predictor)
    plt.legend(fontsize=ANNOT_SIZE)

    subset_corr_res = scipy.stats.pearsonr(
        cl_subset[top_predictor], cl_subset["Arm truncation frequency"]
    )
    full_corr_res = scipy.stats.pearsonr(
        arm_alteration_covariates[top_predictor],
        arm_alteration_covariates["Arm truncation frequency"],
    )
    print(f"{top_cell_line} correlation:", subset_corr_res)
    print("all correlation:", full_corr_res)

    plt.text(
        0.03,
        0.95,
        f"r = {subset_corr_res.statistic:.02f}",
        transform=plt.gca().transAxes,
        fontdict={"fontsize": ANNOT_SIZE, "color": cl_color},
        ha="left",
    )

    plt.text(
        0.03,
        0.9,
        f"r = {full_corr_res.statistic:.02f}",
        transform=plt.gca().transAxes,
        fontdict={"fontsize": ANNOT_SIZE},
        ha="left",
    )
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig(
        os.path.join(FIGURE_DIR, "arm_dependency_truncation_best_prediction.png")
    )


def plot_arm_gain_predictors():
    arm_alteration_summary_table = load_file(
        "arm_alteration_covariates.csv", local_dir=PROCESSED_DIR
    )
    arm_alteration_correlations = load_file(
        "arm_alteration_correlations.csv", local_dir=PROCESSED_DIR
    )

    print(
        "arm gain median frequency:",
        arm_alteration_summary_table["Arm gain frequency"].median(),
    )
    print(
        "arm loss median frequency:",
        arm_alteration_summary_table["Arm truncation frequency"].median(),
    )
    top_cell_line, top_predictor = (
        arm_alteration_correlations[
            arm_alteration_correlations["Outcome"] == "Arm gain frequency"
        ]
        .sort_values("Correlation with outcome")
        .iloc[0, :]
        .loc[["Cell line", "Predictor"]]
    )
    subset = arm_alteration_summary_table[
        [top_predictor, "Arm gain frequency"]
    ].dropna()
    full_corr_res = scipy.stats.pearsonr(
        subset[top_predictor], subset["Arm gain frequency"]
    )
    print(f"{top_predictor} correlation:", full_corr_res)

    fig, axs = plt.subplots(1, 2, figsize=(70 * mm, 60 * mm), width_ratios=[39, 1])
    plt.subplots_adjust(left=0.45, right=0.85, top=0.95, bottom=0.15)

    arm_alteration_subset = arm_alteration_correlations[
        arm_alteration_correlations["Outcome"] == "Arm gain frequency"
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

    axs[0].set_xlabel("Correlation with arm gain frequency (Pearson's r)")
    axs[0].set_ylabel("Feature")

    plt.savefig(os.path.join(FIGURE_DIR, "gene_wise_arm_gain_predictors.png"))


def plot_arm_gain_vs_arm_loss_degs():

    def generate_arm_alteration_differential_genes(fdr_threshold=0.05):
        cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
            "cell_line"
        ].tolist()
        cl_to_sceptre_test = dict()
        for cl in tqdm(cell_lines):
            cl_to_sceptre_test[cl] = load_file(
                "discovery_analysis.csv",
                local_dir=os.path.join(PROCESSED_DIR, "sceptre", "arm_trunc", cl),
            )

        sceptre_test_pass_qc = pd.concat(
            {
                cl: cl_to_sceptre_test[cl].pivot(
                    index="response_id", columns="grna_target", values="pass_qc"
                )
                for cl in cell_lines
            },
            axis=1,
        )
        sceptre_test_pval = pd.concat(
            {
                cl: cl_to_sceptre_test[cl].pivot(
                    index="response_id", columns="grna_target", values="p_value"
                )
                for cl in cell_lines
            },
            axis=1,
        ).loc[sceptre_test_pass_qc.any(axis=1), :]
        sceptre_test_zscore = pd.concat(
            {
                cl: cl_to_sceptre_test[cl].pivot(
                    index="response_id", columns="grna_target", values="z_orig"
                )
                for cl in cell_lines
            },
            axis=1,
        ).loc[sceptre_test_pass_qc.any(axis=1), :]

        arm_zscore = sceptre_test_zscore.copy()
        arm_fdr = recalibrate_pvalue_matrix_columnwise(sceptre_test_pval).reindex_like(
            arm_zscore
        )

        gene_locs = load_file("gene_locations.csv", local_dir=DATA_DIR)
        target_arms = gene_locs[
            gene_locs["Gene name"].isin(
                load_file("knockout_metadata.csv", local_dir=METADATA_DIR)[
                    "knockout"
                ].tolist()
            )
        ].set_index("Gene name")["ChrArm"]
        control_arms = target_arms.loc[target_arms.index.str.startswith("OR")]

        arm_z_mean_change = arm_zscore.T.groupby(level=1).mean().T
        n_cl_response = (
            (~arm_zscore.isna())
            .T.groupby(level=1)
            .sum()
            .T.rename(
                {
                    "GAIN": "n_cell_lines_gain_detected",
                    "TRUNC": "n_cell_lines_trunc_detected",
                },
                axis=1,
            )
            .reset_index()
        )
        n_cl_sig_response = (
            (arm_fdr < 0.05)
            .T.groupby(level=1)
            .sum()
            .T.rename(
                {
                    "GAIN": "n_cell_lines_gain_significant",
                    "TRUNC": "n_cell_lines_trunc_significant",
                },
                axis=1,
            )
            .reset_index()
        )
        arm_z_mean_change = (
            arm_z_mean_change.rename_axis("response_id")
            .reset_index()
            .merge(n_cl_response)
            .merge(n_cl_sig_response)
            .merge(
                gene_locs[["Gene name", "ChrArm"]],
                left_on=["response_id"],
                right_on=["Gene name"],
            )
        )
        arm_z_mean_change["control_arm"] = arm_z_mean_change["ChrArm"].isin(
            control_arms.tolist()
        )

        return arm_zscore, arm_fdr, arm_z_mean_change

    sceptre_zscore, sceptre_fdr, arm_z_mean_change = (
        generate_arm_alteration_differential_genes()
    )

    df_to_plot = arm_z_mean_change[
        (~arm_z_mean_change["control_arm"])
        & (
            (arm_z_mean_change["n_cell_lines_gain_significant"] >= 1)
            | (arm_z_mean_change["n_cell_lines_trunc_significant"] >= 1)
        )
        & (
            (arm_z_mean_change["n_cell_lines_gain_detected"] >= 10)
            | (arm_z_mean_change["n_cell_lines_trunc_detected"] >= 10)
        )
    ]
    xlabel = "TRUNC"
    ylabel = "GAIN"

    plt.figure(figsize=(50 * mm, 60 * mm))
    sns.scatterplot(df_to_plot, x=xlabel, y=ylabel, s=8)
    plt.axhline(0, linestyle="solid", color="black", zorder=0)
    plt.axvline(0, linestyle="solid", color="black", zorder=0)
    plt.xlabel("Mean z-score in control KOs with arm truncation")
    plt.ylabel("Mean z-score in control KOs with arm gain")

    most_diff = (
        df_to_plot.assign(diff=lambda x: (x[xlabel] - x[ylabel]).abs())
        .sort_values("diff", ascending=False)
        .head(1)
    )
    top_x = df_to_plot.sort_values(xlabel, ascending=False).head(2)
    bottom_x = df_to_plot.sort_values(xlabel, ascending=True).head(2)
    top_y = df_to_plot.sort_values(ylabel, ascending=False).head(2)
    bottom_y = df_to_plot.sort_values(ylabel, ascending=True).head(2)
    df_to_annot = pd.concat(
        [most_diff, top_x, bottom_x, top_y, bottom_y], axis=0, ignore_index=True
    ).drop_duplicates()

    texts = [
        plt.text(
            r[xlabel],
            r[ylabel],
            r["response_id"],
            ha="center",
            va="center",
            fontsize=ANNOT_SIZE,
            bbox={
                "facecolor": "white",
                "edgecolor": "white",
                "boxstyle": "round,pad=0.001",
            },
        )
        for i, r in df_to_annot.iterrows()
    ]
    adjust_text(texts, expand=(1.4, 1.5))

    plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.15)
    plt.savefig(os.path.join(FIGURE_DIR, "arm_gain_vs_loss_degs_in_controls.png"))


def plot_cell_calling_heatmap():
    # import cell line metadata
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    cl_to_arxspan_id = cl_metadata.set_index("cell_line")["arxspan_id"].to_dict()
    arxspan_id_to_cl = cl_metadata.set_index("arxspan_id")["cell_line"].to_dict()
    cell_lines = cl_metadata.cell_line.to_list()

    all_cells_table = load_file("all_cells_table.csv", local_dir=PROCESSED_DIR)

    cell_calling_df = pd.DataFrame(
        {
            "Passing\nsinglet": all_cells_table[
                all_cells_table["pass_qc"] & ~all_cells_table["arm_trunc"]
            ].value_counts("cell_line"),
            "Singlet with\narm loss": all_cells_table[
                all_cells_table["pass_qc"] & all_cells_table["arm_trunc"]
            ].value_counts("cell_line"),
            "Multiple\ninfection": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["grna_max_umi"] >= 5)
                        & ((x["grna_max_umi"] / x["grna_n_umis"]) < 0.8)
                        & (x["response_n_umis"] >= x["response_n_umis"].quantile(0.01))
                        & (x["response_n_umis"] <= x["response_n_umis"].quantile(0.99))
                        & (
                            x["response_n_nonzero"]
                            >= x["response_n_nonzero"].quantile(0.01)
                        )
                        & (
                            x["response_n_nonzero"]
                            <= x["response_n_nonzero"].quantile(0.99)
                        )
                        & (x["response_p_mito"] <= 0.25)
                    ]
                )
            ),
            "Ambiguous\ninfection": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["grna_max_umi"] > 1)
                        & (x["grna_max_umi"] < 5)
                        & (x["response_n_umis"] >= x["response_n_umis"].quantile(0.01))
                        & (x["response_n_umis"] <= x["response_n_umis"].quantile(0.99))
                        & (
                            x["response_n_nonzero"]
                            >= x["response_n_nonzero"].quantile(0.01)
                        )
                        & (
                            x["response_n_nonzero"]
                            <= x["response_n_nonzero"].quantile(0.99)
                        )
                        & (x["response_p_mito"] <= 0.25)
                    ]
                )
            ),
            "Unperturbed": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["grna_max_umi"] <= 1)
                        & (x["response_n_umis"] >= x["response_n_umis"].quantile(0.01))
                        & (x["response_n_umis"] <= x["response_n_umis"].quantile(0.99))
                        & (
                            x["response_n_nonzero"]
                            >= x["response_n_nonzero"].quantile(0.01)
                        )
                        & (
                            x["response_n_nonzero"]
                            <= x["response_n_nonzero"].quantile(0.99)
                        )
                        & (x["response_p_mito"] <= 0.25)
                    ]
                )
            ),
            "Other\nQC fail": all_cells_table.groupby("cell_line").apply(
                lambda x: len(
                    x[
                        (x["response_n_umis"] < x["response_n_umis"].quantile(0.01))
                        | (x["response_n_umis"] > x["response_n_umis"].quantile(0.99))
                        | (
                            x["response_n_nonzero"]
                            < x["response_n_nonzero"].quantile(0.01)
                        )
                        | (
                            x["response_n_nonzero"]
                            > x["response_n_nonzero"].quantile(0.99)
                        )
                        | (x["response_p_mito"] > 0.25)
                    ]
                )
            ),
        }
    )

    plt.figure(figsize=(70 * mm, 60 * mm))
    sns.heatmap(
        (cell_calling_df.T / all_cells_table.value_counts("cell_line")).T,
        annot=True,
        fmt=".02f",
        annot_kws={"fontsize": ANNOT_SIZE},
        cbar_kws={"label": "Fraction of total cells"},
    )
    plt.ylabel("Cell line")
    plt.xlabel("")
    for tl in plt.gca().get_xticklabels():
        if "Passing\nsinglet" in tl.get_text():
            tl.set_color("tab:orange")
        elif "Singlet with\narm loss" in tl.get_text():
            tl.set_color("tab:blue")
        elif "Unperturbed" in tl.get_text():
            tl.set_color("tab:green")
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.9)
    plt.savefig(os.path.join(FIGURE_DIR, "cell_calling.png"))


def deep_ctrl_lfc_expr():
    deep_ctrl_sig = load_file("deep_ctrl_results_sig.csv", local_dir=PROCESSED_DIR)

    cell_lines = ["KMRC20", "UMRC3"]
    controls = ["OR13C7", "OR9Q1", "OR13H1", "OR10J3"]
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))[: len(controls)]
    ctrl_palette = {item: tab10_colors[i] for i, item in enumerate(controls)}

    fig, axs = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(50 * mm, 60 * mm))

    for i in range(len(cell_lines)):
        cl = cell_lines[i]
        ax = axs.flat[i]

        sig_ctrl_results = deep_ctrl_sig.query(f'cell_line == "{cl}"')

        # label higher exp, lfc
        outliers = sig_ctrl_results.query(
            "log2_mean_expression > 1.25 & abs_log_2_fold_change > .2"
        ).index
        texts = []
        for outlier in outliers:
            texts.append(
                ax.text(
                    sig_ctrl_results.loc[outlier, "log2_mean_expression"],
                    sig_ctrl_results.loc[outlier, "abs_log_2_fold_change"],
                    sig_ctrl_results.loc[outlier, "response_id"],
                    ha="center",
                    va="center",
                    fontsize=ANNOT_SIZE,
                )
            )

        # plot
        sns.scatterplot(
            sig_ctrl_results,
            x="log2_mean_expression",
            y="abs_log_2_fold_change",
            hue="grna_target",
            ax=ax,
            s=4,
            linewidth=0,
            palette=ctrl_palette,
        )
        adjust_text(
            texts,
            x=sig_ctrl_results["log2_mean_expression"],
            y=sig_ctrl_results["abs_log_2_fold_change"],
            ax=ax,
            arrowprops=annot_arrow_props,
        )
        ax.axhline(1, ls="--", color="gray", zorder=0)
        ax.set_xlabel("$\\log_{2}$(mean expression + 1)")
        ax.set_ylabel("")
        ax.set_title(cl)
        ax.get_legend().remove()

    lines, labels = ax.get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        title="Knockout",
        loc="center left",
        bbox_to_anchor=(0.6, 0.75),
        framealpha=1,
    )
    fig.supylabel(
        "Absolute LFC\nof significant response genes",
        fontsize=axs[0].yaxis.get_label().get_fontsize(),
        ha="center",
        va="center",
        x=0.1,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "rescreen_lfc_expr.png"))


def main():
    # figure 2a
    plot_arm_loss_vector_consistency()

    # figure 2b
    plot_and_print_top_arm_loss_predictor_scatterplot()

    # figure 2c
    plot_and_print_arm_loss_frequency_vs_tp53()

    # figure 2d
    plot_arm_gain_predictors()

    # figure 2e
    plot_arm_gain_vs_arm_loss_degs()

    # figure 2f
    deep_ctrl_lfc_expr()

    # figure 2g
    plot_cellwise_qc_step_summary()

    # figure 2h
    plot_cell_calling_heatmap()


if __name__ == "__main__":
    main()
