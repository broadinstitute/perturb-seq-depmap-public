import os

import numpy as np
import pandas as pd
import scipy.sparse
import tqdm as tqdm
from constants import *
from data_utils import *
from figure_utils import *
from gene_utils import *
from scipy.stats import false_discovery_control

# preprocessing


def calculate_gene_properties(df):
    gene_locs = df.copy()

    chromosome_extreme_gene_bounds = (
        gene_locs.groupby("ChrArm")
        .agg({"Gene end (bp)": "max", "Gene start (bp)": "min"})
        .reset_index()
    )
    centromere_estimates = chromosome_extreme_gene_bounds.assign(
        last_gene_position=lambda x: x.apply(
            lambda y: (
                y["Gene end (bp)"] if "p" in y["ChrArm"] else y["Gene start (bp)"]
            ),
            axis=1,
        )
    ).set_index("ChrArm")["last_gene_position"]
    telomere_estimates = chromosome_extreme_gene_bounds.assign(
        last_gene_position=lambda x: x.apply(
            lambda y: (
                y["Gene start (bp)"] if "p" in y["ChrArm"] else y["Gene end (bp)"]
            ),
            axis=1,
        )
    ).set_index("ChrArm")["last_gene_position"]

    gene_locs["Centromere estimate"] = gene_locs["ChrArm"].map(centromere_estimates)
    gene_locs["Telomere estimate"] = gene_locs["ChrArm"].map(telomere_estimates)
    gene_locs = gene_locs.assign(
        **{
            "Midgene estimate": lambda x: (
                (x["Gene start (bp)"] + x["Gene end (bp)"]) / 2
            ),
            "Distance to centromere": lambda x: (
                x["Midgene estimate"] - x["Centromere estimate"]
            ).abs(),
            "Distance to telomere": lambda x: (
                x["Midgene estimate"] - x["Telomere estimate"]
            ).abs(),
        }
    )

    return gene_locs


def get_gene_loc_info(out_dir=DATA_DIR):
    gene_locs = (
        load_file("gene_locations_biomart", local_dir=METADATA_DIR)
        .sort_values("Gene start (bp)")
        .loc[lambda x: x["Gene name"].notnull() & x["Karyotype band"].notnull()]
        .loc[lambda x: ~x["Gene name"].duplicated()]
    )
    gene_locs["ChrArm"] = (
        gene_locs["Chromosome/scaffold name"] + gene_locs["Karyotype band"].str[0]
    )

    gene_locs["Arm"] = gene_locs["Karyotype band"].str[0]

    gene_locs = calculate_gene_properties(gene_locs)

    if out_dir is not None:
        gene_locs.to_csv(os.path.join(out_dir, "gene_locations.csv"), index=False)

    return gene_locs.reset_index(drop=True)


def generate_crispr_table(out_dir=PROCESSED_DIR):
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    achilles_scores = load_file(
        "CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0
    )

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

    crispr_df["dependency_strength_class"] = discretize_series_by_interval(
        crispr_df["gene_effect"], interval_dict=dependency_intervals
    )
    crispr_df["is_dependent"] = crispr_df["dependency_strength_class"] == "Strong"

    if out_dir is not None:
        crispr_df.to_csv(os.path.join(out_dir, "crispr_table.csv"), index=False)

    return crispr_df


def create_geneset_collection(gene_universe=None, out_dir=DATA_DIR):
    from gseapy.msigdb import Msigdb

    msig = Msigdb()

    # load various gene set databases
    hallmark = msig.get_gmt(category="h.all", dbver="2024.1.Hs")
    cgp = msig.get_gmt(category="c2.cgp", dbver="2024.1.Hs")
    kegg = msig.get_gmt(category="c2.cp.kegg_legacy", dbver="2024.1.Hs")
    reactome = msig.get_gmt(category="c2.cp.reactome", dbver="2024.1.Hs")
    gobp = msig.get_gmt(category="c5.go.bp", dbver="2024.1.Hs")
    hgnc_table = load_file("gene_table", local_dir=DOWNLOADED_DIR)
    hgnc_annotations = (
        hgnc_table.loc[:, ["symbol", "gene_group"]]
        .dropna(subset=["symbol"])
        .reset_index(drop=True)
        .rename({"symbol": "gene", "gene_group": "term"}, axis=1)
    )
    # corum = load_file('corum_humanComplexes.txt', local_dir=METADATA_DIR, delimiter='\t').loc[:, ['complex_name', 'synonyms', 'comment_complex', 'subunits_gene_name', 'subunits_gene_name_synonyms']]

    # produce a subset of that table with valid symbols
    all_genesets = pd.concat(
        [
            pd.Series(hallmark)
            .rename("gene")
            .rename_axis("term")
            .explode()
            .reset_index()
            .assign(collection="Hallmark"),
            pd.Series(cgp)
            .rename("gene")
            .rename_axis("term")
            .explode()
            .reset_index()
            .assign(collection="CGP"),
            pd.Series(kegg)
            .rename("gene")
            .rename_axis("term")
            .explode()
            .reset_index()
            .assign(collection="KEGG"),
            pd.Series(reactome)
            .rename("gene")
            .rename_axis("term")
            .explode()
            .reset_index()
            .assign(collection="Reactome"),
            pd.Series(gobp)
            .rename("gene")
            .rename_axis("term")
            .explode()
            .reset_index()
            .assign(collection="GO:BP"),
            hgnc_annotations.set_index("gene")["term"]
            .str.split("|")
            .explode()
            .reset_index()
            .sort_values("term")
            .dropna(subset=["term"])
            .assign(collection="HGNC"),
            # corum.set_index('complex_name')['subunits_gene_name'].str.split(';').explode().rename_axis('term').rename('gene').reset_index().assign(collection='CORUM')
        ],
        axis=0,
    )

    all_genesets = all_genesets.merge(
        all_genesets.value_counts(["term", "collection"])
        .rename("original_set_size")
        .reset_index()
    )

    if gene_universe is not None:
        all_genesets["present_in_data"] = all_genesets["gene"].isin(gene_universe)
        all_genesets = all_genesets.merge(
            all_genesets[all_genesets["present_in_data"]]
            .value_counts(["term", "collection"])
            .rename("effective_set_size")
            .reset_index()
        )

    if out_dir is not None:
        all_genesets.to_csv(os.path.join(out_dir, "all_genesets.csv"), index=False)

    return all_genesets


all_genesets = create_geneset_collection(out_dir=DATA_DIR)

# reshape sceptre results


def consolidate_cells_table(out_dir=None, max_grnas=1, max_mito_frac=0.25):
    cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
        "cell_line"
    ].tolist()
    cl_to_sceptre_cells = dict()
    for cl in tqdm(cell_lines):
        # cl_to_sceptre_cells[cl] = sc.read_h5ad(os.path.join(PROCESSED_DIR, 'scanpy', 'raw', cl + '.h5ad')).obs
        cell_line_subset = load_file(
            "cells.csv", local_dir=os.path.join(PROCESSED_DIR, "sceptre", cl)
        ).set_index("barcode")
        assigned_passing_cells = cell_line_subset[(cell_line_subset["pass_qc"])].copy()
        unperturbed_mask = (
            (cell_line_subset["grna_n_umis"] <= max_grnas)
            & (
                cell_line_subset["response_n_umis"]
                >= assigned_passing_cells["response_n_umis"].min()
            )
            & (
                cell_line_subset["response_n_umis"]
                <= assigned_passing_cells["response_n_umis"].max()
            )
            & (
                cell_line_subset["response_n_nonzero"]
                >= assigned_passing_cells["response_n_nonzero"].min()
            )
            & (
                cell_line_subset["response_n_nonzero"]
                <= assigned_passing_cells["response_n_nonzero"].max()
            )
            & (cell_line_subset["response_p_mito"] < max_mito_frac)
        )
        cell_line_subset.loc[unperturbed_mask, ["assigned_grna", "assigned_ko"]] = (
            "unperturbed"
        )
        cell_line_subset.loc[unperturbed_mask, "ChrArm"] = "N/A"
        cell_line_subset.loc[
            unperturbed_mask,
            ["distal_frac", "n_distal_genes", "non_distal_frac", "n_non_distal_genes"],
        ] = np.nan
        cell_line_subset.loc[unperturbed_mask, ["control_guide", "arm_trunc"]] = False
        cl_to_sceptre_cells[cl] = cell_line_subset

    all_cells_table = (
        pd.concat(cl_to_sceptre_cells, axis=0)
        .rename_axis(["cell_line", "barcode"], axis=0)
        .reset_index()
    )
    if out_dir is not None:
        all_cells_table.to_csv(
            os.path.join(out_dir, "all_cells_table.csv"), index=False
        )

    return all_cells_table


def generate_pseudobulk_counts(out_dir=PROCESSED_DIR):
    cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
        "cell_line"
    ].tolist()
    grna_to_ko_mapping = []
    cl_to_pseudobulk = dict()
    for cl in tqdm(cell_lines):
        raw_data = load_raw_10x(cl)
        cells_table = raw_data.obs.copy()
        grna_to_ko_mapping.append(
            cells_table.drop_duplicates(subset=["assigned_grna", "assigned_ko"])[
                ["assigned_grna", "assigned_ko"]
            ]
        )
        pseudobulk = anndata_groupby(
            raw_data[
                raw_data.obs["pass_qc"]
                & ~(raw_data.obs["arm_trunc"] | raw_data.obs["arm_gain"]),
                :,
            ],
            axis="obs",
            group_column="assigned_grna",
            agg_func="sum",
        )
        pseudobulk = pd.DataFrame(
            pseudobulk[:, pseudobulk.X.sum(axis=0) > 0].X.todense(),
            index=pseudobulk.obs.index,
            columns=pseudobulk[:, pseudobulk.X.sum(axis=0) > 0].var.index,
        )
        cl_to_pseudobulk[cl] = pseudobulk

    grna_to_ko = (
        pd.concat(grna_to_ko_mapping, axis=0, ignore_index=True)
        .drop_duplicates(subset=["assigned_grna", "assigned_ko"])
        .set_index("assigned_grna")["assigned_ko"]
    )
    pseudobulk_by_ko = pd.concat(
        {cl: cl_to_pseudobulk[cl].groupby(grna_to_ko).sum().T for cl in cell_lines},
        axis=1,
    )
    pseudobulk_by_grna = pd.concat(
        {cl: cl_to_pseudobulk[cl].T for cl in cell_lines}, axis=1
    )
    if out_dir is not None:
        pseudobulk_by_ko.to_csv(os.path.join(out_dir, "pseudobulk_sum_by_ko.csv"))
        pseudobulk_by_grna.to_csv(os.path.join(out_dir, "pseudobulk_sum_by_grna.csv"))

    return pseudobulk_by_ko, pseudobulk_by_grna


def generate_sceptre_results_matrices(out_dir=PROCESSED_DIR):
    cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
        "cell_line"
    ].tolist()
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    cl_to_sceptre_test = dict()
    cl_to_sceptre_calibration = dict()
    for cl in tqdm(cell_lines):
        # sceptre_discovery_calibration = tc.get(name='perturbseq-pilot-sceptre-results-2b29', file=cl)
        cl_to_sceptre_test[cl] = load_file(
            "discovery_analysis.csv",
            local_dir=os.path.join(PROCESSED_DIR, "sceptre", cl),
        )
        cl_to_sceptre_test[cl]["analysis_type"] = "discovery_analysis"
        # cl_to_sceptre_calibration[cl] = load_file('calibration_check.csv', local_dir=os.path.join(PROCESSED_DIR, 'sceptre', cl))
        # cl_to_sceptre_calibration[cl]['analysis_type'] = 'calibration_check'

    sceptre_test_pass_qc = pd.concat(
        {
            cl: cl_to_sceptre_test[cl].pivot(
                index="response_id", columns="grna_target", values="pass_qc"
            )
            for cl in cell_lines
        },
        axis=1,
    )
    sceptre_test_lfc = pd.concat(
        {
            cl: cl_to_sceptre_test[cl].pivot(
                index="response_id", columns="grna_target", values="log_2_fold_change"
            )
            for cl in cell_lines
        },
        axis=1,
    ).loc[sceptre_test_pass_qc.any(axis=1), :]
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

    sceptre_lfc = sceptre_test_lfc.copy()
    sceptre_pval = sceptre_test_pval.copy()
    sceptre_zscore = sceptre_test_zscore.copy()
    sceptre_fdr = recalibrate_pvalue_matrix_columnwise(sceptre_pval).reindex_like(
        sceptre_lfc
    )
    # sceptre_screenwise_fdr = pd.concat({cl: recalibrate_pvalue_matrixwise(sceptre_pval.loc[:, pd.IndexSlice[cl, :]].droplevel(0, axis=1)) for cl in cell_lines}, axis=1).reindex_like(sceptre_pval)

    if out_dir is not None:
        sceptre_lfc.to_csv(os.path.join(out_dir, "lfc_matrix.csv"))
        sceptre_pval.to_csv(os.path.join(out_dir, "pvalue_matrix.csv"))
        sceptre_zscore.to_csv(os.path.join(out_dir, "zscore_matrix.csv"))
        sceptre_fdr.to_csv(os.path.join(out_dir, "fdr_matrix.csv"))

    return sceptre_lfc, sceptre_pval, sceptre_zscore, sceptre_fdr


def generate_l2fc_rpm_matrix(out_dir=PROCESSED_DIR):
    cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
        "cell_line"
    ].tolist()

    rpm_l2fc = dict()
    for cl in tqdm(cell_lines):
        adata = load_raw_10x(cl, drop_zero=False)
        adata = adata[adata.obs["pass_qc"]].copy()
        count_df = adata.to_df()
        ko_df = count_df.groupby(adata.obs["assigned_ko"]).sum()
        rpm_df = ko_df.div(ko_df.sum(axis=1), axis=0) * 1e6 + 1
        cntrl_df = count_df.groupby(adata.obs["control_guide"]).sum()
        cntrl_rpm = (cntrl_df.div(cntrl_df.sum(axis=1), axis=0) * 1e6 + 1).loc[True]
        l2fc = np.log2(rpm_df) - np.log2(cntrl_rpm)
        l2fc.columns.name = "grna_target"
        l2fc.index.name = "response_id"
        rpm_l2fc[cl] = l2fc.T

    rpm_l2fc_df = pd.concat(rpm_l2fc, axis=1, keys=rpm_l2fc.keys())

    if out_dir is not None:
        rpm_l2fc_df.to_csv(os.path.join(out_dir, "l2fc_rpm_matrix.csv"))


# arm loss analysis


def map_gene_locations(df, key_column="gene_ids", separator=";"):
    gene_locations = load_file("gene_locations.csv", local_dir=DATA_DIR)

    extended_df = df.drop(key_column, axis=1).merge(
        df[key_column].str.split(";").explode(), left_index=True, right_index=True
    )
    extended_df = (
        extended_df.reset_index()
        .merge(
            gene_locations, left_on=key_column, right_on="Gene stable ID", how="left"
        )
        .set_index("index")
    )

    string_join = lambda x: (
        np.nan
        if len(x.dropna()) == 0
        else x.dropna()[0]
        if len(set(x.dropna())) == 1
        else ";".join(sorted(list(set(x.dropna()))))
    )
    collapsed_df = extended_df.groupby(level=0).agg(
        {
            "feature_types": "first",
            "genome": "first",
            "gene_ids": string_join,
            "Gene stable ID": string_join,
            "Gene stable ID version": string_join,
            "Karyotype band": string_join,
            "Chromosome/scaffold name": string_join,
            "Gene start (bp)": "min",
            "Gene end (bp)": "max",
            "ChrArm": string_join,
            "Arm": string_join,
        }
    )
    collapsed_df = calculate_gene_properties(collapsed_df)

    return collapsed_df


def mask_targets(
    matrix, cell_metadata, cell_resolution="assigned_ko", use_mutations=True
):
    target_expression = []
    groups = cell_metadata[cell_resolution].unique().tolist()
    for g in groups:
        if cell_resolution == "assigned_grna":
            target_gene = g.split("_")[0]
        else:
            target_gene = g
        if target_gene in matrix.columns.tolist():
            if use_mutations:
                slicer = pd.IndexSlice[g, :]
            else:
                slicer = [g]
            target_expression.append(
                matrix.loc[slicer, target_gene]
                .rename("mean_z_target_expr")
                .reset_index()
                .assign(target_gene=target_gene)
            )
            matrix.loc[slicer, target_gene] = np.nan
    target_expression = pd.concat(target_expression, axis=0, ignore_index=True)
    return matrix, target_expression


def calculate_arm_level_mean_z(
    zscore_matrix,
    cell_metadata,
    gene_metadata,
    cell_resolution="assigned_ko",
    use_mutations=True,
):
    # aggregate over cells
    cell_groupers = [cell_resolution]
    mean_z_by_group = (
        zscore_matrix.join(cell_metadata[cell_groupers]).groupby(cell_groupers).mean()
    )
    if use_mutations:
        cell_groupers.append("mutation_group")
        mean_z_by_group_with_mutations = (
            zscore_matrix.join(cell_metadata[cell_groupers])
            .groupby(cell_groupers)
            .mean()
        )
        mean_z_by_group = pd.concat(
            [
                mean_z_by_group_with_mutations,
                pd.concat({"all": mean_z_by_group}, axis=0)
                .swaplevel(axis=0)
                .rename_axis(cell_groupers),
            ],
            axis=0,
        )

    mean_z_by_group, target_expression = mask_targets(
        mean_z_by_group,
        cell_metadata,
        cell_resolution=cell_resolution,
        use_mutations=use_mutations,
    )

    # aggregate over genes
    gene_groupers = ["ChrArm"]
    if use_mutations:
        # flatten the multi-index for compatibility
        mean_z_by_group.index = mean_z_by_group.index.map(lambda x: ";".join(x))
    mean_z_by_group_arm = (
        mean_z_by_group.T.join(gene_metadata[gene_groupers])
        .groupby(gene_groupers)
        .mean()
        .T
    )
    if use_mutations:
        # recreate the multi-index
        mean_z_by_group_arm.index = mean_z_by_group_arm.index.str.split(";").map(
            lambda x: tuple(x)
        )
    mean_z_by_group_arm = mean_z_by_group_arm.rename_axis(cell_groupers, axis=0)

    return mean_z_by_group_arm, target_expression


def perform_cell_reassignment(adata_obs, max_grnas=1, max_mito_frac=0.25):
    assigned_passing_cells = adata_obs[(adata_obs["pass_qc"])].copy()
    unperturbed_mask = (
        (adata_obs["grna_n_umis"] <= max_grnas)
        & (
            adata_obs["response_n_umis"]
            >= assigned_passing_cells["response_n_umis"].min()
        )
        & (
            adata_obs["response_n_umis"]
            <= assigned_passing_cells["response_n_umis"].max()
        )
        & (
            adata_obs["response_n_nonzero"]
            >= assigned_passing_cells["response_n_nonzero"].min()
        )
        & (
            adata_obs["response_n_nonzero"]
            <= assigned_passing_cells["response_n_nonzero"].max()
        )
        & (adata_obs["response_p_mito"] < max_mito_frac)
    )
    adata_obs.loc[unperturbed_mask, ["assigned_grna", "assigned_ko"]] = "unperturbed"
    adata_obs.loc[unperturbed_mask, "ChrArm"] = "N/A"
    adata_obs.loc[
        unperturbed_mask,
        ["distal_frac", "n_distal_genes", "non_distal_frac", "n_non_distal_genes"],
    ] = np.nan
    adata_obs.loc[unperturbed_mask, ["control_guide", "arm_trunc"]] = False

    adata_obs.loc[adata_obs["assigned_ko"] == "unperturbed", "pass_qc"] = True

    return adata_obs


def generate_single_cell_expression_zscores(cell_line_name):
    from sklearn.utils.sparsefuncs import mean_variance_axis

    gene_locations = load_file("gene_locations.csv", local_dir=DATA_DIR)

    adata = load_raw_10x(cell_line_name)
    adata.obs = perform_cell_reassignment(adata.obs)
    pass_qc_mask = adata.obs["pass_qc"]

    print("calculating z-scored expression...")
    expr_tpm = (adata.X / adata.X.sum(axis=1) * 1e6).tocsr()
    expr_tpm_log1p = np.log1p(expr_tpm)
    # take statistics from high-quality expression and CRISPR cells
    mean_, var_ = mean_variance_axis(expr_tpm_log1p[pass_qc_mask, :], axis=0)

    obs_mask = (adata.obs["assigned_ko"] == "unperturbed") | (adata.obs["pass_qc"])
    var_mask = var_ != 0
    adata = adata[obs_mask, var_mask]

    zscore_cell_level_expr = (
        expr_tpm_log1p[obs_mask, :][:, var_mask] - mean_[var_mask]
    ) / np.sqrt(var_[var_mask])
    zscore_cell_level_expr = pd.DataFrame(
        zscore_cell_level_expr, index=adata.obs.index, columns=adata.var.index
    )
    print(f"after filtering to cells with definitive CRISPR assignments: {adata.shape}")

    print("mapping gene locations...")
    adata.var = map_gene_locations(adata.var).reindex(index=adata.var.index.tolist())

    adata.obs["mutation_group"] = "normal"
    adata.obs.loc[adata.obs["arm_trunc"], "mutation_group"] = "loss"
    adata.obs.loc[adata.obs["arm_gain"], "mutation_group"] = "gain"

    print("calculating arm-level expression...")
    mean_z_by_ko_group_arm, ko_target_expression = calculate_arm_level_mean_z(
        zscore_cell_level_expr, adata.obs, adata.var, cell_resolution="assigned_ko"
    )
    mean_z_by_grna_group_arm, grna_target_expression = calculate_arm_level_mean_z(
        zscore_cell_level_expr, adata.obs, adata.var, cell_resolution="assigned_grna"
    )

    control_cells = adata.obs[
        adata.obs["control_guide"] & (adata.obs["mutation_group"] == "normal")
    ]
    baseline_expression = (
        zscore_cell_level_expr.reindex(index=control_cells.index.tolist())
        .mean()
        .rename("mean_z_control_expr")
    )
    target_expression = grna_target_expression.rename(
        {"mean_z_target_expr": "mean_z_grna_target_expr"}, axis=1
    ).merge(
        ko_target_expression.rename(
            {"mean_z_target_expr": "mean_z_ko_target_expr"}, axis=1
        ),
        how="left",
    )
    target_expression = target_expression.merge(
        baseline_expression, left_on="target_gene", right_index=True, how="left"
    )
    target_expression = target_expression.loc[
        :,
        [
            "assigned_grna",
            "assigned_ko",
            "mutation_group",
            "mean_z_grna_target_expr",
            "mean_z_ko_target_expr",
            "mean_z_control_expr",
            "target_gene",
        ],
    ]

    return (
        adata,
        zscore_cell_level_expr,
        mean_z_by_ko_group_arm,
        mean_z_by_grna_group_arm,
        target_expression,
    )


def calculate_single_cell_arm_zscores(out_dir=os.path.join(PROCESSED_DIR)):
    cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
        "cell_line"
    ].tolist()
    ko_arm_level_zscores = dict()
    grna_arm_level_zscores = dict()
    target_expression_summary = dict()
    for cl in tqdm(cell_lines):
        (
            adata,
            zscore_cell_level_expr,
            mean_z_by_ko_group_arm,
            mean_z_by_grna_group_arm,
            target_expression,
        ) = generate_single_cell_expression_zscores(cl)
        ko_arm_level_zscores[cl] = mean_z_by_ko_group_arm.sort_index()
        grna_arm_level_zscores[cl] = mean_z_by_grna_group_arm.sort_index()
        target_expression_summary[cl] = target_expression
    ko_arm_level_zscores = pd.concat(
        ko_arm_level_zscores, axis=0, ignore_index=False
    ).rename_axis(["cell_line", "assigned_ko", "mutation_group"])
    grna_arm_level_zscores = pd.concat(
        grna_arm_level_zscores, axis=0, ignore_index=False
    ).rename_axis(["cell_line", "assigned_grna", "mutation_group"])
    target_expression_summary = (
        pd.concat(target_expression_summary, axis=0, ignore_index=False)
        .droplevel(1, axis=0)
        .rename_axis("cell_line")
    )

    if out_dir is not None:
        ko_arm_level_zscores.to_csv(
            os.path.join(out_dir, "single_cell_ko_arm_zscores.csv")
        )
        grna_arm_level_zscores.to_csv(
            os.path.join(out_dir, "single_cell_grna_arm_zscores.csv")
        )
        target_expression_summary.to_csv(
            os.path.join(out_dir, "single_cell_target_zscores.csv")
        )

    return ko_arm_level_zscores, grna_arm_level_zscores, target_expression_summary


def prepare_arm_loss_example(
    cell_line_name, knockout, window_size=100, out_dir=PROCESSED_DIR
):
    from sklearn.utils.sparsefuncs import mean_variance_axis

    gene_locations = load_file("gene_locations.csv", local_dir=DATA_DIR)

    adata = load_raw_10x(cell_line_name)
    adata.obs = perform_cell_reassignment(adata.obs)
    pass_qc_mask = adata.obs["pass_qc"]

    print("calculating z-scored expression...")
    expr_tpm = (adata.X / adata.X.sum(axis=1) * 1e6).tocsr()
    expr_tpm_log1p = np.log1p(expr_tpm)
    # take statistics from high-quality expression and CRISPR cells
    mean_, var_ = mean_variance_axis(expr_tpm_log1p[pass_qc_mask, :], axis=0)

    obs_mask = (adata.obs["assigned_ko"] == "unperturbed") | (
        adata.obs["control_guide"]
    )

    print("mapping gene locations...")
    adata.var = map_gene_locations(adata.var).reindex(index=adata.var.index.tolist())
    chrarm = (
        gene_locations[gene_locations["Gene name"] == knockout].loc[:, "ChrArm"].iloc[0]
    )
    var_mask = (adata.var["ChrArm"] == chrarm) & (var_ != 0)
    adata = adata[obs_mask, var_mask]

    zscore_cell_level_expr = (
        expr_tpm_log1p[obs_mask, :][:, var_mask] - mean_[var_mask]
    ) / np.sqrt(var_[var_mask])
    zscore_cell_level_expr = pd.DataFrame(
        zscore_cell_level_expr, index=adata.obs.index, columns=adata.var.index
    )
    print(f"after filtering to relevant cells: {adata.shape}")

    chr_gene_order = adata.var.sort_values("Midgene estimate").index.tolist()
    z_melt = (
        zscore_cell_level_expr.melt(ignore_index=False)
        .reset_index()
        .rename({"index": "gene", "value": "z_expr"}, axis=1)
    )
    rolled_z = (
        zscore_cell_level_expr.loc[:, chr_gene_order]
        .T.rolling(window=window_size, center=True)
        .mean()
        .T
    )
    rolled_z_melt = (
        rolled_z.melt(ignore_index=False)
        .reset_index()
        .rename({"index": "gene", "value": "z_expr_rolled"}, axis=1)
    )
    longform_z = z_melt.merge(rolled_z_melt)

    longform_z = longform_z.merge(
        adata.obs.drop("ChrArm", axis=1),
        left_on="barcode",
        right_index=True,
        how="left",
    ).merge(adata.var, left_on="gene", right_index=True, how="left")
    longform_z["Cell subset"] = "Other control KO"
    longform_z.loc[(longform_z["assigned_ko"] == "unperturbed"), "Cell subset"] = (
        "Unperturbed"
    )
    longform_z.loc[(longform_z["assigned_ko"] == knockout), "Cell subset"] = "Normal KO"
    longform_z.loc[
        (longform_z["assigned_ko"] == knockout) & (longform_z["arm_trunc"]),
        "Cell subset",
    ] = "Arm loss KO"
    longform_z.loc[
        (longform_z["assigned_ko"] == knockout) & (longform_z["arm_gain"]),
        "Cell subset",
    ] = "Arm gain KO"

    if out_dir is not None:
        longform_z.to_csv(
            os.path.join(out_dir, "single_cell_arm_loss_example.csv"), index=False
        )

    return adata, zscore_cell_level_expr, longform_z


def prepare_arm_alteration_frequency_table(out_dir=PROCESSED_DIR):
    all_cells_table = load_file("all_cells_table.csv", local_dir=PROCESSED_DIR)
    gene_locs = load_file("gene_locations.csv", local_dir=DATA_DIR)
    cell_line_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    knockout_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    target_expression = load_file(
        "single_cell_target_zscores.csv", local_dir=PROCESSED_DIR
    )
    achilles_scores = load_file(
        "CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0
    )

    achilles_scores.columns = [x.split(" (")[0] for x in achilles_scores.columns]
    crispr_df = (
        achilles_scores.reindex(index=cell_line_metadata["arxspan_id"])
        .melt(var_name="CRISPRGene", value_name="gene_effect", ignore_index=False)
        .reset_index()
        .replace(cell_line_metadata.set_index("arxspan_id")["cell_line"].to_dict())
        .rename({"arxspan_id": "cell_line"}, axis=1)
    )
    ko_to_chrarm = gene_locs[
        gene_locs["Gene name"].isin(knockout_metadata["knockout"].tolist())
    ].set_index("Gene name")[["ChrArm", "Gene start (bp)", "Gene end (bp)"]]

    all_ko_gene_effects = (
        pd.concat(
            {
                ko: crispr_df.merge(
                    gene_locs[gene_locs["ChrArm"] == ko_to_chrarm.loc[ko, "ChrArm"]],
                    left_on="CRISPRGene",
                    right_on=["Gene name"],
                ).assign(
                    **{
                        "Target start": ko_to_chrarm.loc[ko, "Gene start (bp)"],
                        "Target end": ko_to_chrarm.loc[ko, "Gene end (bp)"],
                    }
                )
                for ko in ko_to_chrarm.index.tolist()
            }
        )
        .droplevel(level=1)
        .rename_axis("assigned_ko")
        .reset_index()
    )

    distal_arm_ess = (
        all_ko_gene_effects[
            (
                (all_ko_gene_effects["Arm"] == "p")
                & (
                    all_ko_gene_effects["Gene end (bp)"]
                    < all_ko_gene_effects["Target start"]
                )
            )
            | (
                (all_ko_gene_effects["Arm"] == "q")
                & (
                    all_ko_gene_effects["Gene start (bp)"]
                    > all_ko_gene_effects["Target end"]
                )
            )
        ]
        .groupby(["cell_line", "assigned_ko"])["gene_effect"]
        .mean()
        .rename("Average distal arm dependency")
        .rename_axis(["cell_line", "assigned_ko"])
        .reset_index()
    )
    whole_arm_ess = (
        all_ko_gene_effects.groupby(["cell_line", "assigned_ko"])["gene_effect"]
        .mean()
        .rename("Average arm dependency")
        .rename_axis(["cell_line", "assigned_ko"])
        .reset_index()
    )
    target_ess = all_ko_gene_effects[
        all_ko_gene_effects["assigned_ko"] == all_ko_gene_effects["CRISPRGene"]
    ][["cell_line", "assigned_ko", "gene_effect"]].rename(
        {"gene_effect": "Target dependency"}, axis=1
    )

    guide_level = (
        all_cells_table[all_cells_table["pass_qc"]]
        .groupby(["cell_line", "assigned_grna"])[["arm_trunc", "arm_gain"]]
        .mean()
        .reset_index()
        .assign(
            assigned_ko=lambda x: x["assigned_grna"].str.split("_").str[0],
            vector=lambda x: x["assigned_grna"].str.split("_").str[1],
        )
        .merge(
            all_cells_table[all_cells_table["pass_qc"]]
            .value_counts(["cell_line", "assigned_grna"])
            .rename("n_cells_per_grna")
            .reset_index()
        )
    )

    arm_alteration_summary_table = (
        pd.concat(
            [
                guide_level[(guide_level["vector"] == "v1")]
                .set_index(["cell_line", "assigned_ko"])[
                    ["arm_trunc", "arm_gain", "n_cells_per_grna"]
                ]
                .rename(
                    {
                        "arm_trunc": "v1_arm_trunc_freq",
                        "arm_gain": "v1_arm_gain_freq",
                        "n_cells_per_grna": "v1_n_cells",
                    },
                    axis=1,
                ),
                guide_level[(guide_level["vector"] == "v2")]
                .set_index(["cell_line", "assigned_ko"])[
                    ["arm_trunc", "arm_gain", "n_cells_per_grna"]
                ]
                .rename(
                    {
                        "arm_trunc": "v2_arm_trunc_freq",
                        "arm_gain": "v2_arm_gain_freq",
                        "n_cells_per_grna": "v2_n_cells",
                    },
                    axis=1,
                ),
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
        .merge(
            gene_locs[
                [
                    "Gene name",
                    "Distance to centromere",
                    "Distance to telomere",
                    "Gene start (bp)",
                    "Gene end (bp)",
                    "ChrArm",
                    "Midgene estimate",
                ]
            ].assign(
                **{
                    "Target gene length": lambda x: (
                        x["Gene end (bp)"] - x["Gene start (bp)"]
                    )
                }
            ),
            left_on="assigned_ko",
            right_on="Gene name",
        )
        .merge(
            all_cells_table[all_cells_table["pass_qc"]]
            .value_counts(["cell_line", "assigned_ko"])
            .rename("Total cells per KO")
            .reset_index()
        )
        .merge(
            all_cells_table[all_cells_table["pass_qc"]]
            .groupby(["cell_line", "assigned_ko"])["arm_trunc"]
            .mean()
            .rename("Arm truncation frequency")
            .reset_index()
        )
        .merge(
            all_cells_table[all_cells_table["pass_qc"]]
            .groupby(["cell_line", "assigned_ko"])["arm_gain"]
            .mean()
            .rename("Arm gain frequency")
            .reset_index()
        )
        .merge(
            target_expression[target_expression["mutation_group"] == "normal"][
                ["cell_line", "assigned_ko", "mean_z_control_expr"]
            ]
            .drop_duplicates()
            .rename({"mean_z_control_expr": "Target expression in controls"}, axis=1),
            how="left",
        )
        .merge(distal_arm_ess)
        .merge(whole_arm_ess)
        .merge(target_ess)
        .merge(
            knockout_metadata[["knockout", "target_class"]].rename(
                {"knockout": "assigned_ko"}, axis=1
            )
        )
        .drop(["Gene name"], axis=1)
    )

    if out_dir is not None:
        arm_alteration_summary_table.to_csv(
            os.path.join(out_dir, "arm_alteration_covariates.csv"), index=False
        )

    return arm_alteration_summary_table, guide_level


def perform_arm_alteration_correlations(out_dir=PROCESSED_DIR):
    arm_alteration_summary_table = load_file(
        "arm_alteration_covariates.csv", local_dir=PROCESSED_DIR
    )
    outcome_features = ["Arm truncation frequency", "Arm gain frequency"]

    coarse_arm_alteration_correlations = (
        pd.concat(
            [
                pd.DataFrame(
                    pd.concat(
                        [
                            fast_cor(
                                arm_alteration_summary_table.groupby("assigned_ko")[
                                    [
                                        feature,
                                        "Distance to centromere",
                                        "Distance to telomere",
                                        "Target gene length",
                                    ]  # one per gene, any stat will do
                                ].median()
                            )
                            .loc[feature]
                            .drop(feature),
                            fast_cor(
                                arm_alteration_summary_table[
                                    outcome_features
                                    + [
                                        "Target dependency",
                                        "Target expression in controls",
                                        "Average arm dependency",
                                        "Average distal arm dependency",
                                    ]
                                ]
                            )
                            .loc[feature]
                            .drop(feature),
                        ],
                        axis=0,
                    )
                    .rename(feature)
                    .rename_axis("Predictor")
                )
                for feature in outcome_features
            ],
            axis=1,
        )
        .melt(
            ignore_index=False,
            var_name="Outcome",
            value_name="Aggregate correlation with outcome",
        )
        .reset_index()
    )

    cell_line_arm_alteration_correlations = (
        pd.concat(
            {
                feature: arm_alteration_summary_table.groupby("cell_line")
                .apply(
                    lambda x: fast_cor(
                        x[
                            [feature]
                            + list(set(outcome_features) - set([feature]))
                            + [
                                "Average arm dependency",
                                "Average distal arm dependency",
                                "Target dependency",
                                "Target expression in controls",
                                "Distance to centromere",
                                "Distance to telomere",
                                "Target gene length",
                            ]
                        ]
                    ).iloc[0, 1:]
                )
                .melt(
                    ignore_index=False,
                    var_name="Predictor",
                    value_name="Correlation with outcome",
                )
                for feature in outcome_features
            },
            ignore_index=False,
        )
        .rename_axis(["Outcome", "Cell line"])
        .reset_index()
    )

    arm_alteration_correlations = cell_line_arm_alteration_correlations.merge(
        coarse_arm_alteration_correlations
    )

    if out_dir is not None:
        arm_alteration_correlations.to_csv(
            os.path.join(out_dir, "arm_alteration_correlations.csv"), index=False
        )

    return arm_alteration_correlations


# single cell high variance


def select_high_variance_genes(gene_stats_df):
    control_single_cell_gene_stats = gene_stats_df.copy()

    median_single_cell_stats = (
        control_single_cell_gene_stats.assign(lines_present=True)
        .groupby("gene")
        .agg(
            {
                "raw_mean_rank": "median",
                "raw_mean_quantile": "median",
                "index_of_dispersion": "median",
                "index_of_dispersion_rank": "median",
                "lines_present": "count",
            }
        )
        .sort_values("index_of_dispersion", ascending=False)
    )

    # above 99th quantile of index_of_dispersion and present in > 5 lines
    disp_quant = np.quantile(median_single_cell_stats["index_of_dispersion"], 0.99)
    print("99th quantile of index_of_dispersion =", disp_quant)
    high_var_common_genes = median_single_cell_stats[
        (median_single_cell_stats["lines_present"] > 5)
        & (median_single_cell_stats["index_of_dispersion"] > disp_quant)
    ]

    high_expr_excluded = high_var_common_genes[
        (high_var_common_genes["raw_mean_quantile"] < 0.97)
    ]
    top_variance_genes = high_expr_excluded.head(5).index.tolist()

    control_single_cell_gene_stats["selected"] = control_single_cell_gene_stats[
        "gene"
    ].isin(top_variance_genes)
    return (
        control_single_cell_gene_stats,
        median_single_cell_stats,
        high_var_common_genes,
        high_expr_excluded,
    )


def generate_single_cell_stats(cell_line_name):
    from sklearn.utils.sparsefuncs import mean_variance_axis

    adata = load_raw_10x(cell_line_name)
    adata = adata[
        (adata.obs["pass_qc"])
        & (adata.obs["control_guide"])
        & (~adata.obs["arm_trunc"])
        & (~adata.obs["arm_gain"]),
        :,
    ]
    print(f"after filtering to relevant cells: {adata.shape}")

    expr_tpm = (adata.X / adata.X.sum(axis=1) * 1e6).tocsr()
    expr_tpm_log1p = np.log1p(expr_tpm)
    mean_, var_ = mean_variance_axis(expr_tpm_log1p, axis=0)
    raw_mean_, raw_var_ = mean_variance_axis(adata.X, axis=0)

    single_cell_stats = pd.DataFrame(
        {
            "mean": mean_,
            "variance": var_,
            "raw_mean": raw_mean_,
            "raw_variance": raw_var_,
        },
        index=adata.var.index,
    ).rename_axis("gene")
    mean_variable = "raw_mean"
    var_variable = "raw_variance"
    nonzero_mask = single_cell_stats[mean_variable] > 0

    single_cell_stats = single_cell_stats[nonzero_mask]
    single_cell_stats["std"] = np.sqrt(single_cell_stats[var_variable])
    single_cell_stats["coeff_of_variation"] = (
        single_cell_stats["std"] / single_cell_stats[mean_variable]
    )
    single_cell_stats["index_of_dispersion"] = (
        single_cell_stats[var_variable] / single_cell_stats[mean_variable]
    )
    single_cell_stats = single_cell_stats.join(
        single_cell_stats.rank(method="average", ascending=False).add_suffix("_rank")
    )
    single_cell_stats[f"{mean_variable}_quantile"] = 1 - (
        single_cell_stats[f"{mean_variable}_rank"]
        / single_cell_stats[f"{mean_variable}_rank"].max()
    )

    adata = adata[:, nonzero_mask]

    return adata, single_cell_stats


def calculate_single_cell_stats(out_dir=os.path.join(PROCESSED_DIR)):
    cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
        "cell_line"
    ].tolist()
    cl_to_single_cell_stats = dict()
    cl_to_raw_data_subset = dict()
    for cl in tqdm(cell_lines):
        cl_to_raw_data_subset[cl], cl_to_single_cell_stats[cl] = (
            generate_single_cell_stats(cl)
        )

    control_single_cell_gene_stats = (
        pd.concat(cl_to_single_cell_stats, axis=0)
        .rename_axis(["cell_line", "gene"])
        .reset_index()
    )

    if out_dir is not None:
        control_single_cell_gene_stats.to_csv(
            os.path.join(PROCESSED_DIR, "single_cell_control_gene_stats.csv"),
            index=False,
        )

    control_single_cell_gene_stats, _, _, _ = select_high_variance_genes(
        control_single_cell_gene_stats
    )
    top_variance_genes = (
        control_single_cell_gene_stats[control_single_cell_gene_stats["selected"]][
            "gene"
        ]
        .unique()
        .tolist()
    )

    cl_to_top_correlates = dict()
    cl_to_selected_gene_raw_expr = dict()
    for cl in tqdm(cell_lines):
        # get top correlates
        subdata_to_densify = cl_to_raw_data_subset[cl]
        dense_mtx = pd.DataFrame(
            subdata_to_densify.X.todense(),
            index=subdata_to_densify.obs.index,
            columns=subdata_to_densify.var.index,
        )
        cor_mtx = fast_cor(
            dense_mtx.reindex(columns=top_variance_genes).dropna(how="all", axis=1),
            dense_mtx,
        )
        cl_to_top_correlates[cl] = cor_mtx
        # get raw data
        selected_gene_mask = cl_to_raw_data_subset[cl].var.index.isin(
            top_variance_genes
        )
        subdata_to_densify = cl_to_raw_data_subset[cl][:, selected_gene_mask]
        dense_mtx = pd.DataFrame(
            subdata_to_densify.X.todense(),
            index=subdata_to_densify.obs.index.tolist(),
            columns=subdata_to_densify.var.index.tolist(),
        )
        cl_to_selected_gene_raw_expr[cl] = dense_mtx

    top_var_top_correlates = pd.concat(cl_to_top_correlates, axis=0).rename_axis(
        ["cell_line", "gene"]
    )
    top_var_selected_expr = pd.concat(cl_to_selected_gene_raw_expr, axis=0).rename_axis(
        ["cell_line", "barcode"]
    )

    if out_dir is not None:
        top_var_top_correlates.to_csv(
            os.path.join(PROCESSED_DIR, "single_cell_high_variance_top_correlates.csv")
        )
        top_var_selected_expr.to_csv(
            os.path.join(PROCESSED_DIR, "single_cell_high_variance_raw_expression.csv")
        )

    return control_single_cell_gene_stats, top_var_top_correlates, top_var_selected_expr


# perturbation qc


def generate_pseudobulk_correlations_with_ccle(n_top_genes=500, out_dir=PROCESSED_DIR):
    pseudobulk_by_ko = load_file(
        "pseudobulk_sum_by_ko.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    controls = ko_metadata[ko_metadata["target_class"] == "Olfactory Control"][
        "knockout"
    ].tolist()
    control_pseudobulk = (
        pseudobulk_by_ko.loc[:, pd.IndexSlice[:, controls]]
        .groupby(level=0, axis=1)
        .sum()
    )
    control_pseudobulk_logtpm = np.log1p(
        (control_pseudobulk / control_pseudobulk.sum()) * 1e6
    )

    ccle_expr_logtpm = load_file(
        "OmicsExpressionProteinCodingGenesTPMLogp1",
        local_dir=DOWNLOADED_DIR,
        index_col=0,
    ).T
    ccle_expr_genes_remapper = remap_genes(
        ccle_expr_logtpm.index.tolist(), output_column="symbol", verbose=False
    )
    ccle_expr_logtpm = ccle_expr_logtpm.rename(ccle_expr_genes_remapper, axis=0)

    common_genes = list(
        set(ccle_expr_logtpm.index.tolist())
        & set(control_pseudobulk_logtpm.index.tolist())
    )
    top_var_genes = (
        ccle_expr_logtpm.reindex(index=common_genes)
        .var(axis=1)
        .sort_values(ascending=False)
        .head(n_top_genes)
        .index.tolist()
    )

    ccle_cor_top_var_profile = fast_cor(
        control_pseudobulk_logtpm.reindex(index=top_var_genes),
        ccle_expr_logtpm.reindex(index=top_var_genes),
    )

    top_var_pseudobulk_cor_longform = (
        ccle_cor_top_var_profile.melt(ignore_index=False)
        .reset_index()
        .rename(
            {"index": "scCellLine", "variable": "CCLE", "value": "TopVarCorrelation"},
            axis=1,
        )
        .assign(
            scArxspanID=lambda x: x["scCellLine"].replace(
                cl_metadata.set_index("cell_line")["arxspan_id"]
            ),
            Identity=lambda x: x["CCLE"] == x["scArxspanID"],
        )
    )

    top_var_pseudobulk_corr_longform = top_var_pseudobulk_cor_longform.merge(
        cl_metadata.set_index("cell_line")["OncotreeLineage"],
        left_on="scCellLine",
        right_index=True,
    )

    if out_dir is not None:
        top_var_pseudobulk_corr_longform.to_csv(
            os.path.join(out_dir, "pseudobulk_correlation_with_ccle.csv"), index=False
        )

    return top_var_pseudobulk_corr_longform


def generate_cell_depletion_table(regtype="lowess", out_dir=PROCESSED_DIR):
    all_cells_table = load_file("all_cells_table.csv", local_dir=PROCESSED_DIR)
    crispr_df = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    controls = ko_metadata[ko_metadata["target_class"] == "Olfactory Control"][
        "knockout"
    ].tolist()

    cl_to_depletion = dict()
    for cl in all_cells_table["cell_line"].unique().tolist():
        cell_count_per_ko = all_cells_table[
            all_cells_table["pass_qc"] & (all_cells_table["cell_line"] == cl)
        ].value_counts("assigned_ko")
        cl_to_depletion[cl] = (
            cell_count_per_ko / cell_count_per_ko.reindex(index=controls).mean()
        )
    cl_to_depletion = (
        pd.concat(cl_to_depletion, axis=0)
        .rename_axis(("cell_line", "assigned_ko"))
        .rename("depletion")
        .reset_index()
    )

    cells_per_ko_table = (
        all_cells_table[all_cells_table["pass_qc"]]
        .groupby(["cell_line", "assigned_ko"])
        .count()["barcode"]
        .reset_index()
        .merge(crispr_df.rename({"guide": "assigned_ko"}, axis=1))
        .rename({"barcode": "number_cells"}, axis=1)
        .merge(cl_to_depletion)
    )

    cells_per_ko_table = cells_per_ko_table.merge(
        ko_metadata.set_index("knockout")["target_class"],
        left_on="assigned_ko",
        right_index=True,
    )

    if regtype == "linear":
        import statsmodels.api as sm

        aligned = cells_per_ko_table.dropna(subset=["depletion", "gene_effect"])
        linear_x = aligned["gene_effect"]
        linear_x = sm.add_constant(linear_x)
        model = sm.OLS(aligned["depletion"].tolist(), linear_x)
        y_pred = model.fit().fittedvalues
        regression_y = aligned[["cell_line", "assigned_ko"]].copy()
        regression_y["y_pred"] = y_pred
    elif regtype == "lowess":
        lowess_x = cells_per_ko_table["gene_effect"]
        lowess_y = lowess_trend(lowess_x, cells_per_ko_table["depletion"])
        ordered_lowess = (
            pd.DataFrame(
                {
                    "x": lowess_x,
                    "y": lowess_y,
                    "cell_line": cells_per_ko_table["cell_line"],
                    "assigned_ko": cells_per_ko_table["assigned_ko"],
                }
            )
            .dropna()
            .sort_values("x")
        )
        regression_y = ordered_lowess.loc[:, ["cell_line", "assigned_ko", "y"]].rename(
            {"y": "y_pred"}, axis=1
        )

    if regression_y is not None:
        cells_per_ko_table = cells_per_ko_table.merge(regression_y)
        cells_per_ko_table["y_pred"] = cells_per_ko_table["y_pred"].clip(lower=0)

    if out_dir is not None:
        cells_per_ko_table.to_csv(
            os.path.join(out_dir, "depletion_vs_crispr_table.csv"), index=False
        )

    return cells_per_ko_table


def generate_cell_quality_covariates(fdr_threshold=0.05, out_dir=PROCESSED_DIR):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    all_cells_table = load_file("all_cells_table.csv", local_dir=PROCESSED_DIR)
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    cell_lines = cl_metadata["cell_line"].tolist()

    knockdown_detected = pd.concat(
        [
            pd.Series(
                {
                    cl: np.nansum(
                        np.diag(
                            ((sceptre_fdr < fdr_threshold) & (sceptre_zscore < 0))
                            .loc[:, pd.IndexSlice[cl, :]]
                            .droplevel(level=0, axis=1)
                            .reindex(
                                index=ko_metadata["knockout"].tolist(),
                                columns=ko_metadata["knockout"].tolist(),
                            )
                        )
                    )
                    for cl in cell_lines
                }
            ).rename("n_self_downregulated"),
            pd.Series(
                {
                    cl: np.nanmean(
                        np.diag(
                            ((sceptre_fdr < fdr_threshold) & (sceptre_zscore < 0))
                            .loc[:, pd.IndexSlice[cl, :]]
                            .droplevel(level=0, axis=1)
                            .reindex(
                                index=ko_metadata["knockout"].tolist(),
                                columns=ko_metadata["knockout"].tolist(),
                            )
                        )
                    )
                    for cl in cell_lines
                }
            ).rename("frac_self_downregulated"),
        ],
        axis=1,
    )

    cell_quality_covariates = (
        all_cells_table[all_cells_table["pass_qc"]]
        .groupby(["cell_line", "assigned_ko"])
        .mean(numeric_only=True)
        .groupby(level=0, axis=0)
        .mean(numeric_only=True)[["response_n_umis", "response_n_nonzero"]]
        .rename(
            {
                "response_n_umis": "avg_umis_per_ko",
                "response_n_nonzero": "avg_genes_per_ko",
            },
            axis=1,
        )
        .join(knockdown_detected)
        .join(
            all_cells_table[all_cells_table["pass_qc"]]
            .groupby(["cell_line", "assigned_ko"])
            .sum(numeric_only=True)
            .groupby(level=0, axis=0)
            .mean(numeric_only=True)[["response_n_umis", "pass_qc"]]
            .rename(
                {"response_n_umis": "total_umis_per_ko", "pass_qc": "n_cells_per_ko"},
                axis=1,
            )
        )
    )
    if out_dir is not None:
        cell_quality_covariates.to_csv(
            os.path.join(out_dir, "cell_quality_covariates.csv")
        )

    return cell_quality_covariates


def prepare_mtpap_validation_table(out_dir=PROCESSED_DIR):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    pseudobulk_by_ko = load_file(
        "pseudobulk_sum_by_ko.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    controls = ko_metadata[ko_metadata["target_class"] == "Olfactory Control"][
        "knockout"
    ].tolist()
    control_pseudobulk = (
        pseudobulk_by_ko.loc[:, pd.IndexSlice[:, controls]]
        .groupby(level=0, axis=1)
        .sum()
    )
    control_pseudobulk_logtpm = np.log1p(
        (control_pseudobulk / control_pseudobulk.sum()) * 1e6
    )

    melt_df = (
        sceptre_zscore.loc[
            sceptre_zscore.index.str.startswith("MT-"), pd.IndexSlice[:, "MTPAP"]
        ]
        .rename_axis(["cell_line", "assigned_ko"], axis=1)
        .melt(value_name="z_orig", ignore_index=False)
        .reset_index()
    )
    control_expr = (
        control_pseudobulk_logtpm.loc[
            control_pseudobulk_logtpm.index.str.startswith("MT-"), :
        ]
        .melt(ignore_index=False)
        .rename_axis("response_id")
        .reset_index()
        .rename({"variable": "cell_line", "value": "mt_expr"}, axis=1)
    )
    melt_df = melt_df.merge(control_expr)

    if out_dir is not None:
        melt_df.to_csv(os.path.join(out_dir, "mtpap_validation_table.csv"), index=False)

    return melt_df


# global analysis


def define_significant_geneset(
    strength_matrix, significance_matrix, min_genes_to_consider=25, n_top_genes=25
):
    genes_to_consider = set()
    for perturbation in significance_matrix.columns:
        sig_genes = (
            significance_matrix.loc[:, perturbation].loc[lambda x: x].index.tolist()
        )
        if len(sig_genes) >= min_genes_to_consider:
            genes_to_add = (
                strength_matrix.loc[sig_genes, perturbation]
                .abs()
                .sort_values(ascending=False)
                .head(n_top_genes)
                .index.tolist()
            )
            genes_to_consider = genes_to_consider.union(genes_to_add)
    return list(genes_to_consider)


def generate_all_by_all_correlations(out_dir=PROCESSED_DIR, fdr_threshold=0.05):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )

    genes_to_correlate = define_significant_geneset(
        sceptre_zscore,
        (sceptre_fdr < fdr_threshold),
        min_genes_to_consider=5,
        n_top_genes=25,
    )
    print(
        f"identifed top differentially expressed genes (n={len(genes_to_correlate):02d}) to correlate..."
    )

    full_corr_matrix = (
        fast_cor(sceptre_zscore.loc[genes_to_correlate, :])
        .rename_axis(("CellLine1", "TestedPerturbation1"), axis=0)
        .rename_axis(("CellLine2", "TestedPerturbation2"), axis=1)
    )
    melted_correlation_matrix = full_corr_matrix.melt(
        ignore_index=False, value_name="Correlation"
    ).reset_index()

    cell_lines = full_corr_matrix.reset_index()["CellLine1"].unique().tolist()
    knockouts = full_corr_matrix.reset_index()["TestedPerturbation1"].unique().tolist()

    import itertools

    ko_combos = [x for x in itertools.combinations_with_replacement(knockouts, 2)]
    cl_combos = [x for x in itertools.combinations_with_replacement(cell_lines, 2)]
    cl_product = [x for x in itertools.product(cell_lines, repeat=2)]

    ko_pairs_to_corr = []
    for ko_pair in tqdm(ko_combos):
        ko1, ko2 = ko_pair
        if ko1 == ko2:
            cl_pairs = cl_combos
        else:
            cl_pairs = cl_product

        for cl_pair in cl_pairs:
            cl1, cl2 = cl_pair
            if (cl1, ko1) in full_corr_matrix.index.tolist() and (
                cl2,
                ko2,
            ) in full_corr_matrix.index.tolist():
                ko_pairs_to_corr.append(
                    [
                        ko1,
                        cl1,
                        ko2,
                        cl2,
                        full_corr_matrix.loc[
                            pd.IndexSlice[cl1, ko1], pd.IndexSlice[cl2, ko2]
                        ],
                    ]
                )
    unique_correlation_combinations = pd.DataFrame(
        ko_pairs_to_corr,
        columns=["grna_target1", "cell_line1", "grna_target2", "cell_line2", "r"],
    )

    unique_correlation_combinations["same_target"] = (
        unique_correlation_combinations["grna_target1"]
        == unique_correlation_combinations["grna_target2"]
    )
    unique_correlation_combinations["same_cell_line"] = (
        unique_correlation_combinations["cell_line1"]
        == unique_correlation_combinations["cell_line2"]
    )
    if out_dir is not None:
        full_corr_matrix.to_csv(os.path.join(PROCESSED_DIR, "full_corr_matrix.csv"))
        melted_correlation_matrix.to_csv(
            os.path.join(PROCESSED_DIR, "melted_corr_matrix.csv"), index=False
        )
        unique_correlation_combinations.to_csv(
            os.path.join(PROCESSED_DIR, "unique_perturbation_pairs.csv"), index=False
        )

    return full_corr_matrix, melted_correlation_matrix, unique_correlation_combinations


def generate_transcriptional_change_table(out_dir=PROCESSED_DIR):
    cl_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    cell_lines = cl_metadata["cell_line"].unique().tolist()
    ko_metadata = load_file("knockout_metadata.csv", local_dir=METADATA_DIR)
    controls = ko_metadata[ko_metadata["target_class"] == "Olfactory Control"][
        "knockout"
    ].tolist()
    pseudobulk_by_ko = load_file(
        "pseudobulk_sum_by_ko.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    control_pseudobulk = (
        pseudobulk_by_ko.loc[:, pd.IndexSlice[:, controls]]
        .groupby(level=0, axis=1)
        .sum()
    )

    pseudobulk_control_correlations = pd.concat(
        [
            fast_cor(
                np.log1p(
                    (
                        pseudobulk_by_ko.loc[:, pd.IndexSlice[cl, :]]
                        / pseudobulk_by_ko.loc[:, pd.IndexSlice[cl, :]].sum()
                    )
                    * 1e6
                ).droplevel(0, axis=1),
                pd.DataFrame(
                    np.log1p(
                        (control_pseudobulk[cl] / control_pseudobulk[cl].sum()) * 1e6
                    ).rename("Controls")
                ),
            ).rename({"Controls": cl}, axis=1)
            for cl in cell_lines
        ],
        axis=1,
    )

    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)
    transcriptional_change_df = (
        np.sqrt(1 - np.square(pseudobulk_control_correlations))
        .melt(
            ignore_index=False, var_name="cell_line", value_name="deviation_from_basal"
        )
        .reset_index()
        .merge(ko_metadata.rename({"knockout": "assigned_ko"}, axis=1))
        .merge(crispr_table.rename({"guide": "assigned_ko"}, axis=1))
    )

    if out_dir is not None:
        transcriptional_change_df.to_csv(
            os.path.join(out_dir, "transcriptional_change_table.csv"), index=False
        )

    return transcriptional_change_df


def prepare_dependency_hallmark_mean_z_matrix(out_dir=PROCESSED_DIR):
    cell_line_metadata = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)
    crispr = load_file("CRISPRGeneEffect", local_dir=DOWNLOADED_DIR, index_col=0)
    all_genesets = load_file("all_genesets.csv", local_dir=METADATA_DIR)
    zscore_matrix = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )

    hallmark_sets = [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_MYC_TARGETS_V2",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_P53_PATHWAY",
        "HALLMARK_APOPTOSIS",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    ]

    hallmark_set_dict = {}
    for set in hallmark_sets:
        hallmark_set_dict[set] = (
            all_genesets[all_genesets["term"] == set]["gene"].unique().tolist()
        )

    crispr.columns = crispr.columns.map(lambda x: x.split(" ")[0])
    crispr = crispr.loc[cell_line_metadata["arxspan_id"]].rename(
        cell_line_metadata.set_index("arxspan_id")["cell_line"]
    )
    crispr_dependent = crispr < -1

    dep_mean_z = {}
    ko_genes = zscore_matrix.columns.get_level_values(1).unique()
    for ko_gene in ko_genes:
        dependent_lines = crispr_dependent.index[crispr_dependent[ko_gene]]
        if len(dependent_lines) > 0:
            dep_z = zscore_matrix.loc[
                :,
                (zscore_matrix.columns.get_level_values(1) == ko_gene)
                & (zscore_matrix.columns.get_level_values(0).isin(dependent_lines)),
            ]
            ko_dep_means = {}
            for set, genes in hallmark_set_dict.items():
                ko_dep_means[set] = (
                    dep_z.loc[dep_z.index.intersection(genes)].mean().mean()
                )
            dep_mean_z[ko_gene] = ko_dep_means

    dependency_hallmark_mean_z_matrix = pd.DataFrame(dep_mean_z).reset_index(
        names=["term"]
    )
    dependency_hallmark_mean_z_matrix["term"] = dependency_hallmark_mean_z_matrix[
        "term"
    ].apply(clean_geneset_name, nth=5)

    if out_dir is not None:
        dependency_hallmark_mean_z_matrix.to_csv(
            os.path.join(out_dir, "dependency_hallmark_mean_z_matrix.csv"), index=False
        )

    return dependency_hallmark_mean_z_matrix


def prepare_top_dependency_diff_expr_gene_table(
    fdr_threshold=0.05, n_top_genes=10, z_thresh=3, out_dir=PROCESSED_DIR
):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    )
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    )
    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)

    dependent_z_matrix = sceptre_zscore.loc[
        :,
        crispr_table[crispr_table["is_dependent"]]
        .apply(lambda x: (x["cell_line"], x["guide"]), axis=1)
        .values.tolist(),
    ]
    dependent_fdr_matrix = sceptre_fdr.reindex_like(dependent_z_matrix)

    n_unique_dep_targets_up = (
        ((dependent_fdr_matrix < fdr_threshold) & (dependent_z_matrix > z_thresh))
        .T.groupby(level=1)
        .any()
        .sum()
    )
    n_unique_dep_targets_down = (
        ((dependent_fdr_matrix < fdr_threshold) & (dependent_z_matrix < -z_thresh))
        .T.groupby(level=1)
        .any()
        .sum()
    )
    n_perturbations_up = (
        (dependent_fdr_matrix < fdr_threshold) & (dependent_z_matrix > z_thresh)
    ).sum(axis=1)
    n_perturbations_down = (
        (dependent_fdr_matrix < fdr_threshold) & (dependent_z_matrix < -z_thresh)
    ).sum(axis=1)
    n_deps_profiled = (~dependent_fdr_matrix.isna()).sum(axis=1)
    avg_strength = dependent_z_matrix.mean(axis=1)

    top_diff_expr_gene_summary = pd.concat(
        [
            n_unique_dep_targets_up.rename("n_targets_up"),
            n_unique_dep_targets_down.rename("n_targets_down"),
            n_perturbations_up.rename("n_dep_perturbations_up"),
            n_perturbations_down.rename("n_dep_perturbations_down"),
            n_deps_profiled.rename("n_dependencies_detected"),
            avg_strength.rename("mean_zscore"),
        ],
        axis=1,
    )

    top_genes_up = (
        top_diff_expr_gene_summary.sort_values(
            ["n_targets_up", "mean_zscore"], ascending=[False, False]
        )
        .head(n_top_genes)
        .index.tolist()
    )
    top_genes_down = (
        top_diff_expr_gene_summary.sort_values(
            ["n_targets_down", "mean_zscore"], ascending=[False, True]
        )
        .head(n_top_genes)
        .index.tolist()
    )
    top_diff_expr_gene_summary = top_diff_expr_gene_summary.reset_index()

    # these values are per target
    n_deps_significant = (
        ((dependent_fdr_matrix < fdr_threshold) & (dependent_z_matrix.abs() > 5))
        .groupby(level=1, axis=1)
        .sum()
    )
    n_deps_profiled = (~dependent_fdr_matrix.isna()).groupby(level=1, axis=1).sum()
    frac_deps_significant = (n_deps_significant / n_deps_profiled).mask(
        n_deps_profiled < 2
    )

    top_dependencies_matrix = (
        dependent_z_matrix.groupby(level=1, axis=1)
        .mean()
        .loc[top_genes_up + top_genes_down[::-1], :]
    )

    melted_top_dependencies_two_sided = (
        top_dependencies_matrix.melt(ignore_index=False, value_name="mean_dep_z")
        .reset_index()
        .merge(
            frac_deps_significant.melt(
                ignore_index=False, value_name="frac_dependent"
            ).reset_index()
        )
        .dropna()
    )

    expr_order = (
        top_dependencies_matrix.loc[
            :, melted_top_dependencies_two_sided["grna_target"].unique().tolist()
        ]
        .mean(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )
    expr_order = pd.Series(
        np.arange(len(expr_order)) + 1, index=expr_order, name="response_id_order"
    )

    ko_order = (
        crispr_table[crispr_table["is_dependent"]]
        .groupby("guide")["gene_effect"]
        .mean()
        .reindex(melted_top_dependencies_two_sided["grna_target"].unique().tolist())
        .sort_values()
        .index.tolist()
    )
    ko_order = pd.Series(
        np.arange(len(ko_order)) + 1, index=ko_order, name="target_order"
    )

    melted_top_dependencies_two_sided = melted_top_dependencies_two_sided.merge(
        expr_order, left_on="response_id", right_index=True
    ).merge(ko_order, left_on="grna_target", right_index=True)

    if out_dir is not None:
        melted_top_dependencies_two_sided.to_csv(
            os.path.join(out_dir, "dependency_top_diff_expr_gene_table.csv"),
            index=False,
        )
        top_diff_expr_gene_summary.to_csv(
            os.path.join(out_dir, "dependency_top_diff_expr_gene_summary.csv"),
            index=False,
        )

    return melted_top_dependencies_two_sided, top_diff_expr_gene_summary


def prepare_mistimed_perturbation_tables(out_dir=PROCESSED_DIR):
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    )
    crispr_table = load_file("crispr_table.csv", local_dir=PROCESSED_DIR)
    dependent_z_matrix = sceptre_zscore.loc[
        :,
        crispr_table[crispr_table["is_dependent"]]
        .apply(lambda x: (x["cell_line"], x["guide"]), axis=1)
        .values.tolist(),
    ]

    cell_cycle_genes = (
        all_genesets[
            all_genesets.term.isin(["HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT"])
        ]["gene"]
        .unique()
        .tolist()
    )
    cell_cycle_mean_z = dependent_z_matrix.reindex(index=cell_cycle_genes).mean()
    cell_cycle_table = (
        cell_cycle_mean_z.rename_axis(("cell_line", "assigned_ko"))
        .rename("CellCycleDependentMeanZ")
        .reset_index()
    )

    cell_depletion_table = load_file(
        "depletion_vs_crispr_table.csv", local_dir=PROCESSED_DIR
    )
    cell_depletion_table = cell_depletion_table.assign(
        **{
            "residual_depletion": lambda x: x["depletion"] - x["y_pred"],
            "expected_cells": lambda x: (
                ((x["number_cells"] / x["depletion"]) * x["y_pred"]) + 1
            ),  # pseudocount
            "observed_vs_expected_cell_ratio": lambda x: (
                x["number_cells"] / x["expected_cells"]
            ),
        }
    )
    timepoint_pred_table = cell_depletion_table.merge(cell_cycle_table)

    mistimed_bar_df = pd.DataFrame(
        {
            "cell_lines_positive_cc_and_residual": timepoint_pred_table[
                (timepoint_pred_table["depletion"] > timepoint_pred_table["y_pred"])
                & (timepoint_pred_table["CellCycleDependentMeanZ"] > 0)
            ].value_counts("assigned_ko"),
            # 'cell_lines_positive_residual': timepoint_pred_table[
            #     (timepoint_pred_table['depletion'] > timepoint_pred_table['y_pred'])
            # ].value_counts('assigned_ko'),
            # 'n_dep_cell_lines': timepoint_pred_table.value_counts('assigned_ko'),
            "mean_cc": timepoint_pred_table.groupby("assigned_ko")[
                "CellCycleDependentMeanZ"
            ].mean(),
            # 'median_cc': timepoint_pred_table.groupby('assigned_ko')['CellCycleDependentMeanZ'].median(),
            "observed_vs_expected_cell_ratio_total": timepoint_pred_table.groupby(
                "assigned_ko"
            )[["number_cells", "expected_cells"]]
            .sum()
            .assign(
                observed_vs_expected_cell_ratio=lambda x: (
                    x["number_cells"] / x["expected_cells"]
                )
            )["observed_vs_expected_cell_ratio"],
            # 'observed_vs_expected_cell_ratio_mean': timepoint_pred_table.groupby('assigned_ko')['observed_vs_expected_cell_ratio'].mean(),
            # 'observed_vs_expected_cell_ratio_weighted_mean': timepoint_pred_table.groupby('assigned_ko').apply(lambda x: ((x['mean_control_cell_count'] / x['mean_control_cell_count'].sum()) * x['observed_vs_expected_cell_ratio']).sum()).sort_values()
        }
    ).fillna(0)
    # mistimed_bar_df = mistimed_bar_df[(mistimed_bar_df['cell_lines_positive_cc_and_residual'] > 0) & (mistimed_bar_df['n_dep_cell_lines'] > 1)]
    mistimed_bar_df = mistimed_bar_df[
        (mistimed_bar_df["cell_lines_positive_cc_and_residual"] > 0)
    ]

    if out_dir is not None:
        timepoint_pred_table.to_csv(
            os.path.join(PROCESSED_DIR, "timepoint_prediction_table.csv"), index=False
        )
        mistimed_bar_df.to_csv(
            os.path.join(PROCESSED_DIR, "mistimed_perturbation_table.csv"), index=True
        )

    return timepoint_pred_table, mistimed_bar_df


# vignettes


def generate_perturbation_pair_correlation_summary(
    high_corr=0.5, medium_corr=0.35, out_dir=PROCESSED_DIR
):
    unique_correlation_combinations = load_file(
        "unique_perturbation_pairs.csv", local_dir=PROCESSED_DIR
    )

    ko_pair_to_melted_corr = dict()
    for ko_pair, idxs in unique_correlation_combinations.groupby(
        ["grna_target1", "grna_target2"]
    ).groups.items():
        ko_pair_to_melted_corr[ko_pair] = unique_correlation_combinations.loc[
            idxs, :
        ].drop(["grna_target1", "grna_target2"], axis=1)

    ko_pair_summary = dict()
    for ko_pair in ko_pair_to_melted_corr:
        if ko_pair[0] == ko_pair[1]:
            pair_subset = ko_pair_to_melted_corr[ko_pair][
                ~ko_pair_to_melted_corr[ko_pair]["same_cell_line"]
            ]
        else:
            pair_subset = ko_pair_to_melted_corr[ko_pair]

        ko_pair_summary[ko_pair] = pd.Series(
            {
                "average_correlation": pair_subset["r"].mean(),
                "skewness_correlation": scipy.stats.skew(
                    pair_subset["r"], nan_policy="omit"
                ),
                "kurtosis_correlation": scipy.stats.kurtosis(
                    pair_subset["r"], fisher=False, nan_policy="omit"
                ),
                "median_correlation": pair_subset["r"].median(),
                "n_pairs_high_corr": (pair_subset["r"] >= high_corr).sum(),
                "fraction_pairs_high_corr": (pair_subset["r"] >= high_corr).mean(),
                "all_cross_lines_high": (
                    ~pair_subset[pair_subset["r"] >= high_corr]["same_cell_line"]
                ).all()
                & (pair_subset[pair_subset["r"] >= high_corr].shape[0] > 0),
                "n_pairs_medium_corr": (pair_subset["r"] >= medium_corr).sum(),
                "fraction_pairs_medium_corr": (pair_subset["r"] >= medium_corr).mean(),
                "all_cross_lines_medium": (
                    ~pair_subset[pair_subset["r"] >= medium_corr]["same_cell_line"]
                ).all()
                & (pair_subset[pair_subset["r"] >= medium_corr].shape[0] > 0),
            }
        )
    ko_pair_summary = (
        pd.concat(ko_pair_summary, axis=1)
        .T.rename_axis(["grna_target1", "grna_target2"])
        .reset_index()
    )
    ko_pair_summary.loc[
        ko_pair_summary["grna_target1"] == ko_pair_summary["grna_target2"],
        ["all_cross_lines_high", "all_cross_lines_medium"],
    ] = False
    ko_pair_summary["bimodality_coefficient"] = (
        np.power(ko_pair_summary["skewness_correlation"], 2) + 1
    ) / ko_pair_summary["kurtosis_correlation"]
    ko_pair_summary["is_bimodal"] = ko_pair_summary["bimodality_coefficient"] > (5 / 9)

    if out_dir is not None:
        ko_pair_summary.to_csv(
            os.path.join(out_dir, "perturbation_pair_correlation_summary.csv"),
            index=False,
        )

    return ko_pair_summary


# gene sets and enrichment


def calculate_mean_geneset_scores(out_dir=PROCESSED_DIR):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    geneset_table = load_file("all_genesets.csv", local_dir=DATA_DIR)
    group_columns = ["collection", "term"]
    geneset_groups = geneset_table.groupby(group_columns).apply(
        lambda x: list(x["gene"])
    )

    geneset_aggregates = dict()
    for grp, genes in tqdm(
        geneset_groups.items(), total=len(geneset_groups), position=0, leave=True
    ):
        geneset_aggregates[grp] = sceptre_zscore.reindex(index=genes).mean()
    mean_geneset_matrix = (
        pd.concat(geneset_aggregates, axis=1).rename_axis(group_columns, axis=1).T
    )

    if out_dir is not None:
        mean_geneset_matrix.to_csv(
            os.path.join(out_dir, "geneset_mean_zscore_matrix.csv")
        )

    return mean_geneset_matrix


def perform_single_cell_high_variance_enrichment_test(
    n_top_genes=50, out_dir=PROCESSED_DIR
):
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    high_variance_top_correlates = load_file(
        "single_cell_high_variance_top_correlates.csv", local_dir=PROCESSED_DIR
    ).set_index(["cell_line", "gene"])

    # do independently per cell line and gene
    geneset_tests_by_high_variance_gene = dict()
    for cl, g in high_variance_top_correlates.index.tolist():
        gene_correlations = (
            high_variance_top_correlates.loc[(cl, g)]
            .drop(g)
            .dropna()
            .sort_values(ascending=False)
        )
        geneset_tests_by_high_variance_gene[(cl, g)] = run_multiple_hypergeometric(
            query_genes=gene_correlations.head(n_top_genes).index.tolist(),
            geneset_table=all_genesets[all_genesets["collection"] == "Hallmark"],
            all_genes=gene_correlations.index.tolist(),
            report_genes=True,
        )
    all_geneset_tests = (
        pd.concat(geneset_tests_by_high_variance_gene)
        .droplevel(level=2)
        .rename_axis(["cell_line", "high_variance_gene"])
        .reset_index()
    )
    all_geneset_tests["FDR"] = scipy.stats.false_discovery_control(
        all_geneset_tests["pval"]
    )
    all_geneset_tests = all_geneset_tests.sort_values("FDR")

    if out_dir is not None:
        all_geneset_tests.to_csv(
            os.path.join(
                out_dir, "single_cell_high_variance_correlate_geneset_enrichment.csv"
            ),
            index=False,
        )

    return all_geneset_tests


def perform_rna_perturbation_enrichment_test(fdr_threshold=0.05, out_dir=PROCESSED_DIR):
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    significant_genes_by_rna_ko = (
        sceptre_fdr[(sceptre_fdr < fdr_threshold)]
        .loc[:, pd.IndexSlice[:, ["SMG6", "XRN1", "ADAR"]]]
        .apply(lambda x: x.dropna().index.tolist())
    )
    tested_genes_by_rna_ko = sceptre_fdr.loc[
        :, pd.IndexSlice[:, ["SMG6", "XRN1", "ADAR"]]
    ].apply(lambda x: x.dropna().index.tolist())

    individual_rna_ko_hgeom_results = dict()
    for cl, ko in significant_genes_by_rna_ko.index.tolist():
        individual_rna_ko_hgeom_results[(cl, ko)] = run_multiple_hypergeometric(
            query_genes=significant_genes_by_rna_ko.loc[(cl, ko)],
            geneset_table=all_genesets[all_genesets["collection"] == "HGNC"],
            all_genes=tested_genes_by_rna_ko.loc[(cl, ko)],
            report_genes=True,
        )
    individual_rna_ko_hgeom_results = (
        pd.concat(individual_rna_ko_hgeom_results, axis=0, ignore_index=False)
        .droplevel(2)
        .rename_axis(["CellLine", "Knockout"])
        .reset_index()
    )
    individual_rna_ko_hgeom_results["FDR"] = scipy.stats.false_discovery_control(
        individual_rna_ko_hgeom_results["pval"]
    )
    individual_rna_ko_hgeom_results = individual_rna_ko_hgeom_results.sort_values("FDR")

    if out_dir is not None:
        individual_rna_ko_hgeom_results.to_csv(
            os.path.join(out_dir, "rna_perturbation_deg_geneset_enrichment.csv"),
            index=False,
        )

    return individual_rna_ko_hgeom_results


def perform_myc_validation_enrichment_test(out_dir=PROCESSED_DIR):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    sceptre_fdr = load_file(
        "fdr_matrix.csv", local_dir=PROCESSED_DIR, header=[0, 1], index_col=0
    )
    all_genesets = load_file("all_genesets.csv", local_dir=None)
    cell_lines = load_file("cell_line_metadata.csv", local_dir=METADATA_DIR)[
        "cell_line"
    ].tolist()
    myc_targets = (
        all_genesets[
            all_genesets["term"].isin(
                ["HALLMARK_MYC_TARGETS_V1", "HALLMARK_MYC_TARGETS_V2"]
            )
        ]["gene"]
        .unique()
        .tolist()
    )
    myc_longform = (
        sceptre_zscore.reindex(index=myc_targets)
        .loc[:, pd.IndexSlice[:, "MYC"]]
        .droplevel(1, axis=1)
        .melt(ignore_index=False, var_name="cell_line", value_name="z")
        .reset_index()
    )

    strong_genes = (sceptre_zscore.abs() > 1).apply(
        lambda x: x.loc[lambda y: y == True].index.tolist()
    )
    # significant_genes = (sceptre_fdr < 0.05).apply(lambda x: x.loc[lambda y: y == True].index.tolist())
    gene_universes = (~(sceptre_zscore.isna())).apply(
        lambda x: x.loc[lambda y: y == True].index.tolist()
    )

    myc_enrichment = (
        pd.concat(
            [
                run_single_hypergeometric(
                    # query_genes=significant_genes.loc[(cl, 'MYC')],
                    query_genes=strong_genes.loc[(cl, "MYC")],
                    geneset_genes=myc_targets,
                    all_genes=gene_universes.loc[(cl, "MYC")],
                ).rename(cl)
                for cl in cell_lines
            ],
            axis=1,
        )
        .T.rename_axis("cell_line")
        .reset_index()
    )

    myc_enrichment["p_transform"] = -np.log10(myc_enrichment["pval"])
    myc_enrichment = myc_enrichment.merge(
        sceptre_zscore.reindex(index=myc_targets)
        .loc[:, pd.IndexSlice[:, "MYC"]]
        .droplevel(1, axis=1)
        .mean()
        .rename("mean_z_myc_targets"),
        left_on="cell_line",
        right_index=True,
    )
    myc_enrichment = myc_enrichment.merge(
        sceptre_zscore.reindex(index=myc_targets)
        .loc[:, pd.IndexSlice[:, "MYC"]]
        .droplevel(1, axis=1)
        .median()
        .rename("median_z_myc_targets"),
        left_on="cell_line",
        right_index=True,
    )

    myc_df = myc_longform.merge(myc_enrichment)

    if out_dir is not None:
        myc_df.to_csv(os.path.join(out_dir, "myc_validation_enrichment_table.csv"))

    return myc_df


def generate_scyl1_knockout_enrichment_test(out_dir=PROCESSED_DIR):
    sceptre_zscore = load_file(
        "zscore_matrix.csv", local_dir=PROCESSED_DIR, index_col=0, header=[0, 1]
    ).rename_axis(["cell_line", "grna_target"], axis=1)
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)

    strongest_scyl1_response_genes = sceptre_zscore.loc[
        (
            (
                (sceptre_zscore.loc[:, pd.IndexSlice[:, "SCYL1"]]).mean(axis=1) >= 1
            )  # has a strong expression change in general with SCYL1 KO
            & (
                (~sceptre_zscore.loc[:, pd.IndexSlice[:, "SCYL1"]].isna()).mean(axis=1)
                >= 0.5
            )  # profiled in at 75% of lines
        ),
        pd.IndexSlice[:, "SCYL1"],
    ].index.tolist()
    geneset_collection = all_genesets[
        all_genesets["collection"].isin(["Reactome"])
        & (all_genesets["original_set_size"] >= 25)
        & (all_genesets["original_set_size"] <= 200)
    ]

    scyl1_enrichment = run_multiple_hypergeometric(
        query_genes=strongest_scyl1_response_genes,
        geneset_table=geneset_collection,
        all_genes=sceptre_zscore.index.tolist(),
    )

    if out_dir is not None:
        scyl1_enrichment.to_csv(
            os.path.join(out_dir, "scyl1_knockout_enrichment_table.csv")
        )

    return scyl1_enrichment


def generate_deep_rescreen_ctrl_results(out_dir=PROCESSED_DIR):
    cell_lines = ["KMRC20", "UMRC3"]
    results_dict, cells_table_dict, expr_dict = {}, {}, {}
    for cl in cell_lines:
        results = pd.read_csv(
            f"{PROCESSED_DIR}/h2h_sceptre/{cl}-Cas9/discovery_analysis.csv",
            low_memory=False,
        )
        results_filt = results.loc[~results["p_value"].isnull()]
        results.loc[~results["p_value"].isnull(), "FDR"] = false_discovery_control(
            results_filt["p_value"]
        )
        results_dict[cl] = results

        cells_table = pd.read_csv(
            f"{PROCESSED_DIR}/h2h_sceptre/{cl}-Cas9/cells.csv"
        ).set_index("barcode")
        cells_table_dict[cl] = cells_table
        expr_matrix = pd.read_csv(
            f"../h2h_data_formatted/{cl}-Cas9_gex_matrix.csv"
        ).set_index("Unnamed: 0")
        expr_dict[cl] = expr_matrix

    # get deep rescreen ctrl results
    controls = ["OR13C7", "OR9Q1", "OR13H1", "OR10J3"]
    deep_ctrl_results = pd.DataFrame()
    for k, df in results_dict.items():
        ctrl_results = df[df["grna_target"].isin(controls)]
        ctrl_results = ctrl_results[~ctrl_results["significant"].isna()].copy()
        ctrl_results["cell_line"] = k
        deep_ctrl_results = pd.concat([deep_ctrl_results, ctrl_results])

    # get deep rescreen sig ctrl results with mean expr
    deep_ctrl_sig = pd.DataFrame()
    for cl in cell_lines:
        ctrl_results = deep_ctrl_results.query(f'cell_line == "{cl}"')
        sig_ctrl_results = ctrl_results[ctrl_results["significant"]].copy()

        ctrl_exp_mean = pd.DataFrame()
        for ko in controls:
            ko_cells = cells_table_dict[cl].query(f"assigned_ko == '{ko}'").index
            ctrl_exp_mean[ko] = expr_dict[cl].loc[ko_cells].mean()

        ctrl_exp_mean = ctrl_exp_mean.stack().reset_index()
        ctrl_exp_mean.columns = ["response", "ko", "mean_expression"]
        ctrl_exp_mean.index = ctrl_exp_mean["ko"] + "_" + ctrl_exp_mean["response"]

        sig_ctrl_results["ko_response"] = (
            sig_ctrl_results["grna_target"] + "_" + sig_ctrl_results["response_id"]
        )
        sig_ctrl_results = sig_ctrl_results.set_index("ko_response")
        sig_ctrl_results["mean_expression"] = ctrl_exp_mean["mean_expression"]
        sig_ctrl_results["log2_mean_expression"] = np.log2(
            ctrl_exp_mean["mean_expression"] + 1
        )
        sig_ctrl_results["abs_log_2_folout_dird_change"] = abs(
            sig_ctrl_results["log_2_fold_change"]
        )
        sig_ctrl_results["cell_line"] = cl

        deep_ctrl_sig = pd.concat([deep_ctrl_sig, sig_ctrl_results])

    if out_dir is not None:
        deep_ctrl_sig.to_csv(f"{out_dir}/deep_ctrl_results_sig.csv")
        deep_ctrl_results.to_csv(f"{out_dir}/deep_ctrl_results.csv")

    return deep_ctrl_sig, deep_ctrl_results


def generate_deep_rescreen_downsample_curve(out_dir=PROCESSED_DIR):
    all_genesets = load_file("all_genesets.csv", local_dir=DATA_DIR)
    HGNC = load_file("gene_table", local_dir=DOWNLOADED_DIR)

    cell_lines = ["KMRC20", "UMRC3"]
    downsamps = [f"downsamp{i + 1}" for i in range(8)]

    downsamp_results = {}
    mean_umi_per_ko = {}
    for cl in cell_lines:
        downsamp_results[cl] = {}
        experiment_mean_umi_per_ko = {}
        for downsamp in downsamps:
            results = pd.read_csv(
                f"{PROCESSED_DIR}/h2h_sceptre/{cl}-Cas9-{downsamp}/discovery_analysis.csv",
                low_memory=False,
            )
            results_filt = results.loc[~results["p_value"].isnull()]
            results.loc[~results["p_value"].isnull(), "FDR"] = false_discovery_control(
                results_filt["p_value"]
            )
            downsamp_results[cl][downsamp] = results

            cells_table = pd.read_csv(
                f"{PROCESSED_DIR}/h2h_sceptre/{cl}-Cas9-{downsamp}/cells.csv"
            ).set_index("barcode")
            cells_table = cells_table[cells_table["pass_qc"]]
            experiment_mean_umi_per_ko[downsamp] = (
                cells_table.groupby("assigned_ko")["response_n_umis"].sum().mean()
            )
        mean_umi_per_ko[cl] = experiment_mean_umi_per_ko

    mean_umi_per_ko = pd.DataFrame(mean_umi_per_ko).stack().reset_index()
    mean_umi_per_ko.columns = ["downsample", "cell_line", "mean_umi_per_ko"]
    mean_umi_per_ko["sample"] = (
        mean_umi_per_ko["cell_line"] + "-" + mean_umi_per_ko["downsample"]
    )
    mean_umi_per_ko = mean_umi_per_ko.set_index("sample")

    # expected signatures
    cell_cycle_terms = ["HALLMARK_G2M_CHECKPOINT", "HALLMARK_E2F_TARGETS"]
    myc_target_terms = ["HALLMARK_MYC_TARGETS_V1", "HALLMARK_MYC_TARGETS_V2"]

    cell_cycle = list(all_genesets[all_genesets["term"].isin(cell_cycle_terms)]["gene"])
    myc_target = list(all_genesets[all_genesets["term"].isin(myc_target_terms)]["gene"])

    snornas = HGNC.loc[
        lambda x: (
            x["gene_group"].str.contains("Small nucleolar RNA .+ host", case=False)
            == True
        ),
        "symbol",
    ].to_list()
    proteasome = HGNC.loc[
        lambda x: x["gene_group"].str.contains("proteasome", case=False) == True,
        "symbol",
    ].to_list()
    ferritin = HGNC.loc[
        lambda x: x["gene_group"].str.contains("ferri", case=False) == True, "symbol"
    ].to_list()

    expected_h2h = pd.DataFrame(
        [
            ("MYC", "MYC --> cell cycle down", "down", cell_cycle),
            ("MYC", "MYC --> MYC targets down", "down", myc_target),
            ("MTOR", "MTOR --> cell cycle down", "down", cell_cycle),
            (
                "MTOR",
                "MTOR --> MTOR signaling down",
                "down",
                list(all_genesets.query('term == "HALLMARK_MTORC1_SIGNALING"')["gene"]),
            ),
            ("PSMA1", "PSMA1 --> proteosome up", "up", proteasome),
            ("SMG6", "SMG6 --> snoRNA host up", "up", snornas),
            (
                "MDM2",
                "MDM2 --> TP53 up",
                "up",
                list(all_genesets.query('term == "HALLMARK_P53_PATHWAY"')["gene"]),
            ),
            (
                "SEC23IP",
                "SEC23IP --> UPR up",
                "up",
                list(
                    all_genesets.query('term == "HALLMARK_UNFOLDED_PROTEIN_RESPONSE"')[
                        "gene"
                    ]
                ),
            ),
            ("TXN", "TXN --> ferritin up", "up", ferritin),
            (
                "TFRC",
                "TFRC --> Hypoxia up",
                "up",
                list(all_genesets.query('term == "HALLMARK_HYPOXIA"')["gene"]),
            ),
        ],
        columns=["target", "name", "direction", "genes"],
    )

    pct_sig_df = pd.DataFrame()
    for cl in cell_lines:
        cl_sig_df = pd.DataFrame()
        for downsamp in downsamps:
            results = downsamp_results[cl][downsamp]
            downsamp_sig_dict = {}
            for idx, row in expected_h2h.iterrows():
                target = row["target"]
                genes = row["genes"]
                direction = row["direction"]
                name = row["name"]

                target_sig = results.query(
                    f'grna_target == "{target}" & significant == True'
                )
                if direction == "up":
                    target_sig = target_sig[target_sig["log_2_fold_change"] > 0]
                else:
                    target_sig = target_sig[target_sig["log_2_fold_change"] < 0]

                sig_genes = set(target_sig["response_id"]).intersection(genes)
                downsamp_sig_dict[name] = sig_genes

            cl_sig_df[downsamp] = pd.Series(downsamp_sig_dict)

        cl_n_sig_df = pd.DataFrame()
        for downsamp in downsamps:
            # count len intersection with downsamp8
            cl_n_sig_df[downsamp] = cl_sig_df.apply(
                lambda row: len(row[downsamp] & row["downsamp8"]), axis=1
            )

        cl_n_sig_df["target"] = cl_n_sig_df.index.to_series().map(
            expected_h2h.set_index("name")["target"]
        )
        cl_n_sig_df = cl_n_sig_df.groupby("target").sum().T

        cl_pct_sig_df = (cl_n_sig_df / cl_n_sig_df.loc["downsamp8"]).fillna(0)
        cl_pct_sig_df = cl_pct_sig_df.stack().reset_index()
        cl_pct_sig_df.columns = ["downsamp", "target", "pct_sig"]
        cl_pct_sig_df["cell_line"] = cl
        cl_pct_sig_df["n_sig"] = cl_n_sig_df.stack().reset_index()[0]

        pct_sig_df = pd.concat([pct_sig_df, cl_pct_sig_df])

    pct_sig_df["pct_sig"] = pct_sig_df["pct_sig"] * 100
    pct_sig_df["sample"] = pct_sig_df["cell_line"] + "-" + pct_sig_df["downsamp"]
    pct_sig_df["mean_umi_per_ko"] = pct_sig_df["sample"].map(
        mean_umi_per_ko["mean_umi_per_ko"]
    )

    # exclude low
    to_exclude = pct_sig_df[
        (pct_sig_df["downsamp"] == "downsamp8") & (pct_sig_df["n_sig"] < 5)
    ]
    for idx, row in to_exclude.iterrows():
        target = row["target"]
        cl = row["cell_line"]
        pct_sig_df = pct_sig_df.query(f'target != "{target}" | cell_line != "{cl}"')

    if out_dir is not None:
        pct_sig_df.to_csv(f"{PROCESSED_DIR}/deep_downsample_curve.csv")

    return pct_sig_df


def main():
    print("preprocess")
    # preprocess
    gene_locs = get_gene_loc_info(DATA_DIR)
    crispr_df = generate_crispr_table(PROCESSED_DIR)

    print("summarize sceptre")
    # summarize sceptre
    all_cells_table = consolidate_cells_table(out_dir=PROCESSED_DIR)
    pseudobulk_by_ko, pseudobulk_by_grna = generate_pseudobulk_counts(
        out_dir=PROCESSED_DIR
    )
    sceptre_lfc, sceptre_pval, sceptre_zscore, sceptre_fdr = (
        generate_sceptre_results_matrices(PROCESSED_DIR)
    )
    l2fc_rpm_matrix = generate_l2fc_rpm_matrix(out_dir=PROCESSED_DIR)
    geneset_mean_zscores = calculate_mean_geneset_scores(out_dir=PROCESSED_DIR)

    print("figure files")
    # generate files used by figure plotting functions
    ko_arm_level_zscores, grna_arm_level_zscores, target_expression_summary = (
        calculate_single_cell_arm_zscores(out_dir=PROCESSED_DIR)
    )
    _, _, longform_z = prepare_arm_loss_example(
        "RPMI7951", "OR13C4", window_size=50, out_dir=PROCESSED_DIR
    )

    arm_alteration_summary_table, guide_level = prepare_arm_alteration_frequency_table(
        out_dir=PROCESSED_DIR
    )
    arm_alteration_correlations = perform_arm_alteration_correlations(
        out_dir=PROCESSED_DIR
    )

    control_single_cell_gene_stats, top_var_top_correlates, top_var_selected_expr = (
        calculate_single_cell_stats(out_dir=PROCESSED_DIR)
    )

    top_var_pseudobulk_corr_longform = generate_pseudobulk_correlations_with_ccle(
        n_top_genes=500
    )
    cells_per_ko_table = generate_cell_depletion_table(
        regtype="lowess", out_dir=PROCESSED_DIR
    )
    cell_quality_covariates = generate_cell_quality_covariates(
        fdr_threshold=0.05, out_dir=PROCESSED_DIR
    )
    mtpap_validation_df = prepare_mtpap_validation_table(out_dir=PROCESSED_DIR)

    transcriptional_change_df = generate_transcriptional_change_table(
        out_dir=PROCESSED_DIR
    )
    dependency_hallmark_mean_z_matrix = prepare_dependency_hallmark_mean_z_matrix()
    top_dependency_diff_expr_gene_df, top_diff_expr_gene_summary_df = (
        prepare_top_dependency_diff_expr_gene_table(
            fdr_threshold=0.05, n_top_genes=10, z_thresh=3, out_dir=PROCESSED_DIR
        )
    )
    timepoint_pred_table, mistimed_bar_df = prepare_mistimed_perturbation_tables(
        out_dir=PROCESSED_DIR
    )

    full_corr_matrix, melted_correlation_matrix, unique_correlation_combinations = (
        generate_all_by_all_correlations(out_dir=PROCESSED_DIR)
    )
    ko_pair_summary = generate_perturbation_pair_correlation_summary(
        high_corr=0.5, medium_corr=0.35, out_dir=PROCESSED_DIR
    )

    deep_ctrl_sig, deep_ctrl_results = generate_deep_rescreen_ctrl_results(
        out_dir=PROCESSED_DIR
    )
    deep_downsample_curve = generate_deep_rescreen_downsample_curve(
        out_dir=PROCESSED_DIR
    )

    print("genesets")
    # genesets
    single_cell_high_variance_correlate_enrichment_df = (
        perform_single_cell_high_variance_enrichment_test(
            n_top_genes=50, out_dir=PROCESSED_DIR
        )
    )
    myc_enrichment_df = perform_myc_validation_enrichment_test(out_dir=PROCESSED_DIR)
    rna_perturbation_enrichment_df = perform_rna_perturbation_enrichment_test(
        fdr_threshold=0.05, out_dir=PROCESSED_DIR
    )
    scyl_enrichment_df = generate_scyl1_knockout_enrichment_test(out_dir=PROCESSED_DIR)


if __name__ == "__main__":
    main()
