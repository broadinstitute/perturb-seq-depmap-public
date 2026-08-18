import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from constants import *
from taigapy import create_taiga_client_v3
from taigapy.client_v3 import LocalFormat

tc = create_taiga_client_v3()

VERSION = 11

FILE_MAP = {
    # metadata
    "knockout_metadata.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/knockout_metadata",
    "crispr_table.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/crispr_table",
    "cell_line_metadata.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/cell_line_metadata",
    # derived
    "all_cells_table.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/all_cells_table",
    "all_genesets.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/all_genesets",
    "arm_loss_covariates.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/arm_loss_covariates",
    "cell_quality_covariates.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/cell_quality_covariates",
    "control_single_cell_gene_stats.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/control_single_cell_gene_stats",
    "deviation_from_basal_score.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/deviation_from_basal_score",
    "mean_z_expression_change_by_chr_arm.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/mean_z_expression_change_by_chr_arm",
    "mean_z_expression_change_by_chr_arm_corrected.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/mean_z_expression_change_by_chr_arm_corrected",
    "melted_correlation_matrix.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/melted_correlation_matrix",
    "mistimed_perturbation_table.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/mistimed_perturbation_table",
    "timepoint_prediction.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/timepoint_prediction",
    "top_variance_raw_expression.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/top_variance_raw_expression",
    "top_variance_top_correlates.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/top_variance_top_correlates",
    "unique_perturbation_pairs.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/unique_perturbation_pairs",
    "dependency_top_diff_expr_gene_table.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/dependency_top_diff_expr_gene_table",
    "dependency_top_diff_expr_gene_summary.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/dependency_top_diff_expr_gene_summary",
    "dependency_hallmark_mean_z_matrix.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/dependency_hallmark_mean_z_matrix",
    "deep_ctrl_results.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/deep_ctrl_results",
    "deep_ctrl_results_sig.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/deep_ctrl_results_sig",
    "deep_downsample_curve.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/deep_downsample_curve",
    # derived and formatted with multi-indexes
    "pseudobulk_sum_by_grna.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/pseudobulk_sum_by_grna",
    "pseudobulk_sum_by_ko.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/pseudobulk_sum_by_ko",
    "fdr_matrix.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/fdr_matrix",
    "lfc_matrix.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/lfc_matrix",
    "pvalue_matrix.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/pvalue_matrix",
    "zscore_matrix.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/zscore_matrix",
    # downloaded from DepMap
    "Chronos_Combined_predictability_results.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/Chronos_Combined_predictability_results",
    "CRISPRGeneEffect": "public-25q2-c5ef.114/CRISPRGeneEffect",
    "OmicsExpressionProteinCodingGenesTPMLogp1": "public-25q2-c5ef.114/OmicsExpressionProteinCodingGenesTPMLogp1",
    "OmicsSomaticMutationsMatrixDamaging": "public-25q2-c5ef.114/OmicsSomaticMutationsMatrixDamaging",
    "OmicsSomaticMutationsMatrixHotspot": "public-25q2-c5ef.114/OmicsSomaticMutationsMatrixHotspot",
    "OmicsProfiles": "public-25q2-c5ef.114/OmicsProfiles",
    "OmicsCNSegmentsProfileWGS": "public-25q2-c5ef.114/OmicsCNSegmentsProfileWGS",
    "OmicsExpressionTPMLogp1Virus_e6_subset.csv": f"perturb-seq-pilot-2025-c2aa.{VERSION}/OmicsExpressionTPMLogp1Virus_e6_subset",
    # derived from NCBI
    "gene_locations_biomart": "perturbseq-fluent-pilot-1-ac75.14/locations_biomart",
    # derived from HGNC
    "gene_table": "hgnc-gene-table-e250.4/hgnc_complete_set",
    # derived from UCSC
    "cytoband": f"perturb-seq-pilot-2025-c2aa.{VERSION}/cytoband",
}

double_header_files = [
    "fdr_matrix.csv",
    "lfc_matrix.csv",
    "pvalue_matrix.csv",
    "zscore_matrix.csv",
]

double_index_files = ["pseudobulk_sum_by_grna.csv", "pseudobulk_sum_by_ko.csv"]

downloaded_files = [
    "CRISPRGeneEffect",
    "OmicsExpressionProteinCodingGenesTPMLogp1",
    "OmicsSomaticMutationsMatrixDamaging",
    "OmicsSomaticMutationsMatrixHotspot",
    "OmicsProfiles",
    "OmicsCNSegmentsProfileWGS",
    "gene_table",
]


def load_file(filename, local_dir=None, **kwargs):
    """
    Read a file in from a local data folder, and pull it from figshare if absent
    """
    if filename == "OmicsCNGene":
        return load_cn(local_dir, **kwargs)
    if filename == "cytoband":
        if local_dir is None:
            return tc.download_to_cache(
                FILE_MAP[filename], requested_format=LocalFormat.RAW
            )
        else:
            return f"{local_dir}/cytoband.tsv"

    if (filename in downloaded_files) and (local_dir is not None):
        filename = filename + ".csv"

    if filename not in os.listdir(local_dir):
        if filename in double_header_files or filename in double_index_files:
            if filename in double_header_files:
                header = [0, 1]
            else:
                header = 0
            if filename in double_index_files:
                index_col = [0, 1]
            else:
                index_col = 0
            filepath = tc.download_to_cache(
                FILE_MAP[filename], requested_format=LocalFormat.RAW
            )
            table = pd.read_csv(filepath, header=header, index_col=index_col, **kwargs)
        else:
            table = tc.get(FILE_MAP[filename])
    else:
        filepath = os.path.join(local_dir, filename)
        table = pd.read_csv(filepath, **kwargs)
    return table


def load_cn(
    local_dir=None, taiga_permaname="public-25q2-c5ef", taiga_version=114, **kwargs
):
    CN_FILES = ["OmicsCNGeneWGS", "OmicsCNGeneWES"]
    local_files = os.listdir(local_dir)
    cn_tables = []
    for cnf in CN_FILES:
        if cnf not in local_files:
            cn_tables.append(tc.get(f"{taiga_permaname}.{taiga_version}/{cnf}"))
        else:
            cn_tables.append(
                pd.read_csv(os.path.join(local_dir, cnf + ".csv"), **kwargs)
            )
    cn_matrix = pd.concat(cn_tables, axis=0, ignore_index=False)
    return cn_matrix


# anndata processing


def anndata_groupby(adata, axis="var", group_column="index", agg_func="sum"):
    import anndata as ad

    if axis == "obs":
        metadata_table = adata.obs
    if axis == "var":
        metadata_table = adata.var

    if group_column is None or group_column == "index":
        groupby_object = metadata_table.groupby(level=0)
    else:
        groupby_object = metadata_table.groupby(group_column)

    X = adata.X
    if axis == "obs":
        N_obs = groupby_object.ngroups
        N_var = X.shape[1]
    if axis == "var":
        N_obs = X.shape[0]
        N_var = groupby_object.ngroups
    X_agg = scipy.sparse.lil_matrix((N_obs, N_var))

    group_names = []
    index_names = []
    counter = 0
    for group_columns, idx_ in groupby_object.indices.items():
        if axis == "obs":
            X_agg[counter, :] = X[idx_, :].sum(axis=0)
        if axis == "var":
            X_agg[:, counter] = X[:, idx_].sum(axis=1)

        counter += 1
        group_names.append(group_columns)
        index_names.append("_".join(map(str, group_columns)))

    collapsed_metadata = pd.DataFrame(index=groupby_object.indices.keys())
    if axis == "obs":
        return ad.AnnData(X=X_agg.tocsr(), obs=collapsed_metadata, var=adata.var)
    if axis == "var":
        return ad.AnnData(X=X_agg.tocsr(), obs=adata.obs, var=collapsed_metadata)


# data imports
def locate_raw_data(cell_line, dataset="perturbseq-pilot-raw-bbdd", version=17):
    if cell_line in [
        "C4I",
        "JVE127",
        "KMRC20",
        "MG63",
        "ONE58",
        "SKGII",
        "SLR23",
        "SNU8",
        "T3M5",
    ]:  # these have only been sequenced once
        ge_file = f"{cell_line}_GE_filtered_feature_bc_matrix"
        shorten_index = True
    elif cell_line in [
        "HCT15",
        "RPMI7951",
        "U343",
        "KYSE140",
        "HKGZCC",
    ]:  # these have been sequenced multiple times
        ge_file = f"{cell_line}_combined_GE_filtered_feature_bc_matrix"
        shorten_index = True
    elif cell_line in ["KYSE450", "UMRC3"]:  # these have been screened multiple times
        ge_file = f"{cell_line}_all_GE_filtered_feature_bc_matrix"
        shorten_index = False
    else:
        raise ValueError(f"cannot locate data for {cell_line}")

    fp = tc.download_to_cache(
        f"{dataset}.{version}/{ge_file}", requested_format=LocalFormat.RAW
    )
    return fp


def load_raw_10x(cell_line_name, drop_zero=True):
    print(f"Loading {cell_line_name}...")
    fp = locate_raw_data(cell_line_name)
    raw_data = sc.read_10x_h5(fp)
    print(f"raw shape: {raw_data.shape}")
    resolved_genes = ad.concat(
        [
            raw_data[:, ~raw_data.var.index.duplicated(keep=False)],
            anndata_groupby(
                raw_data[:, raw_data.var.index.duplicated(keep=False)],
                axis="var",
                group_column="index",
                agg_func="sum",
            ),
        ],
        axis="var",
    )
    print(f"after summing duplicated gene names: {resolved_genes.shape}")
    cl_subset = load_file(
        "cells.csv", local_dir=os.path.join(PROCESSED_DIR, "sceptre", cell_line_name)
    ).set_index("barcode")
    if "-" not in cl_subset.index[0]:
        cl_subset.index = cl_subset.index + "-1"

    # drop cells that are missing crispr data (and therefore not in the cells table)
    common_cells = [
        x for x in resolved_genes.obs.index.tolist() if x in cl_subset.index.tolist()
    ]
    resolved_genes = resolved_genes[common_cells, :]
    print(f"after dropping cells without CRISPR data: {resolved_genes.shape}")

    # add back metadata
    resolved_genes.obs = cl_subset.reindex(index=resolved_genes.obs.index.tolist())
    resolved_genes.var = (
        raw_data.var.groupby(level=0)
        .agg(lambda x: ";".join(sorted(list(set(x)))))
        .reindex(resolved_genes.var.index.tolist())
    )

    # drop genes with 0 counts
    if drop_zero:
        resolved_genes = resolved_genes[:, resolved_genes.X.sum(axis=0) > 0]
        print(f"after dropping genes with 0 counts: {resolved_genes.shape}")

    return resolved_genes


def read_hdf5(filename):
    src = h5py.File(filename, "r")
    try:
        dim_0 = [x.decode("utf8") for x in src["dim_0"]]
        dim_1 = [x.decode("utf8") for x in src["dim_1"]]
        data = np.array(src["data"])

        return pd.DataFrame(index=dim_0, columns=dim_1, data=data)
    finally:
        src.close()


def ensure_filepath_exists(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def download_crispr_data(cell_line, dataset="perturbseq-pilot-raw-bbdd", version=17):
    if cell_line in [
        "C4I",
        "JVE127",
        "KMRC20",
        "MG63",
        "ONE58",
        "SKGII",
        "SLR23",
        "SNU8",
        "T3M5",
        "HCT15",
        "RPMI7951",
        "U343",
        "KYSE140",
        "HKGZCC",
    ]:
        crispr_file = f"{cell_line}_CRISPR.filt"
    elif cell_line in ["KYSE450", "UMRC3"]:
        crispr_file = f"{cell_line}_all_CRISPR.filt"
    else:
        raise ValueError(f"cannot locate data for {cell_line}")
    fp = tc.download_to_cache(
        f"{dataset}.{version}/{crispr_file}", requested_format=LocalFormat.RAW
    )
    print(f"loading {cell_line} from {fp}")
    return pd.read_csv(fp, index_col=0)


# data munging


def discretize_series_by_interval(value_series, interval_dict):
    discretized_matrix = value_series.copy()
    discretized_matrix.loc[:] = np.nan
    replacements = dict()
    i = 0
    for lb, interval in interval_dict.items():
        left_boolean = (
            value_series >= interval.left
            if interval.closed_left
            else value_series > interval.left
        )
        right_boolean = (
            value_series <= interval.right
            if interval.closed_left
            else value_series < interval.right
        )
        in_interval = (left_boolean & right_boolean).astype(bool)
        discretized_matrix[in_interval] = i
        replacements[i] = lb
        i += 1
    return discretized_matrix.replace(replacements)


def recalibrate_pvalue_matrixwise(matrix):
    # melt the table
    index_name = matrix.index.name
    column_name = matrix.columns.name
    melted_table = matrix.melt(ignore_index=False).reset_index().dropna()
    # compute the fdr correction
    melted_table["value"] = scipy.stats.false_discovery_control(melted_table["value"])
    # pivot back into a matrix
    fdrs = melted_table.pivot(
        index=index_name, columns=column_name, values="value"
    ).reindex_like(matrix)
    return fdrs


def recalibrate_pvalue_matrix_columnwise(matrix):
    fdr_series_dict = dict()
    for c in matrix.columns.tolist():
        nonnull_elements = matrix.loc[:, c].dropna().index.tolist()
        fdr_series_dict[c] = pd.Series(
            scipy.stats.false_discovery_control(matrix.loc[nonnull_elements, c]),
            index=nonnull_elements,
        )
    fdrs = pd.concat(fdr_series_dict, axis=1).reindex_like(matrix)
    return fdrs


# statistical operations


def np_cor_no_missing(x, y):
    """Full column-wise Pearson correlations of two matrices with no missing values."""
    xv = (x - x.mean(axis=0)) / x.std(axis=0)
    yv = (y - y.mean(axis=0)) / y.std(axis=0)
    result = np.dot(xv.T, yv) / len(xv)
    return result


def group_cols_with_same_mask(x):
    """
    Group columns with the same indexes of NAN values.

    Return a sequence of tuples (mask, columns) where columns are the column indices
    in x which all have the mask.
    """
    per_mask = {}
    for i in range(x.shape[1]):
        o_mask = np.isfinite(x[:, i])
        o_mask_b = np.packbits(o_mask).tobytes()
        if o_mask_b not in per_mask:
            per_mask[o_mask_b] = [o_mask, []]
        per_mask[o_mask_b][1].append(i)
    return per_mask.values()


def fast_cor_core(x, y):
    """
    x (`np.array`): 2D array. All columns will be correlated with all columns of y.
    y (`np.array`): 2D array. All columns will be correlated with all columns of x.
                    Must have save length as x.
    returns: `np.array` of shape (x.shape[1], y.shape[1]), where the ith, jth element
        is the pearson correlation of x[:, i] and y[:, j] with null elements removed.
    """
    result = np.zeros(shape=(x.shape[1], y.shape[1]))

    x_groups = group_cols_with_same_mask(x)
    y_groups = group_cols_with_same_mask(y)
    for x_mask, x_columns in x_groups:
        for y_mask, y_columns in y_groups:
            # print(x_mask, x_columns, y_mask, y_columns)
            combined_mask = x_mask & y_mask

            # not sure if this is the fastest way to slice out the relevant subset
            x_without_holes = x[:, x_columns][combined_mask, :]
            y_without_holes = y[:, y_columns][combined_mask, :]

            try:
                c = np_cor_no_missing(x_without_holes, y_without_holes)
            except ValueError:
                raise ValueError(
                    "trying to correlate two groups with shapes %r and %r"
                    % (x_without_holes.shape, y_without_holes.shape)
                )
            # update result with these correlations
            result[np.ix_(x_columns, y_columns)] = c
    return result


def fast_cor(x, y=None):
    """
    x (`pd.DataFrame`): Numerical matrix. All columns will be correlated with all columns of y.
    y (`pd.DataFrame`): Numerical matrix. All columns will be correlated with all columns of x.
                    Index must overlap x.
    returns: `pd.DataFrame` of shape (x.shape[1], y.shape[1]), where the ith, jth element
        is the pearson correlation of x[:, i] and y[:, j] with null elements removed.
    """
    if y is None:
        y = x
    if x is y:
        shared = x.index
    else:
        shared = sorted(set(x.index) & set(y.index))
    if len(shared) < 2:
        raise ValueError("x and y don't have at least two rows in common")
    out = pd.DataFrame(
        fast_cor_core(x.loc[shared].values, y.loc[shared].values),
        index=x.columns,
        columns=y.columns,
    )
    return out


def lowess_trend(
    x, y, frac=0.25, max_points=300, min_points=20, delta_frac=0.005, **kwargs
):
    """
    A wrapper for statsmodel's lowess with a somewhat more useful parameterization
    Parameters:
        `x`, `y`: the points. `y` will be smoothed as a function of `x`.
        `frac`: `float` in [0, 1]. The fraction of the points used for each linear regression.
        `max_points`: `int`. The maximum number of points to be used for each linear regression.
                    Overrides `frac` when smaller.
        `delta_frac`: the fraction of the range of `x` within which to use linear interpolation
        `              instead of a new regression.
        Other args passed to lowess.
    Returns:
        The unsorted smoothed y values.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    frac = min(max_points / len(x), frac)
    frac = max(frac, min_points / len(x))
    rng = x.max() - x.min()
    delta = min(delta_frac * rng, 50 / len(x) * rng)
    delta = min(delta, rng)
    return lowess(
        y, x, frac, delta=delta, is_sorted=False, return_sorted=False, **kwargs
    )


##### Gene sets #####


def run_single_hypergeometric(
    query_genes,
    geneset_genes,
    all_genes,
    report_genes=False,
    preprocess_query=True,
    preprocess_geneset=True,
):
    original_geneset_size = len(geneset_genes)

    if preprocess_query:
        query_genes = list(set(query_genes).intersection(set(all_genes)))

    if preprocess_geneset:
        geneset_genes = list(set(geneset_genes).intersection(set(all_genes)))

    overlap = list(set(query_genes).intersection(set(geneset_genes)))

    # if a/b|c/d is the contingency table crossing membership to geneset and query, then
    success = len(overlap)  # = a
    total = len(all_genes)  # = a + b + c + d
    max_successes = len(geneset_genes)  # = a + b
    draws = len(query_genes)  # = a + c
    odds_ratio_numerator = success * (total - max_successes - draws + success)
    odds_ratio_denominator = (max_successes - success) * (draws - success)

    pval = scipy.stats.hypergeom.sf(success, total, max_successes, draws, loc=1)

    if (
        odds_ratio_numerator == 0 and odds_ratio_denominator == 0
    ):  # nothing tested or no genes outside the query or geneset
        odds_ratio = np.nan
    elif (
        odds_ratio_denominator == 0
    ):  # all members of geneset are in query or vice versa
        odds_ratio = np.inf
    else:
        odds_ratio = odds_ratio_numerator / odds_ratio_denominator

    res = pd.Series(
        {
            "n_overlap": success,
            "original_set_size": original_geneset_size,
            "effective_set_size": max_successes,
            "odds_ratio": odds_ratio,
            "pval": pval,
        }
    )
    if report_genes:
        res.loc["overlap"] = ";".join(overlap)
    return res


def run_multiple_hypergeometric(
    query_genes,
    geneset_table,
    all_genes,
    report_genes=False,
    preprocess_query=True,
    preprocess_geneset=True,
):
    assert (
        "gene" in geneset_table.columns.tolist()
        and "term" in geneset_table.columns.tolist()
    ), "geneset table must have columns 'gene' and 'term'"
    support_table = geneset_table.copy()
    original_set_sizes = support_table.groupby("term")["gene"].count()

    if preprocess_query:
        query_genes = list(set(query_genes).intersection(set(all_genes)))
    if preprocess_geneset:
        support_table = support_table[support_table["gene"].isin(all_genes)]

    support_table["_in_set"] = support_table["gene"].isin(query_genes)

    # if a/b|c/d is the contingency table crossing membership to geneset and query, then
    successes = support_table.groupby("term")["_in_set"].sum()  # = a
    totals = len(all_genes)  # = a + b + c + d
    max_successes = support_table.groupby("term")["gene"].count()  # a + c
    draws = len(query_genes)  # = a + b

    odds_ratio_numerator = successes * (totals - max_successes - draws + successes)
    odds_ratio_denominator = (max_successes - successes) * (draws - successes)
    odds_ratios = odds_ratio_numerator / odds_ratio_denominator
    odds_ratios.loc[(odds_ratio_numerator == 0) & (odds_ratio_denominator == 0)] = (
        np.nan
    )
    odds_ratios.loc[(odds_ratio_denominator == 0)] = np.inf

    pvals = pd.Series(
        scipy.stats.hypergeom.sf(successes, totals, max_successes, draws, loc=1),
        index=successes.index,
    )

    res = pd.concat(
        [
            successes.rename("n_overlap"),
            original_set_sizes.reindex(index=successes.index).rename(
                "original_set_size"
            ),
            max_successes.rename("effective_set_size"),
            odds_ratios.rename("odds_ratio"),
            pvals.rename("pval"),
        ],
        axis=1,
    )

    if report_genes:
        if support_table[support_table["_in_set"]].shape[0] > 0:
            res["overlap"] = (
                support_table[support_table["_in_set"]]
                .groupby("term")
                .apply(lambda x: ";".join(sorted(list(x["gene"]))))
                .reindex(index=res.index)
            )
        else:
            res["overlap"] = np.nan

    res["FDR"] = scipy.stats.false_discovery_control(res["pval"])
    return res.sort_values("pval").rename_axis("term").reset_index()


def clean_geneset_name(string, nth=2, remove_prefix=True):
    string = string.replace("_", " ")
    split_str = string.split(" ")[remove_prefix:]
    new_str_list = []
    for j, x in enumerate(split_str):
        if j % nth == 0:
            new_str_list.append("\n")
        else:
            new_str_list.append(" ")
        new_str_list.append(x)
    return "".join(new_str_list).strip()


def divide_geneset_name_into_two_lines(string, remove_prefix=True):
    string = string.replace("_", " ")
    split_str = string.split(" ")[remove_prefix:]
    if len(split_str) > 1:
        breakpoint_to_length_diff = dict()
        for i in range(1, len(split_str)):
            pre_length = np.sum([len(y) for y in split_str[:i]])
            post_length = np.sum([len(y) for y in split_str[i:]])
            breakpoint_to_length_diff[i] = np.abs(post_length - pre_length)
        min_breakpoint = pd.Series(breakpoint_to_length_diff).idxmin()
        return (
            " ".join(split_str[:min_breakpoint])
            + "\n"
            + " ".join(split_str[min_breakpoint:])
        )
    else:
        return split_str[0].strip()
