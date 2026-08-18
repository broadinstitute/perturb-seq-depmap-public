library(sceptre)
library(tidyverse)
library(rhdf5)
library(Matrix)

INPUT_DIRECTORY <- file.path('.', 'single_cell_data')
OUTPUT_DIRECTORY <- file.path('.', 'processed', 'sceptre')
cell_line_data_manifest <- read_csv(file.path(INPUT_DIRECTORY, 'raw_data_file_manifest.csv'))

read_hdf5 <- function(filename) {
  expr_handle <- H5Fopen(filename)
  expression <- sparseMatrix(i=expr_handle$matrix$indices, p=expr_handle$matrix$indptr, x=expr_handle$matrix$data, dims=expr_handle$matrix$shape, index1=FALSE, repr='C')
  gene_features <- expr_handle$matrix$features %>% as_tibble()
  expression@Dimnames[[1]] <- gene_features$id
  expression@Dimnames[[2]] <- expr_handle$matrix$barcodes
  H5Fclose(expr_handle)
  
  uniques <- gene_features %>% filter(!name %in% gene_features$name[gene_features$name %>% duplicated()])
  duplicates <- gene_features %>% filter(name %in% gene_features$name[gene_features$name %>% duplicated()])
  
  unique_expression <- expression[uniques$id %>% c(), ]
  unique_expression@Dimnames[[1]] <- uniques$name
  
  duplicate_expression <- list()
  for (g in duplicates$name %>% unique()) {
    duplicate_expression[[g]] <- expression[duplicates %>% filter(name == g) %>% .$id %>% c(), ] %>% colSums()
  }
  duplicate_expression <- duplicate_expression %>% bind_rows() %>% as("sparseMatrix")
  duplicate_expression@Dimnames[[1]] <- duplicates$name %>% unique()
  
  expression <- rbind(unique_expression, duplicate_expression)
  
  return(expression)
}

for (i in 1:nrow(cell_line_data_manifest)) {
  # parse file locations
  cl <- cell_line_data_manifest[i, ]$CellLine
  expr_file <- file.path(INPUT_DIRECTORY, cell_line_data_manifest[i, ]$GeneExpression)
  crispr_file <- file.path(INPUT_DIRECTORY, cell_line_data_manifest[i, ]$CRISPR)
  
  print(cl)
  set.seed(1) # set seed for reproducibility
  
  tryCatch({
    print('reading files...')
    expression <- read_hdf5(expr_file)
    crispr <- read_csv(crispr_file) %>% column_to_rownames('CRISPR') %>% as("sparseMatrix")
    if (!(crispr %>% colnames() %>% grepl('-', .) %>% any())) {
      expression@Dimnames[[2]] <- expression@Dimnames[[2]] %>% str_split_i('-', 1)
    }
    
    # filter to passing cells and controls only, excluding the suspected off-target guide
    cells_table <- read_csv(file.path(OUTPUT_DIRECTORY, cl, 'cells.csv')) %>% filter(pass_qc & control_guide & if_else((cl == "SKGII") & (assigned_grna == "OR11H1"), FALSE, TRUE))
    
    print('formatting input data...')
    
    modified_grna_matrix <- cells_table %>% mutate(assigned_grna = if_else(arm_trunc, "TRUNC", if_else(arm_gain, "GAIN", assigned_grna)), indicator=1) %>% pivot_wider(id_cols=assigned_grna, names_from=barcode, values_from=indicator) %>% column_to_rownames("assigned_grna")
    modified_grna_matrix[is.na(modified_grna_matrix)] <- 0
    modified_grna_matrix <- modified_grna_matrix %>% as("sparseMatrix")
    
    # align gene expression and assignment matrices
    common_barcodes <- cells_table$barcode
    response_matrix <- expression[, common_barcodes]
    modified_grna_matrix <- modified_grna_matrix[, common_barcodes]
    
    # sceptre checks that the cell barcodes are IDENTICAL between matrices, rather than equal; set the column names to the same vector
    colnames(response_matrix) <- common_barcodes
    colnames(modified_grna_matrix) <- common_barcodes
    
    # map guides to their gene targets
    guide_to_gene <- data.frame(guide = rownames(modified_grna_matrix), gene = rownames(modified_grna_matrix) %>% str_split_i('_', 1))
    grna_target_data_frame <- data.frame(grna_id=guide_to_gene$guide, grna_target=guide_to_gene$gene) %>%
      mutate(grna_target = ifelse(startsWith(grna_target, 'OR'), 'non-targeting', grna_target)) # consider control genes as non-targeting for compatibility
    
    # 1. Create the sceptre object
    sceptre_object <- import_data(
      response_matrix = response_matrix,
      grna_matrix = modified_grna_matrix,
      grna_target_data_frame = grna_target_data_frame,
      moi = 'low'
    )
    
    # 2. Set analysis parameters
    
    print('setting analysis parameters...')
    
    positive_control_pairs <- construct_positive_control_pairs(sceptre_object = sceptre_object)
    # discovery pairs - we choose not to exclude the positive control pairs, as it is not a given that the target gene's expression will be dysregulated
    discovery_pairs <- construct_trans_pairs(
      sceptre_object = sceptre_object,
      positive_control_pairs = positive_control_pairs,
      pairs_to_exclude = "none"
    )
    
    # set these and other parameters
    sceptre_object <- set_analysis_parameters(
      sceptre_object = sceptre_object,
      discovery_pairs = discovery_pairs,
      positive_control_pairs = positive_control_pairs,
      side = "both",
      grna_integration_strategy = "union",
      formula_object = "default",
      resampling_approximation = "skew_normal",
      control_group = "nt_cells",
      resampling_mechanism = "permutations",
      multiple_testing_method = "BH",
      multiple_testing_alpha = 0.1
    )
    
    # 3. Assign gRNAs to cells
    
    print('assigning guides to cells...')
    
    # use the maximum strategy, using the default requirements for minimum counts and minimum guide fraction
    sceptre_object <- assign_grnas(
      sceptre_object = sceptre_object,
      method = 'maximum',
      umi_fraction_threshold = 0.8,
      min_grna_n_umis_threshold = 1
    )
    
    # 4. Run cell-wise and gRNA-response pair QC
    
    print('running cell QC and response pair QC...')
    
    # use default parameters, but increase the mitochondrial percentage threshold to account for expected cell stress and mitochondrial dysfunction for some knockouts
    sceptre_object <- run_qc(
      sceptre_object = sceptre_object,
      n_nonzero_trt_thresh = 7,
      n_nonzero_cntrl_thresh = 7,
      response_n_umis_range = c(0, 1),
      response_n_nonzero_range = c(0, 1),
      p_mito_threshold = 0.25,
    )
    
    # 5. Run calibration check
    
    print('running negative control checks...')
    
    # use increased verbosity to extract test statistics for comparison
    sceptre_object <- run_calibration_check(
      sceptre_object = sceptre_object,
      parallel = TRUE,
      output_amount = 2 # semi-verbose outputs (model specification without all permutation results)
    )
    
    # 6. Run power check and discovery analysis
    
    print('running novel discovery analysis...')
    
    # run the discovery analysis to identify novel responses
    sceptre_object <- run_discovery_analysis(
      sceptre_object = sceptre_object,
      parallel = TRUE,
      output_amount = 2 # semi-verbose outputs
    )
    
    # 7. Export results
    
    print('saving results...')
    
    write_outputs_to_directory(
      sceptre_object = sceptre_object,
      directory = file.path(OUTPUT_DIRECTORY, 'arm_trunc', cl)
    )
    
    # reformat the data into csv
    
    # discovery analysis (all non-control pairs)
    infile <- file.path(file.path(OUTPUT_DIRECTORY, 'arm_trunc', cl, 'results_run_discovery_analysis.rds'))
    outfile <- file.path(file.path(OUTPUT_DIRECTORY, 'arm_trunc', cl, 'discovery_analysis.csv'))
    read_rds(infile) %>% write_csv(outfile)
    
    # calibration check (negative controls)
    infile <- file.path(file.path(OUTPUT_DIRECTORY, 'arm_trunc', cl, 'results_run_calibration_check.rds'))
    outfile <- file.path(file.path(OUTPUT_DIRECTORY, 'arm_trunc', cl, 'calibration_check.csv'))
    read_rds(infile) %>% write_csv(outfile)
    
  }, error = function(e) print(paste0('Error in ', cl, ': ', e)))
}


