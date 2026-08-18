library(sceptre)
library(tidyverse)
library(rhdf5)
library(Matrix)

INPUT_DIRECTORY <- file.path('.', 'h2h_data_formatted')
OUTPUT_DIRECTORY <- file.path('.', 'processed', 'h2h_sceptre')

gene_locations <- read_csv(file.path('.', 'metadata', 'gene_locations_biomart.csv')) %>% filter(!is.na(`Gene name`), !is.na(`Karyotype band`)) %>% arrange(`Gene start (bp)`) %>% distinct(`Gene name`, .keep_all=TRUE)
gene_locations$`ChrArm` <- paste0(gene_locations$`Chromosome/scaffold name`, substr(gene_locations$`Karyotype band`, start = 1, stop = 1))
cell_line_data_manifest <- read_csv(file.path(INPUT_DIRECTORY, 'h2h_data_file_manifest.csv')) %>% column_to_rownames('...1')

prepare_cells_table <- function(sceptre_object, gene_locations) {
  # extract covariates
  cells_table <- sceptre_object@covariate_data_frame
  cells_table$sceptre_index <- seq(nrow(cells_table))
  cells_table$barcode <- common_barcodes
  cells_table$pass_qc <- FALSE
  cells_table[sceptre_object@cells_in_use, "pass_qc"] <- TRUE
  
  # append guide asssignments
  grna_assignment_matrix <- get_grna_assignments(sceptre_object)
  for (grna in (rownames(grna_assignment_matrix))) {
    cells_table[which(grna_assignment_matrix[grna, ]), "assigned_grna"] <- grna
  }
  cells_table$grna_max_umi <- grna_matrix %>% apply(2, function(x) max(x))
  

  cells_table$assigned_ko <- str_split_i(cells_table$assigned_grna, "_", 2)
  # change AAVS1 to alias PPP1R12C for gene_location table
  cells_table$assigned_ko_geneloc <- cells_table %>% 
    mutate(assigned_ko = ifelse(assigned_ko == "AAVS1", "PPP1R12C", assigned_ko)) %>% 
    pull(assigned_ko)
  cells_table$control_guide <- str_split_i(cells_table$assigned_grna, "_", 2) == "non-targeting"
  
  # merge gene target locations
  cells_table <- cells_table %>% left_join(gene_locations %>% transmute(assigned_ko_geneloc=`Gene name`, ChrArm = `ChrArm`), by = join_by("assigned_ko_geneloc"))
  return(cells_table)
}

call_arm_truncations_by_ko <- function(cells_table, ko, distal_read_quantile, gene_locations) {
  # get ko information
  target <- gene_locations %>% filter(`Gene name` == ko)
  
  # identify all genes towards the telomere from the targeted gene
  distal_gene_info <- gene_locations %>% filter(`ChrArm` == target$`ChrArm`, `Gene name` != ko) %>% 
    filter(((substr(`Karyotype band`, start = 1, stop = 1) == "p") & (`Gene end (bp)` <= target$`Gene end (bp)`)) | 
             ((substr(`Karyotype band`, start = 1, stop = 1) == "q") & (`Gene start (bp)` >= target$`Gene end (bp)`))) %>%
    filter(`Gene name` %in% rownames(sceptre_object@response_matrix[[1]]))
  distal_genes <- distal_gene_info$`Gene name`
  
  # identify all other genes on the same arm
  other_arm_gene_info <- gene_locations %>% filter(`ChrArm` == target$`ChrArm`, `Gene name` != ko, `Gene name` %in% rownames(sceptre_object@response_matrix[[1]]), !(`Gene name` %in% distal_genes))
  other_arm_genes <- other_arm_gene_info$`Gene name`
  
  # subset to the knockout cells
  ko_cells <- cells_table %>% filter(assigned_ko_geneloc == ko)
  ko_response_matrix <- sceptre_object@response_matrix[[1]][, ko_cells$`sceptre_index`]
  
  # subset to the control cells 
  control_cells <- cells_table %>% filter(control_guide, assigned_ko_geneloc != ko)
  control_response_matrix <- sceptre_object@response_matrix[[1]][, control_cells$`sceptre_index`]
  
  # compute the fraction of reads in both populations that map to genes more distal than the targeted gene on the same arm
  ko_distal_coverage <- (colSums(ko_response_matrix[distal_genes, ]) + 1) / (colSums(ko_response_matrix) + 1)
  ko_proximal_coverage <- (colSums(ko_response_matrix[other_arm_genes, ]) + 1) / (colSums(ko_response_matrix) + 1)
  control_distal_coverage <- (colSums(control_response_matrix[distal_genes, ]) + 1) / (colSums(control_response_matrix) + 1)
  
  # use a quantile-based threshold on the high-quality control cells to determine a threshold for arm loss inferred from expression
  lower_distal_frac_threshold <- control_cells %>% select(sceptre_index, barcode, pass_qc) %>% mutate(distal_fraction = control_distal_coverage) %>% filter(`pass_qc`) %>% .$`distal_fraction` %>% quantile(probs=distal_read_quantile, names=FALSE)
  upper_distal_frac_threshold <- control_cells %>% select(sceptre_index, barcode, pass_qc) %>% mutate(distal_fraction = control_distal_coverage) %>% filter(`pass_qc`) %>% .$`distal_fraction` %>% quantile(probs=1 - distal_read_quantile, names=FALSE)
  
  # map these values back to the targeted cells
  ko_cell_table_subset <- ko_cells %>% select(sceptre_index, barcode) %>% mutate(distal_frac = ko_distal_coverage, n_distal_genes = length(distal_genes), non_distal_frac = ko_proximal_coverage, n_non_distal_genes = length(other_arm_genes), lower_distal_threshold = lower_distal_frac_threshold, arm_trunc = distal_frac < lower_distal_threshold, upper_distal_threshold = upper_distal_frac_threshold, arm_gain = distal_frac > upper_distal_threshold)
  
  return(ko_cell_table_subset)
}

call_arm_truncations <- function(cells_table, distal_read_quantile, gene_locations) {
  # loop through all knockouts to identify arm truncation events
  skip_arm_guides = unique((cells_table %>% filter(is.na(ChrArm)))$assigned_grna)
  skip_arm_kos <- str_split_i(skip_arm_guides, "_", 2)
  all_arm_truncations <- NULL
  for (ko in unique(cells_table$assigned_ko_geneloc)) {
    # 
    if (ko %in% skip_arm_kos) {
      next
    }
    if (is_null(all_arm_truncations)) {
      all_arm_truncations <- call_arm_truncations_by_ko(cells_table, ko, distal_read_quantile, gene_locations)
    } else {
      all_arm_truncations <- rbind(all_arm_truncations, call_arm_truncations_by_ko(cells_table, ko, distal_read_quantile, gene_locations))
    }
  }
  # join these calls with the original cell metadata table
  return(cells_table %>% left_join(all_arm_truncations, join_by("sceptre_index", "barcode")))
}


for (cl in c('KMRC20-Cas9', 'UMRC3-Cas9')) {
  # parse file locations
  expr_file <- file.path(INPUT_DIRECTORY, cell_line_data_manifest[cl, ]$GeneExpression)
  crispr_file <- file.path(INPUT_DIRECTORY, cell_line_data_manifest[cl, ]$CRISPR)
  
  print(cl)
  set.seed(1) # set seed for reproducibilitiy
  
  tryCatch({
    print('reading files...')
    expression <- read_csv(expr_file) %>% column_to_rownames('...1')
    expression <- data.matrix(expression)
    expression <- t(Matrix(expression, sparse = TRUE))
    crispr <- t(read_csv(crispr_file) %>% column_to_rownames('...1'))
    if (!(crispr %>% colnames() %>% grepl('-', .) %>% any())) {
      expression@Dimnames[[2]] <- expression@Dimnames[[2]] %>% str_split_i('-', 1)
    }
    crispr[is.na(crispr)] <- 0 
    
    print('formatting input data...')
    
    # align gene expression and CRISPR guide count matrices
    common_barcodes <- intersect(expression %>% colnames(), crispr %>% colnames())
    response_matrix <- expression[, common_barcodes]
    grna_matrix <- crispr[, common_barcodes]

    # sceptre checks that the cell barcodes are IDENTICAL between matrices, rather than equal; set the column names to the same vector
    colnames(response_matrix) <- common_barcodes
    colnames(grna_matrix) <- common_barcodes

    # map guides to their gene targets
    guide_to_gene <- data.frame(guide = rownames(crispr), gene = rownames(crispr) %>% str_split_i('_', 2)) 
    grna_target_data_frame <- data.frame(grna_id=guide_to_gene$guide, grna_target=guide_to_gene$gene)
    
    # 1. Create the sceptre object
    sceptre_object <- import_data(
      response_matrix = response_matrix,
      grna_matrix = grna_matrix,
      grna_target_data_frame = grna_target_data_frame,
      moi = 'low'
    )

    # 2. Set analysis parameters

    print('setting analysis parameters...')

    # positive control pairs
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
      min_grna_n_umis_threshold = 5
    )

    # 4. Run cell-wise and gRNA-response pair QC

    print('running cell QC and response pair QC...')

    # use default parameters, but increase the mitochondrial percentage threshold to account for expected cell stress and mitochondrial dysfunction for some knockouts
    sceptre_object <- run_qc(
      sceptre_object = sceptre_object,
      n_nonzero_trt_thresh = 7,
      n_nonzero_cntrl_thresh = 7,
      response_n_umis_range = c(0.01, 0.99),
      response_n_nonzero_range = c(0.01, 0.99),
      p_mito_threshold = 0.25,
    )
    
    # based on the qc-passing cells at this stage, further remove cells with inferred chromosome arm truncations
    cells_table <- prepare_cells_table(sceptre_object, gene_locations)
    cells_table <- call_arm_truncations(cells_table, distal_read_quantile = 0.025, gene_locations=gene_locations)
    
    # re-run qc to remove the arm loss cells
    sceptre_object <- run_qc(
      sceptre_object = sceptre_object,
      n_nonzero_trt_thresh = 7,
      n_nonzero_cntrl_thresh = 7,
      response_n_umis_range = c(0.01, 0.99),
      response_n_nonzero_range = c(0.01, 0.99),
      p_mito_threshold = 0.25,
      additional_cells_to_remove = (cells_table %>% filter(arm_trunc | arm_gain) %>% .$`sceptre_index`)
    )
    
    # filter discovery pairs for OR and nongenic cutting gRNAs only
    or_pairs <- sceptre_object@discovery_pairs[startsWith(grna_target, "OR"), ]
    nongenic_pairs <- sceptre_object@discovery_pairs[grepl("Chr2-4|Chr2-2", sceptre_object@discovery_pairs$grna_target)]
    sceptre_object@discovery_pairs <- rbind(or_pairs, nongenic_pairs)

    # 5. Run calibration check

    print('running negative control checks...')

    # use increased verbosity to extract test statistics for comparison
    sceptre_object <- run_calibration_check(
      sceptre_object = sceptre_object,
      parallel = TRUE,
      calibration_group_size = 2, # manually lowered for low n NT guides
      output_amount = 2 # semi-verbose outputs (model specification without all permutation results)
    )

    # 6. Run power check and discovery analysis

    print('running novel discovery analysis...')

    # run the power check to compare p-values for positive control responses and negative control responses
    sceptre_object <- run_power_check(
      sceptre_object = sceptre_object,
      parallel = TRUE,
      output_amount = 2 # semi-verbose outputs
    )

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
      directory = file.path(OUTPUT_DIRECTORY, cl)
    )

    # reformat the data into csv

    # discovery analysis (all non-control pairs)
    infile <- file.path(file.path(OUTPUT_DIRECTORY, cl, 'results_run_discovery_analysis.rds'))
    outfile <- file.path(file.path(OUTPUT_DIRECTORY, cl, 'discovery_analysis.csv'))
    read_rds(infile) %>% write_csv(outfile)

    # calibration check (negative controls)
    infile <- file.path(file.path(OUTPUT_DIRECTORY, cl, 'results_run_calibration_check.rds'))
    outfile <- file.path(file.path(OUTPUT_DIRECTORY, cl, 'calibration_check.csv'))
    read_rds(infile) %>% write_csv(outfile)

    # power check (all controls)
    infile <- file.path(file.path(OUTPUT_DIRECTORY, cl, 'results_run_power_check.rds'))
    outfile <- file.path(file.path(OUTPUT_DIRECTORY, cl, 'power_check.csv'))
    read_rds(infile) %>% write_csv(outfile)

    # cell metadata table
    outfile <- file.path(file.path(OUTPUT_DIRECTORY, cl, 'cells.csv'))
    cells_table %>% write_csv(outfile)
    
  }, error = function(e) print(paste0('Error in ', cl)))
}


