import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from data_utils import *

import requests

from taigapy import create_taiga_client_v3
tc = create_taiga_client_v3()

hgnc_annotations = tc.get('hgnc-gene-table-e250.4/hgnc_complete_set')
hgnc_annotations['cds_gene_id'] = hgnc_annotations['symbol'] + hgnc_annotations['entrez_id'].apply(lambda x: ' (Unknown)' if np.isnan(x) else f' ({str(int(x))})')

def remap_genes(gene_tag_list, global_id_mapping=hgnc_annotations[['symbol', 'entrez_id', 'alias_symbol', 'prev_symbol', 'cds_gene_id']], 
                id_column='entrez_id', symbol_column='symbol', output_column='cds_gene_id', alternative_symbol_columns=['alias_symbol', 'prev_symbol'],
                allow_duplicated_out=False, preserve_unmapped=True, verbose=True):
    '''
    Ingests a list of CDS-formatted gene tags, expected to be "<HGNC Symbol> (<Entrez ID>)", but will also 
    accept only symbols. Then maps the genes to the current stable names according to the provided gene table. 
    Matches genes in order of Entrez ID, followed by HGNC symbol, then by any of the alternative symbols.
    '''
    
    assert output_column in global_id_mapping.columns.tolist(), f'"`{output_column}`" not found in `global_id_mapping`, please specify an output format'
    global_id_mapping = global_id_mapping.copy()
    
    # if the desired output is one of the inputs, this resolves the problem of duplicated columns
    if output_column in [id_column, symbol_column]:
        _output_column = '_' + output_column
        global_id_mapping[_output_column] = global_id_mapping[output_column]
    else:
        _output_column = output_column
    
    # process the ground truth labels and summarize
    ground_truth_mapping = []
    if verbose: 
        print('Processing the ground truth mapping...')
        print(f'Aligning `{output_column}` using `{id_column}` and `{symbol_column}`...')
    ground_truth_mapping.append(global_id_mapping[[symbol_column, id_column, _output_column]].dropna())
    for alt_sym_col in alternative_symbol_columns:
        if alt_sym_col in global_id_mapping.columns.tolist():
            if verbose:
                print(f'Supplementing additional symbols from `{alt_sym_col}`...')
            map_supplement = global_id_mapping[[alt_sym_col, id_column, _output_column]].dropna().rename({alt_sym_col: symbol_column}, axis=1)
            map_supplement[symbol_column] = map_supplement[symbol_column].str.split('|')
            map_supplement = map_supplement.explode(symbol_column)
            ground_truth_mapping.append(map_supplement)
    ground_truth_mapping = pd.concat(ground_truth_mapping, axis=0, ignore_index=True).drop_duplicates(subset=[symbol_column], keep='first')
    if verbose: 
        print(f'Found {len(ground_truth_mapping[symbol_column].unique())} unique entries for `{symbol_column}` mapping to {len(ground_truth_mapping[id_column].unique())} unique entries for `{id_column}`')
    
    # for the inputs, split on whitespace and interpret the components as symbol and id
    if verbose:
        print('Processing the inputs...')
    mapping_to_fix = pd.DataFrame({
        'input_tag': gene_tag_list,
        'input_symbol': [x.split(' ')[0] for x in gene_tag_list],
        'input_id': [x.split(' ')[1].strip('()') if ' ' in x else 'Unknown' for x in gene_tag_list]
    })
    mapping_to_fix['input_id'] = mapping_to_fix['input_id'].replace({'Unknown': np.nan}).astype(float)
    
    output_mappings = []

    to_fix_by_id = mapping_to_fix[~mapping_to_fix['input_id'].isna()]
    to_fix_by_symbol = []
    to_fix_by_symbol.append(mapping_to_fix[mapping_to_fix['input_id'].isna()])
    
    # attempt to resolve genes with ids first
    if len(to_fix_by_id) > 0:
        to_fix_by_id = to_fix_by_id.merge(ground_truth_mapping[[id_column, _output_column]].drop_duplicates(), 
                                          left_on='input_id', right_on=id_column, how='left').drop(id_column, axis=1)
        mapped_by_id = to_fix_by_id[~to_fix_by_id[_output_column].isna()]
        output_mappings.append(mapped_by_id)
        if verbose:
            print(f'Mapped {(len(mapped_by_id) / len(mapping_to_fix)) * 100 :.02f}% of inputs by `{id_column}`...')
        
        # send the genes that failed id matching to the symbol matching inputs
        unmapped = to_fix_by_id[to_fix_by_id[_output_column].isna()].drop(_output_column, axis=1)
        if len(unmapped) > 0:
            to_fix_by_symbol.append(unmapped)
            if verbose:
                print(f'Attempting to map the remainder by `{symbol_column}`...')

    # resolve the remaining genes with symbols
    if len(to_fix_by_symbol) > 0:
        to_fix_by_symbol = pd.concat(to_fix_by_symbol, axis=0)
        to_fix_by_symbol = to_fix_by_symbol.merge(ground_truth_mapping[[symbol_column, _output_column]].drop_duplicates(), 
                                                  left_on='input_symbol', right_on=symbol_column, how='left').drop(symbol_column, axis=1)
        mapped_by_symbol = to_fix_by_symbol[~to_fix_by_symbol[_output_column].isna()]
        output_mappings.append(mapped_by_symbol)
        if verbose:
            print(f'Mapped {(len(mapped_by_symbol) / len(mapping_to_fix)) * 100 :.02f}% of inputs by `{symbol_column}`...')
        
        # remaining genes have failed all mapping attempts, summarize below
        unmapped = to_fix_by_symbol[to_fix_by_symbol[_output_column].isna()]
        
    if verbose:
        print(f'{(len(unmapped) / len(mapping_to_fix)) * 100 :.02f}% of inputs remain unmapped')
    if len(unmapped) > 0:
        if verbose:
            print('\t Unmapped inputs: {}'.format(', '.join(unmapped['input_tag'].tolist())))
    
    output_mappings = pd.concat(output_mappings, axis=0, ignore_index=True)
    if not allow_duplicated_out: # keep only the first instance of duplicate output ids
        output_mappings = output_mappings.drop_duplicates(subset=_output_column, keep='first')
    if preserve_unmapped: # for unmapped genes, keep them in the output mapping in their input forms
        output_mappings = pd.concat([
            output_mappings,
            pd.DataFrame({'input_tag': unmapped['input_tag'].tolist(), _output_column: unmapped['input_tag'].tolist()})
        ], axis=0)
    
    return output_mappings.set_index('input_tag')[_output_column]

def get_enriched_terms(gene_list, n_top_terms=10, term_groups=True, **kwargs):
    """
    Get the top enriched terms / term groups from GeneTEA for the gene list provided.
    Parameters:
        `gene_list` (`list`): list of genes to pass to GeneTEA.
        `n_top_terms` (`int`): max number of top terms or term groups to request from GeneTEA; default 10.
    """
    genetea_url = "https://cds.team/genetea-api/enriched-terms/"
    params = {
        'gene_list': gene_list, 
        'model':'v2', 
        'group_terms':term_groups, #returns top 10 term groups if true, top 10 terms if false
        "n":n_top_terms,
        **kwargs
    }
    r = requests.get(genetea_url, params=params)
    enriched_terms = pd.DataFrame(r.json()['enriched_terms'])
    if enriched_terms.shape[0]==0:
        return None
    enriched_terms['Matching Genes in List'] = enriched_terms['Matching Genes in List'].apply(lambda x: x.split(' '))
    if not term_groups:
        enriched_terms = enriched_terms.drop(columns=['Term Group'])
    return enriched_terms

