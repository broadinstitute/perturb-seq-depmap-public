import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from constants import *

from matplotlib.transforms import ScaledTranslation
def offset_x_labels(fig, ax, dx, dy):
    '''
    Adjusts x-axis labels horizontally in dpi units. Primarily used to fix alignment with rotated labels.
    '''
    offset = ScaledTranslation(dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

def prepare_gridspec(ax_map, height_ratios=None, width_ratios=None, figsize = (8, 10), 
                     inner_gs_dict=dict(), **outer_gs_kws):
    '''
    Creates a nested gridspec from a boolean matrix `ax_map` defining the subplots. This function only goes two layers deep in gridspecs.
    Named arguments to this function are applied to the outer gridspec, while inner gridspec arguments are supplied as a dictionary
    mapping coordinate positions in `ax_map` to dictionaries of GridSpecFromSubplotSpec keywords.
    '''
    fig = plt.figure(figsize=figsize)
    axis_matrix = ax_map.loc[ax_map.any(axis=1), ax_map.any()]
    nrows, ncols = axis_matrix.shape
    
    if height_ratios is None:
        height_ratios = [1 for _ in range(nrows)]
    elif width_ratios is None:
        width_ratios = [1 for _ in range(ncols)]
    
    outer = gridspec.GridSpec(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios, **outer_gs_kws)
    ax_map = dict()
    for i in range(nrows):
        ax_map[i] = dict()
        for j in range(ncols):
            if axis_matrix.iloc[i, j]:
                if (i, j) in inner_gs_dict:
                    gs = gridspec.GridSpecFromSubplotSpec(subplot_spec=outer[i, j], **inner_gs_dict[(i, j)])
                    ax_map[i][j] = dict()
                    for mi in range(inner_gs_dict[(i, j)]['nrows']):
                        for mj in range(inner_gs_dict[(i, j)]['ncols']):
                            ax_map[i][j][(mi, mj)] = fig.add_subplot(gs[mi, mj])
                else:
                    gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i, j])
                    ax_map[i][j] = fig.add_subplot(gs[0])
    return fig, ax_map

def vertical_stacked_barplot(df, x, y, stack_variable, stack_order=None, figsize=None, ax=None, **kwargs):
    pivot_matrix = df.pivot(index=stack_variable, columns=x, values=y)
    if stack_order is not None:
        pivot_matrix = pivot_matrix.reindex(index=stack_order)
    cumulative_matrix = np.cumsum(pivot_matrix)
    cumulative_df = cumulative_matrix.melt(ignore_index=False, value_name=y).reset_index()
    
    if ax is None:
        if figsize is None:
            figsize=(180 * mm, 60 * mm)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
    ax = sns.barplot(cumulative_df, x=x, y=y, hue=stack_variable, hue_order=stack_order[::-1], dodge=False, **kwargs)
    return ax

def identify_label_centers(labels):
    label_series = pd.Series(labels).rename('label')
    breakpoints = np.array([0] + (np.where((label_series.iloc[:-1].values != label_series.iloc[1:].values))[0] + 1).tolist() + [len(label_series)])
    midpoints = (breakpoints[:-1] + (breakpoints[1:] - 1)) / 2
    midpoint_labels = label_series.iloc[breakpoints[:-1]].values.tolist()
    return midpoints, midpoint_labels, breakpoints

def manually_annotate(ax, text, df, xcol, ycol, xyoffset, xycoords='data', 
                      arrowprops=dict(arrowstyle='-', color='black', alpha=1, linewidth=0.25, shrinkA=1, shrinkB=1),
                      ha='center', va='center', color='black', weight='normal', 
                      bbox_kws=None):
    ax.annotate(
        text, (df.loc[text, xcol], df.loc[text, ycol]),
        xytext=(df.loc[text, xcol] + xyoffset[0], df.loc[text, ycol] + xyoffset[1]), xycoords=xycoords, 
        arrowprops=arrowprops, horizontalalignment=ha, verticalalignment=va,
        c=color, fontsize=5, fontweight=weight, bbox=bbox_kws
    )

def identify_gene_centers(perturbation_order):
    ko_order = []
    ko_start_idx = dict()
    for i, x in enumerate(perturbation_order):
        if x[1] not in ko_start_idx:
            ko_order.append(x[1])
            ko_start_idx[x[1]] = i
    ko_start_idx['end'] = len(perturbation_order) + 1
    
    center_idxs = []
    for j, ko in enumerate(ko_order):
        if j < len(ko_order) - 1:
            next_gene_start = ko_start_idx[ko_order[j+1]]
        else:
            next_gene_start = ko_start_idx['end']
        center_idxs.append(((next_gene_start - ko_start_idx[ko]) // 2) + ko_start_idx[ko])
    return [perturbation_order[k] for k in center_idxs], center_idxs

def identify_cell_centers(perturbation_order):
    cell_order = []
    cell_start_idx = dict()
    for i, x in enumerate(perturbation_order):
        if x[0] not in cell_start_idx:
            cell_order.append(x[0])
            cell_start_idx[x[0]] = i
    cell_start_idx['end'] = len(perturbation_order) + 1
    
    center_idxs = []
    for j, ko in enumerate(cell_order):
        if j < len(cell_order) - 1:
            next_cell_start = cell_start_idx[cell_order[j+1]]
        else:
            next_cell_start = cell_start_idx['end']
        center_idxs.append(((next_cell_start - cell_start_idx[ko]) // 2) + cell_start_idx[ko])
    return [perturbation_order[k] for k in center_idxs], center_idxs

def categorical_scatterplot(df, xlabel, ylabel, hue=None, dodge=True, dodge_shift=(-0.3, 0.3), jitter=(0, 0), is_x_cat=None, is_y_cat=None, xorder=None, yorder=None, hue_order=None, xtickrot=0, ytickrot=0, ax=None, figsize=(8, 6), seed=None, **plt_kwargs):
    from pandas.api.types import is_string_dtype, is_numeric_dtype
    
    rng = np.random.default_rng(seed=seed)
    
    df = df.copy()
    if is_x_cat is None:
        is_x_cat = not is_numeric_dtype(df[xlabel])
    if is_y_cat is None:
        is_y_cat = not is_numeric_dtype(df[ylabel])
    if jitter is None:
        jitter = (0, 0)
    elif isinstance(jitter, bool):
        if jitter:
            jitter = (0.1, 0.1)
        else:
            jitter = (0, 0)
    elif isinstance(jitter, float) or isinstance(jitter, int):
        print('yes')
        jitter = (jitter, jitter)
        
    # fix the coordinate positions if either axis is categorical
    if is_x_cat:
        if xorder is None:
            xorder = df[xlabel].unique().tolist()
        xmapping = {v:k for k,v in dict(enumerate(xorder)).items()}
        df['_xcoord'] = df[xlabel].replace(xmapping)
        if hue is not None and dodge:
            if hue_order is None:
                hue_order = df[hue].unique().tolist()
            hue_shifts = dict(zip(hue_order, np.linspace(dodge_shift[0], dodge_shift[1], len(hue_order))))
            df['_xcoord'] = df['_xcoord'] + df[hue].replace(hue_shifts)
        df['_xcoord'] = df['_xcoord'] + rng.uniform(-np.abs(jitter[0]), np.abs(jitter[0]), len(df['_xcoord']))
    else:
        df['_xcoord'] = df[xlabel]
        
    if is_y_cat:
        if yorder is None:
            yorder = df[ylabel].unique().tolist()
        ymapping = {v:k for k,v in dict(enumerate(yorder)).items()}
        df['_ycoord'] = df[ylabel].replace(ymapping)
        if hue is not None and dodge:
            if hue_order is None:
                hue_order = df[hue].unique().tolist()
            hue_shifts = dict(zip(hue_order, np.linspace(dodge_shift[0], dodge_shift[1], len(hue_order))))
            df['_ycoord'] = df['_ycoord'] + df[hue].replace(hue_shifts)
        df['_ycoord'] = df['_ycoord'] + rng.uniform(-np.abs(jitter[1]), np.abs(jitter[1]), len(df['_ycoord']))
    else:
        df['_ycoord'] = df[ylabel]
        
    if ax is None:
        if figsize is None:
            figsize = (8, 6)
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        
    # make the plot
    sns.scatterplot(
        df, x='_xcoord', y='_ycoord', hue=hue, ax=ax, **plt_kwargs
    )
    
    # replace the axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if is_x_cat:
        ax.set_xticks(list(xmapping.values()), list(xmapping.keys()), rotation=xtickrot)
    if is_y_cat:
        ax.set_yticks(list(ymapping.values()), list(ymapping.keys()), rotation=ytickrot)
    
    return df, ax

# networks and graph helpers

def plot_loop(xy_start, r, n_points=200, offset_angle=0, ax=None, **kwargs):
    if isinstance(r, tuple):
        r1 = r[0]
        r2 = r[1]
    else:
        r1 = r
        r2 = r
    if ax is None:
        plt.figure()
        ax = plt.gca()
    thetas = np.linspace(0, 2 * np.pi, n_points) + offset_angle
    x_base = r1 * np.cos(thetas)
    y_base = r2 * np.sin(thetas)
    x_offset = r1 * np.cos(offset_angle)
    y_offset = r2 * np.sin(offset_angle)
    xs = x_base + xy_start[0] + x_offset
    ys = y_base + xy_start[1] + y_offset
    ax.plot(xs, ys, **kwargs)
    
def get_angle_from_xaxis(xy_vector, fix=False):
    signed_angle = np.arctan2(xy_vector[1], xy_vector[0])
    if signed_angle < 0:
        return (2 * np.pi) + signed_angle
    else:
        return signed_angle
    
def scale_weights(series, vmin=None, vmax=None, clip=True, lims=(0.5, 2)):
    from matplotlib.colors import Normalize
    if vmin is None:
        vmin = series.min()
    if vmax is None:
        vmax = series.max()
    norm = Normalize(vmin=vmin, vmax=vmax, clip=clip)
    return series.apply(lambda x: np.interp(norm(x), (0, 1), lims))