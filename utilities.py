import numpy as np
import pyPLS
import time
import sys
sys.path.append("../Broad_collaboration")
sys.path.append("/")
import broad_utils as utils
import seaborn as sns
import pandas as pd
import random
from scipy import interpolate
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
import string
from cycler import cycler
import matplotlib as mpl

params = {'mathtext.default': 'regular', 
          'figure.facecolor': 'white', 
          'font.size': 20}
plt.rcParams.update(params)


def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_featurecols(df):
    """returna  list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_metacols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]


def load_df(df_path, condition_query):
    df = pd.read_csv(df_path,
                     low_memory=False,
                     index_col=False)
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.query(condition_query)
    return df

def log_transform(s):
    if s >= 0:
        return np.log(s + 1)
    if s < 0:
        return -np.log(abs(s) + 1)

def pca_with_svd(X, ndim, exp_variance=False, return_loadings=False):
    U, S, V = np.linalg.svd(X)
    P, D, Q = np.linalg.svd(X)
    Xv = np.dot(X, V.T[:, 0:ndim])   
    xus = np.matmul(U[:, 0:ndim], np.diag(S)[0:ndim, 0:ndim])
    
    assert np.isclose(Xv, xus).all()
    if not exp_variance and not return_loadings: 
        return xus
    if return_loadings:
        return xus, np.array(V.T[:, 0:ndim])
    if exp_variance:
        return xus, S[0:ndim]**2
        
def calc_eq_score_df(fit_df, pred_df, ncp=10):
    results_df = get_metadata(pred_df)
    ess_df = get_metadata(pred_df)
    test = np.array(get_featuredata(pred_df), dtype=float)
    eqs = fit_df[fit_df['Metadata_control_type'] != 'negcon']['Metadata_pert_iname'].unique().tolist()
    for eq in eqs:
        work = fit_df[(fit_df["Metadata_pert_iname"] == eq) | (fit_df["Metadata_control_type"] == 'negcon')]
        X = np.array(get_featuredata(work), dtype=float)
        Y = np.array(work["Metadata_control_type"] != 'negcon', dtype=float)
        pls = pyPLS.pls(X, Y, ncp=ncp)
        results, stats = pls.predict(test, statistics=True)
        results_df[f"{eq}_eq"] = results
        ess_df[f"{eq}_SSE"] = stats['ESS']
    return results_df, ess_df

def calc_eq_score_df_with_cv(fit_df, pred_df, ncp=10):
    results_df = get_metadata(pred_df)
    ess_df = get_metadata(pred_df)
    eqs = fit_df[fit_df['Metadata_control_type'] != 'negcon']['Metadata_pert_iname'].unique().tolist()
    for eq in eqs:
        work = fit_df[(fit_df["Metadata_pert_iname"] == eq) | (fit_df["Metadata_control_type"] == 'negcon')]
        full_tmp_results_df = pd.DataFrame()
        full_tmp_ess_df = pd.DataFrame()
        for replicate_ind in work[work["Metadata_pert_iname"] == eq].index:
            tmp_work = work[work.index != replicate_ind].copy()
            used_replicates_ind = tmp_work[tmp_work["Metadata_control_type"] != "negcon"].index
            X = np.array(get_featuredata(tmp_work), dtype=float)
            Y = np.array(tmp_work["Metadata_control_type"] != 'negcon', dtype=float)
            pls = pyPLS.pls(X, Y, ncp=ncp)
            tmp_test = pred_df.copy()
            if eq in tmp_test.Metadata_pert_iname.unique().tolist():
                for uri in used_replicates_ind:
                    tmp_test.drop(uri, axis=0, inplace=True)
            tmp_results_df = get_metadata(tmp_test)
            tmp_ess_df = get_metadata(tmp_test)

            results, stats = pls.predict(np.array(get_featuredata(tmp_test), dtype=float), statistics=True)
            tmp_results_df[f"{eq}_eq"] = results
            full_tmp_results_df = pd.concat([full_tmp_results_df, tmp_results_df], axis=0)
            tmp_ess_df[f"{eq}_SSE"] = stats['ESS']
            full_tmp_ess_df = pd.concat([full_tmp_ess_df, tmp_ess_df], axis=0)
        
        results_df = pd.concat([results_df, get_featuredata(full_tmp_results_df).groupby(level=0).mean()], axis=1)
        ess_df = pd.concat([ess_df, get_featuredata(full_tmp_ess_df).groupby(level=0).mean()], axis=1)
    return results_df, ess_df

def calc_eq_score_df_with_cv_all(fit_df, pred_df, ncp=10):
    results_df = get_metadata(pred_df)
    ess_df = get_metadata(pred_df)
    eqs1 = fit_df['Metadata_pert_iname'].unique().tolist()
    eqs2 = fit_df['Metadata_pert_iname'].unique().tolist()
    for eq1 in eqs1:
        eqs2.remove(eq1)
        for eq2 in eqs2:
            work = fit_df[(fit_df["Metadata_pert_iname"] == eq1) | (fit_df["Metadata_pert_iname"] == eq2)]
            full_tmp_results_df = pd.DataFrame()
            full_tmp_ess_df = pd.DataFrame()
            # for replicate_ind in work[work["Metadata_pert_iname"] == eq1].index:
            if 'negcon' in work.Metadata_control_type.unique().tolist():
                negcon_ind = work[work['Metadata_control_type'] == 'negcon'].index
                len_work = len(work[work['Metadata_control_type'] != 'negcon'])
                negcon = True
            else:
                negcon = False
            for i, replicate_ind in enumerate(work[work['Metadata_control_type'] != 'negcon'].index):
                if negcon:
                    drop_nc_ind = negcon_ind[range(i, len(negcon_ind)-len_work+i), len_work]
                    drop_ind = [drop_nc_ind, replicate_ind]
                else:
                    drop_ind = replicate_ind
                tmp_work = work[work.index != drop_ind].copy()
                used_replicates_ind = tmp_work.index
                X = np.array(get_featuredata(tmp_work), dtype=float)
                Y = np.array(tmp_work["Metadata_pert_iname"] != eq1, dtype=float)
                pls = pyPLS.pls(X, Y, ncp=ncp)
                tmp_test = pred_df.copy()
                if eq1 in tmp_test.Metadata_pert_iname.unique().tolist() or eq2 in tmp_test.Metadata_pert_iname.unique().tolist():
                    for uri in used_replicates_ind:
                        tmp_test.drop(uri, axis=0, inplace=True)
                tmp_results_df = get_metadata(tmp_test)
                tmp_ess_df = get_metadata(tmp_test)

                results, stats = pls.predict(np.array(get_featuredata(tmp_test), dtype=float), statistics=True)
                tmp_results_df[f"{eq}_eq"] = results
                full_tmp_results_df = pd.concat([full_tmp_results_df, tmp_results_df], axis=0)
                tmp_ess_df[f"{eq}_SSE"] = stats['ESS']
                full_tmp_ess_df = pd.concat([full_tmp_ess_df, tmp_ess_df], axis=0)
        
        results_df = pd.concat([results_df, get_featuredata(full_tmp_results_df).groupby(level=0).mean()], axis=1)
        ess_df = pd.concat([ess_df, get_featuredata(full_tmp_ess_df).groupby(level=0).mean()], axis=1)
    return results_df, ess_df


def calc_eq_score_df_with_cv_ep(fit_df, pred_df, ncp=1):
    results_df = get_metadata(pred_df)
    ess_df = get_metadata(pred_df)
    eqs = fit_df[fit_df['Metadata_control_type'] != 'negcon']['Metadata_pert_iname'].unique().tolist()
    negcon_mean = np.mean(np.array(get_featuredata(fit_df[fit_df['Metadata_control_type'] == 'negcon']), dtype=float), axis=0)
    print(negcon_mean.max(), negcon_mean.mean())
    for eq in eqs:
        work = fit_df[(fit_df["Metadata_pert_iname"] == eq)]
        # work = work - fit_df["Metadata_control_type"] == 'negcon']
        full_tmp_results_df = pd.DataFrame()
        full_tmp_ess_df = pd.DataFrame()
        for replicate_ind in work[work["Metadata_pert_iname"] == eq].index:
            tmp_work = work[work.index != replicate_ind].copy()
            used_replicates_ind = tmp_work[tmp_work["Metadata_control_type"] != "negcon"].index
            X = np.array(get_featuredata(tmp_work), dtype=float) - negcon_mean
            X = np.concatenate([X, np.zeros(X.shape)], axis=0)
            Y = np.array(tmp_work["Metadata_control_type"] != 'negcon', dtype=float)
            Y = np.concatenate([Y, np.zeros(Y.shape)], axis=0)
            pls = pyPLS.pls(X, Y, ncp=ncp, scaling=1)
            tmp_test = pred_df.copy()
            if eq in tmp_test.Metadata_pert_iname.unique().tolist():
                for uri in used_replicates_ind:
                    tmp_test.drop(uri, axis=0, inplace=True)
            tmp_results_df = get_metadata(tmp_test)
            tmp_ess_df = get_metadata(tmp_test)

            results, stats = pls.predict(np.array(get_featuredata(tmp_test) - negcon_mean, dtype=float), statistics=True)
            tmp_results_df[f"{eq}_eq"] = results
            full_tmp_results_df = pd.concat([full_tmp_results_df, tmp_results_df], axis=0)
            tmp_ess_df[f"{eq}_SSE"] = stats['ESS']
            full_tmp_ess_df = pd.concat([full_tmp_ess_df, tmp_ess_df], axis=0)
        
        results_df = pd.concat([results_df, get_featuredata(full_tmp_results_df).groupby(level=0).mean()], axis=1)
        ess_df = pd.concat([ess_df, get_featuredata(full_tmp_ess_df).groupby(level=0).mean()], axis=1)
    return results_df, ess_df,


def calc_corr_score_df(crisp_df, comp_crisp_df):
    results_df = comp_crisp_df.iloc[:, :26].copy()
    err_df = comp_crisp_df.iloc[:, :26].copy()
    for gene, genes in crisp_df.groupby('Metadata_pert_iname'):
        if genes['Metadata_control_type'].unique() != 'negcon':
            corr_list = []
            gene_vals = np.array(genes.iloc[:, 26:].values, dtype=float)
            for pert, perts in comp_crisp_df.groupby('Metadata_pert_iname'):
                pert_vals = np.array(perts.iloc[:, 26:].values, dtype=float)
                corr=np.corrcoef(pert_vals, gene_vals, rowvar=True)
                corr = corr[len(pert_vals):, :len(pert_vals)]
                corr_list.extend(np.nanmedian(corr, axis=0))
            err_df[f"{gene}_corr"] = corr_list
            results_df[f"{gene}_corr"] = corr_list
    return results_df, err_df

def calc_comp_corr_score_df(crisp_df, comp_crisp_df):
    results_df = comp_crisp_df.iloc[:, :26].copy()
    err_df = comp_crisp_df.iloc[:, :26].copy()
    for gene, genes in crisp_df.groupby('Metadata_pert_iname'):
        if genes['Metadata_control_type'].unique() != 'negcon':
            corr_list = []
            corr_dict = {}
            gene_vals = np.array(genes.iloc[:, 26:].values, dtype=float)
            for pert, perts in comp_crisp_df.groupby('Metadata_pert_iname'):
                pert_vals = np.array(perts.iloc[:, 26:].values, dtype=float)
                corr = np.corrcoef(pert_vals, gene_vals, rowvar=True)
                corr = corr[len(pert_vals):, :len(pert_vals)]
                corr_list.extend(np.nanmedian(corr, axis=0))
                corr_dict.update(zip(perts.index, np.nanmedian(corr, axis=0)))  
            err_df[f"{gene}_corr"] = pd.Series(corr_dict)
            results_df[f"{gene}_corr"] = pd.Series(corr_dict)
    return results_df, err_df


def calc_top_eq_scores(results_df, crisp_df, df_targets, ess_df, n_top=10, feat_type='eq'):
    target_match_list = []
    both_targets = []
#     c_results = results_df[results_df['Metadata_experiment_type'] == 'Compound'].copy()
    for pert, perts in results_df.groupby('Metadata_pert_iname'):
        if f"{pert}_{feat_type}" in perts.columns:
            perts = perts.drop(f"{pert}_{feat_type}", axis=1)
        m_eq = perts.iloc[:, 26:].mean(axis=0).abs().copy()
        m_ess = ess_df.iloc[:, 26:].loc[perts.index].mean(axis=0).copy()
        top_cols = m_eq.nlargest(n=n_top)
        c_ess = m_ess[top_cols.keys()]
#         trts = cols.keys().tolist()
        top_matches = [x[:-3] for x in top_cols.keys().tolist()]
#         targets = df_targets[df_targets['pubchem_cid'] == perts['Metadata_pubchem_cid'].unique()[0]]['target_list'].unique()
        targets = df_targets[df_targets['pert_iname'] == perts['Metadata_pert_iname'].unique()[0]]['target_list'].unique()
        if targets:
            crisp_pert = []
            ind_crisp_pert = []
            crisp_ess = []
            for target in targets[0].split('|'):  
                crisp_pert.extend(crisp_df[crisp_df['Metadata_target'] == target]['Metadata_pert_iname'].unique().tolist())
            if crisp_pert:
                target_match_list.insert(-1, any(x in top_matches for x in crisp_pert))
                control_t = perts['Metadata_control_type'].unique()[0]
                cps = [f"{x}_{feat_type}" for x in crisp_pert]
                both_targets.append([top_matches, m_eq[top_cols.keys()].tolist(), c_ess.tolist(), crisp_pert, targets[0].split('|'), pert, control_t])
                #both_targets.append([top_matches, m_eq[top_cols.keys()].tolist(), c_ess.tolist(), crisp_pert, targets[0].split('|'), m_eq[cps].tolist(), pert, control_t])
        else: 
            print('No match', targets, perts['Metadata_pubchem_cid'].unique()[0], pert)
    percent_top = np.sum(target_match_list)/len(target_match_list)
    #cmp_gene_df = pd.DataFrame(both_targets, columns=['Top_ranked_genes', f"{feat_type}_Score", 'ESS', 'Target_genes', 'Targets', 'Target_eq', 'Compound', 'Control_type'])
    cmp_gene_df = pd.DataFrame(both_targets, columns=['Top_ranked_genes', f"{feat_type}_Score", 'ESS', 'Target_genes', 'Targets', 'Compound', 'Control_type'])
    def color_rows(s):
        if any(x in s.Target_genes for x in s.Top_ranked_genes):
            return ['background-color: #5eae76']*len(s)
        else:
            return ['background-color: #de796e']*len(s)
    cmp_gene_df.style.apply(color_rows, axis=1)
    return cmp_gene_df, percent_top


def pairwise_median_comp_gene_eqs(comp_res_df, crisp_df, feat_type='eq'):
    non_pair_val_pls = []
    non_pair_medians = []
    pair_medians = []
    pair_val_pls = []
    for target, targets in comp_res_df.groupby("Metadata_target"):
#         print('____________ \n Target:', target)
        pert_names = targets["Metadata_pert_iname"].unique().tolist()
        pair_median = []
        pair_val = 0
        if (len(pert_names) > 1) & (target != 'NO_SITE'):
            for pert, perts in targets.groupby("Metadata_pert_iname"):
#                 print('Pert:', pert)
                genes = crisp_df[(crisp_df['Metadata_target'] == target)]['Metadata_pert_iname'].unique()
                for gene in genes:
                    if gene == pert:
                        pass
                    else:
    #                     print('Gene:', gene)
                        if abs(np.median(pair_val)) < abs(np.mean(targets[targets["Metadata_pert_iname"] != pert][f"{gene}_{feat_type}"].values)):
                            pair_val = (targets[targets["Metadata_pert_iname"] != pert][f"{gene}_{feat_type}"].values)
                            pair_median = np.median(targets[targets["Metadata_pert_iname"] != pert][f"{gene}_{feat_type}"].values)
            pair_val_pls.extend(pair_val)
            pair_medians.append(pair_median)
    return pair_medians, pair_val_pls


def pairwise_median_comp_comp_eqs(comp_res_df, feat_type='eq'):
    non_pair_val_pls = []
    non_pair_medians = []
    pair_medians = []
    pair_val_pls = []
    for target, targets in comp_res_df.groupby("Metadata_target"):
        pert_names = targets["Metadata_pert_iname"].unique().tolist()
        pair_median = []
        pair_val = 0
        if (len(pert_names) > 1) & (target != 'NO_SITE'):
            match_perts = comp_res_df[(comp_res_df['Metadata_target'] == target)]['Metadata_pert_iname'].unique()
            for match_pert in match_perts:
                if abs(np.median(pair_val)) < abs(np.median(targets[targets["Metadata_pert_iname"] != match_pert][f"{match_pert}_{feat_type}"].values)):
                    pair_val = (targets[targets["Metadata_pert_iname"] != match_pert][f"{match_pert}_{feat_type}"].values)
                    pair_median = np.median(targets[targets["Metadata_pert_iname"] != match_pert][f"{match_pert}_{feat_type}"].values)
            pair_val_pls.extend(pair_val)
            pair_medians.append(pair_median)
    return pair_medians, pair_val_pls


def nonpairwise_median_comp_gene_eqs(comp_res_df, crisp_df, n_samples=1000, seed=9000, feat_type='eq'):
    n_samples_median = []
    pert_names = comp_res_df[comp_res_df["Metadata_control_type"] != "negcon"]["Metadata_pert_iname"].unique()
    random.seed(seed)
    while len(n_samples_median) < n_samples:
        pert_name = random.choices(pert_names, k=2)
        if (pert_name[0] != pert_name[1]) & (comp_res_df[comp_res_df["Metadata_pert_iname"] == pert_name[0]]["Metadata_target"].unique() != (comp_res_df[comp_res_df["Metadata_pert_iname"] == pert_name[1]]["Metadata_target"].unique())):
            target = comp_res_df[comp_res_df['Metadata_pert_iname'] == pert_name[0]]['Metadata_target'].unique()
            genes = crisp_df[crisp_df['Metadata_target'] == target[0]]['Metadata_pert_iname'].unique()
            n_samples_median.append(np.median(comp_res_df[comp_res_df["Metadata_pert_iname"] == pert_name[1]][f"{genes[0]}_{feat_type}"].values))
    return n_samples_median


def plot_percent_95_5_strong_df(df=None, null=None, corr=None, 
                               metadata_common='Metadata_target',
                               metadata_perturbation="Metadata_pert_iname", n_samples=1000,
                               xlab='xlabel', cumul=False, n_bins=20):
    plt.style.use("seaborn-ticks")
    plt.rcParams["image.cmap"] = "Set1"
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    corr_df = pd.DataFrame()
    if not null:
        print('yo')
        compound_df = utils.remove_negcon_empty_wells(df)
        corr = utils.corr_between_perturbation_pairs(df=compound_df, metadata_common="Metadata_target", metadata_perturbation="Metadata_pert_iname")
        null = utils.corr_between_perturbation_non_pairs(df=compound_df, n_samples=n_samples, metadata_common="Metadata_target", metadata_perturbation="Metadata_pert_iname")
    cell_type = df['Metadata_cell_line'].unique().tolist()[0]
    experiment_time = df['Metadata_timepoint'].unique().tolist()[0]
    prop_5_95, value_95, value_5 = utils.percent_score(null, corr, how='both')
    corr_df = corr_df.append({'Experiment':f'Compound_{cell_type}_{experiment_time}',
                                      'Corr':corr,
                                      'Null':null,
                                      'Percent_Strong':'%.3f'%prop_5_95,
                                      'Value_5':value_5,
                                      'Value_95':value_95}, ignore_index=True)
    fs = [18, 9]
    plt.figure(figsize=fs)
    dens = True
    plt.hist(corr_df.loc[0, 'Null'], label='non-pairs', density=dens, bins=n_bins, alpha=0.5, cumulative=cumul)
    plt.hist(corr_df.loc[0, 'Corr'], label='pairs', density=dens, bins=n_bins, alpha=0.5, cumulative=cumul)
    plt.axvline(corr_df.loc[0,'Value_95'], label='95% threshold')
    plt.axvline(corr_df.loc[0,'Value_5'], label='5% threshold', color='blue')
    plt.legend()
    plt.title(
        f"Experiment = {corr_df.loc[0,'Experiment']}\n" +
        f"Percent Strong Match = {corr_df.loc[0,'Percent_Strong']}"
    )
    plt.ylabel("density")
    plt.xlabel(xlab)
    sns.despine()
    return plt.gca()

               
class ClusterPlot():

    def _get_interpolate_plot_(self, points, interpolate=False):
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices,0],
                           points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                           points[hull.vertices,1][0])
        if interpolate:
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull], 
                                            u=dist_along, s=0)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)
        if not interpolate:
            interp_x, interp_y = (x_hull, y_hull)
        return(interp_x, interp_y)

    def _get_lines_plot_(self, points):
        center = np.sum(points, axis=0)/len(points[:, 0])
        cen_x, cen_y = center
        x = [points[:, 0], cen_x*np.ones(len(points))]
        y = [points[:, 1], cen_y*np.ones(len(points))]
        linexy = (x, y)
        cenxy = (cen_x, cen_y)
        return (cenxy, linexy)
    
    def plot_fig_with_clusters(self, df, metadata_grouping, x_axis, y_axis, figure_size=(15, 10), 
                              markersize=15, add_df=None, plot_add=False, plot_add_circle=False, bw=False, ax=None):
        if bw:
            cm = plt.get_cmap('Greys')
            abcs = list(string.ascii_lowercase)
            abc_i = 0
        else:
            cm = plt.get_cmap('tab20')
        n_colors = len(df[metadata_grouping].unique())
        if add_df is not None:
            n_colors = len(df[metadata_grouping].unique()) + len(add_df[metadata_grouping].unique())
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[cm(1.*i/n_colors) for i in range(n_colors)])
        aa = 0.2
        if ax == None:
            fig, ax = plt.subplots()
        
#         if plot_add:
#             fig, axs = plt.subplots(1, 2, figsize=figure_size)
        for pert, perts in df.groupby(metadata_grouping):
            points = perts[[x_axis, y_axis]].values
            interp_x, interp_y = self._get_interpolate_plot_(points)
            cenxy, linexy = self._get_lines_plot_(points)
            if not bw:
                ax.fill(interp_x, interp_y, alpha=aa)
                p = ax.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize)
                ax.plot(linexy[0], linexy[1], c=p[0].get_color(), alpha=aa*2)
                ax.plot(perts[x_axis], perts[y_axis], '.', label=pert, c=p[0].get_color(), markersize=markersize)
            elif bw:
                all_lines = [np.sqrt(x**2 + y**2) for x, y in zip(linexy[0]-cenxy[0], linexy[1]-cenxy[1])]
                longest_line = np.max(all_lines)
                plt.text(cenxy[0]-longest_line*0.8, cenxy[1]+longest_line, f"{abcs[abc_i]}")            
                plt.fill(interp_x, interp_y, alpha=aa, c='black')
                p = plt.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize, c='black')
                plt.plot(linexy[0], linexy[1], c='black', alpha=aa*2)
                plt.plot(perts[x_axis], perts[y_axis], '.', label=f"{abcs[abc_i]} - {pert}", c='black', markersize=markersize)
                abc_i+=1
#             plt.title('Uncentered PCA using pca_with_svd')
        if add_df is not None and plot_add:
            for pert, perts in add_df.groupby(metadata_grouping):
                    points = perts[[x_axis, y_axis]].values
                    interp_x, interp_y = self._get_interpolate_plot_(points)
                    cenxy, linexy = self._get_lines_plot_(points)
                    if not bw:
                        plt.fill(interp_x, interp_y, alpha=aa)
                        p = plt.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize)
                        plt.plot(linexy[0], linexy[1], c=p[0].get_color(), alpha=aa*2)
                        plt.plot(perts[x_axis], perts[y_axis], '.', label=pert, c=p[0].get_color(), markersize=markersize)
                    elif bw:
                        all_lines = [np.sqrt(x**2 + y**2) for x, y in zip(linexy[0]-cenxy[0], linexy[1]-cenxy[1])]
                        longest_line = np.max(all_lines)
                        plt.text(cenxy[0]-longest_line*0.8, cenxy[1]+longest_line, f"{abcs[abc_i]}")       
                        plt.fill(interp_x, interp_y, alpha=aa, c='black')
                        p = plt.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize, c='black')
                        plt.plot(linexy[0], linexy[1], c='black', alpha=aa*2)
                        plt.plot(perts[x_axis], perts[y_axis], '.', label=f"{abcs[abc_i]} - {pert}", c='black', markersize=markersize)
                        abc_i+=1                   
                    if plot_add_circle:
                        all_lines = [np.sqrt(x**2 + y**2) for x, y in zip(linexy[0]-cenxy[0], linexy[1]-cenxy[1])]
                        longest_line = np.max(all_lines)
                        if bw:
                            circ = plt.Circle(cenxy, longest_line*1.2, color='black', fill=False)
                        elif not bw:
                            circ = plt.Circle(cenxy, longest_line*1.2, color='red', fill=False)
                        plt.gca().add_patch(circ)
        ax.legend(loc=(1.05,0))
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        return plt.gca()

    def plot_fig_with_mean_clusters(self, df, metadata_grouping, x_axis, y_axis, figure_size=(15, 10)):
        cm = plt.get_cmap('tab20')
        n_colors = len(df[metadata_grouping].unique())
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[cm(1.*i/n_colors) for i in range(n_colors)])
        aa = 0.2
        fig = plt.figure(figsize=figure_size)
        negcon_points = df[df['Metadata_pert_iname'] == 'DMSO'][[x_axis, y_axis]].values
        negcon_center = np.sum(negcon_points, axis=0)/len(negcon_points[:, 0])
        cen_dict = {}
        dist_dict = {}
        mean_dist_dict = {}
        for pert, perts in df.groupby('Metadata_pert_iname'):
            if pert == 'DMSO':
                p = plt.plot(negcon_center[0], negcon_center[1], marker='o')#, c='black')
                points = perts[[x_axis, y_axis]].values
                cenxy, linexy = self._get_lines_plot_(points)
                mean_dist_dict.update({pert:  np.mean(np.sqrt(((linexy[0][0]-cenxy[0]))**2 + (linexy[1][0]-cenxy[1])**2))})
                circ = plt.Circle(cenxy, mean_dist_dict[pert], color=p[0].get_color())
                plt.gca().add_patch(circ)
            else:
                points = perts[[x_axis, y_axis]].values
                cenxy, linexy = self._get_lines_plot_(points)
                cen_dict.update({pert: cenxy - negcon_center})
                dist_dict.update({pert:  np.sqrt(cen_dict[pert][0]**2 + cen_dict[pert][1]**2)})
                mean_dist_dict.update({pert:  np.mean(np.sqrt(((linexy[0][0]-cenxy[0]))**2 + (linexy[1][0]-cenxy[1])**2))})
                p = plt.plot(cenxy[0], cenxy[1], marker='o')
                circ = plt.Circle(cenxy, mean_dist_dict[pert], alpha=aa, color=p[0].get_color())
                plt.gca().add_patch(circ)
                plt.plot([negcon_center[0], cenxy[0]], [negcon_center[1], cenxy[1]], '-', label=pert, c=p[0].get_color())
#                 plt.title('Values from Centered PCA using pyPLS.pca')
        plt.legend(loc=(1.05,0))
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        return plt.gca()
