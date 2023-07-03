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
