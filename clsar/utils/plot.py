import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
sns.set(style='white',  font='sans-serif', font_scale=2)
    
    

def plot_dfa_save(dfa, save_dir):

    dfa.alpha = dfa.alpha.apply(lambda x:'%.e' % float(x))
    fig, ax = plt.subplots(figsize=(8, 6))
    #im = ax.scatter(dfa.alpha, dfa.rmse, c=dfa.rmse, cmap='jet', s=200)
    #colors = sns.color_palette('jet_r', len(dfa)).as_hex()

    ax.plot(dfa.alpha, dfa.rmse)
    ax.scatter(dfa.alpha, dfa.rmse, c=dfa.rmse, cmap='rainbow', s=200, alpha = 1)
    ax.errorbar(dfa.alpha, dfa.rmse, yerr=dfa.rmse_err, capsize = 6, ecolor='#8c8c8c', color = '#999999')

    
    ax.set_xlabel('alpha')
    ax.set_ylabel('CV RMSE')

    ax.tick_params(left='off',  bottom='off', pad=.5,)
    ax.set_title('Alpha performance')

    ax.tick_params(axis='x', labelrotation=60)

    fig.savefig(os.path.join(save_dir,'alpha_performance.png'), dpi=300, bbox_inches='tight')
    dfa.to_csv(os.path.join(save_dir,'alpha_performance.csv'))
    

    
def plot_dfc_save(dfc, save_dir):

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.scatter(dfc.cl, dfc.cu, c=dfc.rmse, cmap='jet', s=200)
    ax.set_xlabel('cliff lower')
    ax.set_ylabel('cliff upper')
    cbar2 = fig.colorbar(im, ax=ax, aspect=40, pad=0.02)
    cbar2.set_label('CV RMSE', rotation=90)
    fig.tight_layout()
    ax.tick_params(left='off',  bottom='off', pad=.5,)
    vm = ax.get_ylim()[1]
    ax.set_xlim(0, vm)
    ax.set_ylim(0.1, vm)
    ax.set_title('Cliff performance')

    fig.savefig(os.path.join(save_dir,'cliff_performance.png'), dpi=300, bbox_inches='tight')
    dfc.to_csv(os.path.join(save_dir,'cliff_performance.csv'))
    
    
def plot_dfp_save(dfp1, save_dir):
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib import ticker

    dfp1['trps2'] = dfp1.trps.replace(0, np.nan)
    dfp1['trps2'] = np.log2(dfp1['trps2'])
    ticks = list(np.arange(dfp1.lower.min(), dfp1.lower.max() + 0.3, 0.5).round(2))
    v = dfp1.trps2.dropna().sort_values().astype(int)
    bds = [0]

    bds1 = np.linspace(0, v.max(), 20)
    bds.extend(bds1)
    bds = pd.Series(bds).astype(float).to_list()

    base_cmaps = ['Greys', 'gist_ncar_r'] #gist_ncar_r 

    n_base = len(base_cmaps)
    N=[1, len(bds1)]# number of colors  to extract from each cmap, sum(N)=len(classes)
    colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.1, 0.9, N[i])) for i,name in zip(range(n_base),base_cmaps)])
    cmap = ListedColormap(colors)
    boundary_norm = BoundaryNorm(bds, cmap.N)
    fig, ax = plt.subplots(figsize=(9, 7))
    s = 70
    marker = 'o'
    lw = 0

    im = ax.scatter(x = dfp1.lower, 
                     y = dfp1.upper, #vmax = dfp1.trps.max(), 
                     c = np.log2(dfp1.trps+1), 
                     norm = boundary_norm, #marker = ',',
                     marker = marker,
                     edgecolors='k', 
                     lw=lw, 
                     s = s, cmap= cmap, label = 'trps2')

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    fmt.format = '$\\mathdefault{%1.2f}$'
    cbar = fig.colorbar(im, ax=ax, aspect=40, pad = 0.02, format = fmt,)# 
    cbar.set_label('Log2 No. of mined triplets')

    ax.tick_params(left='off',  bottom='off', pad=.3,)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title('Cliff vs. Triplets')
    ax.set_ylabel('cliff upper')
    ax.set_xlabel('cliff lower')
    fig.tight_layout()
    
    fig.savefig(os.path.join(save_dir,'triplets_distribution.png'), dpi=300, bbox_inches='tight')
    dfp1.to_csv(os.path.join(save_dir,'triplets_distribution.csv'))
    
    
def plot_dfe_save(dfe, save_dir):

    best_epoch = dfe.idxmin()
    
    vmax = dfe.max()#-1
    vmin = dfe.min()-0.2

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = dfe.plot()
    ax.set_xlabel('epochs')
    ax.set_ylabel('CV RMSE')
    
    ax.vlines(best_epoch, 0, 100, ls = '--', color = 'red', lw = 2)
    ax.text(best_epoch, 0.0, '%s' % best_epoch, color='red', transform=ax.get_xaxis_transform(),
            ha='center', va='top')
         
    fig.tight_layout()
    ax.tick_params(left='off',  bottom='off', pad=.5,)

    ax.set_ylim(vmin, vmax)
    
    ax.set_title('CV performance')

    fig.savefig(os.path.join(save_dir,'epoch_performance.png'), dpi=300, bbox_inches='tight')
    dfe.to_csv(os.path.join(save_dir,'epoch_performance.csv'))