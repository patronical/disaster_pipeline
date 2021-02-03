# buildvisuals.py
# builds visualizations of train data
# generates a png file for the front end

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_theme(style="whitegrid")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
register_matplotlib_converters()


def MessCountGraph(axs, data, title):
    '''
    import data and title
    build stats dataframe
    build visualization
    return plot axis
    '''
    df = data.copy()
    t_cols = df.columns.tolist()
    # visualize breakdown of classes
    counts = []
    counts.append(('not_related',(len(df)-df['related'].sum())))
    for t in t_cols:
        counts.append((t, df[t].sum()))
    df_stats = pd.DataFrame(counts, columns = ['category', 'counts'])
    # build visualization
    df_stats.plot(x='category',y='counts',kind='bar',
             legend=False, grid=True, figsize=(10,5),
             color=sns.color_palette(), ax=axs)
    axs.set_title(title)
    axs.set_ylabel('Counts')
    axs.set_xlabel('Category')
    
    return axs


def TokCountGraph(axs, data, label, num_bins):
    '''
    import axis, data, and title
    build visualization
    return plot axis
    '''
    # build plot
    sns.histplot(data, color = 'purple', bins=num_bins,
                 line_kws=dict(edgecolor="k", linewidth=1),
                 label=label, legend=True, ax=axs)
    axs.set_ylim(0,3700)
    axs.text(55, 1100,label, fontsize=9)

    return axs


def BuildFig(data, filename):
    '''
    import data
    build visualizations
    save figure as png file
    '''
    # Build Framework
    plt.close('all')
    fig = plt.figure(constrained_layout=False, 
                     figsize=(14, 2), dpi=150)
    gs = gridspec.GridSpec(2, 7, figure=fig)
    gs.update(wspace=0.01, hspace=0.05)
    
    # Build Message Counts
    ax3 = fig.add_subplot(gs[0:2, 0:4])
    ax3_label = 'Train Data: Message Counts'
    ax3 = MessCountGraph(ax3, data, ax3_label)
    
    # Build Token Counts Related
    ax1 = fig.add_subplot(gs[1,5:7])
    dftr = dft[dft.index.isin(data.index)].copy()
    data_ir = dftr[dftr['related']==1]['prep'].str.len()
    ax1_label = 'Related'
    ax1.set_xlabel('Tokens Per Message')
    ax1 = TokCountGraph(ax1, data_ir, ax1_label,22)
    left, right = ax1.get_xlim()
    
    # Build Token Counts Not Related
    ax2 = fig.add_subplot(gs[0,5:7])
    ax2.set_xlim(left,right)
    data_nr = dftr[dftr['related']==0]['prep'].str.len().tolist()
    ax2.set_title('Train Data: Token Counts')
    ax2_label = 'Not Related'
    ax2 = TokCountGraph(ax2, data_nr, ax2_label,15)
    
    # Save Visualizations
    plt.savefig(filename, bbox_inches='tight', dpi=150,
                facecolor='white', edgecolor='black')
    
    print('figure saved...') 