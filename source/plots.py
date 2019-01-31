import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_heatmap(data_df):
    corr_df = data_df.corr()
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    fig, ax = plt.subplots(figsize=(30, 25))
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=1.0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    fig.savefig('../figures/correlation-heat-map.png')

def probability_density_function(y_pred, z_test, target_col, bias_name, bias_types, filename):
    sns.set()
    for i in [0,1]:
        ax = sns.distplot(y_pred[z_test==i], hist=False, label=bias_types[i])
    ax.set_xlim(0,1)
    ax.set_title('Model Prediction Probability Density Function')
    ax.set_ylabel('density')
    ax.set_xlabel('P('+target_col+'|'+bias_name+')')
    fig = ax.get_figure()
    fig.savefig('../figures/'+filename+'.png')
