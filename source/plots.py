import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def basic_stats(data_df, bias_col, target_col, pos_target):
    print('Number of data points =', len(data_df.index))
    print('Number of features =', len(data_df.columns)-1)

    target_names = tuple(data_df[target_col].unique())
    if target_names[0]!=pos_target:
        target_names = target_names[1], target_names[0]
    bias_names = tuple(data_df[bias_col].unique())

    n_target = [0]*len(target_names)
    for i, t in enumerate(target_names):
        n_target[i] = data_df[data_df[target_col] == t].shape[0]
        print('Number of', target_col, t, '= ', n_target[i])
    n_bias = [0]*len(bias_names)
    for i, b in enumerate(bias_names):
        n_bias[i] = data_df[data_df[bias_col] == b].shape[0]
        print('Number of {} {} = {}'.format(bias_col, b, n_bias[i]))

    columns = list(data_df)
    target_by_bias = data_df.groupby([bias_col, target_col]).count()[columns[0]]

    temp = [[0,0],[0,0]]
    for i, t in enumerate(target_names):
        for j, b in enumerate(bias_names):
            temp[i][j] = target_by_bias[b, t]
    target_by_bias_df = pd.DataFrame(data = temp, index = target_names, columns=bias_names)
    display(target_by_bias_df)
    target_by_bias_df.plot.bar()
    plt.legend(loc='upper left')
    plt.xticks(rotation='horizontal')
    plt.xlabel(target_col)
    plt.ylabel('count')

    print('')
    pos_bias_frac = [0.]*len(bias_names)
    for i, b in enumerate(bias_names):
        pos_bias_frac[i] = target_by_bias[b, pos_target]
        print('Proportion {} with {} {} = {:2.2%}'.format(b, target_col, pos_target, pos_bias_frac[i]/n_bias[i]))


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


def get_bias_factor(y_pred, z_test, target_name, bias_name, categories, filename):
    temp = [[0,0],[0,0]]
    for i in [0,1]:
        for j in [0,1]:
            temp[i][j] = np.logical_and((y_pred>0.5)==i, z_test==j).sum()
    target_by_bias_df = pd.DataFrame(data = temp, index = categories[target_name], columns=categories[bias_name])
    display(target_by_bias_df)

    display(target_by_bias_df.sum())
    print(target_by_bias_df.sum().sum())

    target_by_bias_df.plot.bar()
    #plt.legend(loc='upper left')
    plt.xticks(rotation='horizontal')
    plt.xlabel(target_name)
    plt.ylabel('count')
    plt.savefig('../figures/'+filename+'.png')

    pos_bias_frac = [0,0]
    prop = [0,0]
    for i, b in enumerate(categories[bias_name]):
        pos_bias_frac[i] = target_by_bias_df[b][categories[target_name][1]]
        prop[i] =  pos_bias_frac[i]/target_by_bias_df.sum()[i]
        print('Proportion {} with {} {} = {:2.2%}'.format(b, target_name, categories[target_name][1], prop[i]))
    bias_factor = prop[1]/prop[0]

    return bias_factor


def probability_density_function(y_pred, z_test, target_col, bias_name, bias_names, filename):
    sns.set()
    for i in [0,1]:
        ax = sns.distplot(y_pred[z_test==i], hist=False, label=bias_names[i])
    ax.set_xlim(0,1)
    ax.set_title('Model Prediction Probability Density Function')
    ax.set_ylabel('density')
    ax.set_xlabel('P('+target_col+'|'+bias_name+')')
    fig = ax.get_figure()
    fig.savefig('../figures/'+filename+'.png')
