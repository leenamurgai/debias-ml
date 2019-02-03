import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


sns.set()

def target_by_bias_table_histogram(target_by_bias_df, target_name, figname):
    st.write(target_by_bias_df)

    bias_names = list(target_by_bias_df)
    pos_target = target_by_bias_df.index[1]
    n_bias = target_by_bias_df.sum()

    pos_bias_frac = [0.]*len(bias_names)
    prop = [0.]*len(bias_names)
    for i, b in enumerate(bias_names):
        pos_bias_frac[i] = target_by_bias_df[b][pos_target]
        prop[i] = pos_bias_frac[i]/n_bias[i]
        st.write('Proportion {} with {} {} = {:2.2%}'.format(b, target_name, pos_target, prop[i]))
    bias_factor = max(prop)/min(prop)
    st.write('bias factor = {:.2f}'.format(bias_factor))

    target_by_bias_df.plot.bar()
    plt.xticks(rotation='horizontal')
    plt.xlabel(target_name)
    plt.ylabel('count')
    plt.savefig('../figures/'+figname+'.png')
    st.pyplot()
    plt.gcf().clear()
    plt.close()


def heatmap(corr_df, figname):
    for i in corr_df.index:
        corr_df.at[i,i]=0
    st.write('Values range = [[{:.3f}, {:.3f}]]'.format(corr_df.min().min(), corr_df.max().max()))
    st.write('')
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(28,28))
    ax = sns.heatmap(corr_df, mask=mask, cmap=cmap, linewidths=.5, cbar=False)
    #sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=1.0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    #cbar = ax.collections[0].colorbar
    #cbar.ax.tick_params(labelsize=40)
    #ax.set_title(title)
    plt.savefig('../figures/'+figname+'.png')
    st.pyplot()
    plt.gcf().clear()
    plt.close()


def get_bias_factor(y_pred, z_test, target_name, bias_name, categories, figname):
    temp = [[0,0],[0,0]]
    for i in [0,1]:
        for j in [0,1]:
            temp[i][j] = np.logical_and((y_pred>0.5)==i, z_test==j).sum()
    target_by_bias_df = pd.DataFrame(data = temp, index = categories[target_name], columns=categories[bias_name])
    #st.write(target_by_bias_df.sum())
    #st.write(target_by_bias_df.sum().sum())
    target_by_bias_table_histogram(target_by_bias_df, target_name, figname)


def probability_density_function(y_pred, z_test, target_col, bias_name, bias_names, figname):
    for i in [0,1]:
        ax = sns.distplot(y_pred[z_test==i], hist=False, label=bias_names[i])
    ax.set_xlim(0,1)
    ax.set_title('Model Prediction Probability Density Function')
    ax.set_ylabel('density')
    ax.set_xlabel('P('+target_col+'|'+bias_name+')')
    fig = ax.get_figure()
    fig.savefig('../figures/'+figname+'.png')
    st.pyplot()
    plt.gcf().clear()
    plt.close()
