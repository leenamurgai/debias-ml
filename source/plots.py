import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from data import categories_to_columns

sns.set()

def target_by_bias_table_histogram(target_by_bias_df, target_name, bias_name, figname):
    st.write(target_by_bias_df)

    bias_names = list(target_by_bias_df)
    pos_target = target_by_bias_df.index[1]

    st.write('Bias factor = ', target_by_bias_df.iloc[1].max()/target_by_bias_df.iloc[1].min())

    target_by_bias_df.rename(index=str, columns={'Asian-Pac-Islander':'Asian', 'Amer-Indian-Eskimo':'Am-In-Es'}, inplace=True)

    target_by_bias_df.loc[pos_target].plot.bar()
    plt.xticks(rotation='horizontal')
    plt.xlabel(bias_name)
    plt.ylabel('Proportion '+target_name+pos_target)
    plt.savefig('../figures/'+figname+'-hist.png')
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
    plt.savefig('../figures/'+figname+'.png')
    st.pyplot()
    plt.gcf().clear()
    plt.close()


def probability_density_function(y_pred, z_test, target_name, bias_name, categories, figname):
    categories_col = categories_to_columns(categories)
    target_col = categories_col[target_name][1]
    for i in [0,1]:
        ax = sns.distplot(y_pred[z_test==i], hist=False, label=categories[bias_name][i])
    ax.set_xlim(0,1)
    ax.set_title('Model Prediction Probability Density Function')
    ax.set_ylabel('density')
    ax.set_xlabel('P('+target_col+'|'+bias_name+')')
    fig = ax.get_figure()
    fig.savefig('../figures/'+figname+'-dist.png')
    st.pyplot()
    plt.gcf().clear()
    plt.close()
    get_bias_factor(y_pred, z_test, target_name, bias_name, categories, figname)

def probability_density_functions(y_pred, Z_test, target_name, bias_names, categories, figname):
    categories_col = categories_to_columns(categories)
    bias_cols = [categories_col[b][1] for b in bias_names]
    for i, b in enumerate(bias_names):
        probability_density_function(y_pred, Z_test[bias_cols[i]], target_name, b, categories, figname+'-'+b)
    z_test = np.logical_and( Z_test[bias_cols[0]]==1 , Z_test[bias_cols[1]]==1 )
    bias_name = bias_names[0]+' & '+bias_names[1]
    joint_bias = categories[bias_names[0]][0]+' & '+categories[bias_names[1]][0]
    bias_types = [ joint_bias , 'others' ]
    joint_dict = {target_name: categories[target_name], bias_name: bias_types}
    probability_density_function(y_pred, z_test, target_name, bias_name, joint_dict, figname+'-'+bias_names[0]+'-'+bias_names[1])


def get_bias_factor(y_pred, z_test, target_name, bias_name, categories, figname):
    temp = [[0,0],[0,0]]
    for i in [0,1]:
        for j in [0,1]:
            temp[i][j] = np.logical_and((y_pred>0.5)==i, z_test==j).sum()
    target_by_bias_df = pd.DataFrame(data = temp, index = categories[target_name], columns=categories[bias_name])
    #st.write(target_by_bias_df.sum())
    #st.write(target_by_bias_df.sum().sum())
    target_by_bias_df = target_by_bias_df/target_by_bias_df.sum()
    target_by_bias_table_histogram(target_by_bias_df, target_name, bias_name, figname)
