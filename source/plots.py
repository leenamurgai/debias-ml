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

    st.write('Bias factor (', bias_name, ')= ', target_by_bias_df.iloc[1].max()/target_by_bias_df.iloc[1].min())

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
    temp = get_bias_factor(y_pred, z_test, target_name, bias_name, categories)

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


def get_bias_factor(y_pred, z_test, target_name, bias_name, categories):
    temp = [[0,0],[0,0]]
    for i in [0,1]:
        for j in [0,1]:
            temp[i][j] = np.logical_and((y_pred>0.5)==i, z_test==j).sum()
    target_by_bias_df = pd.DataFrame(data = temp, index = categories[target_name], columns=categories[bias_name])
    #st.write(target_by_bias_df.sum())
    #st.write(target_by_bias_df.sum().sum())
    target_by_bias_df = target_by_bias_df/target_by_bias_df.sum()
    #st.write(target_by_bias_df)
    bias_factor = target_by_bias_df.iloc[1].max()/target_by_bias_df.iloc[1].min()
    st.write('Bias factor = ', bias_factor)
    return bias_factor


def plot_distributions(y, Z, target_name, bias_names, categories, factor, results_df, figname):
    categories_col   = categories_to_columns(categories)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    #legend={'race': ['not asian & white','asian & white'],
    #        'sex': ['female','male']}
    #for i, b in enumerate(Z.columns):
    for i, b in enumerate(bias_names):
        col = categories_col[b][1]
        results_df.at[factor, 'Bias factor: '+b] = get_bias_factor(y, Z[col], target_name, b, categories)
        for b_val in [0, 1]:
            ax = sns.distplot(y[Z[col]== b_val], hist=False,
                              kde_kws={'shade': True,},
                              label='{}'.format(categories[b][b_val]),
                              ax=axes[i])
        ax.set_xlim(0,1)
        ax.set_ylim(0,7)
        ax.set_yticks([])
        ax.set_title("sensitive attibute: {}".format(b))
        if i == 0:
            ax.set_ylabel('prediction distribution')
        ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(b))
    fig.text(1.0, 0.9, f"Training oversample factor #{factor}", fontsize='16')
    fig.text(1.0, 0.65, '\n'.join([#"Prediction performance:",
                                   #f"- ROC AUC: {results_df['ROC AUC']:.2f}",
                                   f"- Accuracy: {results_df['Accuracy'][factor]:.2f}",
                                   f"- F1 score: {results_df['F1 score'][factor]:.2f}",
                                   f"- Precision: {results_df['Precision'][factor]:.2f}",
                                   f"- Recall: {results_df['Recall'][factor]:.2f}"]), fontsize='14')
    fig.text(1.0, 0.4, '\n'.join(["Bias factor:"] +
                                 [f"- {b}: {results_df['Bias factor: '+b][factor]:.2f}" for b in bias_names]), fontsize='14')
    fig.tight_layout()
    plt.savefig('../figures/'+figname+'-dist.png', bbox_inches='tight')
    st.pyplot()
    plt.gcf().clear()
    plt.close()
