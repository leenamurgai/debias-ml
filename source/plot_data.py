import numpy as np
import seaborn as sns
import streamlit as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set()

def pie_chart(data, labels, label_name):
    sum_data = np.sum(data)
    pct_data = ['{:.1f}%'.format(100.0*d/sum_data) for d in data]
    new_labels = [labels[i]+' ('+pct_data[i]+')' for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts = ax.pie(data,textprops=dict(color="w"))
    ax.legend(wedges, new_labels,
              title=label_name,
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.savefig('../figures/'+label_name+'-pie.png', bbox_inches = 'tight')
    st.pyplot(bbox_inches='tight')
    plt.gcf().clear()
    plt.close()

def target_by_bias_table_histogram(target_by_bias_df, target_feature,
                                   sensitive_feature, figname):
    st.write(target_feature, ' proportions by ', sensitive_feature,':')
    st.write(target_by_bias_df)

    bias_names = list(target_by_bias_df)
    pos_target = target_by_bias_df.index[1]

    st.write('Bias factor (', sensitive_feature, ') = ',
             target_by_bias_df.iloc[1].max()/target_by_bias_df.iloc[1].min())

    target_by_bias_df.rename(index=str,
                             columns={'Asian-Pac-Islander':'Asian',
                                      'Amer-Indian-Eskimo':'Am-In-Es'},
                             inplace=True)

    target_by_bias_df.loc[pos_target].plot.bar()
    plt.xticks(rotation='horizontal')
    plt.xlabel(sensitive_feature)
    plt.ylabel('Proportion '+target_feature+pos_target)
    plt.savefig('../figures/'+figname+'-hist.png')
    st.pyplot()
    plt.gcf().clear()
    plt.close()
