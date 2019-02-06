import streamlit as st

from adult_uci import ingest_data
from adult_uci import explore_data
from adult_uci import process_data

st.title('Eliminating Bias in Machine Learning')

filename = 'adult-data.csv'

data_df = ingest_data(filename)

pos_bias_labels = explore_data(data_df)

n_train = 30000
process_data(data_df, pos_bias_labels, filename, n_train)
