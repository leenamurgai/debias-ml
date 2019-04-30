import pandas as pd

from plots import pie_chart
from plots import target_by_bias_table_histogram

class Data:
    def __init__(self,
                 data_df,
                 sensitive_features = ['sex', 'race'],
                 target_feature = 'ann_salary',
                 pos_target = '>50K',
                 prefix_sep = ' is '):

        self.sensitive_features = sensitive_features
        self.target_feature = target_feature
        self.pos_target = pos_target
        self.prefix_sep = prefix_sep

        self.bias_names, self.target_names = self.getNamesNumbers(data_df)
        self.pos_bias_labels = self.getPosBiasLabels(data_df)

        self.data_df, self.categories = self.preprocess_data(data_df)
        self.binarise_bias_cols()
        self.remove_redundant_cols()
        self.categories_col = self.categories_to_columns()
        self.bias_col_types   = [self.categories[b] for b in self.sensitive_features]
        self.bias_cols        = [self.categories_col[b][1] for b in self.sensitive_features]
        self.target_col_types = self.categories[self.target_feature]
        self.target_col       = self.categories_col[self.target_feature][1]
        self.feature_cols = list(self.data_df.columns)
        self.feature_cols.remove(self.target_col) # leave bias_cols in features
        self.move_target_col_to_end()


    def getNamesNumbers(self, data_df):
        # get sensitive catagories in no particular order
        bias_names = [None]*len(self.sensitive_features)
        for i, col in enumerate(self.sensitive_features):
            bias_names[i] = tuple(data_df[col].unique())

        # get target catagories and make sure they are in the right order
        target_names = tuple(data_df[self.target_feature].unique())
        if target_names[0] == self.pos_target:
            target_names = target_names[1], target_names[0]

        # print how many in each sensitive catagory there are in the data
        n_bias = [None]*len(self.sensitive_features)
        for j, col in enumerate(self.sensitive_features):
            n_bias[j] = [0]*len(bias_names[j])
            for i, b in enumerate(bias_names[j]):
                n_bias[j][i] = data_df[data_df[col] == b].shape[0]
                #st.write('Number of {} {} ='.format(col, b), n_bias[j][i])
            pie_chart(n_bias[j], bias_names[j], self.sensitive_features[j])

        # print how many in each target catagory there are in the data
        n_target = [0]*len(target_names)
        for i, t in enumerate(target_names):
            n_target[i] = data_df[data_df[self.target_feature] == t].shape[0]
            #st.write('Number of', target_feature, t, '= ', n_target[i])
        pie_chart(n_target, target_names, self.target_feature)

        return bias_names, target_names


    def getPosBiasLabels(self, data_df):
        """We transform each bias_name to a binary data type
        pos_bias_labels is a dict mapping each bias_name to a list of the types
        we associate to the postive label"""
        pos_bias_labels = {}
        columns = list(data_df)
        for k, col in enumerate(self.sensitive_features):
            pos_bias_labels[col] = []
            target_by_bias = data_df.groupby([col, self.target_feature]).count()[columns[0]]
            temp = [None]*len(self.target_names)
            for i, t in enumerate(self.target_names):
                temp[i] = [None]*len(self.bias_names[k])
                for j, b in enumerate(self.bias_names[k]):
                    temp[i][j] = target_by_bias[b, t]
            target_by_bias_df = pd.DataFrame(data = temp, index = self.target_names, columns = self.bias_names[k])
            target_by_bias_df = target_by_bias_df / target_by_bias_df.sum()

            # figure out how to split the bias categories so there are only 2 and ultimately 1 feature to train on
            mean_prop = target_by_bias_df.mean(axis = 1)
            for b in list(target_by_bias_df):
                if target_by_bias_df[b].loc[self.pos_target] > mean_prop[self.pos_target]:
                    pos_bias_labels[col].append(b)

            target_by_bias_table_histogram(target_by_bias_df, self.target_feature, col, 'original-data-'+col)

        return pos_bias_labels


    def preprocess_data(self, data_df):
        """returns data_df with catagorical features converted to binary and categories
        which is a dict whihch maps the original categorical features to the possible
        types"""
        out = pd.DataFrame(index=data_df.index)  # output dataframe, initially empty
        categories = {}
        for col, col_data in data_df.iteritems():
            # If non-numeric, convert to one or more dummy variables
            if col_data.dtype == object:
                categories[col] = list(data_df[col].fillna('Unknown').unique())
                col_data = pd.get_dummies(col_data, prefix=col, prefix_sep=self.prefix_sep)
            out = out.join(col_data)
        [v.remove('Unknown') for v in categories.values() if 'Unknown' in v]
        return out.fillna('Unknown'), categories


    def categories_to_columns(self):
        """returns data_df with catagorical features converted to binary and categories
        which is a dict whihch maps the original categorical features to the possible
        types"""
        categories_col = {}
        for k, v in self.categories.items():
            val = [k + self.prefix_sep + vi for vi in v]
            categories_col[k] = val
        return categories_col


    def binarise_bias_cols(self):
        categories_col = self.categories_to_columns()
        pos_bias_cols = {}
        for key in self.pos_bias_labels.keys():
            pos_bias_cols[key] = [key + self.prefix_sep + name for name in self.pos_bias_labels[key]]
            if len(self.pos_bias_labels[key])>1:
                new_col1 = key + self.prefix_sep
                new_col0 = key + self.prefix_sep + 'not '
                for i in self.pos_bias_labels[key]:
                    new_col1 = new_col1 + i + '-'
                    new_col0 = new_col0 + i + '-'
                new_col1 = new_col1[:-1]
                new_col0 = new_col0[:-1]
                for i in self.data_df.index:
                    self.data_df.at[i, new_col1] = 0
                    self.data_df.at[i, new_col0] = 0
                    if self.data_df[pos_bias_cols[key]].loc[i].sum()==1:
                        self.data_df.at[i, new_col1] = 1
                    else:
                        self.data_df.at[i, new_col0] = 1
                for col in categories_col[key]:
                    del self.data_df[col]
                self.categories[key] = [new_col0[len(key)+4:], new_col1[len(key)+4:]]


    def remove_redundant_cols(self):
        for k, v in self.categories.items():
            val = [k+' is '+vi for vi in v]
            col = self.data_df[val].sum().idxmin()
            if col == (self.target_feature+' is '+self.pos_target):
                col = self.data_df[val].sum().idxmax()
            the_val = col[(len(k)+4):]
            if v[0]!=the_val:
                v.remove(the_val)
                v.insert(0,the_val)
            del self.data_df[col]

    def move_target_col_to_end(self):
        target_df = self.data_df[self.target_col]
        del self.data_df[self.target_col]
        self.data_df[self.target_col] = target_df
