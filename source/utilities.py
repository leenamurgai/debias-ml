from configparser import ConfigParser


def write_params_to_file(bias_cols, target_col, bias_col_types, categories):
    parser = ConfigParser()
    parser.add_section('new_data_values')
    parser.set('new_data_values', 'bias_cols', str(bias_cols))
    parser.set('new_data_values', 'target_col', str(target_col))
    parser.set('new_data_values', 'bias_col_types', str(bias_col_types))
    parser.add_section('categories')
    for k, v in categories.items():
        parser.set('categories', k, str(v))
    with open('../config/new_params.ini', 'w') as file:
        parser.write(file)


class DataParams:
    def __init__(self):
        parser = ConfigParser()
        parser.read('../config/params.ini')
        self.filename = parser.get('data_values', 'filename')
        self.sensitive_features  = parser.get('data_values', 'sensitive_features').split()
        self.target_feature = parser.get('data_values', 'target_feature')
        self.pos_target  = parser.get('data_values', 'pos_target')
        self.n_train = int(parser.get('train_values', 'n_train'))


class ProcessedDataParams:
    def __init__(self):
        self.parser = ConfigParser()
        self.parser.read('../config/new_params.ini')
        self.bias_cols = self.get_bias_cols()
        self.target_col = self.parser.get('new_data_values', 'target_col')
        self.bias_col_types = self.get_bias_col_types()
        self.categories = self.get_categories()

    def transform_string(self, string):
        new_string = ''
        for char in string:
            if not(char == "'" or char == '[' or char == ']'):
                new_string = new_string + char
        return new_string

    def get_bias_cols(self):
        temp = self.parser.get('new_data_values', 'bias_cols').split("', '")
        return [self.transform_string(s) for s in temp]

    def get_bias_col_types(self):
        temp = self.parser.get('new_data_values', 'bias_col_types').split("], [")
        temp = [t.split("', '") for t in temp]
        return [[self.transform_string(s) for s in t] for t in temp]

    def get_categories(self):
        categories = {}
        for k, v in self.parser.items('categories'):
            categories[k] = [self.transform_string(s) for s in v.split("', '")]
        return categories
