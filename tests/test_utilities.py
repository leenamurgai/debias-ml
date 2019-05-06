from utilities import write_params_to_file
from utilities import DataParams
from utilities import ProcessedDataParams

def test_DataParams():
    my_params = DataParams('config/adult-data_params.ini')
    assert my_params.filename == 'adult-data.csv'
    assert my_params.sensitive_features == ['sex', 'race']
    assert my_params.target_feature == 'ann_salary'
    assert my_params.pos_target == '>50K'
    assert my_params.n_train == 30000

def test_ProcessedDataParams(tmp_path):

    bias_cols = ['sex is Male', 'race is White-Asian']
    target_col = 'ann_salary is >50K'
    bias_col_types = [['Female', 'Male'], ['not White-Asian', 'White-Asian']]
    categories = {'key1': ['val11', 'val12'], 'key2': ['val21', 'val22']}
    write_params_to_file(bias_cols, target_col, bias_col_types, categories,
                         tmp_path/'test.ini')

    #my_params = ProcessedDataParams('config/test.ini')
    my_params = ProcessedDataParams(tmp_path/'test.ini')
    assert my_params.bias_cols == ['sex is Male', 'race is White-Asian']
    assert my_params.target_col == 'ann_salary is >50K'
    assert my_params.bias_col_types == [['Female', 'Male'], ['not White-Asian', 'White-Asian']]
    assert my_params.categories == {'key1': ['val11', 'val12'], 'key2': ['val21', 'val22']}
