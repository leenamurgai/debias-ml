from data import categories_to_columns

def test_categories_to_columns():
    categories = {'key1': ['val11', 'val12'], 'key2': ['val21', 'val22']}
    assert categories_to_columns(categories) == {'key1': ['key1 is val11', 'key1 is val12'],
                                                 'key2': ['key2 is val21', 'key2 is val22']}
