import pandas as pd

from rwanda.src.utils import get_repo_root

def impute_data(train, test):
    train['is_train'] = 1
    test['is_train'] = 0

    # Concat train and test
    concat_df = pd.concat([train, test], axis=0)

    # Fill nas by using the mean of the neighboring cells
    # First sort the data by ID_LAT_LON, year, week_no
    # Then fill nas by using the mean of the neighboring cells
    concat_df = concat_df.sort_values(by=['ID_LAT_LON', 'year', 'week_no'])
    concat_df = concat_df.fillna(method='ffill')
    concat_df = concat_df.fillna(method='bfill')

    # Separate train and test again
    train = concat_df[concat_df['is_train']==1]
    test = concat_df[concat_df['is_train']==0]
    test = test.drop(['is_train', 'emission'], axis=1)
    train = train.drop(['is_train'], axis=1)

    return train, test
