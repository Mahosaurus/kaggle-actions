import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rwanda.src.utils import get_repo_root

def reduce_dimenions(train, test):
    train['is_train'] = 1
    test['is_train'] = 0

    # Concat train and test
    concat_df = pd.concat([train, test], axis=0)

    transformer_dict = {
        '^Sulphur': 6,
        '^Nitrogen': 4,
        '^Ozone': 4,
        '^Carbon': 4,
        '^Cloud': 3,
        '^Uv': 3,
        '^Form': 5

    }

    for key, comps in transformer_dict.items():
        pca = PCA(n_components=comps)
        # Standardize concat_df.filter(regex=key)) columns
        tmp = StandardScaler().fit_transform(concat_df.filter(regex=key))
        # Fit the PCA model
        pca.fit(tmp)
        # Replace the original variables with the principal components
        components = pca.transform(tmp)
        ## Drop the original variables
        concat_df = concat_df.drop(concat_df.filter(regex=key).columns, axis=1)
        ## Add the principal components to the dataframe
        for i in range(comps):
            concat_df[f'{key[1:]}_{i+1}'] = components[:, i]

    # Separate train and test again
    train = concat_df[concat_df['is_train']==1]
    test = concat_df[concat_df['is_train']==0]
    test = test.drop(['is_train', "emission"], axis=1)
    train = train.drop(['is_train'], axis=1)

    return train, test
