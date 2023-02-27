from copy import deepcopy
from pipeline.preproc import preprocess_data
from pipeline.encoders import encode, decode
from pipeline.imputers import get_imputers, impute

# Hints
# -> Conditional Generation of new column
# data['VIP'] = data.apply(lambda row: True if row["Destination"] == "TRAPPIST-1e" else False, axis=1)
# -> Conditional Filling of NAs
# cond = data["HomePlanet"] == "Earth"
# data['VIP'] = data['VIP'].fillna(cond.map({True:'manual', False: 'automatic'}))

def get_pipeline(data):
    """Runs transformer Pipeline"""
    data_new = deepcopy(data)
    data_new = preprocess_data(data_new)
    imputers = get_imputers(data_new)
    data_new = impute(data_new, imputers)
    encoders = encode(data_new)
    return encoders, imputers

def apply_pipeline(data, encoders, imputers):
    data_new = deepcopy(data)
    data_new = preprocess_data(data_new)
    data_new = impute(data_new, imputers)
    data_new = decode(data_new, encoders)
    return data_new
