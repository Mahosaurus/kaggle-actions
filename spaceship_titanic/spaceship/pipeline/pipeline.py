import pandas as pd
import sklearn.pipeline as sk_pipeline
from sklearn.preprocessing import OneHotEncoder

# Hints
# -> Conditional Generation of new column
# data['VIP'] = data.apply(lambda row: True if row["Destination"] == "TRAPPIST-1e" else False, axis=1)
# -> Conditional Filling of NAs
# cond = data["HomePlanet"] == "Earth"
# data['VIP'] = data['VIP'].fillna(cond.map({True:'manual', False: 'automatic'}))

def encode_pass_id(data):
    enc = OneHotEncoder(handle_unknown='ignore', min_frequency=7, sparse_output=False)
    values = data["PassengerId"].apply(lambda x: x[0:4])
    enc.fit(values.to_numpy().reshape(-1, 1))
    return enc

def get_missing_value_indicators(data):
    # Get missing_value indicators
    for column in data.columns.values:
        data[f"No_{column}_Indicator"] = data[column].isnull().astype(int)
    return data

def initial_indicators(data):
    # Indicator for RoomService Payment
    data["RoomService_paid"] = data.apply(lambda row: 1 if row["RoomService"] > 0 else 0, axis=1)
    # Indicator for FoodCourt Payment
    data["FoodCourt_paid"] = data.apply(lambda row: 1 if row["FoodCourt"] > 0 else 0, axis=1)
    return data

def impute(data):
    # Earth -> VIP = False
    cond = data["HomePlanet"] != "Earth"
    data["VIP"] = data["VIP"].fillna(cond.map({True: True, False: False}))
    # Fill NAs, # TODO: Use mode without NA
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["RoomService"] = data["RoomService"].fillna(data["RoomService"].median())
    data["ShoppingMall"] = data["ShoppingMall"].fillna(data["ShoppingMall"].median())
    data["Spa"] = data["Spa"].fillna(data["Spa"].median())
    data["VRDeck"] = data["VRDeck"].fillna(data["VRDeck"].median())
    data["FoodCourt"] = data["FoodCourt"].fillna(data["FoodCourt"].median())
    return data

def binarize_passenger_id(data, enc):
    data["PassGroup"] = data["PassengerId"].apply(lambda x: x[0:4])
    labels = enc.transform(data["PassGroup"].to_numpy().reshape(-1, 1))
    data = pd.concat([data, pd.DataFrame(labels, columns=enc.get_feature_names_out())], axis=1)
    return data

def get_dummies(data):
    # One hot encode HomePlanet
    data = pd.get_dummies(data, prefix=["HomePlanet"], columns=["HomePlanet"], drop_first=False)
    # One hot encode VIP
    data = pd.get_dummies(data, prefix=["VIP"], columns=["VIP"], drop_first=False)
    # One hot encode Destination
    data = pd.get_dummies(data, prefix=["Destination"], columns=["Destination"], drop_first=False)
    ## One hot encode Cabin
    data[["Deck", "Num", "Side"]] = data["Cabin"].str.split("/", 3, expand=True)
    data["Num"] = (data["Num"].fillna(data["Num"].median()).astype(float))  # TODO: Use mode without NA
    data = pd.get_dummies(data, prefix=["Deck"], columns=["Deck"], drop_first=False)
    data = pd.get_dummies(data, prefix=["Side"], columns=["Side"], drop_first=False)
    # One hot encode CryoSleep
    data = pd.get_dummies(data, prefix=["CryoSleep"], columns=["CryoSleep"], drop_first=False)
    return data

def get_encoders(data):
    enc_pass_id = encode_pass_id(data)
    return enc_pass_id

def make_pipeline(data, enc):
    data = get_missing_value_indicators(data)
    data = initial_indicators(data)
    print(data.columns.values[0:1000])
    data = impute(data)
    data = binarize_passenger_id(data, enc)
    data = get_dummies(data)
    return data
