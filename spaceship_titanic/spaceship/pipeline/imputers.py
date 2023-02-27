import pandas as pd

import sklearn.linear_model as sk_linear_model

def _impute_VIP(data: pd.DataFrame) -> pd.DataFrame:
    # Earth -> VIP = False
    cond = data["HomePlanet"] != "Earth"
    data["VIP"] = data["VIP"].fillna(cond.map({True: True, False: False}))
    return data

def impute_model_age(data: pd.DataFrame):
    tmp = data[["Age", "VIP", "Spa"]].notnull()
    y = tmp["Age"]
    X = tmp[["VIP", "Spa"]]
    reg = sk_linear_model.LinearRegression().fit(X, y)
    return reg

def _impute_age(data: pd.DataFrame) -> pd.DataFrame:
    # Earth -> VIP = False
    ...
    return data

def get_imputers(data: pd.DataFrame) -> dict:
    """Gets different encodes"""
    imputers = {}
    imputers["Age"] = impute_model_age(data)
    return imputers

def impute(data: pd.DataFrame, imputers: dict):
    """Fill NAs, # TODO: Use mode without NA"""
    data = _impute_VIP(data)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Cabin_Num"] = (data["Cabin_Num"].fillna(data["Cabin_Num"].median()).astype(float))  # TODO: Use mode without NA
    data["Cabin_Deck"] = data["Cabin_Deck"].fillna(data["Cabin_Deck"].mode().values[0])
    data["Cabin_Side"] = data["Cabin_Side"].fillna(data["Cabin_Side"].mode().values[0])
    data["Destination"] = data["Destination"].fillna(data["Destination"].mode())
    data["CryoSleep"] = data["CryoSleep"].fillna(data["CryoSleep"].mode())
    data["RoomService"] = data["RoomService"].fillna(data["RoomService"].median())
    data["ShoppingMall"] = data["ShoppingMall"].fillna(data["ShoppingMall"].median())
    data["Spa"] = data["Spa"].fillna(data["Spa"].median())
    data["VRDeck"] = data["VRDeck"].fillna(data["VRDeck"].median())
    data["FoodCourt"] = data["FoodCourt"].fillna(data["FoodCourt"].median())
    return data
