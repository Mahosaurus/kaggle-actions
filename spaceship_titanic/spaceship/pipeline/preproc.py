import numpy as np
import pandas as pd

def get_missing_value_indicators(data: pd.DataFrame):
    # Get missing_value indicators
    for column in data.columns.values:
        data[f"No_{column}_Indicator"] = data[column].isnull().astype(int)
    return data

def split_features(data):
    data['Cabin_Deck'] = data['Cabin'].str.split("/", expand = True)[0].astype("string")
    data['Cabin_Num']  = data['Cabin'].str.split("/", expand = True)[1].astype("float")
    data['Cabin_Side'] = data['Cabin'].str.split("/", expand = True)[2].astype("string")
    data['PassengerId_Group'] = data['PassengerId'].str.split("_", expand = True)[0].astype(str)
    data['PassengerId_GroupNumber'] = data['PassengerId'].str.split("_", expand=True)[1].astype(str)
    data['Total_expenses'] = data["RoomService"] + data["FoodCourt"] + data["ShoppingMall"] + data["Spa"] + data["VRDeck"]
    return data

def initial_pay_indicators(data: pd.DataFrame):
    # Indicator for RoomService Payment
    data["RoomService_paid"] = data.apply(lambda row: 1 if row["RoomService"] > 0 else 0, axis=1)
    # Indicator for FoodCourt Payment
    data["FoodCourt_paid"] = data.apply(lambda row: 1 if row["FoodCourt"] > 0 else 0, axis=1)
    return data

def remove_outliers(data):
    pass

def preprocess_data(data):
    data = get_missing_value_indicators(data)
    data = split_features(data)
    data = initial_pay_indicators(data)
    return data
