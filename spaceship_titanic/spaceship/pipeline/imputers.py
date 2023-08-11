import pandas as pd
#pd.options.mode.chained_assignment = None

import sklearn.linear_model as sk_linear_model
import sklearn.impute as sk_impute #(SimpleImputer, IterativeImputer)

def _impute_VIP(data: pd.DataFrame) -> pd.DataFrame:
    # Earth -> VIP = False
    #data.loc[data.VIP.isna() & data["HomePlanet"] != "Earth" & ~data["Age"] < 18.0, 'VIP'] = True
    data["Age"] = data["Age"].astype(float)
    data.loc[(data.VIP.isna()) & (data["HomePlanet"] != "Earth") & (data["Age"] > 18.0), 'VIP'] = True
    data.loc[data.VIP.isna(), 'VIP'] = False
    return data

def _impute_cryo_sleep(data):
    data.loc[(data.CryoSleep.isna()) & (data.Total_expenses.gt(0)), 'CryoSleep'] = False
    return data["CryoSleep"]

def _impute_age(data: pd.DataFrame, reg) -> pd.DataFrame:
    """Apply Reg Model"""
    tmp = data[["VIP", "Total_expenses", "Age"]]

    medians = {
        "VIP": float(tmp.loc[:, "VIP"].mode()),
        "Total_expenses": float(tmp.loc[:, "Total_expenses"].mode())
        }
    tmp.loc[:,:].fillna(value=medians, inplace=True)

    age_imputed = reg.predict(tmp[["VIP", "Total_expenses"]])

    # Fill only if NA
    tmp.loc[:, 'Age'] = tmp.loc[:, 'Age'].mask(tmp.loc[:, 'Age'].isna(), age_imputed)
    return tmp["Age"]

def estimate_age_model(data: pd.DataFrame):
    """Estimate Reg Model"""
    tmp = data[["Age", "VIP", "Total_expenses"]].notnull()
    y = tmp["Age"]
    X = tmp[["VIP", "Total_expenses"]]
    reg = sk_linear_model.LinearRegression().fit(X, y)
    return reg

def get_imputers(data: pd.DataFrame) -> dict:
    """Gets different encodes"""
    imputers = {}
    imputers["Age"] = estimate_age_model(data)
    return imputers

def impute(data: pd.DataFrame, imputers: dict):
    """Fill NAs, # TODO: Use mode without NA"""
    data = _impute_VIP(data)
    data["Age"] = _impute_age(data, imputers["Age"])

    data.loc[data.CryoSleep.eq(True), "RoomService"] = 0
    data.loc[data.CryoSleep.eq(True), "FoodCourt"] = 0
    data.loc[data.CryoSleep.eq(True), "ShoppingMall"] = 0
    data.loc[data.CryoSleep.eq(True), "Spa"] = 0
    data.loc[data.CryoSleep.eq(True), "VRDeck"] = 0
    data.loc[data.CryoSleep.eq(True), "Total_expenses"] = 0

    data["CryoSleep"] = _impute_cryo_sleep(data)

    data["Cabin_Num"] = (data["Cabin_Num"].fillna(data["Cabin_Num"].median()).astype(float))
    data["Cabin_Deck"] = data["Cabin_Deck"].fillna(data["Cabin_Deck"].mode().values[0])
    data["Cabin_Side"] = data["Cabin_Side"].fillna(data["Cabin_Side"].mode().values[0])
    data["Destination"] = data["Destination"].fillna(data["Destination"].mode())

    data["RoomService"] = data["RoomService"].fillna(data["RoomService"].median())
    data["ShoppingMall"] = data["ShoppingMall"].fillna(data["ShoppingMall"].median())
    data["Spa"] = data["Spa"].fillna(data["Spa"].median())
    data["VRDeck"] = data["VRDeck"].fillna(data["VRDeck"].median())
    data["FoodCourt"] = data["FoodCourt"].fillna(data["FoodCourt"].median())
    return data
