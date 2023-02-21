import pandas as pd
import sklearn.pipeline as sk_pipeline

# Hints
# -> Conditional Generation of new column
# data['VIP'] = data.apply(lambda row: True if row["Destination"] == "TRAPPIST-1e" else False, axis=1)
# -> Conditional Filling of NAs
# cond = data["HomePlanet"] == "Earth"
# data['VIP'] = data['VIP'].fillna(cond.map({True:'manual', False: 'automatic'}))

def impute(data):
    # Earth -> VIP = False
    cond = data["HomePlanet"] != "Earth"
    data['VIP'] = data['VIP'].fillna(cond.map({True: True, False: False}))
    # Get indicators
    data["No_Cabin_Indicator"] = data["Cabin"].isnull().astype(int)
    # Get indicators
    data["No_Destination_Indicator"] = data["Destination"].isnull().astype(int)
    return data

def make_pipeline(data):
    # Fill NAs
    data["Age"] = data["Age"].fillna(data["Age"].median())
    # Indicator for RoomService Payment
    data["RoomService_paid"] = data.apply(lambda row: 1 if row["RoomService"] > 0 else 0, axis=1)
    # Fill NAs
    data["RoomService"] = data["RoomService"].fillna(data["RoomService"].median()) #TODO: Use mode without NA
    # Indicator for FoodCourt Payment
    data["FoodCourt_paid"] = data.apply(lambda row: 1 if row["FoodCourt"] > 0 else 0, axis=1)
    # Fill NAs
    data["FoodCourt"] = data["FoodCourt"].fillna(data["FoodCourt"].median()) #TODO: Use mode without NA
    # Fill NAs
    data["ShoppingMall"] = data["ShoppingMall"].fillna(data["ShoppingMall"].median()) #TODO: Use mode without NA
    # Fill NAs
    data["Spa"] = data["Spa"].fillna(data["Spa"].median()) #TODO: Use mode without NA
    # Fill NAs
    data["VRDeck"] = data["VRDeck"].fillna(data["VRDeck"].median()) #TODO: Use mode without NA
    # One hot encode HomePlanet
    data = pd.get_dummies(data, prefix=['HomePlanet'], columns = ['HomePlanet'], drop_first=False)
    # One hot encode VIP
    data = pd.get_dummies(data, prefix=['VIP'], columns = ['VIP'], drop_first=False)
    # One hot encode Destination
    data = pd.get_dummies(data, prefix=['Destination'], columns = ['Destination'], drop_first=False)
    ## One hot encode Cabin
    data[["Deck", "Num", "Side"]] = data["Cabin"].str.split('/', 3, expand=True)
    data["Num"] = data["Num"].fillna(data["Num"].median()).astype(float) #TODO: Use mode without NA
    data = pd.get_dummies(data, prefix=['Deck'], columns = ['Deck'], drop_first=False)
    data = pd.get_dummies(data, prefix=['Side'], columns = ['Side'], drop_first=False)
    # One hot encode CryoSleep
    data = pd.get_dummies(data, prefix=['CryoSleep'], columns = ['CryoSleep'], drop_first=False)
    return data
