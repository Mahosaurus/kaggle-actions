import pandas as pd
import sklearn.pipeline as sk_pipeline

def make_pipeline(data):
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["RoomService"] = data["RoomService"].fillna(data["RoomService"].median()) #TODO: Use mode without NA
    data["FoodCourt"] = data["FoodCourt"].fillna(data["FoodCourt"].median()) #TODO: Use mode without NA
    data["ShoppingMall"] = data["ShoppingMall"].fillna(data["ShoppingMall"].median()) #TODO: Use mode without NA
    data["Spa"] = data["Spa"].fillna(data["Spa"].median()) #TODO: Use mode without NA
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