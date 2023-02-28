import pandas as pd

import sklearn.preprocessing as sk_preproc

def encode_generic(data: pd.DataFrame, var: str, freq: int = 1):
    """Encodes generic"""
    enc = sk_preproc.OneHotEncoder(handle_unknown='ignore', min_frequency=freq, sparse_output=False)
    enc.fit(data[[var]])
    return enc

def decode_generic(data: pd.DataFrame, enc, var: str):
    """Decodes generic"""
    labels = enc.transform(data[var].to_numpy().reshape(-1, 1))
    data = pd.concat([data, pd.DataFrame(labels, columns=enc.get_feature_names_out())], axis=1)
    return data

def encode(data: pd.DataFrame) -> dict:
    """Gets different encodes"""
    encoders = {}
    encoders["PassengerId_Group"] = encode_generic(data, "PassengerId_Group", freq=7)
    encoders["HomePlanet"] = encode_generic(data, "HomePlanet")
    encoders["CryoSleep"] = encode_generic(data, "CryoSleep")
    encoders["VIP"] = encode_generic(data, "VIP")
    encoders["Destination"] = encode_generic(data, "Destination")
    encoders["Cabin_Deck"] = encode_generic(data, "Cabin_Deck")
    encoders["Cabin_Side"] = encode_generic(data, "Cabin_Side")
    encoders["encode_deck_trans_ratio"] = encode_deck_trans_ratio(data)
    return encoders

def decode(data: pd.DataFrame, encoder) -> dict:
    """Gets different encodes"""
    data = decode_generic(data, encoder["PassengerId_Group"], "PassengerId_Group")
    data = decode_generic(data, encoder["HomePlanet"], "HomePlanet")
    data = decode_generic(data, encoder["CryoSleep"],  "CryoSleep")
    data = decode_generic(data, encoder["VIP"], "VIP")
    data = decode_generic(data, encoder["Destination"], "Destination")
    data = decode_generic(data, encoder["Cabin_Deck"], "Cabin_Deck")
    data = decode_generic(data, encoder["Cabin_Side"], "Cabin_Side")
    data = decode_deck_trans_ratio(data, encoder["encode_deck_trans_ratio"])
    return data

def encode_deck_trans_ratio(data):
    # Total passengers by Deck
    deck_total_pass = data.groupby('Cabin_Deck').Cabin_Deck.count()
    # Total Transported by Deck
    deck_total_transported = data.groupby('Cabin_Deck').Transported.sum()
    # Dictionary with Deck_transp_ratio
    deck_transp_ratio = (deck_total_transported / deck_total_pass).to_dict()
    return deck_transp_ratio

def decode_deck_trans_ratio(data, deck_transp_ratio):
    # Create Deck_transp_ratio
    data['deck_transp_ratio'] = data.Cabin_Deck.map(deck_transp_ratio)
    data['deck_transp_ratio'].fillna(data['deck_transp_ratio'].mean(), inplace=True)
    return data
