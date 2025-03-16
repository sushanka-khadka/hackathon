import joblib
import pandas as pd


model= joblib.load('../completely new/models/model.pkl')
all_freq_map= joblib.load('../completely new/models/encoders/all_freq_map.pkl')
ct= joblib.load('../completely new/models/encoders/col_transformer.pkl')
sc= joblib.load('../completely new/models/encoders/scaler.pkl')
feature_names= joblib.load('../completely new/models/feature_names.pkl')    # after one-hot encoded
print(len(feature_names))

print(feature_names)

# print(model)




def make_prediction(pred_set_x):
    for col, freq in all_freq_map.items():
        pred_set_x[col] = pred_set_x[col].map(freq).fillna(min(freq.values()))

    pred_set_x= ct.transform(pred_set_x)
    pred_set_x= sc.transform(pred_set_x)
    # print(pd.DataFrame(pred_set_x, columns=feature_names))
    result= model.predict(pd.DataFrame(pred_set_x, columns=feature_names))
    # return le.inverse_transform(result)
    for pred in result:
        if pred:
            print("Danger")
        else:
            print('ok ')


pred= model.predict(pd.DataFrame(sc.transform(ct.transform(pd.DataFrame([['Healthcare', 3124, 812, 724949.27, 7143, 7143, 47401, 'medium', 'Basic Debit', False, 'Chrome', 'web', 1, 1, False, 0, False]],
                columns=['merchant_category', 'merchant_type', 'merchant', 'amount', 'currency','country', 'city', 'city_size', 'card_type', 'card_present', 'device','channel', 'device_fingerprint', 'distance_from_home',
       'high_risk_merchant', 'transaction_hour', 'weekend_transaction']))), columns=feature_names))

make_prediction(pd.DataFrame([['Healthcare', 3124, 812, 724949.27, 7143, 7143, 47401, 'medium', 'Basic Debit', False, 'Chrome', 'web', 1, 1, False, 0, False]],
                columns=['merchant_category', 'merchant_type', 'merchant', 'amount', 'currency','country', 'city', 'city_size', 'card_type', 'card_present', 'device','channel', 'device_fingerprint', 'distance_from_home',
       'high_risk_merchant', 'transaction_hour', 'weekend_transaction']))


print(pred)