from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, FloatField, BooleanField, IntegerField
from wtforms.validators import DataRequired, NumberRange
import joblib
import pandas as pd

app= Flask(__name__)
app.config['SECRET_KEY']= 'secret'
# take input
class InputForm(FlaskForm):
    mer_cat_choices = [('Restaurant', 'Restaurant'), ('Entertainment','Entertainment'), ('Grocery','Grocery'), ('Gas','Gas'), ('Healthcare','Healthcare'), ('Education', 'Education'), ('Travel', 'Travel'), ('Retail', 'Retail')]
    merchant_category= SelectField('merchant_category', choices= mer_cat_choices)
    mer_type_choices= ['fast_food', 'gaming', 'physical', 'major', 'medical', 'online', 'hotels', 'pharmacy', 'premium', 'events', 'supplies', 'airlines', 'local', 'booking', 'streaming', 'transport', 'casual']
    mer_type_choices= [(item, item) for item in mer_type_choices ]
    print(mer_type_choices)
    merchant_type= SelectField('merchant_type', choices=mer_type_choices)
    merchant= StringField('merchant', default='MasterClass')
    amount= FloatField('amount')
    cur_choices= ['GBP', 'BRL', 'JPY', 'AUD', 'NGN', 'EUR', 'MXN', 'RUB', 'CAD', 'SGD', 'USD']
    cur_choices = [(item, item) for item in cur_choices]
    currency= SelectField('currency', choices= cur_choices)
    countries= ['UK', 'Brazil', 'Japan', 'Australia', 'Nigeria', 'Germany', 'Mexico', 'Russia', 'France', 'Canada', 'Singapore', 'USA']
    countries = [(item, item) for item in countries]
    country= SelectField('country', choices= countries)
    #  need to perform grouping
    cities= ['Unknown City', 'San Antonio', 'Philadelphia', 'Phoenix', 'San Diego', 'Los Angeles', 'Chicago', 'Dallas', 'New York', 'San Jose', 'Houston']
    cities = [(item, item) for item in cities]
    city= SelectField('city', choices= cities)
    city_size_choices= ['medium', 'large']
    city_size_choices = [(item, item) for item in city_size_choices]
    city_size= SelectField('city_size', choices= city_size_choices)
    card_types= ['Platinum Credit', 'Premium Debit', 'Basic Debit', 'Gold Credit', 'Basic Credit']
    card_types = [(item, item) for item in card_types]
    card_type= SelectField('card_type', choices=card_types)
    card_present= BooleanField('card_present', default=False)
    devices= ['iOS App', 'Edge', 'Firefox', 'Chrome', 'Android App', 'NFC Payment', 'Chip Reader', 'Safari', 'Magnetic Stripe']
    devices = [(item, item) for item in devices]
    device= SelectField('device', choices= devices)
    channels= ['mobile', 'web', 'pos']
    channels = [(item, item) for item in channels]
    channel= SelectField('channel', choices=channels)
    # device fingerprint too large
    device_fingerprint= StringField('device_fingerprint', default='71f718bb0d6fd8e44feb86150fd846f0')
    distance_from_home= BooleanField('distance form home')
    high_risk_merchant= BooleanField('high risk merchant')
    transaction_hour= IntegerField('transaction hour', validators=[NumberRange(min=0, max= 23, message='Hour must be within [0,24]') ])
    weekend_transaction= BooleanField('weekend transaction')
    submit= SubmitField('Predict!!!')


model= joblib.load('../completely new/models/model.pkl')
all_freq_map= joblib.load('../completely new/models/encoders/all_freq_map.pkl')
ct= joblib.load('../completely new/models/encoders/col_transformer.pkl')
sc= joblib.load('../completely new/models/encoders/scaler.pkl')
feature_names= joblib.load('../completely new/models/feature_names.pkl')    # after one-hot encoded


def make_prediction(pred_set_x):
    for col, freq in all_freq_map.items():
        pred_set_x[col] = pred_set_x[col].map(freq).fillna(min(freq.values()))
    pred_set_x= ct.transform(pred_set_x)
    pred_set_x= sc.transform(pred_set_x)
    result= model.predict(pd.DataFrame(pred_set_x, columns=feature_names))
    return result
    # for pred in result:
    #     if pred:
    #         print("Danger")
    #     else:
    #         print('ok ')


@app.route('/', methods= ['GET', 'POST'])
def home():
    form = InputForm()
    if form.validate_on_submit():
        data= request.form
        print(data)
        features= {
            'merchant_category': data['merchant_category'],
            'merchant_type': data['merchant_type'],
            'merchant': data['merchant'], # should be removed from features or use some default value (most frequent or random)
            'amount': form.amount.data, # flask form data instead or raw data
            'currency' :data['currency'],
            'country': data['country'],
            'city': data['city'],
            'city_size': data['city_size'],
            'card_type': data['card_type'],
            'card_present': form.card_present.data,
            'device': data['device'],
            'channel': data['channel'],
            'device_fingerprint': data['device_fingerprint'],  # should use some default value (most frequent or random)
            'distance_from_home': form.distance_from_home.data,
            'high_risk_merchant': form.high_risk_merchant.data,
            'transaction_hour': data['transaction_hour'],
            'weekend_transaction': form.weekend_transaction.data
        }
        print(type(form.transaction_hour.data)) # form field data
        print(type(data['transaction_hour'])) # raw data request
        print(type(data.get('transaction_hour')))
        print('success')
        print(features)

        print(features.values())
        pred_set_x= pd.DataFrame([(features)])  # into linear list of dict
        print('hereereerererere\n')
        print(pred_set_x)
        pred= make_prediction(pred_set_x)
        print(pred)

        table_html = pred_set_x.to_html(classes='table table-bordered', index=False)
        for value in pred:
            if value:
                status = 'Fraud Transaction'
            else:
                status= 'Normal Transaction '



        # pred = model.predict(sc.transform(ct.transform(
        #     [['Healthcare', 3124, 812, 724949.27, 7143, 7143, 47401, 'medium', 'Basic Debit', False, 'Chrome', 'web', 1,
        #      1, False, 0, False]])))
        # print(pred)

        return render_template('status.html', table_html= table_html, status= status)

    return render_template('home.html', form= form)


if __name__ == '__main__':
    app.run(debug=True, port=5000)


# df columns narakhda pani predict bhako xa tara order chai khyal garnu paro
#

# model.predict(pd.DataFrame([['Healthcare',	3124,	812,	724949.27,	7143,	7143,	47401,	'medium',	'Basic Debit',	False,	'Chrome',	'web',	1,	1,	False,	0,	False]],
#                        columns=X_train.columns))