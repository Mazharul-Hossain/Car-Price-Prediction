import pickle
from io import BytesIO

import numpy as np
from flask import Flask, request, render_template, send_file

import model_utils
# from flask import Flask,
from plotting import price_graph

app = Flask(__name__, static_url_path='/static')
model_utils.first_run()

# regressor = pickle.load(open(model_utils.data_info["Random Forest Regressor"], 'rb'))
regressor = pickle.load(open('linear.pkl', 'rb'))
# classifier = pickle.load(open(model_utils.data_info["Random Forest Regressor"], 'rb'))

label_encoder = {}
# label_encoder = pickle.load(open(model_utils.data_info["label_encoder"], 'rb'))

scaler = None
# scaler = pickle.load(open(model_utils.data_info["scaler"], 'rb'))

read_data = model_utils.load_dataset_frame()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return '<h3>Hello World!....it works....testing</h3>'


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/bootstrap-elements')
def bootstrap():
    return render_template('bootstrap-elements.html')


@app.route('/predicting-car-price')
def car_price():
    global read_data
    if model_utils.data_info is None or len(model_utils.data_info) <= 0:
        model_utils.first_run()

    if read_data is None or len(read_data) <= 0:
        read_data = model_utils.load_dataset_frame()

    city = read_data.City.unique()
    state = read_data.State.unique()
    make = read_data.Make.unique()
    model = read_data.Model.unique()
    return render_template('car-price.html', city=city, state=state, make=make, model=model)


@app.route('/plots/car_price_data/correlation_matrix')
def images():
    return render_template("home.html", title='testing')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ['Year', 'Mileage', 'City', 'State', 'Make', 'Model']

        features, columns = [], model_utils.data_info['columns']
        for column in columns:
            feature = request.form[column]
            if column == 'Year':
                feature = int(feature) - int(model_utils.data_info['car_year_min'])
            elif column == 'Mileage':
                feature = float(feature) / float(model_utils.data_info['car_mileage_mean'])
            # elif column in ['City', 'State', 'Make', 'Model']:
            #     feature = label_encoder[column].transform(str(feature))
            features.append(feature)

        # features = scaler.transform(features)
        final_features = [np.array(features)]
        # print(final_features)
        # return render_template('car-price.html', prediction_text=final_features)

        prediction_text = ""
        prediction = regressor.predict(final_features)
        output = round(prediction[0], 3)
        prediction_text += 'Car price should be around $ {}'.format(output)

        # prediction = classifier.predict(final_features)
        # output = round(prediction[0])
        # prediction_text = '\n Car price should be between $ {} and $ {}'.format(
        #     model_utils.data_info['price_bins'][output], model_utils.data_info['price_bins'][output + 1])

        return render_template('car-price.html', form=request.form,
                               prediction_text=prediction_text)
    except Exception as e:
        prediction_text = 'Exception occurred: {}'.format(e)
        return render_template('car-price.html', form=request.form,
                               prediction_text=prediction_text)


@app.route('/correlation_matrix')
def correlation_matrix():
    bytes_obj = price_graph()
    img = BytesIO()
    bytes_obj.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')
    # plot= base64.b64encode(img.getvalue()).decode('utf8')

    # render_template('home.html', title= "testing", plot=plot)
    # return send_file(img, mimetype='image/png')
    # return send_file(bytes_obj,
    # attachment_filename='plot.png',
    # mimetype='image/png')


if __name__ == '__main__':
    app.debug = True
    app.run()
