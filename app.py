from flask import Flask,request,render_template,send_file,make_response,url_for
from io import BytesIO
import base64
import numpy as np
import pickle
#from flask import Flask, 
from plotting import price_graph
app = Flask(__name__)
model = pickle.load(open('linear.pkl', 'rb'))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/hello')
def hello():
    return '<h3>Hello World!....it works....testing</h3>'

@app.route('/bootstrap-elements')
def bootstrap():
    return render_template('bootstrap-elements.html')


@app.route('/predicting-car-price')
def car_price():
    return render_template('car-price.html')

@app.route('/plots/car_price_data/correlation_matrix')
def images():
    return render_template("home.html", title= 'testing')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #print(final_features)
    prediction = model.predict(final_features)
    #return render_template('car-price.html', prediction_text=final_features)

    output = round(prediction[0], 2)

    return render_template('car-price.html', prediction_text='Car price should be $ {}'.format(output))



@app.route('/correlation_matrix')
def correlation_matrix():
    bytes_obj = price_graph()
    img = BytesIO()
    bytes_obj.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')
    #plot= base64.b64encode(img.getvalue()).decode('utf8')
    
    
    #render_template('home.html', title= "testing", plot=plot)
    #return send_file(img, mimetype='image/png')
    #return send_file(bytes_obj,
                    # attachment_filename='plot.png',
                    # mimetype='image/png')


if __name__ == '__main__':
    app.debug = True
    app.run()