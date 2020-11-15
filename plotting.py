import pandas as pd 
import numpy as np
import pickle
from matplotlib import pyplot as plt
import io


def price_graph():

    df = pd.read_csv('C:/Fall20/Data Mining/true_car_listings.csv')

    #df.head()
    dp=df[df['Price']<50000]
    #dp.head()
    only_price=dp['Price']
    #only_price.head()
    only_price.hist(bins=100, figsize=[14,6])

    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    price_image = io.BytesIO()
    plt.savefig(price_image, format='png')
    price_image.seek(0)
    return price_image
    