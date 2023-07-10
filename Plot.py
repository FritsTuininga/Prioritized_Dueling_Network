import torch
import numpy as np
import yfinance as yf
import torchvision.transforms as transforms
from pandas.tseries.offsets import BDay
from datetime import datetime
from joblib import dump
from tqdm import tqdm
from PIL import Image

def normaliser(data):

    #get maximum and minimum of highs and lows in 28 days
    max_val = data["High"].max()
    min_val = data["Low"].min()

    #normalise outputs and multiply with 84 (to scale candles in figure)
    output = ((data[["High", "Low", "Open", "Close"]]-min_val) / (max_val-min_val))*84

    #round all values to nearest integer
    output = output.round(0).astype("int")

    #reset window index
    return output.reset_index(drop=True)

#PARAMETER SELECTION-----------------------------------------------------------
SYMBOLS = ["^GSPC"]
TYPE = "Test"
CANDLES = 28
START_DAY = 31
START_MONTH = 12
START_YEAR = 2019
END_DAY = 1
END_MONTH = 1
END_YEAR = 2022
#PARAMETER SELECTION-----------------------------------------------------------

#define path
path = f"C:\\Users\\Frits\\Prioritized_Dueling_Network\\{TYPE}"

#create 2d numpy array (84x84) with only white pixels
fig = np.zeros((84,84))
fig.fill(255)

#define torch transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale()
])

for symbol in tqdm(SYMBOLS):

    #import data
    df = yf.download(
            symbol,
            start=datetime(START_YEAR, START_MONTH, START_DAY)-BDay(CANDLES+1),
            end=datetime(END_YEAR, END_MONTH, END_DAY)+BDay(1),
            interval='1d',
            progress=False
            ).reset_index()

    #calculate number of images to generate
    N = len(df) - 28

    #initialisation
    images = []
    labels = []

    for i in tqdm(range(N)):

        #create window
        window = normaliser(df[i:i+CANDLES].copy())

        #copy white canvas
        canvas = fig.copy()

        for j in range(CANDLES):

            #calculate body of the candle
            body = window["Close"][j] - window["Open"][j]

            if body < 0:

                #calculate start and end body (negative body implies Open > Close)
                start_body = window["Close"][j]
                end_body = window["Open"][j]

                #to generate images body must by nonnegative
                body = abs(body)

                #black color is assigned to negative candles
                color = 0

            else:

                #calculate start and end body (positive body implies Close > Open)
                start_body = window["Open"][j]
                end_body = window["Close"][j]

                #gray color is assigned to positive candles
                color = 105

            #add body to canvas for corresponding day
            canvas[start_body:end_body,3*j:3*j+3] = color

            #add high to canvas for corresponding day
            canvas[end_body:window["High"][j],3*j+1] = color

            #add low to canvas for corresponding day
            canvas[window["Low"][j]:start_body,3*j+1] = color

        #horizontal flip to ensure correct form
        canvas = np.flip(canvas, axis=0)

        #convert numpy array to PIL image (RGB)
        canvas = Image.fromarray(canvas)
        canvas = canvas.convert('RGB')

        #append images with transformed canvas (now a grayscale tensor)
        images.append(transform(canvas.copy()))

        #append labels with percental change of this day to the next
        labels.append(torch.tensor([(df["Close"][i+CANDLES]-df["Close"][i+CANDLES-1])/df["Close"][i+CANDLES-1]]))

    #store images and targets
    dump(torch.stack(images, dim=0), f"{path}\\{symbol}_X.joblib")
    dump(torch.stack(labels, dim=0), f"{path}\\{symbol}_y.joblib")
