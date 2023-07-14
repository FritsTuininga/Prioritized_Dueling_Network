# Prioritized_Dueling_Network2

The goal of this project is to evaluate the ability of Convolutional Neural Networks (CNNs) to predict stock price movements based on candlestick imagery. This project is inspired by the paper 'Deep reinforcement learning stock market trading, utilizing a CNN with candlestick images' (Brim et al. 2022). The stock price movements are expressed as the percentage change from day t up and until day t+1. Candlestick imagery is created with a user-specified number of candlesticks ending at day t. We have found no evidence that this approach generates above average returns consistently, but users might change parameter setting to obtain better results. The CNNs are combined with a reinforcement learning framework called the Prioritized Dueling Double Deep Q-learning Network (PDN). The PDN is a Double Deep Q-learning Network (DDQN) with a Prioritized Sampling method and Dueling Neural Network architecture. To read more about these models we refer to 'Deep reinforcement learning with double q-learning' (Van Hasselt et al. 2016), 'Prioritized experience replay' (Schaul et al. 2015) and 'Dueling network
architectures for deep reinforcement learning' (Want et al. 2016)

-Plot.py
This file creates candlestick images of stock/index data. In this case, candlestick images are created for the S&P500 index (^GSPC) and the daily percentage changes are stored. Feel free to add or remove stocks/indices in the list 'SYMBOLS'. 

The parameters (under #PARAMETER SELECTION-----) which can be adjusted by the user include:
1. SYMBOLS:       a list which indicates which stocks/indices are transformed into candlestick images
2. CANDLES:       an integer indicating the number of candles included in one image
3. TYPE:          a string which determines whether the data is used as 'Train' or 'Test' data
4. START_DAY:     an integer which indicates the starting day of the time-period the user wants to transform into candlestick plots
5. START_MONTH:   an integer which indicates the starting month of the time-period the user wants to transform into candlestick plots
6. START_YEAR:    an integer which indicates the starting year of the time-period the user wants to transform into candlestick plots
7. END_DAY:       an integer which indicates the ending day of the time-period the user wants to transform into candlestick plots
8. END_MONTH:     an integer which indicates the ending month of the time-period the user wants to transform into candlestick plots
9. END_YEAR:      an integer which indicates the ending year of the time-period the user wants to transform into candlestick plots

The output of this file is:
1. A joblib file of all candlestick images
2. A joblib file of all daily percentage changes

**Note that the user should define the path (under #define path) to store the candlestick images (X) and daily percentage changes (y)


-PDN.py
This file runs the Prioritized Dueling Double Deep Q-learning Network (PDN). This Reinforcement Learning model takes candlestick images as input and outputs Q-values. The Q-values can be used to select the most appropriate action (i.e. take a long, neutral or short position). In this file, the model is trained and tested on the data created by the Plot.py file. 

The parameters (under #PARAMETER SELECTION-----) which can be adjusted by the user include:
1. RUNS:          an integer indicating the number of models trained and tested with varying initialization parameters
2. EPOCHS:        an integer indicating the number of epochs/episodes/games
3. UPDATES:       an integer indicating the number of updates executed during the update step of the PDN
4. EPSEXP:        a float indicating ...
5. USE_CUDA:      a boolean indicating whether the Graphics card can be used to process images

The output of this file is:
1. Trained PyTorch models
2. Pandas dataframe with generated rewards during last testing period
