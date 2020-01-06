import pandas as pd
import numpy as np

#load average audio data and merge with ‚goodforairplane‘ attribute

data_dev = pd.read_excel('data/CoE_dataset/Dev_Set/dev_set_groundtruth_and_trailers.xls').set_index('movie')
data_test = pd.read_csv('data/CoE_dataset/Test_Set/test_set_inclGoodforairplane.csv').set_index('movie')

indices = ['movie','MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5',
           'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13']

def audio_selection(file_name, set_selection):
    try:
        if set_selection == 'test':
            my_data = np.genfromtxt('data/CoE_dataset/Test_Set/audio_avg/'+ file_name + '.csv', delimiter=',')
        elif set_selection == 'dev':
            my_data = np.genfromtxt('data/CoE_dataset/Dev_Set/audio_avg/'+ file_name + '.csv', delimiter=',')
        else: 
            return np.full((13), np.nan)
    except:
        return np.full((13), np.nan)
    return my_data[:,1]

def audio_data(df, set_selection):
    titles = df.index.tolist()
    split_data = pd.DataFrame(columns = indices).set_index('movie')
    for title in titles:
        split_data.loc[title] = audio_selection(df.loc[title]['filename'], set_selection)
    split_data = split_data.dropna()
    return(split_data)

# test data import
dev_data_audio = audio_data(data_dev, 'dev')

# dev data import
test_data_audio = audio_data(data_test, 'test')
