import numpy as np
import csv
import sys
import time
import datetime

import matplotlib
matplotlib.use('pdf')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pandas as pd
import glob
import os
import pickle

from dtw import dtw_distance

def load_labelled(csv_file_path):

    x_data = []
    label_data = []
    csv.field_size_limit(500 * 1024 * 1024)

    with open(csv_file_path, 'rU') as csvfile:
        uavreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in uavreader:
            label_data.append(int(row[0]))
            x_data.append([float(ts) for ts in row[1].split()])

        # Convert to numpy for efficiency
        x_data     = np.array(x_data)
        label_data = np.array(label_data)

        return [x_data, label_data]

def load_test(csv_file_path):
    x_data = []
    csv.field_size_limit(500 * 1024 * 1024)

    with open(csv_file_path, 'rU') as csvfile:
        uavreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in uavreader:
            x_data.append([float(ts) for ts in row[1].split()])

        # Convert to numpy for efficiency
        x_data = np.array(x_data)

        return x_data



def print_confusion_matrix(tp, fp, fn, tn):
        print( '         | Predicted + | Predicted -')
        print( 'Actual + |      '+str(tp)+'      |    ' + str(fn))
        print( 'Actual - |      '+str(fp)+'      |    ' + str(tn))
        print( '')


def evaluate(labels, label, test_label):
    accuracies = np.zeros(len(labels))
    precisions = np.zeros(len(labels))
    recalls = np.zeros(len(labels))
    f1scores = np.zeros(len(labels))

    for index,l in enumerate(labels):
        count = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for i in range(0,len(label)):
            if label[i] == l:
                count += 1
                if test_label[i] == label[i]:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                #if validation_label_data[i] == label[i]:
                if test_label[i] != l:
                    true_negative += 1
                else:
                    false_negative += 1

        print( 'Label:', l)
        print_confusion_matrix(true_positive, false_positive, true_negative, false_negative)

        acc = float(true_positive + true_negative)/len(label)

        precision = 0.0
        recall = 0.0
        f1score = 0.0
        
        if (true_positive + false_positive) > 0:
            precision = float(true_positive) / float(true_positive + false_positive)
        
        if (true_positive + false_negative) > 0:
            recall = float(true_positive)/float(true_positive + false_negative)
    
        if precision != 0 or recall != 0:
            f1score = float(2 * (precision * recall))/float(precision + recall)
    
        accuracies[index] = acc
        precisions[index] = precision
        recalls[index] = recall
        f1scores[index] = f1score

        print ('Accuracy:', acc)
        print ('Precision:', precision)
        print ('Recall:', recall)
        print ('F1 Score', f1score)
        # ...
        print ('--------------')

    return [accuracies.mean(), precisions.mean(), recalls.mean(), f1scores.mean()]

def get_distances(data, data_array, max_warping_window):
    a = np.zeros(len(data_array))
    for i in range(0,len(data_array)):
        dist = dtw_distance(data_array[i], data, max_warping_window)
        a[i] = dist
        print (str(i) + " - " + str(dist))
    return a




# Online detection

def preprocess(dataframe, threshold=0.001):
    print('Starting preprocessing')
    drop_values = [] # True for delete, False to keep

    previous_row = pd.Series()
    for index, row in dataframe.iterrows():
        should_drop = True
        if  not previous_row.empty:
            if previous_row['header_stamp_secs'] != row['header_stamp_secs']:
                should_drop = False
            elif abs(previous_row['pose_position_x'] - row['pose_position_x']) > 0.001:
                should_drop = False
            elif abs(previous_row['pose_position_y'] - row['pose_position_y']) > 0.001:
                should_drop = False
            elif abs(previous_row['pose_position_z'] - row['pose_position_z']) > 0.001:
                should_drop = False
        else: 
            should_drop = False

        if not should_drop:   
            previous_row = row

        drop_values.append(should_drop)

    drop_values = pd.Series(drop_values)

    dataframe['drop_values'] = drop_values    

    dataframe = dataframe[dataframe.drop_values != False]



# loading csvs and compyting DTW for each row
# adding new dtw values in a csv





# loading all csv files in the folder
# writing in a dictionary [{'window': df1, 'label': '1'}, {'window': df2, 'label': '2'}, ...{..}]

def load_csvs(path, should_preprocess=False):
    files = glob.glob(path)
    
    dictionaries = []
    
    for csv_file in files:

        #print('Loading CSV file: ' + csv_file)
        df = pd.read_csv(csv_file, sep=';')
        
        if should_preprocess:
            preprocess(df)

        filename_split = os.path.splitext(os.path.split(csv_file)[1])[0].split('_')
        label = filename_split[-1]

        flight = {
            'label': int(label),
            'data': df
        }

        dictionaries.append( flight )
    
    return dictionaries


def load_labelled_csvs(path, should_preprocess=False):
    files = glob.glob(path)
    
    dictionaries = []
    
    for csv_file in files:

        df = pd.read_csv(csv_file, sep=';')
        
        if should_preprocess:
            preprocess(df)

        flight = {
            'label': df['label'],
            'data': df
        }

        dictionaries.append( flight )
    
    return dictionaries



def save_data_object(path, data_object):
    with open(path, 'wb') as file_data:
        pickle.dump(file_data, data_object )

def load_data_object(path):
    data = None
    with open(path, 'rb') as file_data:
        data = pickle.load(file_data)

    return data



def get_data_between_seconds(data, start, end):
    return data[ data.seconds.between(start, end, inclusive=True) ]


# returns data for a given window (given seconds)

def get_windows(flights, start, end, threshold=50):
    ''' Return list of flight with the window start to end.
    '''
    windows = []
    for flight in flights:
        data = flight['data']
        window_data = get_data_between_seconds(data, start, end)
        if len(window_data) >= threshold:
            windows.append({
                'label': flight['label'],
                'window': window_data
            })
    
    return windows



def window_to_lists(windows, column_name):
    ''' Return array of data in a given window and array of labels
    '''
    train_data = []
    train_label = []
    for data in windows:
        window = data['window']
        label = data['label']

        train_data.append(window[column_name].tolist())
        train_label.append(label)
    
    return train_data, train_label


def get_windows_values(data, column, start, end, threshold=50):
    windows = []
    for d in data:
        window = get_data_between_seconds(d, start, end)
        if len (window) >= threshold:
            windows.append(window[column].tolist())
    return windows

def get_windows_labels(data, start, end, threshold=50):
    windows = []
    for d in data:
        window = get_data_between_seconds(d, start, end)
        if len (window) >= threshold:
            windows.append(window['label'].iloc[0])
    return windows

# DTW labelling and plots

def dtw_plots(dtws):
    sorted_dtws = np.sort(dtws)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H%M%S')

    #plt.plot(sorted_dtws, '-')
    #plt.xlabel('Experiment ID')
    #plt.ylabel('DTW Value')
    #plt.title('DTW value between first show and each other show')
    #plt.savefig('dtw_plots/dtwValues_' + st + '.pdf')

    plt.figure()
    plt.hist(sorted_dtws, bins=20)
    plt.xlabel('DTW Value')
    plt.ylabel('Number of Experiments')
    plt.title('Distribution of experiments grouped by DTW value')
    plt.savefig('dtwHistogram_' + st + '.pdf')

def get_label_for_value(value, good_limit, bad_limit):
    label = -1
    if value < good_limit:
        label = 1
    elif value < bad_limit:
        label = 2
    else:
        label = 3
    
    return label

def label_dtws(dtws, good_limit, bad_limit, outputFileName):
    labels = np.zeros(len(dtws))
    for i in range(0,len(dtws)):
        labels[i] = get_label_for_value(dtws[i], good_limit, bad_limit)

    with open(outputFileName, 'wb') as f:
        writer = csv.writer(f)
        for l in labels:
            writer.writerow(str(int(l)))

    print ('Labels: ' , labels)




