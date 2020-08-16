import click
import numpy as np
import pandas as pd

import time
import datetime

import matplotlib
#matplotlib.use('jpeg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from kNNDTW import KnnDtw
from utils import load_labelled, load_test, get_distances, dtw_plots, label_dtws, load_csvs, get_windows, window_to_lists
from utils import save_data_object, load_data_object, load_labelled_csvs
from trainer import Trainer

from argparse import Namespace as ns

from dtw import dtw_distance

@click.group()
def cli():
    pass

@cli.command('findbestk')
@click.option('--kmin', default=1, help='Minimum value for k Nearest Neighbours')
@click.option('--kmax', default=5, help='Maximum value for k Nearest Neighbours')
@click.option('-w', default=4000, help='Maximum Warping Window')
@click.option('--seed', default=108, help='Seed for random shuffle')
@click.option('--train', help='Training Data as CSV file path')
def find_best_k(kmin, kmax, w, seed, train):
    click.echo('--- Find best k ---')
    
    kmin = int(min(kmin, kmax))
    kmax = int(max(kmin, kmax))

    ks = range(kmin, kmax + 1)

    train_data, train_label = load_labelled(train)

    click.echo('  - ks    : %s ' % str(ks))
    click.echo('  - w     : %d ' % w)
    click.echo('  - seed  : %d ' % seed)
    click.echo('  - train : %s ' % train)

    click.echo('  - Training data size: %d' % len(train_data))
    click.echo('\nRunning...')

    trainer = Trainer(seed=seed, data=train_data, data_labels=train_label)
    trainer.find_best_k(ks, w)

    click.echo('\nDone.')

@cli.command('findbestw')
@click.option('--wmin', default=100, help='Minimum value for Warping Window')
@click.option('--wmax', default=4000, help='Maximum value for Warping Window')
@click.option('--step', default=100, help='Step for Warping Window')
@click.option('-k', default=3, help='k Nearest Neighbours')
@click.option('--seed', default=108, help='Seed for random shuffle')
@click.option('--train', help='Training Data as CSV file path')
def find_best_w(wmin, wmax, step, k, seed, train):
    click.echo('--- Find best w ---')
    
    wmin = int(min(wmin, wmax))
    wmax = int(max(wmin, wmax))

    ws = range(wmin, wmax, step)

    train_data, train_label = load_labelled(train)

    click.echo('  - ws    : %s ' % str(ws))
    click.echo('  - k     : %d ' % k)
    click.echo('  - seed  : %d ' % seed)
    click.echo('  - train : %s ' % train)

    click.echo('  - Training data size: %d' % len(train_data))
    click.echo('\nRunning...')

    trainer = Trainer(seed=seed, data=train_data, data_labels=train_label)
    trainer.find_best_w(k, ws)

    click.echo('\nDone.')

@cli.command('predict')
@click.option('-k', default=3, help='k Nearest Neighbours')
@click.option('-w', default=200, help='Maximum Warping Window')
@click.option('--train', help='Training Data as CSV file path')
@click.option('--test', help='Test Data as CSV file path')
def predict(k, w, train, test):
    click.echo('--- Predicting a label ---')
    #click.echo('Predicting with k=%d and w=%d.' % (k,w))

    train_data, train_label = load_labelled(train)
    test_data = load_test(test)

    click.echo('  - k     : %d ' % k)
    click.echo('  - w     : %d ' % w)
    click.echo('  - train : %s ' % train)
    click.echo('  - test  : %s ' % test)

    click.echo('\nRunning...')


    model = KnnDtw(k_neighbours = k, max_warping_window = w)
    model.fit(train_data, train_label)
    
    predicted_label, probability = model.predict(test_data, parallel=False)
    
    click.echo('\nPredicted label : %s ' % str(predicted_label))
    click.echo('\nDone.')

@cli.command('dtw')
@click.option('--data', help='Single timeseries data as CSV file path')
@click.option('--dataarray', help='List of timeseries data as CSV file path')
@click.option('-w', default=200, help='Maximum Warping Window')
def compute_dtw(data, dataarray, w):
    click.echo('--- Compute DTW ---')

    timeseries, timeseries_label = load_labelled(dataarray)
    timeserie_1 = load_test(data)

    click.echo('  - data        : %s ' % data)
    click.echo('  - dataarray   : %s ' % dataarray)
    click.echo('  - w           : %d ' % w)

    click.echo('\nRunning...')

    unsorted_dtws = get_distances(timeserie_1[0], data_array=timeseries, max_warping_window=w)

    # Save plots
    dtw_plots(unsorted_dtws)
    click.echo('Done. Plots have been saved.')

    click.echo('Choose a maximum number for labelling good and bad data based on DTW values.')
    click.echo('Check the plots to take better decsision.')
    click.echo('  Example: If value for Good is 150, all data with DTW 0-150 will be labelled "Good".')
    # Enter limit for 'Good'
    good_value = raw_input(' > Enter a value for "Good" (Ex: 150) : ')
    # Enter limit for 'Bad'
    bad_value = raw_input(' > Enter a value for "Bad" (Ex: 350) : ')
    # Print and save results to CSV
    fileName = raw_input(' > Enter a file name (add .csv at the end) : ')

    label_dtws(unsorted_dtws, int(good_value), int(bad_value), fileName)

    click.echo('\nDone.')


# Online anomaly detection
@cli.command('online')
@click.option('-k', default=3, help='k Nearest Neighbours')
@click.option('-w', default=200, help='Maximum Warping Window')
@click.option('--window-size', default=2, help='Maximum Warping Window')
@click.option('--train', help='Training Data as CSV file path')
@click.option('--test', help='Test Data as CSV file path')
def online_identification(k, w, window_size, train, test):

    click.echo('\n--- Online flight degradation identification ---')

    train_csv = load_csvs(train, should_preprocess=False)
    test_csv = load_csvs(test, should_preprocess=False)
    
    click.echo('  - k     : %d ' % k)
    click.echo('  - w     : %d ' % w)
    click.echo('  - train : %s ' % train)
    click.echo('  - test  : %s ' % test)

    click.echo('\nRunning...\n')
    
    # For Test
    last_second = test_csv[0]['data']['seconds'].iloc[-1]
    #print('[TEST] Last second: {}'.format(last_second))

    for time_window in range(0, last_second, window_size):
        click.echo('Time window is: %d ' % time_window)

        train_window = get_windows(train_csv, time_window, time_window + window_size - 1)
        test_window = get_windows(test_csv, time_window, time_window + window_size - 1)
        
        train_data, train_label = window_to_lists(train_window, 'pose_position_z')
        test_data, test_label = window_to_lists(test_window, 'pose_position_z')


        model = KnnDtw(k_neighbours = k, max_warping_window = w)
        model.fit(np.array(train_data), np.array(train_label))
        
        #click.echo(train_label)

        predicted_label, probability = model.predict(test_data, parallel=True)
        click.echo('Predicted label : %s ' % str(predicted_label))
    
        click.echo('\n')

    click.echo('\nDone.')


@cli.command('findbestk_online')
@click.option('--kmin', default=1, help='Minimum value for k Nearest Neighbours')
@click.option('--kmax', default=5, help='Maximum value for k Nearest Neighbours')
@click.option('-w', default=4000, help='Maximum Warping Window')
@click.option('--window-size', default=2, help='Window size')
@click.option('--seed', default=108, help='Seed for random shuffle')
@click.option('--train', help='Training Data as CSV file path')
@click.option('--feature', help='The feature to use (e.g. pose_position_z)')
def find_best_k_online(kmin, kmax, w, window_size, seed, train, feature):
    click.echo('--- Find best k ---')
    
    kmin = int(min(kmin, kmax))
    kmax = int(max(kmin, kmax))

    ks = range(kmin, kmax + 1)

    train_csv = load_labelled_csvs(train, should_preprocess=False) # [{labal;..., data},...]
    
    #train_label = [flight['label'][0] for flight in train_csv] # Wrong, dont need
    train_data = [flight['data'] for flight in train_csv]
    #print(train_label)
    
    click.echo('  - ks    : %s ' % str(ks))
    click.echo('  - w     : %d ' % w)
    click.echo('  - seed  : %d ' % seed)
    click.echo('  - train : %s ' % train)

    click.echo('  - Training data size: %d' % len(train_data))
    click.echo('\nRunning...')

    trainer = Trainer(seed=seed, data=train_data, data_labels=None)
    trainer.find_best_k_online(ks, w, window_size, feature)
    

    click.echo('\nDone.')


@cli.command('findbestw_online')
@click.option('--wmin', default=100, help='Minimum value for Warping Window')
@click.option('--wmax', default=4000, help='Maximum value for Warping Window')
@click.option('--step', default=100, help='Step for Warping Window')
@click.option('-k', default=3, help='k Nearest Neighbours')
@click.option('--window-size', default=2, help='Window size')
@click.option('--seed', default=108, help='Seed for random shuffle')
@click.option('--train', help='Training Data as CSV file path')
@click.option('--feature', help='The feature to use (e.g. pose_position_z)')
def find_best_w_online(wmin, wmax, step, window_size, k, seed, train, feature):
    click.echo('--- Find best w ---')
    
    wmin = int(min(wmin, wmax))
    wmax = int(max(wmin, wmax))

    ws = range(wmin, wmax, step)

    train_csv = load_labelled_csvs(train, should_preprocess=False)

    train_data = [flight['data'] for flight in train_csv]

    click.echo('  - ws    : %s ' % str(ws))
    click.echo('  - k     : %d ' % k)
    click.echo('  - seed  : %d ' % seed)
    click.echo('  - train : %s ' % train)

    click.echo('  - Training data size: %d' % len(train_data))
    click.echo('\nRunning...')

    trainer = Trainer(seed=seed, data=train_data, data_labels=None)
    trainer.find_best_w_online(k, ws, window_size, feature)

    click.echo('\nDone.')

    
@cli.command('dtw_time_window')
@click.option('--data', help='Single timeseries data as CSV file path')
@click.option('--dataarray', help='List of timeseries data as CSV file path')
@click.option('-w', default=200, help='Maximum Warping Window')
@click.option('--window-size', default=2, help='Window size')
@click.option('--coordinate', default='pose_position_z', help='X/Y/Z coordinate')
@click.option('--should-plot', default=False)
def compute_labels(data, dataarray, w, window_size, coordinate, should_plot):
    click.echo('--- Compute labels ---')

    desired_csv = load_csvs(data, should_preprocess=False)
    data_csv = load_csvs(dataarray, should_preprocess=False)

    for index, flight_data_number in enumerate(data_csv):
        flight_data_number['data']['flight_number'] = index

    # For Test
    last_second = desired_csv[0]['data']['seconds'].iloc[-1]

    #click.echo('  - data        : %s ' % data)
    click.echo('  - dataarray   : %s ' % dataarray)
    click.echo('  - w           : %d ' % w)

    click.echo('\nRunning...')

    # For plot-pdfs title
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H%M%S')
    
    flight_dict = {}
    #flight_full_data = pd.DataFrame()
    flight_dataframe = pd.DataFrame()

    for time_window in range(0, last_second, window_size):
        click.echo('Time window is: %d ' % time_window)

        desired_window = get_windows(desired_csv, time_window, time_window + window_size - 1)
        data_window = get_windows(data_csv, time_window, time_window + window_size - 1)
        #print(data_window)


        # Compute all DTWs

        desired_flight = desired_window[0]['window'][coordinate].tolist()

        dtws = []

        for index, flight_window in enumerate(data_window):
            # Compute DTW of window with desired_window ; window[coordinate].tolist()
            pos_data = flight_window['window'][coordinate].tolist()
            flight_seconds = flight_window['window']['seconds'].tolist()
            
            dtw = dtw_distance(desired_flight, pos_data, w)
            # Add key DTW to window ; window['dtw']
            flight_window['dtw'] = dtw
            # Track all DTWS ; append dtw value
            dtws.append( dtw )
            #click.echo('Done. Plots have been saved.')
            
            print(dtw)

            if should_plot:
                # plotting desired data with each flight for each window
                plt.plot(desired_flight, label='Desired')
                plt.plot(pos_data, label='Unlabelled')
                #plt.xlim(0,15)
                plt.ylim(0,2.1)
                plt.xlabel('Seconds')
                plt.ylabel('Meters')
                plt.title('Desired and unlabelled data. DTW = {}'.format(dtw))
                plt.legend()
                plt.grid(True)
                plt.savefig('plots/z/desired_each_flight/dtwValues_{}_{}_{}.jpeg'.format(st, time_window, index))
                plt.cla()
       
        
        print(dtws)
        if should_plot:
            print(dtws)
            dtws = [550 if x==9.223372036854776e+18 else x for x in dtws]
            # plotting dtw values of all flights for each window
            plt.figure(figsize=(20, 10))
            plt.plot(dtws, color='#006600', alpha=.9) # marker='o', linestyle='--',
            plt.ylim([0, 600])

            flights_x = [i for i in range(0, 53)]
            for flight_x, dtw in zip(flights_x, dtws):
                plt.text(flight_x, dtw, str(dtw))

            plt.grid(True)
            #plt.show()
            plt.savefig('plots/z/dtw_values/dtwValues_{}_{}.pdf'.format(st, time_window))
            plt.cla()

            click.echo('Done. Plots have been saved.')


        # Loop again and update labels
            
        for flight_window in data_window:

            if flight_window['dtw'] < 100:
                flight_window['label'] = 1
            elif flight_window['dtw'] < 299:
                flight_window['label'] = 2
            else: 
                flight_window['label'] = 3

            # Save to flight_dict
            fn = flight_window['window']['flight_number'].iloc[0]
            flight_full_data = flight_window['window']
            flight_full_data.is_copy = False
            flight_full_data['label'] = flight_window['label']
            flight_full_data['dtw'] = flight_window['dtw']

            if fn not in flight_dict.keys():
                flight_dict[fn] = flight_full_data # flight_window['window']
            else:
                flight_dict[fn] = pd.concat([flight_dict[fn], flight_full_data]) # flight_window['window']

    for flight_index in flight_dict:
        flight_dataframe = pd.DataFrame(flight_dict[flight_index])
        flight_dataframe.to_csv('data/labelled/z/f_{}.csv'.format(flight_index), sep=';', encoding='utf-8')
        print('f_{}.csv saved!'.format(flight_index))

   
    click.echo('\nDone.')


if __name__ == '__main__':
    cli()
