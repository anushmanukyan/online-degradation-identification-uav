import random
import numpy as np
import pandas as pd
import pickle

from kNNDTW import KnnDtw
from utils import evaluate
from utils import get_windows_values, get_windows_labels

class Trainer():

    def __init__(self, split_test=0.9, split_validation=0.7, seed=108, labels={1:'GOOD', 2:'BAD', 3:'WORST'}, data=None, data_labels=None):
        self.split_test = split_test
        self.split_validation = split_validation
        self.seed = seed
        self.labels = labels

        self.data = data
        self.data_labels = data_labels

        self._split()

    def _take(self, data, indices):
        ''' Returns only the data of the indices.
        '''
        if not data:
            return None

        if(isinstance(data, (np.ndarray))):
            return np.take(data, indices)
        else:
            result = []
            for index in indices: 
                result.append(data[index])
            
            return result

    def _split(self):
        random.seed(self.seed)

        indices = np.arange(len(self.data))
        random.shuffle(indices)

        # Splitting into training and test

        split = int(round(len(self.data) * self.split_test)) # Split of 90%

        train_validation_index = indices[:split]
        test_index             = indices[split:]

        
        # Splitting into training and validation

        training_split = int(round(len(self.data) * self.split_validation)) # Split of 70%

        training_index   = train_validation_index[:training_split]
        validation_index = train_validation_index[training_split:]
        
        self.training_validation_data          = self._take(self.data, train_validation_index)
        self.training_validation_label_data    = self._take(self.data_labels, train_validation_index)        

        self.test_data          = self._take(self.data, test_index)
        self.test_label_data    = self._take(self.data_labels, test_index)
        
        self.training_data          = self._take(self.data, training_index)
        self.training_label_data    = self._take(self.data_labels, training_index)

        self.validation_data        = self._take(self.data, validation_index)
        self.validation_label_data  = self._take(self.data_labels, validation_index)


    def evaluate_model(self, k, max_warping_window, train_data, train_label, test_data, test_label):
        print ('--------------------------')
        print ('--------------------------\n')
        print ('Running for k = ', k)
        print ('Running for w = ', max_warping_window)
        
        model = KnnDtw(k_neighbours = k, max_warping_window = max_warping_window)
        model.fit(train_data, train_label)
        
        predicted_label, probability = model.predict(test_data, parallel=False)
        
        print ('\nPredicted : ', predicted_label)
        print ('Actual    : ', test_label)
        
        accuracy, precision, recall, f1score = evaluate(self.labels, predicted_label, test_label)
        
        print ('Avg/Total Accuracy  :', accuracy)
        print ('Avg/Total Precision :', precision)
        print ('Avg/Total Recall    :', recall)
        print ('Avg/Total F1 Score  :', f1score)
        
        # result = np.zeros((len(ks),4))
        # result[0] = accuracy
        # result[1] = precision
        # result[2] = recall
        # result[3] = f1score


    # evaluate the model for each window
    def evaluate_online_model(self, k, max_warping_window, train_data, test_data, window_size, feature):
        print('--------------------------')
        print('--------------------------\n')
        print('Running for k = ', k)
        print('Running for w = ', max_warping_window)
        print('Running for window_size = ', window_size)
        print('Running for feature = ', feature)
        
        evaluation_results = []

        # For Test
        last_second = test_data[0]['seconds'].iloc[-1]
        #print('[TEST] Last second: {}'.format(last_second))

        for time_window in range(0, last_second, window_size):
            print('Time window is: %d ' % time_window)

            columns = [feature, 'label']
            train = get_windows_values(train_data, feature, time_window, time_window + window_size - 1)
            test = get_windows_values(test_data, feature, time_window, time_window + window_size - 1)

            train_label = get_windows_labels(train_data, time_window, time_window + window_size - 1)
            test_label = get_windows_labels(test_data, time_window, time_window + window_size - 1)

            print('Train labels {}'.format(train_label))
            print('Test labels {}'.format(test_label))

            print(len(train))
            print(len(test))
            for i in range(0, len(test)):
                print('Flight_{}:{}'.format(i, len(test[i])))

            model = KnnDtw(k_neighbours = k, max_warping_window = max_warping_window)
            model.fit(np.array(train), np.array(train_label))
            
            predicted_label, probability = model.predict(test, parallel=False)

            
            print ('\nPredicted : ', predicted_label)
            print ('Actual    : ', test_label)
            
            accuracy, precision, recall, f1score = evaluate(self.labels, predicted_label, test_label)

            print ('Avg/Total Accuracy  :', accuracy)
            print ('Avg/Total Precision :', precision)
            print ('Avg/Total Recall    :', recall)
            print ('Avg/Total F1 Score  :', f1score)


            results = {
                'time_window': time_window,
                'predicted_labels': predicted_label.tolist(),
                'actual_labels': test_label,
                'total_accuracy': accuracy,
                'total_precision': precision,
                'total_recall': recall,
                'total_f1score': f1score
            }

            evaluation_results.append( results )
        
        return evaluation_results



    def find_best_k(self, ks, max_warping_window):
        for index, k in enumerate(ks):
            self.evaluate_model(k, max_warping_window, self.training_data, self.training_label_data, self.validation_data, self.validation_label_data)

    def find_best_k_online(self, ks, max_warping_window, window_size, feature):
        results = [] 
        for index, k in enumerate(ks):
            evaluation = self.evaluate_online_model(k, max_warping_window, self.training_data, self.validation_data, window_size, feature)
            results_with_k = {
                'k' : k,
                'evaluation_all_ks' : evaluation
            }
            print(results_with_k)
            # Save each k result in csv
            df_k_res = pd.DataFrame(results_with_k['evaluation_all_ks'])
            df_k_res['k'] = results_with_k['k']
            df_k_res.to_csv('results/k_results/k_{}_{}.csv'.format(feature, df_k_res['k'][0]), sep=';', encoding='utf-8')
            print('CSV saved!')
            results.append(results_with_k)

        #Save in csv
        print(results) 
        df_result = pd.DataFrame()
        for res in results: 
            df_all_k = pd.DataFrame(res['evaluation_all_ks'])
            df_all_k['k'] = res['k']
            df_result = df_result.append(df_all_k)
            df_result.to_csv('results/k_results/k_{}_all.csv'.format(feature), sep=';', encoding='utf-8')
            print('CSV saved!')
        
        #with open('C:\\Users\\anush.manukyan\\Desktop\\Online_detection_uav_degradation\\k_res.txt', 'wb') as f:
        #    pickle.dump(results, f)
        #print(results)

    def find_best_w(self, k, max_warping_windows):
        for index, w in enumerate(max_warping_windows):
            self.evaluate_model(k, w, self.training_data, self.training_label_data, self.validation_data, self.validation_label_data)


    def find_best_w_online(self, k, max_warping_windows, window_size, feature):
        results = [] 
        for index, w in enumerate(max_warping_windows):
            evaluation = self.evaluate_online_model(k, w, self.training_data, self.validation_data, window_size, feature)
            results_with_w = {
                'w' : w,
                'evaluation_all_ws' : evaluation
            }
            #results.append(results_with_w)
            print(results_with_w)
            df_w_res = pd.DataFrame(results_with_w['evaluation_all_ws'])
            df_w_res['w'] = results_with_w['w']
            df_w_res.to_csv('results/w_results/w_{}_{}.csv'.format(df_w_res['w'][0], feature), sep=';', encoding='utf-8')
            print('CSV saved!')
            results.append(results_with_w)

        #Save in csv
        print(results) 
        df_result = pd.DataFrame()
        for res in results: 
            df_all_w = pd.DataFrame(res['evaluation_all_ws'])
            df_all_w['w'] = res['w']
            df_result = df_result.append(df_all_w)
            df_result.to_csv('results/w_results/w_{}_all.csv'.format(feature), sep=';', encoding='utf-8')
            print('CSV saved!')
        #with open('C:\\Users\\anush.manukyan\\Desktop\\Online_detection_uav_degradation\\k_res.txt', 'wb') as f:
        #    pickle.dump(results, f)
        #print(results)

    def evalute_best_model(k, max_warping_window):
        self.evaluate_model(k, max_warping_window, self.training_validation_data, self.training_validation_label_data, self.test_data, self.test_label_data)


 
