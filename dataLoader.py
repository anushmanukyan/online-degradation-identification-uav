import numpy as np
import csv
import sys

class DataLoader():

    def __init__(self):
        self.x_data = []
        self.label_data = []
        csv.field_size_limit(sys.maxsize)

    def load_labelled(self, csv_file_path):

        with open(csv_file_path, 'rU') as csvfile:
            uavreader = csv.reader(csvfile, delimiter=';', quotechar='|')

            for row in uavreader:
                self.label_data.append(int(row[0]))
                self.x_data.append([float(ts) for ts in row[1].split()])
                #print(x_data)
            # Convert to numpy for efficiency

            self.x_data     = np.array(x_data)
            self.label_data = np.array(label_data)

            return [self.x_data, self.label_data]