
import numpy as np
import pandas as pd
import os
from ETD import TrackingbyDetection

image_size = (180, 240)
def load_dataset(path_dataset, sequence, fname=None):
    if sequence=='star_tracking': # 1ms
        events = pd.read_csv(
            '{}/{}/{}'.format(path_dataset, sequence, fname), sep=",", header=None)  
        events.columns = ['timestamp', 'y', 'x', 'polarity'] 
        events_set = events.to_numpy()
        events_set = events_set[:, [0, 2, 1, 3]] # [t, y, x, p] -> [t, x, y, p]
        take_id = np.logical_and(np.logical_and(np.logical_and(events_set[:, 1] >= 0, \
                                                               events_set[:, 2] >= 0), \
                                                               events_set[:, 1] < 240), \
                                                               events_set[:, 2] < 180)
        events_set = events_set[take_id]
        print("Time duration of the sequence: {} s".format(events_set[-1, 0]*1e-3))
        events_set[:, 0] *= 1e+3 # us
        print("Events total count: ", len(events_set))
    else:
        events = pd.read_csv(
            '{}/{}/events.txt'.format(path_dataset,sequence), sep=" ", header=None)  
        events_set = events.to_numpy() # [t, x, y, p]
        print("Events total count: ", len(events_set))
        print("Time duration of the sequence: {} s".format(events_set[-1, 0] - events_set[0, 0]))
        events_set[:, 0] *= 1e+6    # s -> us
    events_set = events_set.astype(np.int64)
    events_set[:, 0] -= events_set[0, 0]
    return events_set

if __name__ == '__main__':
    # dataset_path, sequence, fname = 'dataset', 'shapes_translation', ''
    # dataset_path, sequence, fname = 'dataset', 'shapes_rotation', ''
    dataset_path, sequence, fname = 'dataset', 'shapes_6dof', ''
    # dataset_path, sequence, fname = 'dataset', 'star_tracking', 'Sequence4.csv' # us
    events_set = load_dataset(dataset_path, sequence, fname=fname)
    if sequence == 'star_tracking':
        save_dir = './' + sequence + '/' + fname[:-4] + '_tracking_res'
    else:
        save_dir = './' + sequence + '_tracking_res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Start Tracking-by-Detection...')
    TD = TrackingbyDetection(events_set)
    TD.forward(save_dir)
        
