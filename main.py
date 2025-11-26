
import numpy as np
import pandas as pd
import os
import glob
from ETD import TrackingbyDetection

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
        events_set = events_set.astype(np.int64)
        events_set[:, 0] -= events_set[0, 0]
        print("Events total count: ", len(events_set))
    elif 'shapes' in sequence:
        events = pd.read_csv(
            '{}/{}/events.txt'.format(path_dataset,sequence), sep=" ", header=None)  
        events_set = events.to_numpy() # [t, x, y, p]
        print("Events total count: ", len(events_set))
        print("Time duration of the sequence: {} s".format(events_set[-1, 0] - events_set[0, 0]))
        events_set[:, 0] *= 1e+6    # s -> us
        events_set = events_set.astype(np.int64)
        events_set[:, 0] -= events_set[0, 0]
    else:
        # 从csv文件中读取数据   (us)
        print(f'Reading from {path_dataset}/{sequence}/{fname}')
        events = pd.read_csv(
            '{}/{}/{}'.format(path_dataset, sequence, fname), sep=",", header=None)  
        events.columns = ['x', 'y', 'p', 't']  
        events_set = events.to_numpy()[:, [3, 0, 1, 2]] # t, x, y, p
        events_set = events_set[np.argsort(events_set[:, 0])]
        print("Events total count: ", len(events_set))
        print("Time duration of the sequence: {} s".format((events_set[-1, 0] - events_set[0, 0])*1e-6))
    return events_set

if __name__ == '__main__':
    # dataset_path, sequence, fname, image_size = 'dataset', 'shapes_translation', '', (180, 240)
    # dataset_path, sequence, fname, image_size = 'dataset', 'shapes_rotation', '', (180, 240)
    dataset_path, sequence, fname, image_size = 'dataset', 'shapes_6dof', '', (180, 240)
    # 需要提供初始化跟踪坐标
    gt_files = glob.glob(f'./dataset/{sequence}/locations/*.txt')
    gt_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    events_set = load_dataset(dataset_path, sequence, fname=fname)
    save_dir = './' + sequence + '_tracking_res1126'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Start Tracking-by-Detection...')
    TD = TrackingbyDetection(events_set, gt_files=gt_files)
    TD.forward(save_dir)
    
    
    
