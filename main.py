
import numpy as np
import pandas as pd
import os
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
    dataset_path, sequence, fname, image_size = 'dataset', 'shapes_rotation', '', (180, 240)
    # dataset_path, sequence, fname, image_size = 'dataset', 'shapes_6dof', '', (180, 240)
    # # dataset_path, sequence, fname = 'dataset', 'star_tracking', 'Sequence4.csv' # us
    # 需要提供初始化跟踪坐标
    events_set = load_dataset(dataset_path, sequence, fname=fname)
    '''truth loading'''
    gt_file = f'./dataset/{sequence}/locations/frame_00000000.txt'
    gts = np.loadtxt(gt_file, delimiter=',').reshape(-1, 5)
    gts = gts[:, :-1]
    # cvt xyxy2xywh
    gts[:, 2] -= gts[:, 0]
    gts[:, 3] -= gts[:, 1]
    
    if sequence == 'star_tracking':
        save_dir = './' + sequence + '/' + fname[:-4] + '_tracking_res'
    else:
        save_dir = './' + sequence + '_tracking_res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Start Tracking-by-Detection...')
    TD = TrackingbyDetection(events_set, prev_boxes=gts)
    TD.forward(save_dir)
    
    
    # SOD
    # mid = 'near'
    # dataset_path, sequence, image_size = 'dataset', '1-left', (720, 1280) # dt
    # dataset_path, sequence, image_size = 'dataset', '2-middle', (720, 1280) # dt
    # dataset_path, sequence, image_size = 'dataset', '3-right', (720, 1280) # dt
    # files = os.listdir(f'./{dataset_path}/{mid}/{sequence}')
    # dataset_path, sequence, image_size = 'dataset', 'waterdrops', (720, 1280) # dt
    # files = os.listdir(f'./{dataset_path}//{sequence}')
    # for id, fname in enumerate(files):
    #     events_set = load_dataset(dataset_path, sequence, fname)
    #     save_dir = './' + '/' + sequence + '/' +fname[:-4] + '_tracking_res'
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     print('Start Tracking-by-Detection...')
    #     TD = TrackingbyDetection(events_set, image_size)
    #     TD.forward(save_dir)
            
