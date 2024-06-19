import numpy as np
import cv2
import numba as nb

colormap = ['c', 'g', 'r', 'm', 'y', 'b', 'w', 'k']
shapemap = ['o', '1', 's', 'p', '+', '*', 'x', 'v']
    
@nb.jit(nopython=True)
def calcEntropy(frame):
    q, p, r = 45, 60, 4
    NZGE = []
    for i in range(0, q-1):
        for j in range(0, p-1):
            grayhist = np.zeros(256, np.float32)
            for a in range(i*r, i*r+r):
                for b in range(j*r, j*r+r):
                    grayhist[frame[a, b]] += 1
            grayhist /= r*r
            log_g = np.log2(grayhist+1e-6)
            entropy = grayhist*log_g
            entropy = -entropy.sum()
            NZGE.append(entropy)
    NZGE = np.array(NZGE)
    return NZGE.mean()
    
class TrackingbyDetection():
    def __init__(self, events=None, im_size=(180, 240)):
        self.gamma = 1.5 # search scale
        self.mu = 0.3 # tracking threshold
        self.alpha=0.0832
        self.beta=0.0927
        self.lamda = 0.1 # 论文上为0.7，但0.7检测不到任何框
        self.prev_num = 0
        self.prev_boxes = []
        self.im_size = im_size
        self.model = './model.yml.gz'
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(self.model)
        self.edge_boxes = cv2.ximgproc.createEdgeBoxes(maxBoxes=1000)
        # additionally
        self.prev_save_nameid = np.array([])
        self.events = events
        self.last_name_id = 0

    def AdaptiveTimeSurface(self):
        '''
        Return [np.array([h, w, 2])]*N, N means the number of ATSLTD frames.
        '''
        F = np.zeros((*self.im_size, 3), np.uint8)
        entropy = np.zeros(2)
        prevt = self.events[self.prev_num, 0]
        for i in range(self.prev_num, len(self.events)):
            t, x, y, p = self.events[i]
            F[:, :, p] = np.rint(F[:, :, p]*1.*prevt/(t+1E-6))
            prevt = t
            F[y, x, p] = 255
            if i % 50 == 0:
                entropy[p] = calcEntropy(F[:, :, p])
                if entropy.mean() >= self.alpha and entropy.mean() <= self.beta:
                    self.prev_num = i+1
                    return F
        self.prev_num = len(self.events) - 1
        return F

    def detect(self, im):
        edges = self.edge_detection.detectEdges(np.float32(im) / 255.0)
        orimap = self.edge_detection.computeOrientation(edges)
        edges = self.edge_detection.edgesNms(edges, orimap)
        boxes, scores = self.edge_boxes.getBoundingBoxes(edges, orimap) # (x, y, w, h)
        return boxes[scores[:, 0]>self.lamda]
    
    def track(self, boxes):
        
        def cvtxyxy(regions):
            boxes = []
            for region in regions:
                x, y, w, h = region
                box = (x, y, x+w, y+h)
                boxes.append(box)
            return np.stack(boxes, 0)

        def compute_IoU(prev, curr):
            '''
            Inputs
                previous - xyxy np.array([M, 4])
                current - xywh List[[N, 4]]
            Return
                iou [M, N]
            '''
            A = prev.shape[0]
            B = curr.shape[0]
            curr = cvtxyxy(curr)
            xy_max = np.minimum(prev[:, np.newaxis, 2:].repeat(B, axis=1),
                                np.broadcast_to(curr[:, 2:], (A, B, 2)))
            xy_min = np.maximum(prev[:, np.newaxis, :2].repeat(B, axis=1),
                                np.broadcast_to(curr[:, :2], (A, B, 2)))

            # 计算交集面积
            inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
            inter = inter[:, :, 0]*inter[:, :, 1]

            # 计算每个矩阵的面积
            area_0 = ((prev[:, 2]-prev[:, 0])*(
                prev[:, 3] - prev[:, 1]))[:, np.newaxis].repeat(B, axis=1)
            area_1 = ((curr[:, 2] - curr[:, 0])*(
                curr[:, 3] - curr[:, 1]))[np.newaxis, :].repeat(A, axis=0)
            return inter/(area_0+area_1-inter)
        
        IoU = compute_IoU(self.prev_boxes, boxes)
        vals = IoU.max(1)
        ids = np.argmax(IoU, axis=1)
        success_track = []
        curr_save_nameid = np.full(len(boxes), -1)
        for i, id in enumerate(ids):
            if vals[i] > self.mu:
                success_track.append(boxes[id])
                curr_save_nameid[id] = self.prev_save_nameid[i]
        if len(success_track) > 0:
            self.prev_boxes = np.stack(success_track, 0)
        return curr_save_nameid
        
    def unwarp_events(self, start, end, boxes):
        '''
        To get the original trajectories composed of events.
        boxes - np.array((xyxy))
        '''
        start = np.nonzero(self.events[:, 0] >= start)[0][0]
        end = np.nonzero(self.events[:, 0] >= end)[0][0]
        indice = self.events[start: end]
        trajectories = []
        for b in boxes:
            take_id = np.logical_and(np.logical_and(indice[:, 1] >= b[0], indice[:, 1] <= b[2]), \
                                     np.logical_and(indice[:, 2] >= b[1], indice[:, 2] <= b[3]),)
            trajectories.append(indice[take_id])
        return trajectories
        
    def forward(self, savedir='./tracking_res/'):
        cnt = 0
        while self.prev_num < len(self.events) - 1:
            start = self.events[self.prev_num, 0]
            im = self.AdaptiveTimeSurface()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            end = self.events[self.prev_num, 0]
            print(f'{cnt}-th ATSLD frame is done. Left {len(self.events) - self.prev_num} events to deal.')
            cnt += 1
            if len(self.prev_boxes) == 0:
                boxes = self.detect(im)
                self.prev_boxes = boxes
                curr_save_nameid = np.full(len(boxes), -1)
            else:
                new_boxes = []
                for j in self.prev_boxes:
                    x, y, w, h = j
                    c_x = x + w//2
                    c_y = y + h//2
                    w *= self.gamma
                    h *= self.gamma
                    im_ = im[int(c_y-h//2):int(c_y+h//2), int(c_x-w//2):int(c_x+w//2)]
                    boxes = self.detect(im_)
                    new_boxes.extend(boxes)
                boxes = np.stack(new_boxes, 0)
                curr_save_nameid = self.track(boxes)
            # saving trajectories
            trajectories = self.unwarp_events(start, end, boxes)
            self.prev_save_nameid = curr_save_nameid
            for j, k in enumerate(curr_save_nameid):
                if k == -1:
                    with open(f'{savedir}/{self.last_name_id}.txt', 'a+') as f:
                        np.savetxt(f, np.c_[trajectories[j]], fmt='%d', delimiter=',') # us, x, y, p
                    self.prev_save_nameid[j] = self.last_name_id
                    self.last_name_id += 1
                else:
                    with open(f'{savedir}/{k}.txt', 'a+') as f:
                        np.savetxt(f, np.c_[trajectories[j]], fmt='%d', delimiter=',') # us, x, y, p