import numpy as np
import cv2
import numba as nb
import matplotlib.pyplot as plt

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
            entropy = grayhist*np.log2(grayhist+1e-6)
            entropy = -entropy.sum()
            NZGE.append(entropy)
    NZGE = np.array(NZGE)
    return NZGE.sum() / np.count_nonzero(NZGE)
    
class TrackingbyDetection():
    def __init__(self, events=None, im_size=(180, 240), prev_boxes=[]):
        self.gamma = 1.5 # search scale
        self.mu = 0.3 # tracking threshold
        self.alpha=0.0832
        self.beta=0.0927
        self.lamda = 0.7
        self.prev_num = 0
        self.prev_boxes = prev_boxes
        self.im_size = im_size
        self.model = './model.yml.gz'
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(self.model)
        self.edge_boxes = cv2.ximgproc.createEdgeBoxes(maxBoxes=1000, minBoxArea=100)
        # additionally
        if len(prev_boxes) > 0:
            self.last_name_id = len(prev_boxes)
            self.prev_save_nameid = np.array(range(0, len(prev_boxes)))
        else:
            self.prev_save_nameid = np.array([])
            self.last_name_id = 0
        self.events = events

    def AdaptiveTimeSurface(self):
        '''
        Return [np.array([h, w, 2])]*N, N means the number of ATSLTD frames.
        '''
        F = np.zeros((*self.im_size, 3), np.float32)
        entropy = np.zeros(2)
        prevt = self.events[self.prev_num, 0]
        for i in range(self.prev_num, len(self.events)):
            t, x, y, p = self.events[i]
            # F[:, :, p] = np.rint(F[:, :, p]*1.*prevt/(t+1E-6))
            F = np.round(F*1.*prevt/(t+1E-6))
            prevt = t
            F[y, x, p] = 255
            if i % 50 == 0:
                entropy[p] = calcEntropy(F[:, :, p].astype(np.uint8))
                if entropy.mean() >= self.alpha and entropy.mean() <= self.beta:
                    self.prev_num = i+1
                    return F
        self.prev_num = len(self.events) - 1
        return F

    def detect(self, im, prev_box=None):
        im = im.astype(np.float32) / 255.0 
        edges = self.edge_detection.detectEdges(im)
        orimap = self.edge_detection.computeOrientation(edges)
        edges = self.edge_detection.edgesNms(edges, orimap)
        boxes, scores = self.edge_boxes.getBoundingBoxes(edges, orimap) # (x, y, x, y)
        if prev_box is None:
            return boxes[scores[:, 0] > 0.1]
        refined_boxes, refined_scores = self.refine_proposals(boxes, scores, prev_box)
        if len(refined_scores) == 0:
            return refined_boxes
        return refined_boxes[refined_scores.argmax()]
    
    def track(self, boxes):
        
        def cvtxyxy(regions):
            boxes = []
            for region in regions:
                x, y, w, h = region
                box = np.array((x, y, x+w, y+h))
                boxes.append(box)
            if len(boxes) > 0:
                boxes = np.stack(boxes, 0)
            else:
                boxes = np.empty((0, 4), np.int32)
            return boxes

        def compute_IoU(curr, prev):
            '''
            Inputs
                current - xyxy List[[N, 4]]
                previous - xyxy np.array([M, 4])
            Return
                iou [N, M]
            '''
            A = curr.shape[0]
            B = prev.shape[0]
            curr = cvtxyxy(curr)
            prev = cvtxyxy(prev)
            xy_max = np.minimum(curr[:, np.newaxis, 2:].repeat(B, axis=1),
                                np.broadcast_to(prev[:, 2:], (A, B, 2)))
            xy_min = np.maximum(curr[:, np.newaxis, :2].repeat(B, axis=1),
                                np.broadcast_to(prev[:, :2], (A, B, 2)))

            # 计算交集面积
            inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
            inter = inter[:, :, 0]*inter[:, :, 1]

            # 计算每个矩阵的面积
            area_0 = ((curr[:, 2]-curr[:, 0])*(
                curr[:, 3] - curr[:, 1]))[:, np.newaxis].repeat(B, axis=1)
            area_1 = ((prev[:, 2] - prev[:, 0])*(
                prev[:, 3] - prev[:, 1]))[np.newaxis, :].repeat(A, axis=0)
            return inter/(area_0+area_1-inter+1e-6)
        
        IoU = compute_IoU(boxes, self.prev_boxes)
        vals = IoU.max(1)
        ids = np.argmax(IoU, axis=1)
        success_track = []
        curr_save_nameid = np.full(len(boxes), -1)
        for i, id in enumerate(ids):
            if vals[i] > self.mu:
                success_track.append(boxes[i])
                curr_save_nameid[i] = self.prev_save_nameid[id]
        if len(success_track) > 0:
            self.prev_boxes = np.stack(success_track, 0)
        return curr_save_nameid
        
    def unwarp_events(self, start, end, boxes):
        '''
        To get the original trajectories composed of events.
        boxes - np.array((xywh))
        '''
        start = np.nonzero(self.events[:, 0] >= start)[0][0]
        end = np.nonzero(self.events[:, 0] >= end)[0][0]
        indice = self.events[start: end]
        trajectories = []
        for b in boxes:
            take_id = np.logical_and(np.logical_and(indice[:, 1] >= b[0], indice[:, 1] <= b[0]+b[2]), \
                                     np.logical_and(indice[:, 2] >= b[1], indice[:, 2] <= b[1]+b[3]),)
            trajectories.append(indice[take_id])
        return trajectories
    
    def refine_proposals(self, proposals, scores, previous_bbox):
        """
        论文公式9
        
        score(O_{i-1}, P_i) = φ(w_{i-1}×h_{i-1} / w_{p_i}×h_{p_i}) × φ((w_{i-1}/h_{i-1}) / (w_{p_i}/h_{p_i}))
        """
        def _phi_function(x: float) -> float:
            """
            φ函数定义 (论文公式10)
            
            φ(x) = { x    if 0 < x < 1
                    1/x  if x ≥ 1 }
            """
            if x <= 0:
                return 0
            elif 0 < x < 1:
                return x
            else:  # x >= 1
                return 1.0 / x
        if proposals is None or len(proposals) == 0:
            return np.array([]), np.array([])
        
        if previous_bbox is None:
            return proposals, scores
        
        x_prev, y_prev, w_prev, h_prev = previous_bbox
        area_prev = w_prev * h_prev
        aspect_ratio_prev = w_prev / h_prev if h_prev > 0 else 1.0
        
        refined_boxes = []
        refined_scores = []
        
        for i, box in enumerate(proposals):
            x, y, w, h = box
            area_current = w * h
            aspect_ratio_current = w / h if h > 0 else 1.0
            
            # 计算面积比例得分
            if area_current > 0:
                area_ratio = area_prev / area_current
                area_score = _phi_function(area_ratio)
            else:
                area_score = 0
            
            # 计算宽高比得分
            if aspect_ratio_current > 0:
                aspect_ratio = aspect_ratio_prev / aspect_ratio_current
                aspect_score = _phi_function(aspect_ratio)
            else:
                aspect_score = 0
            
            # 综合得分 (论文公式9)
            total_score = area_score * aspect_score
            
            # 过滤低分候选框
            if total_score > self.lamda:
                refined_boxes.append(box)
                refined_scores.append(total_score)
        
        if len(refined_boxes) == 0:
            return np.array([]), np.array([])
        
        return np.array(refined_boxes), np.array(refined_scores)
    
    def forward(self, savedir='./tracking_res/'):
        cnt = 0
        while self.prev_num < len(self.events) - 1:
            print(f'Converting {cnt}-th ATSLD frame...')
            start = self.events[self.prev_num, 0]
            im = self.AdaptiveTimeSurface()
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
                    # search regions
                    x, y, w, h = j
                    c_x = x + w//2
                    c_y = y + h//2
                    w *= self.gamma
                    h *= self.gamma
                    new_y1 = max(int(c_y-h//2), 0)
                    new_x1 = max(int(c_x-w//2), 0)
                    new_y2 = min(int(c_y+h//2), self.im_size[0])
                    new_x2 = min(int(c_x+w//2), self.im_size[0])
                    if new_y1 > new_y2 or new_x1 > new_x2:
                        continue
                    im_ = im.copy()
                    mask = np.zeros(self.im_size)
                    mask[new_y1:new_y2, new_x1:new_x2] = 1
                    im_[:, :, 0] *= mask
                    im_[:, :, 1] *= mask
                    boxes = self.detect(im_, j)
                    if len(boxes) > 0:
                        new_boxes.append(boxes)
                if len(new_boxes) > 0:
                    boxes = np.stack(new_boxes, 0)
                    curr_save_nameid = self.track(boxes)
                else:
                    self.prev_boxes = np.empty((0, 4))
                    continue
            # saving trajectories
            trajectories = self.unwarp_events(start, end, boxes)
            self.prev_save_nameid = curr_save_nameid
            for j, k in enumerate(curr_save_nameid):
               if len(trajectories[j]) > 0:
                    if k == -1:
                        with open(f'{savedir}/{self.last_name_id}.txt', 'a+') as f:
                            np.savetxt(f, np.c_[trajectories[j]], fmt='%d', delimiter=',') # us, x, y, p
                        self.prev_save_nameid[j] = self.last_name_id
                        self.last_name_id += 1
                    else:
                        with open(f'{savedir}/{k}.txt', 'a+') as f:
                            np.savetxt(f, np.c_[trajectories[j]], fmt='%d', delimiter=',') # us, x, y, p