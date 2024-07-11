import cv2, os
import numpy as np
from PIL import Image, ImageDraw
import torch
import detectron2.utils.comm as comm

WIDTH = 200
DEBUG_SIZE = (WIDTH, WIDTH)

class Debug:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()
    
    def reset(self):
        self.enable = True
        self.count = 0
        self.figures = []
        self.names = []
    
    def tick(self):
        self.figures = []
        self.names = []
        self.enable = (self.count % self.cfg.DEBUG.TICK_PERIOD)==0
        self.count += 1

        if not comm.is_main_process():
            self.enable = False

    def clear(self):
        self.figures = []
        self.names = []

    def add_figures(self, name, figures, size=DEBUG_SIZE):
        """ figures: B, H, W, [C] """
        if self.enable==False: return
        lst = [
            cv2.resize( (m * 255).astype(np.uint8), size[::-1], interpolation=cv2.INTER_LINEAR)
            for m in figures
        ]
        lst = [np.array(Image.fromarray(m).convert("RGB")) for m in lst]
        self.figures.append(lst)
        self.names.append(name)
    
    def add_points(self, name, points, size=DEBUG_SIZE):
        if self.enable==False: return
        """ points \in [0.0, 1.0] \in B, K, 2 """
        lst = []
        for ps in points:
            fig = np.zeros(size+(3,))
            ps = (ps[:, 0:2] * np.array(size).reshape(1, 2)).astype(int)
            fig[ps[:,0], ps[:,1]] = np.array([255, 0, 0])
            lst.append(fig.astype(np.uint8))
        self.figures.append(lst)
        self.names.append(name)
    
    def add_skeleton(self, name, keypoints, size=DEBUG_SIZE):
        if self.enable==False: return
        skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        lst = []
        for ps in keypoints:
            img = Image.fromarray(np.zeros(size+(3,)).astype(np.uint8))
            xy = (ps[:, 0:2] * np.array(size[::-1]).reshape(1, 2)).astype(int)
            for line in skeleton:
                i, j = line[0]-1, line[1]-1
                if ps[i, 2] > 0.5 and ps[j, 2] > 0.5:
                    ImageDraw.Draw(img).line( tuple(xy[i])+tuple(xy[j]), fill="blue")
            lst.append(np.array(img).astype(np.uint8))
        self.figures.append(lst)
        self.names.append(name)

    def add_str(self, name, text, size=(100, WIDTH), color="black"):
        if self.enable==False: return
        """ text is a list/tuple of string """
        lst = []
        for s in text:
            fig = np.ones(size+(3,)) * 255
            img = Image.fromarray(fig.astype(np.uint8))
            ImageDraw.Draw(img).text((0, 0), str(s), fill=color)
            lst.append(np.array(img).astype(np.uint8))
        self.figures.append(lst)
        self.names.append(name)

    def get_fig(self):
        if self.enable==False: return
        rows = [np.concatenate(lst, axis=1) for lst in self.figures]
        width = max([x.shape[1] for x in rows])
        heights = [x.shape[0] for x in rows]
        for i in range(len(rows)):
            h, w, c = rows[i].shape
            tmp = np.ones((h, width, c)) * 255
            tmp[:, 0:w, :] = rows[i]
            rows[i] = tmp
        fig = np.concatenate(rows, axis=0)

        captions = [Image.fromarray((np.ones((h, 200, 3)) * 255).astype(np.uint8)) for h in heights]
        for name, cap in zip(self.names, captions):
            ImageDraw.Draw(cap).text((0, 0), str(name), fill="blue")
        captions = np.concatenate([np.array(cap) for cap in captions], axis=0)
        out = np.concatenate([captions, fig], axis=1)
        return out.astype(np.uint8)
    
    def show(self):
        if self.enable==False: return
        fig = self.get_fig()
        Image.fromarray(fig).show()
    
    def save(self, filename="latest.png"):
        if self.enable==False: return
        fig = self.get_fig()
        filename = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.DEBUG.DEBUG_DIR, "iter_{}_{}".format(self.count-1, filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        Image.fromarray(fig).save(filename)

        self.clear()
