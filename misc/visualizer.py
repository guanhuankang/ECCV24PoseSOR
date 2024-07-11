from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog
import matplotlib.cm as cm
import numpy as np
import copy

class Visualization(object):
    def __init__(self):
        pass

    def drawInstances(self, image, pred_masks, pred_classes, pred_scores):
        """
        image: H,W,C (rgb format) numpy.ndarray
        pred_masks: list of array of np.uint8
        pred_classes: list of int/str
        pred_scores: list of float
        """
        visualizer = Visualizer(image, instance_mode=ColorMode.IMAGE)
        instances = Instances(
            image.shape[0:2],
            pred_masks=pred_masks,
            pred_scores=pred_scores,
            pred_classes=pred_classes
        )
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return vis_output.get_image()

    def overlayInstances(self, image, pred_masks, pred_classes):
        labels = copy.deepcopy(pred_classes)
        labels.sort()
        labels = labels[::-1]
        color_maps = dict( (l, cm.jet(x)[0:3]) for l,x in zip(labels, np.linspace(0.0, 1.0, len(pred_classes))) )
        colors = [color_maps[l] for l in pred_classes]

        visualizer = Visualizer(image, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.overlay_instances(masks=pred_masks, labels=pred_classes, assigned_colors=colors)

        return vis_output.get_image()