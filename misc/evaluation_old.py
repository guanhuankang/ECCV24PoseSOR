import os
import datetime
import numpy as np
import scipy.stats
from PIL import Image
from scipy import stats
import datetime
import tqdm, json
import pandas as pd

def loadJson(file):
    if not os.path.exists(file):
        return []
    with open(file, "r") as f:
        data = json.load(f)
    return data

def dumpJson(data, file):
    with open(file, "w") as f:
        json.dump(data, f)

def dumpXlsx(data, file):
    pd.DataFrame(data).set_index("name").to_excel(file)

class Evaluation:
    def __init__(self):
        super().__init__()

    def IOU(self, pred, gt):
        assert pred.shape==gt.shape
        pred = (pred * 1).astype(np.int32)
        gt = (gt * 1).astype(np.int32)
        inter = np.logical_and(pred>0, gt>0).sum()
        union = np.logical_or(pred>0, gt>0).sum()
        return inter / (union + 1e-6)

    def mae(self, pred, gt):
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE
        p = pred.astype(float) / 255.
        g = gt.astype(float) / 255.
        return np.mean(np.abs(p - g))

    def acc(self, pred, gt):
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE
        p = (pred > 0.0) * 1.0
        g = (gt > 0.0) * 1.0
        return 1.0 - np.mean(np.abs(p - g))

    def fbeta(self, pred, gt):
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE
        p = (pred > 0.0) * 1.0
        g = (gt > 0.0) * 1.0
        tp = (p * g).sum()
        fp = (p * (1.0-g)).sum()
        fn = ((1.0-p) * g).sum()
        pre = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        return ( 1.3 * pre * rec + 1e-6 ) / ( 0.3 * pre + rec + 1e-6 )

    def AP(self, pred, gt, thres = 0.5):
        BG = 0
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE

        gt_uni = np.unique(np.append(np.unique(gt), BG))
        gt_map = dict( (x,r) for r,x in enumerate(gt_uni) )
        pred_uni = np.unique(np.append(np.unique(pred), BG))
        pred_map = dict( (x,r) for r,x in enumerate(pred_uni) )

        matrixs = []
        for gval in gt_uni:
            if gval==BG: continue
            pval = stats.mode(pred[np.where(gt == gval)], keepdims=False).mode
            if pval==BG: continue
            iou = self.IOU(pred==pval, gt==gval)
            if iou >= thres:
                matrixs.append( (iou, gt_map[gval], pred_map[pval]) )

        n_gt = len(gt_uni) - 1
        n_pred = len(pred_uni) - 1
        hit_pred = np.unique([x[-1] for x in matrixs])
        n_hit = len(hit_pred)

        if n_gt<=0:
            ap = 0.0 if n_pred>0 else 1.0
            rec = 1.0
        else:
            ap = (n_hit / n_pred) if n_pred>0 else 0.0
            rec = n_hit / n_gt
        return ap, rec

    def SOR(self, pred, gt, thres = 0.5):
        BG = 0
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE

        gt_uni = np.unique(np.append(np.unique(gt), BG))
        gt_map = dict( (x,r) for r,x in enumerate(gt_uni) )
        pred_uni = np.unique(np.append(np.unique(pred), BG))
        pred_map = dict( (x,r) for r,x in enumerate(pred_uni) )

        pred_rank = []
        gt_rank = []
        for gval in gt_uni:
            if gval==BG: continue
            pred_inside_seg = np.where(gt==gval, pred, BG)
            pred_inside_pixel = pred_inside_seg[np.where(pred_inside_seg > BG)]
            pred_inside_num = len(pred_inside_pixel)
            r = BG
            if pred_inside_num > int((gt==gval).sum() * thres):
                r = stats.mode(pred_inside_pixel, keepdims=False).mode
            if r > BG:
                pred_rank.append(pred_map[r])
                gt_rank.append(gt_map[gval])
        if len(gt_rank) > 1:
            spr = stats.spearmanr(pred_rank, gt_rank).statistic
            return spr ## normalize
        elif len(gt_rank) == 1:
            return 1
        else:
            return np.nan

    def saSOR(self, pred, gt, thres = 0.5):
        BG = 0
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE

        gt_uni = np.unique(np.append(np.unique(gt), BG))
        gt_map = dict( (x,r) for r,x in enumerate(gt_uni) )
        pred_uni = np.unique(np.append(np.unique(pred), BG))
        pred_map = dict( (x,r) for r,x in enumerate(pred_uni) )

        matrixs = []
        for gval in gt_uni:
            if gval==BG: continue
            pval = stats.mode(pred[np.where(gt == gval)], keepdims=False).mode
            if pval==BG: continue
            iou = self.IOU(pred==pval, gt==gval)
            if iou >= thres:
                matrixs.append( (iou, gt_map[gval], pred_map[pval]) )
        matrixs.sort(reverse=True)

        ## calc SA-SOR
        gt_rank = []
        pred_rank = []
        for item in matrixs:
            iou, g_r, p_r = item
            if (g_r not in gt_rank) and (p_r not in pred_rank):
                gt_rank.append(g_r)
                pred_rank.append(p_r)

        for r in range(1, len(gt_uni)):
            if r not in gt_rank:
                gt_rank.append(r)
                pred_rank.append(0)
        return np.corrcoef(pred_rank, gt_rank)[0, 1]

    def __call__(self, test_name, pred_path, gt_path):
        lst = [name for name in os.listdir(gt_path) if name.endswith(".png")]
        print("#test_set={}".format(len(lst)), flush=True)

        mae_scores = []
        acc_scores = []
        fbeta_scores = []
        iou_scores = []

        saSor_scores = []
        sor_scores = []

        ap_scores = []
        rec_scores = []

        for name in tqdm.tqdm(lst):
            gt = np.array(Image.open(os.path.join(gt_path, name)).convert("L"))
            if os.path.exists(os.path.join(pred_path, name)):
                pred = np.array(Image.open(os.path.join(pred_path, name)).convert("L"))
            else:
                pred = np.zeros_like(gt)

            mae_scores.append(self.mae(pred, gt))
            acc_scores.append(self.acc(pred, gt))
            fbeta_scores.append(self.fbeta(pred, gt))
            iou_scores.append(self.IOU(pred, gt))

            ap, rec = self.AP(pred, gt)
            ap_scores.append(ap)
            rec_scores.append(rec)

            sa_sor = self.saSOR(pred, gt)
            if not np.isnan(sa_sor):
                saSor_scores.append(sa_sor)

            sor = self.SOR(pred, gt)
            if not np.isnan(sor):
                sor_scores.append(sor)

        ## save results
        file_name = "evaluation.json"
        time_indice = str(datetime.datetime.now()).replace(" ", "_")
        history = loadJson(file_name)
        history.append({
            "name": test_name,
            "time": time_indice,
            "len": len(lst),
            "accuracy": np.mean(acc_scores),
            "mae": np.mean(mae_scores),
            "fbeta": np.mean(fbeta_scores),
            "iou": np.mean(iou_scores),
            "AP": np.mean(ap_scores),
            "recall": np.mean(rec_scores),
            "SA-SOR": np.sum(saSor_scores) / len(lst),
            "sa_sor_valid": len(saSor_scores),
            "SOR(valid)": (np.mean(sor_scores) + 1.0)/2.0,
            "sor_valid": len(sor_scores)
        })
        print(history[-1], flush=True)
        dumpJson(history, file_name)
        dumpXlsx(history, file_name.replace(".json", ".xlsx"))

if __name__=="__main__":
    eval = Evaluation()
    irsr_gt = r"D:\SaliencyRanking\dataset\irsr\Images\test\gt"
    assr_gt = r"D:\SaliencyRanking\dataset\ASSR\ASSR\gt\test"

    eval(
        test_name="ASSR_on_ASSR",
        pred_path=r"D:\SaliencyRanking\retrain_compared_results\ASSR\ASSR\saliency_maps",
        gt_path=assr_gt
    )
    eval(
        test_name="Siris_on_ASSR",
        pred_path=r"D:\SaliencyRanking\comparedResults\ASSR\ASSR\predicted_saliency_maps",
        gt_path=assr_gt
    )
    eval(
        test_name="Liu_on_IRSR",
        pred_path=r"D:\SaliencyRanking\comparedResults\IRSR\saliency_maps",
        gt_path=irsr_gt
    )
