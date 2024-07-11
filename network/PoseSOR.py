import os, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .neck import build_neck_head
from .modules import build_sis_head, build_actor_interaction_module, build_pose_shift_module
from .other import PositionEmbeddingLearnable, PositionEmbeddingRandom, PositionEmbeddingSine

from .utils import calc_iou, pad1d, xyhw2xyxy, arange_from_ascending_seq
from .debug import Debug
from .loss import hungarianMatcherInPoints, batch_mask_loss_in_points, batch_bbox_loss, heatmap_loss_in_points


def calculate_bbox_iou(box1, box2):
    # Unpack the coordinates of box1 and box2
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the area of box1 and box2
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    # Calculate the coordinates of the intersection rectangle
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    # Check if there is an intersection
    if xi1 >= xi2 or yi1 >= yi2:
        return 0.0  # No intersection

    # Calculate the area of the intersection
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate the IOU score
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou

def post_process_one(result, maskness_threshold):
    """
    Remove redundant objects:
    if two objects' bbox ious are larger than a threshold then, remove the less salient one
    """
    masks = result["masks"]
    scores = result["scores"]
    n = result["num"]
    
    iskeep = [True for _ in range(n)]
    for i in range(n):
        if iskeep[i]==False: continue
        for j in range(i+1, n):
            if iskeep[j]==False: continue
            maskness = (masks[i] * masks[j]).sum() / min(masks[i].sum()+1e-6, masks[j].sum()+1e-6)
            objectness = scores[i] + scores[j]
            tolerance = 1.0 - maskness
            if maskness > maskness_threshold and objectness < (2.0 - tolerance):
                ## same object
                masks[i] = np.maximum(masks[i], masks[j])
                scores[i] = np.maximum(scores[i], scores[j])
                iskeep[j] = False
    ## merge over
    result = dict(
        (k, [v[i] for i in range(n) if iskeep[i]]) if isinstance(v, list) else (k, v)
        for k, v in result.items()
    )
    result["num"] = int(sum([1 for i in range(n) if iskeep[i]]))
    return result

def post_process(results, maskness_threshold=0.9):
    n = results["num"]
    while n > 0:
        results = post_process_one(results, maskness_threshold=maskness_threshold)
        if results["num"] < n:
            n = results["num"]
        else:
            break
    return results

@META_ARCH_REGISTRY.register()
class PoseSOR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck_head(cfg)
        self.instance_seg = build_sis_head(cfg)
        self.actor_interaction = build_actor_interaction_module(cfg)
        self.pose_shift = build_pose_shift_module(cfg)

        self.pe_layer = {
            "SINE": PositionEmbeddingSine(cfg.MODEL.COMMON.EMBED_DIM // 2, normalize=True),
            "RANDOM": PositionEmbeddingRandom(cfg.MODEL.COMMON.EMBED_DIM // 2),
            "APE": PositionEmbeddingLearnable(cfg.MODEL.COMMON.EMBED_DIM)
        }[cfg.MODEL.PE]

        self.debug = Debug(cfg=cfg)
        self.test_debug = Debug(cfg=cfg)
        self.cfg = cfg
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def get_rank_weights(self):
        rank_weights = 1.0 / (torch.arange(self.cfg.MODEL.COMMON.MAX_RANK_LEVEL, device=self.device) + 1e-6)
        rank_weights[0] = 0.0
        return rank_weights

    def hmip(self, out, targets):
        """
        Args:
            out: {"masks": B, n, h, w; "scores": B, n, 1}
            targets: masks in list of n_max, H, W
        Return:
            bi, qi, ti
        """
        size = targets[0][0].shape
        out_masks = F.interpolate(out["masks"], size=size, mode="bilinear")
        out_scores = out["scores"]
        bi, qi, ti = hungarianMatcherInPoints(preds={"masks": out_masks, "scores": out_scores}, targets=targets,
                                              cfg=self.cfg)
        return bi, qi, ti

    def forward(self, batch_list, *args, **argw):
        if self.training:
            self.debug.tick()
            loss = {}
            targets = [x["masks"].to(self.device) for x in batch_list]  ## list of k_i, Ht, Wt
            n_max = max([len(x) for x in targets])
            masks = torch.stack([pad1d(m, dim=0, num=n_max, value=0.0) for m in targets], dim=0)  ## B, n_max, H, W
            train_size = tuple(masks.shape[2::])

            joints = [x["keypoints"].to(self.device) for x in batch_list]  ## list of K, 17, 3
            joints = torch.stack([pad1d(j, dim=0, num=n_max, value=0.0) for j in joints], dim=0)  ## B, n_max, 17, 3
            joints_xy = joints[:, :, :, 0:2]  ## B, n_max, 17, 2
            joints_label = joints[:, :, :, 2:3].gt(0.5).float()  ## B, n_max, 17, 1

            bboxes = [x["bboxes"].to(self.device) for x in batch_list]  ## list of K, 4 [xyxy]
            bboxes = torch.stack([pad1d(b, dim=0, num=n_max, value=0.0) for b in bboxes], dim=0)  ## B,n_max, 4

            next_targets = [x["next_targets"].to(self.device) for x in batch_list]  ## list of K, H, W
            next_targets = torch.stack([pad1d(nxt, dim=0, num=n_max, value=0.0) for nxt in next_targets],
                                       dim=0)  ## B, n_max, H, W

            ranks = [x["category_ids"].to(self.device) for x in batch_list]  ## list of Ki
            ranks = torch.stack([pad1d(r, dim=0, num=n_max, value=0.0) for r in ranks], dim=0)  ## B, n_max
        else:
            self.test_debug.tick()

        ## prepare image
        images = torch.stack([s["image"] for s in batch_list], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std

        ## Multi-Scale
        if self.training:
            train_size = self.cfg.INPUT.DYNAMIC_SIZES[
                np.random.randint(len(self.cfg.INPUT.DYNAMIC_SIZES))
            ]
            images = F.interpolate(images, size=train_size, mode="bilinear")
            masks = F.interpolate(masks, size=train_size, mode="bilinear")
            next_targets = F.interpolate(next_targets, size=train_size, mode="bilinear")

        feats = self.backbone(images)
        feats = self.neck(feats)

        feats_pe = dict((k, self.pe_layer(feats[k])) for k in feats)
        q, qpe, instances_out, instances_aux = self.instance_seg(
            feats=feats,
            feats_pe=feats_pe,
        )  ## q, qpe: B, nq, C

        if self.training:  ## Instance: mask_loss + obj_loss
            bi, qi, ti = self.hmip(instances_out, targets=targets)
            pred_mask = F.interpolate(instances_out["masks"], size=train_size, mode="bilinear")  ## B, nq, H, W
            mask_loss = batch_mask_loss_in_points(pred_mask[bi, qi], masks[bi, ti], cfg=self.cfg).mean()

            labels = torch.zeros_like(instances_out["scores"])
            labels[bi, qi, 0] = 1.0
            weights = torch.ones_like(instances_out["scores"]) * self.cfg.LOSS.OBJ_NEG
            weights[bi, qi, 0] = self.cfg.LOSS.OBJ_POS
            obj_loss = (F.binary_cross_entropy_with_logits(instances_out["scores"], labels,
                                                           reduction="none") * weights).mean()

            if "bboxes" in instances_out:
                loss["ins_bbox_loss"] = batch_bbox_loss(
                    xyhw2xyxy(torch.sigmoid(instances_out["bboxes"][bi, ti])),
                    bboxes[bi, ti],
                    cfg=self.cfg
                ).mean()

            ## bi, qi, ti, n_max | mask_loss, obj_loss
            loss["mask_loss"] = mask_loss * self.cfg.LOSS.MASK_COST
            loss["obj_loss"] = obj_loss * self.cfg.LOSS.CLS_COST

            ## debug
            topk = int(max((bi == 0).sum(), 1))
            self.debug.add_figures("GT_Mask", masks[bi, ti][0:topk].detach().cpu().numpy())
            self.debug.add_figures("pred_mask", pred_mask[bi, qi][0:topk].sigmoid().detach().cpu().numpy())
            self.debug.add_str("pred_obj", instances_out["scores"][bi, qi, 0][0:topk].sigmoid().detach().cpu().numpy())
        else:
            top1_threshold = torch.max(instances_out["scores"].sigmoid(), dim=1, keepdim=True)[0] - 1e-6  ## B, 1, 1
            loose_threshold = torch.minimum(top1_threshold,
                                            torch.ones_like(top1_threshold) * self.cfg.TEST.OBJ_THRESHOLD)
            hits = instances_out["scores"].sigmoid() >= loose_threshold  ## B, nq, 1
            n_max = int(hits.float().sum(dim=1).max().cpu().detach())
            bi, qi, _ = torch.where(hits)
            ti = arange_from_ascending_seq(bi)
            ## bi, qi, ti, n_max
            ## debug
            self.test_debug.add_figures("infer_masks", instances_out["masks"][bi, qi].sigmoid().detach().cpu().numpy())
            self.test_debug.add_str("infer_scores", instances_out["scores"][bi, qi, 0].sigmoid().detach().cpu().numpy())

        ## Prepare hit queries (which is smaller than full q)
        B, nq, C = q.shape
        q_hit = torch.zeros((B, n_max, C), dtype=q.dtype, device=q.device)
        qpe_hit = torch.randn((B, n_max, C), dtype=qpe.dtype, device=qpe.device)
        q_hit[bi, ti, :] = q[bi, qi, :]
        qpe_hit[bi, ti, :] = qpe[bi, qi, :]
        q_mask = torch.zeros((B, n_max, 1), device=self.device).float()  ## B, n_max, 1
        q_mask[bi, ti, 0] = 1.0

        ## Pose-Aware Actor Interaction Module
        pose, pose_pe, pai, pai_pe, joints_out, joints_aux = self.actor_interaction(
            q=q_hit,
            qpe=qpe_hit,
            feats=feats,
            feats_pe=feats_pe,
            q_mask=q_mask,
        )
        ## pose, pose_pe: B, n, k, C; pai, pai_pe: B, n, C; out/aux: {
        # "keypoints": B, n, k, 3,
        # "heatmaps": B, n, k, 4H, 4W
        # "masks": B, n, 4H, 4W
        # }

        if self.training:  ## joint_loss + joint_cls_loss + refined mask loss + heatmaps loss
            joint_loss = F.smooth_l1_loss(input=torch.sigmoid(joints_out["keypoints"][:, :, :, 0:2]), target=joints_xy,
                                          reduction="none")
            joint_cls_loss = F.binary_cross_entropy_with_logits(joints_out["keypoints"][:, :, :, 2:3], joints_label,
                                                                reduction="none")

            joint_mask = q_mask.unsqueeze(-1) * joints_label  ## B, n_max, 17, 1
            joint_cls_loss = (joint_cls_loss[bi, ti]).mean()
            j1, j2, j3, _ = torch.where(joint_mask.gt(0.5))  ## b,n,k,~
            if len(j1) > 0:
                joint_loss = joint_loss[j1, j2, j3].mean()
                heatmap_loss = heatmap_loss_in_points(joints_out["heatmaps"][j1, j2, j3], joints_xy[j1, j2, j3],
                                                      cfg=self.cfg)
            else:
                joint_loss = joint_loss.mean() * 0.0
                heatmap_loss = joints_out["heatmaps"].sigmoid().mean() * 0.0

            refined_masks = F.interpolate(joints_out["masks"], size=train_size, mode="bilinear")
            refined_mask_loss = batch_mask_loss_in_points(refined_masks[bi, ti], masks[bi, ti], cfg=self.cfg).mean()

            if "bboxes" in joints_out:
                loss["bbox_loss"] = batch_bbox_loss(
                    xyhw2xyxy(torch.sigmoid(joints_out["bboxes"][bi, ti])),
                    bboxes[bi, ti],
                    cfg=self.cfg
                ).mean()

            loss["joint_loss"] = joint_loss * self.cfg.LOSS.JOINT_COST
            loss["joint_cls_loss"] = joint_cls_loss * self.cfg.LOSS.JOINT_CLS_COST
            loss["heatmap_loss"] = heatmap_loss * self.cfg.LOSS.HEATMAP_COST
            loss["refined_mask_loss"] = refined_mask_loss * self.cfg.LOSS.MASK_COST

            ## debug
            self.debug.add_skeleton("KP_GT", joints[bi, ti, :, :][0:topk].detach().cpu().numpy())
            self.debug.add_skeleton("keypoints",
                                    joints_out["keypoints"][bi, ti, :, :][0:topk].sigmoid().detach().cpu().numpy())
            self.debug.add_figures("JMasks", joints_out["masks"][bi, ti][0:topk].sigmoid().detach().cpu().numpy())
            # self.debug.add_figures("heatmap", joints_out["heatmaps"][bi, ti].sum(dim=1).sigmoid()[0:topk].detach().cpu().numpy())
        else:
            self.test_debug.add_skeleton("keypoints",
                                         joints_out["keypoints"][bi, ti, :, :].sigmoid().detach().cpu().numpy())
            self.test_debug.add_figures("JMasks", joints_out["masks"][bi, ti].sigmoid().detach().cpu().numpy())
            # self.debug.add_figures("heatmap", joints_out["heatmaps"][bi, ti, 0].sum(dim=1).sigmoid().detach().cpu().numpy())

        partition_out, partition_aux = self.pose_shift(
            q=pai,
            qpe=pai_pe,
            pose=pose,
            pose_pe=pose_pe,
            feats=feats,
            feats_pe=feats_pe,
            q_mask=q_mask,
        )  ## out: {"partitions": B, n, 4H, 4W; "ranks": B, n, max_rank_level}

        if self.training:  ## next targets loss + rank loss
            partition = F.interpolate(partition_out["partitions"], size=train_size, mode="bilinear")  ## B, n_max, H, W
            partition_loss = batch_mask_loss_in_points(partition[bi, ti], next_targets[bi, ti], cfg=self.cfg).mean()
            rank_loss = F.cross_entropy(partition_out["ranks"][bi, ti], ranks[bi, ti].long())

            loss["partition_loss"] = partition_loss * self.cfg.LOSS.PARTITION_COST
            loss["rank_loss"] = rank_loss * self.cfg.LOSS.RANK_COST

            ## debug
            self.debug.add_figures("partitions_GT", next_targets[bi, ti][0:topk].detach().cpu().numpy())
            self.debug.add_figures("partitions", partition[bi, ti][0:topk].sigmoid().detach().cpu().numpy())
            self.debug.add_str("rank",
                               torch.argmax(partition_out["ranks"][bi, ti][0:topk].detach().cpu(), dim=-1).numpy())
            self.debug.add_str("rank_GT", ranks[bi, ti][0:topk].detach().cpu().numpy())
        else:
            self.test_debug.add_figures("partitions",
                                        partition_out["partitions"][bi, ti].sigmoid().detach().cpu().numpy())
            self.test_debug.add_str("rank", torch.argmax(partition_out["ranks"][bi, ti].detach().cpu(), dim=-1).numpy())

        if self.training:
            """
                mask_loss, obj_loss, joint_loss, joint_cls_loss, partition_loss, rank_loss
            """
            self.debug.save("{}.png".format(batch_list[0].get("image_name")))
            return loss
        else:  ## Inference
            """
                instance_out:  {"masks": (B, n, H, W), "scores": (B, n, 1)}
                joints_out:    {"keypoints": (B, n, k, 3)}
                partition_out: {"partitions": (B, n, H, W)}
                rank_out:      {"ranks": (B, n, rank_max)}
            """
            self.test_debug.save("TEST_{}.png".format(batch_list[0].get("image_name")))
            results = []
            for i, sample in enumerate(batch_list):
                H, W = sample["height"], sample["width"]
                image_name = sample.get("image_name", "unkonwn")
                rank_scores = (torch.softmax(partition_out["ranks"][i], dim=-1) * self.get_rank_weights()[None, :]).sum(
                    dim=-1)  ## n

                masks = []
                scores = []
                joints = []
                explict_joints = []
                partitions = []
                heatmaps = []
                saliency = []
                bboxes = []
                num = 0

                pred_masks = F.interpolate(joints_out["masks"], size=(H, W), mode="bilinear")
                pred_scores = torch.zeros((B, n_max, 1), device=self.device) - 1e9
                pred_scores[bi, ti] = instances_out["scores"][bi, qi]
                pred_heatmaps = F.interpolate(joints_out["heatmaps"][i], size=(H, W), mode="bilinear")  ## n, K, H, W
                pred_heatmaps = torch.softmax(pred_heatmaps.flatten(2), dim=-1).unflatten(-1, (H, W))

                # scalar_loose_threshold = 0.0
                # for j in torch.argsort(-rank_scores):
                #     score = float(pred_scores[i, j, 0].sigmoid().detach().cpu())
                #     if q_mask[i, j, 0] < .5: continue
                #     scalar_loose_threshold = max(scalar_loose_threshold, score)
                # scalar_loose_threshold = min(scalar_loose_threshold, self.cfg.TEST.OBJ_READOUT_THRESHOLD)

                for j in torch.argsort(-rank_scores):
                    score = float(pred_scores[i, j, 0].sigmoid().detach().cpu())
                    # if q_mask[i, j, 0] < .5 or score < scalar_loose_threshold: continue
                    if q_mask[i, j, 0] < .5: continue

                    hms = pred_heatmaps[j]  ## K, H, W
                    row_max, row_idx = torch.max(hms, dim=2)  ## K, H
                    map_max, map_idx = torch.max(row_max, dim=1)  ## K
                    y_coord = map_idx.float() / H  ## K
                    x_coord = row_idx[torch.arange(len(map_idx)).long(), map_idx].float() / W  ## K
                    pred_joints = joints_out["keypoints"][i, j].float().sigmoid()  ## K, 3

                    bbox = xyhw2xyxy(torch.sigmoid(joints_out["bboxes"][i, j]).unsqueeze(0))[0]  ## x1y1x2y2
                    bbox = (bbox.detach().cpu() * torch.tensor([W, H, W, H])).numpy()  ## x1y1x2y2

                    masks.append(
                        pred_masks[i, j].sigmoid().detach().cpu().gt(.5).float().numpy()
                    )
                    scores.append(score)
                    joints.append(
                        torch.stack([x_coord, y_coord, pred_joints[:, -1]], dim=-1).detach().cpu().numpy()  ## K, 3
                    )
                    explict_joints.append(
                        pred_joints.detach().cpu().numpy()
                    )  ## K, 3
                    heatmaps.append(
                        pred_heatmaps[j].detach().cpu().numpy()  ## K, H, W
                    )
                    partitions.append(
                        F.interpolate(
                            partition_out["partitions"][i, j][None, None, :, :], size=(H, W), mode="bilinear"
                        )[0, 0].sigmoid().detach().cpu().float().numpy()
                    )
                    saliency.append(
                        float(
                            rank_scores[j].detach().cpu()
                        )
                    )
                    bboxes.append(bbox)
                    num += 1
                results.append({
                    "image_name": image_name,
                    "masks": masks,
                    "scores": scores,
                    "joints": joints,
                    "explict_joints": explict_joints,
                    "heatmaps": heatmaps,
                    "partitions": partitions,
                    "saliency": saliency,
                    "bboxes": bboxes,
                    "num": num,
                })
            return [post_process(result, maskness_threshold=self.cfg.TEST.MASKNESS_THRESHOLD) for result in results]
            # return results
            # end inference
        pass