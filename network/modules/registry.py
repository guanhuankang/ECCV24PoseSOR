from detectron2.utils.registry import Registry

SIS_HEAD_REG = Registry("SIS_HEAD_REG")
SIS_HEAD_REG.__doc__ = """
Saliency Instance Segmentation (SIS) head
"""

def build_sis_head(cfg):
    name = cfg.MODEL.SIS_HEAD.NAME
    return SIS_HEAD_REG.get(name)(cfg)


ACTOR_INTERACTION_REG = Registry("ACTOR_INTERACTION_REG")
ACTOR_INTERACTION_REG.__doc__ = """
Pose-Aware Actor Interaction Module
"""

def build_actor_interaction_module(cfg):
    name = cfg.MODEL.ACTOR_INTERACTION.NAME
    return ACTOR_INTERACTION_REG.get(name)(cfg)

POSE_SHIFT_REG = Registry("POSE_SHIFT_REG")
POSE_SHIFT_REG.__doc__ = """
Pose-Driven Attention Shift/Selection Module
"""

def build_pose_shift_module(cfg):
    name = cfg.MODEL.POSE_SHIFT.NAME
    return POSE_SHIFT_REG.get(name)(cfg)
