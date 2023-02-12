import bisect
import random

import torch
import torch.utils.data as data
import numpy as np

from .h36m_smpl import H36mSMPL
from .hp3d import HP3D
from .mscoco import Mscoco
from .pw3d import PW3D

s_mpii_2_smpl_jt = [
    6, 3, 2,
    -1, 4, 1,
    -1, 5, 0,
    -1, -1, -1,
    8, -1, -1,
    -1,
    13, 12,
    14, 11,
    15, 10,
    -1, -1
]
s_3dhp_2_smpl_jt = [
    4, -1, -1,
    -1, 19, 24,
    -1, 20, 25,
    -1, -1, -1,  # TODO: foot point
    5, -1, -1,
    -1,
    9, 14,
    10, 15,
    11, 16,
    -1, -1,  # 23
    # 7, 
    # -1, -1,
    # 21, 26
]
s_coco_2_smpl_jt = [
    -1, -1, -1,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

s_smpl24_jt_num = 24


class MixDataset2Cam(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 24
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )
    data_domain = set([
        'type',
        'target_theta',
        'target_theta_weight',
        'target_beta',
        'target_smpl_weight',
        'target_uvd_29',
        'target_xyz_24',
        'target_weight_24',
        'target_weight_29',
        'target_xyz_17',
        'target_weight_17',
        'trans_inv',
        'intrinsic_param',
        'joint_root',
        'target_twist',
        'target_twist_weight',
        'depth_factor',
        'target_xyz_weight_24',
        'img_center',
        'camera_scale',
        'camera_trans',
        'camera_valid',
        'camera_error',
        'target_uvd_61',
        'target_weight_61'
    ])
    # [   'camera_scale',
    #     'camera_trans',
    #     'camera_valid',
    #     'target_uvd_61',
    #     'target_weight_61',
    #     'target_xyz_weight_24',
    #     'target_theta',
    #     'target_theta_weight',
    #     'target_beta',
    #     'target_smpl_weight',
    #     'target_uvd_29',
    #     'target_xyz_24',
    #     'target_weight_24',
    #     'target_weight_29',
    #     'target_xyz_17',
    #     'target_weight_17',
    #     'trans_inv',
    #     'intrinsic_param',
    #     'joint_root',
    #     'target_twist',
    #     'target_twist_weight',
    #     'depth_factor',
    #     ]
    def __init__(self,
                 cfg,
                 train=True):
        self._train = train
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))

        # if train:
        #     self.db0 = H36mSMPL(
        #         cfg=cfg,
        #         ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
        #         train=True)
        #     self.db1 = Mscoco(
        #         cfg=cfg,
        #         ann_file=f'person_keypoints_{cfg.DATASET.SET_LIST[1].TRAIN_SET}.json',
        #         train=True)
        #     self.db2 = HP3D(
        #         cfg=cfg,
        #         ann_file=cfg.DATASET.SET_LIST[2].TRAIN_SET,
        #         train=True)
        #     self.db3 = PW3D(
        #         cfg=cfg,
        #         ann_file='3DPW_train_new.json',
        #         train=True
        #     )
        #
        #     self._subsets = [self.db0, self.db1, self.db2, self.db3]
        #     self._2d_length = len(self.db1)
        #     self._3d_length = len(self.db0) + len(self.db2) + len(self.db3)
        # else:
        # self.db0 = H36mSMPL(
        #     cfg=cfg,
        #     ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
        #     train=train)

        # self.db0 = H36mSMPL(
        #     cfg=cfg,
        #     ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
        #     train=True)
        # self.dataset_idx = 0

        self.db0 = PW3D(
            cfg=cfg,
            ann_file='3DPW_train_new_tcmr.json',
            train=True)
        self.dataset_idx = 3

        # self.db0 = HP3D(
        #     cfg=cfg,
        #     ann_file='annotation_mpi_inf_3dhp_train_v2_tcmr.json',
        #     train=True)
        # self.dataset_idx = 2

        self.dbh36 = H36mSMPL(
            cfg=cfg,
            ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
            train=train)

        self._subsets = [self.db0]

        self._subset_size = [len(item) for item in self._subsets]
        self._db0_size = len(self.db0)

        # if train:
        #     self.max_db_data_num = max(self._subset_size)
        #     print('max_data_set', np.argmax(np.array(self._subset_size)))
        #     self.tot_size = (2 * max(self._subset_size))
        #     self.partition = [0.3, 0.4, 0.1, 0.2]
        # else:
        self.tot_size = self._db0_size
        self.partition = [1]

        self.cumulative_sizes = self.cumsum(self.partition)

        self.joint_pairs_24 = self.dbh36.joint_pairs_24
        self.joint_pairs_17 = self.dbh36.joint_pairs_17
        self.root_idx_17 = self.dbh36.root_idx_17
        self.root_idx_smpl = self.dbh36.root_idx_smpl
        self.evaluate_xyz_17 = self.dbh36.evaluate_xyz_17
        self.evaluate_uvd_24 = self.dbh36.evaluate_uvd_24
        self.evaluate_xyz_24 = self.dbh36.evaluate_xyz_24

        self.uvd29_to_joint61 = torch.tensor([-1, 12, 17, 19, 21, 16, 18, 20,  0,  2,  5,  8,  1,  4,  7, -1, -1, -1,
        -1, 27, -1, -1, 28, -1, -1,  8,  5, -1, -1,  4,  7, 21, 19, 17, 16, 18,
        20, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3,  6,  9, 10, 11,
        13, 14, 15, 22, 23, 25, 26], dtype=torch.long)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        # assert idx >= 0
        # if self._train:
        #     p = random.uniform(0, 1)
        #
        #     dataset_idx = bisect.bisect_right(self.cumulative_sizes, p)
        #
        #     _db_len = self._subset_size[dataset_idx]
        #
        #     # last batch: random sampling
        #     if idx >= _db_len * (self.tot_size // _db_len):
        #         sample_idx = random.randint(0, _db_len - 1)
        #     else:  # before last batch: use modular
        #         sample_idx = idx % _db_len
        # else:
        dataset_idx = 0
        sample_idx = idx

        results = self._subsets[dataset_idx][sample_idx]
        if type(results) != tuple:
            return results
        img, target, img_id, bbox = results
        if self.dataset_idx > 0 and self.dataset_idx < 3:
            # COCO, 3DHP
            label_jts_origin = target.pop('target')
            label_jts_mask_origin = target.pop('target_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)
            label_xyz_29_mask = torch.zeros(29, 3)

            if self.dataset_idx == 1:
                # COCO
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 17 * 2, label_jts_origin.shape

                label_jts_origin = label_jts_origin.reshape(17, 2)
                label_jts_mask_origin = label_jts_mask_origin.reshape(17, 2)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_coco_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :2] = label_jts_origin[id2, :2].clone()
                        label_uvd_29_mask[id1, :2] = label_jts_mask_origin[id2, :2].clone()
            elif self.dataset_idx == 2:
                # 3DHP
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 28 * 3, label_jts_origin.shape

                label_xyz_origin = target.pop('target_xyz').reshape(-1, 3)
                label_xyz_mask_origin = target.pop('target_xyz_weight').reshape(-1, 3)

                label_jts_origin = label_jts_origin.reshape(28, 3)
                label_jts_mask_origin = label_jts_mask_origin.reshape(28, 3)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_3dhp_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :3] = label_jts_origin[id2, :3].clone()
                        label_uvd_29_mask[id1, :3] = label_jts_mask_origin[id2, :3].clone()
                        label_xyz_24[id1, :3] = label_xyz_origin[id2, :3].clone()
                        label_xyz_29_mask[id1, :3] = label_xyz_mask_origin[id2, :3].clone()

            label_uvd_61 = label_uvd_29[self.uvd29_to_joint61]
            label_uvd_61[self.uvd29_to_joint61 == -1] = torch.tensor([0., 0., 0.])
            label_uvd_61 = label_uvd_61.reshape(-1)
            label_uvd_61_mask = label_uvd_29_mask[self.uvd29_to_joint61]
            label_uvd_61_mask[self.uvd29_to_joint61 == -1] = torch.tensor([0., 0., 0.])
            label_uvd_61_mask = label_uvd_61_mask.reshape(-1)

            label_uvd_29 = label_uvd_29.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_uvd_24_mask = label_uvd_29_mask[:24, :].reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)
            label_xyz_24_mask = label_xyz_29_mask[:24, :].reshape(-1)

            target['target_uvd_29'] = label_uvd_29
            target['target_xyz_24'] = label_xyz_24
            target['target_weight_24'] = label_uvd_24_mask
            target['target_weight_29'] = label_uvd_29_mask
            target['target_xyz_17'] = label_xyz_17
            target['target_weight_17'] = label_xyz_17_mask
            # target['target_theta'] = torch.zeros(24 * 4)
            target['target_theta'] = torch.zeros(24 * 9)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)
            # target['target_theta_weight'] = torch.zeros(24 * 4)
            target['target_theta_weight'] = torch.zeros(24 * 9)
            target['target_twist'] = torch.zeros(23, 2)
            target['target_twist_weight'] = torch.zeros(23, 2)
            target['target_xyz_weight_24'] = label_xyz_24_mask
            target['target_uvd_61'] = label_uvd_61
            target['target_weight_61'] = label_uvd_61_mask
        else:
            assert set(target.keys()).issubset(self.data_domain), (set(target.keys()) - self.data_domain, self.data_domain - set(target.keys()),)
        target.pop('type')

        return img, target, img_id, bbox