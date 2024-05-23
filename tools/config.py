# coding: utf-8
from os import path as osp
root = osp.dirname(osp.abspath(__file__))


OHAZE_ROOT = osp.abspath(osp.join(root, '../data', 'O-Haze'))

# RESIDE
TRAIN_ITS_ROOT = osp.abspath(osp.join(root, '../data', 'RESIDE', 'ITS'))  # ITS
TEST_SOTS_ROOT = osp.abspath(osp.join(root, '../data', 'RESIDE', 'SOTS', 'indoor'))  # SOTS indoor
TEST_HAZERD_ROOT = osp.abspath(osp.join(root, '../data', 'HAZERD')) 
