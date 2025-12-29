# from __future__ import division
# import argparse
# import logging
# import numpy as np
# import cv2
# from PIL import Image
# from os import makedirs
# from os.path import join, isdir, isfile
#
# from utils.log_helper import init_log, add_file_handler
# from utils.load_helper import load_pretrain
# from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
# from utils.benchmark_helper import load_dataset, dataset_zoo
#
# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
#
# from utils.anchors import Anchors
# from utils.tracker_config import TrackerConfig
# from utils.config_helper import load_config
# from utils.pyvotkit.region import vot_overlap, vot_float2str
#
#
# def to_torch(ndarray):
#     if type(ndarray).__module__ == 'numpy':
#         return torch.from_numpy(ndarray)
#     elif not torch.is_tensor(ndarray):
#         raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
#     return ndarray
#
#
# def im_to_torch(img):
#     img = np.transpose(img, (2, 0, 1))  # C*H*W
#     img = to_torch(img).float()
#     return img
#
#
# def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
#     if isinstance(pos, float): pos = [pos, pos]
#     sz = original_sz
#     im_sz = im.shape
#     c = (original_sz + 1) / 2
#     context_xmin = round(pos[0] - c)
#     context_xmax = context_xmin + sz - 1
#     context_ymin = round(pos[1] - c)
#     context_ymax = context_ymin + sz - 1
#     left_pad = int(max(0., -context_xmin))
#     top_pad = int(max(0., -context_ymin))
#     right_pad = int(max(0., context_xmax - im_sz[1] + 1))
#     bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
#
#     context_xmin, context_xmax = context_xmin + left_pad, context_xmax + left_pad
#     context_ymin, context_ymax = context_ymin + top_pad, context_ymax + top_pad
#
#     r, c, k = im.shape
#     if any([top_pad, bottom_pad, left_pad, right_pad]):
#         te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
#         te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
#         te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
#         te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
#         te_im[:, 0:left_pad, :] = avg_chans
#         te_im[:, c + left_pad:, :] = avg_chans
#         im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
#     else:
#         im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
#
#     if not np.array_equal(model_sz, original_sz):
#         im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
#     else:
#         im_patch = im_patch_original
#     return im_to_torch(im_patch) if out_mode in 'torch' else im_patch
#
#
# def generate_anchor(cfg, score_size):
#     anchors = Anchors(cfg)
#     anchor = anchors.anchors
#     x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
#     anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
#     total_stride = anchors.stride
#     anchor_num = anchor.shape[0]
#     anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
#     ori = - (score_size // 2) * total_stride
#     xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
#                          [ori + total_stride * dy for dy in range(score_size)])
#     xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
#         np.tile(yy.flatten(), (anchor_num, 1)).flatten()
#     anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
#     return anchor
#
#
# def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu'):
#     state = dict()
#     state['im_h'], state['im_w'] = im.shape[0], im.shape[1]
#     p = TrackerConfig()
#
#     # --- תיקון שורש ל-AttributeError ---
#     # בודקים אם model הוא מילון או אובייקט
#     m_anchors = model['anchors'] if isinstance(model, dict) else model.anchors
#
#     if isinstance(m_anchors, dict):
#         class AnchorsObj:
#             pass
#
#         anchor_obj = AnchorsObj()
#         for k, v in m_anchors.items(): setattr(anchor_obj, k, v)
#         p.update(hp, anchor_obj)
#     else:
#         p.update(hp, m_anchors)
#
#     p.renew()
#     net = model['model'] if isinstance(model, dict) else model
#
#     p.scales = m_anchors['scales'] if isinstance(m_anchors, dict) else m_anchors.scales
#     p.ratios = m_anchors['ratios'] if isinstance(m_anchors, dict) else m_anchors.ratios
#     p.anchor_num = net.anchor_num
#     p.anchor = generate_anchor(m_anchors, p.score_size)
#     avg_chans = np.mean(im, axis=(0, 1))
#
#     wc_z = target_sz[0] + p.context_amount * sum(target_sz)
#     hc_z = target_sz[1] + p.context_amount * sum(target_sz)
#     s_z = round(np.sqrt(wc_z * hc_z))
#     z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
#     z = Variable(z_crop.unsqueeze(0))
#     net.template(z.to(device))
#
#     if p.windowing == 'cosine':
#         window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
#     else:
#         window = np.ones((p.score_size, p.score_size))
#
#     state.update({'p': p, 'net': net, 'avg_chans': avg_chans, 'window': np.tile(window.flatten(), p.anchor_num),
#                   'target_pos': target_pos, 'target_sz': target_sz})
#     return state
#
#
# def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
#     p, net, avg_chans, window, target_pos, target_sz = state['p'], state['net'], state['avg_chans'], state['window'], \
#     state['target_pos'], state['target_sz']
#     wc_x = target_sz[1] + p.context_amount * sum(target_sz)
#     hc_x = target_sz[0] + p.context_amount * sum(target_sz)
#     s_x = np.sqrt(wc_x * hc_x)
#     scale_x = p.exemplar_size / s_x
#     d_search = (p.instance_size - p.exemplar_size) / 2
#     pad = d_search / scale_x
#     s_x = s_x + 2 * pad
#     crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
#     x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
#
#     if mask_enable:
#         score, delta, mask = net.track_mask(x_crop.to(device))
#     else:
#         score, delta = net.track(x_crop.to(device))
#
#     delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
#     score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:, 1].cpu().numpy()
#
#     delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
#     delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
#     delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
#     delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]
#
#     def sz(w, h):
#         return np.sqrt((w + (w + h) * 0.5) * (h + (w + h) * 0.5))
#
#     target_sz_in_crop = target_sz * scale_x
#     s_c = np.maximum(sz(delta[2, :], delta[3, :]) / sz(target_sz_in_crop[0], target_sz_in_crop[1]),
#                      sz(target_sz_in_crop[0], target_sz_in_crop[1]) / sz(delta[2, :], delta[3, :]))
#     r_c = np.maximum((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]),
#                      (delta[2, :] / delta[3, :]) / (target_sz_in_crop[0] / target_sz_in_crop[1]))
#     penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
#     pscore = penalty * score * (1 - p.window_influence) + window * p.window_influence
#     best_id = np.argmax(pscore)
#
#     lr = penalty[best_id] * score[best_id] * p.lr
#     res_x, res_y = delta[0, best_id] / scale_x + target_pos[0], delta[1, best_id] / scale_x + target_pos[1]
#     res_w, res_h = target_sz[0] * (1 - lr) + delta[2, best_id] / scale_x * lr, target_sz[1] * (1 - lr) + delta[
#         3, best_id] / scale_x * lr
#     target_pos, target_sz = np.array([res_x, res_y]), np.array([res_w, res_h])
#
#     if mask_enable:
#         best_pscore_id_mask = np.unravel_index(best_id, (5, p.score_size, p.score_size))
#         delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]
#         if refine_enable:
#             mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(p.out_size,
#                                                                                             p.out_size).cpu().data.numpy()
#         else:
#             mask = mask[0, :, delta_y, delta_x].sigmoid().squeeze().view(p.out_size, p.out_size).cpu().data.numpy()
#
#         s = crop_box[2] / p.instance_size
#         sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
#                    crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s, s * p.exemplar_size,
#                    s * p.exemplar_size]
#         s_back = p.out_size / sub_box[2]
#         mapping = np.array([[s_back, 0, -sub_box[0] * s_back], [0, s_back, -sub_box[1] * s_back]]).astype(np.float32)
#         mask_in_img = cv2.warpAffine(mask, mapping, (state['im_w'], state['im_h']), flags=cv2.INTER_LINEAR,
#                                      borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
#         target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
#         contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         if len(contours) != 0:
#             contour = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
#             rbox_in_img = cv2.boxPoints(cv2.minAreaRect(contour))
#         else:
#             loc = cxy_wh_2_rect(target_pos, target_sz)
#             rbox_in_img = np.array([[loc[0], loc[1]], [loc[0] + loc[2], loc[1]], [loc[0] + loc[2], loc[1] + loc[3]],
#                                     [loc[0], loc[1] + loc[3]]])
#
#     state.update({'target_pos': target_pos, 'target_sz': target_sz, 'score': score[best_id],
#                   'mask': mask_in_img if mask_enable else [], 'ploygon': rbox_in_img if mask_enable else []})
#     return state


# --------------------------------------------------------
# SiamMask - Fixed for RL Integration
# --------------------------------------------------------
from __future__ import division
import sys
import os
from pathlib import Path

# --- תיקון נתיבים אגרסיבי ---
# מוצאים את תיקיית SiamMask ואת תיקיית הניסוי (שבה נמצא utils)
current_file = Path(__file__).resolve()
tools_dir = current_file.parent
siammask_root = tools_dir.parent
# הנתיב שבו באמת נמצאת תיקיית utils
experiment_path = siammask_root / "experiments" / "siammask_sharp"

if str(siammask_root) not in sys.path:
    sys.path.insert(0, str(siammask_root))
if str(experiment_path) not in sys.path:
    sys.path.insert(0, str(experiment_path))

import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

# ניסיון ייבוא עם טיפול בשגיאות למקרה שהנתיבים ב-Windows בעייתיים
try:
    from utils.log_helper import init_log, add_file_handler
    from utils.load_helper import load_pretrain
    from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
    from utils.benchmark_helper import load_dataset, dataset_zoo
    from utils.anchors import Anchors
    from utils.tracker_config import TrackerConfig
    from utils.config_helper import load_config
except ImportError:
    # פתרון חירום: ייבוא ישיר אם PYTHONPATH לא התעדכן בזמן
    sys.path.append(str(experiment_path))
    from utils.config_helper import load_config
    from utils.log_helper import init_log, add_file_handler
    from utils.load_helper import load_pretrain
    from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
    from utils.benchmark_helper import load_dataset, dataset_zoo
    from utils.anchors import Anchors
    from utils.tracker_config import TrackerConfig

import torch
from torch.autograd import Variable
import torch.nn.functional as F


# פונקציות עזר של SiamMask
def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float): pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin, context_xmax = context_xmin + left_pad, context_xmax + left_pad
    context_ymin, context_ymax = context_ymin + top_pad, context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        te_im[:, 0:left_pad, :] = avg_chans
        te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
    total_stride = anchors.stride
    anchor_num = anchor.shape[0]
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
        np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu'):
    state = dict()
    state['im_h'], state['im_w'] = im.shape[0], im.shape[1]
    p = TrackerConfig()

    # חילוץ נתונים חכם (תואם לשינויים ב-droneEnv)
    m_anchors = model['anchors'] if isinstance(model, dict) else model.anchors

    if isinstance(m_anchors, dict):
        class AnchorsObj:
            pass

        anchor_obj = AnchorsObj()
        for k, v in m_anchors.items(): setattr(anchor_obj, k, v)
        p.update(hp, anchor_obj)
    else:
        p.update(hp, m_anchors)

    p.renew()
    net = model['model'] if isinstance(model, dict) else model

    # תיקון גישה ל-scales/ratios
    if isinstance(m_anchors, dict):
        p.scales = m_anchors['scales']
        p.ratios = m_anchors['ratios']
    else:
        p.scales = m_anchors.scales
        p.ratios = m_anchors.ratios

    p.anchor_num = net.anchor_num
    p.anchor = generate_anchor(m_anchors, p.score_size)
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    z = Variable(z_crop.unsqueeze(0))
    net.template(z.to(device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    else:
        window = np.ones((p.score_size, p.score_size))

    state.update({
        'p': p, 'net': net, 'avg_chans': avg_chans,
        'window': np.tile(window.flatten(), p.anchor_num),
        'target_pos': target_pos, 'target_sz': target_sz
    })
    return state


def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    p, net, avg_chans, window, target_pos, target_sz = state['p'], state['net'], state['avg_chans'], state['window'], \
    state['target_pos'], state['target_sz']
    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad

    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.to(device))
    else:
        score, delta = net.track(x_crop.to(device))

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:, 1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def sz(w, h):
        return np.sqrt((w + (w + h) * 0.5) * (h + (w + h) * 0.5))

    target_sz_in_crop = target_sz * scale_x
    s_c = np.maximum(sz(delta[2, :], delta[3, :]) / sz(target_sz_in_crop[0], target_sz_in_crop[1]),
                     sz(target_sz_in_crop[0], target_sz_in_crop[1]) / sz(delta[2, :], delta[3, :]))
    r_c = np.maximum((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]),
                     (delta[2, :] / delta[3, :]) / (target_sz_in_crop[0] / target_sz_in_crop[1]))
    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score * (1 - p.window_influence) + window * p.window_influence
    best_id = np.argmax(pscore)

    lr = penalty[best_id] * score[best_id] * p.lr
    res_x = delta[0, best_id] / scale_x + target_pos[0]
    res_y = delta[1, best_id] / scale_x + target_pos[1]
    res_w = target_sz[0] * (1 - lr) + delta[2, best_id] / scale_x * lr
    res_h = target_sz[1] * (1 - lr) + delta[3, best_id] / scale_x * lr

    target_pos, target_sz = np.array([res_x, res_y]), np.array([res_w, res_h])

    state.update({'target_pos': target_pos, 'target_sz': target_sz, 'score': score[best_id]})
    return state