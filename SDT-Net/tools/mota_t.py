import motmetrics as mm  # 导入该库
import numpy as np

acc = mm.MOTAccumulator(auto_id=True) 
gt_file="/ailab/user/zhuangguohang/data/train_v4_ztf/ztf_20190531260590_000816_zr_c08_o_q1_sciimg_distance_1_00/tracks.txt"
ts_file="/ailab/user/zhuangguohang/data/train_v4_ztf/ztf_20190531260590_000816_zr_c08_o_q1_sciimg_distance_1_00/det.txt"


gt_file1 = '/ailab/user/zhuangguohang/data/train_v4_ztf/ztf_20190531253704_000371_zr_c02_o_q2_sciimg_distance_3_00/tracks.txt'
ts_file1 = '/ailab/user/zhuangguohang/ai4stronomy/zhuangguohang/ai4astronomy/sd_docs/PJ-Astronomy-spacedebris/check.txt'
gt1 = mm.io.loadtxt(gt_file1, fmt="mot15-2D", min_confidence=-1)  # 读入GT
ts1 = mm.io.loadtxt(ts_file1, fmt="mot15-2D")  # 读入自己生成的跟踪结果
# ts1[['X', 'Y', 'Width', 'Height']] = ts1[['X', 'Y', 'Width', 'Height']] *4


gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=-1)  # 读入GT
ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")  # 读入自己生成的跟踪结果
ts[['X', 'Y', 'Width', 'Height']] = ts[['X', 'Y', 'Width', 'Height']] *4
import pdb
# pdb.set_trace()

mh = mm.metrics.create()
acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=1)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值
summary = mh.compute(acc,
                     metrics=['num_frames', 'idf1','mota', 'motp'], # 一个list，里面装的是想打印的一些度量
                     name='acc') # 起个名
print(summary)
acc=mm.utils.compare_to_groundtruth(gt1, ts1, 'iou', distth=1)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值



# 打印单个accumulator
summary = mh.compute(acc,
                     metrics=['num_frames','idf1', 'mota', 'motp'], # 一个list，里面装的是想打印的一些度量
                     name='acc') # 起个名
print(summary)
summary = mh.compute_many([acc, acc.events.loc[0:1]], # 多个accumulators组成的list
                          metrics=['num_frames', 'idf1','mota', 'motp'], 
 ) # 起个名
print(summary)