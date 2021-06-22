import datetime
import secrets
import socket
from mpi4py import MPI
import numpy as np

world = MPI.COMM_WORLD
world_size = world.Get_size()
world_rank = world.Get_rank()
start_time = None
end_time = None

if world_rank == 0:
  start_time = datetime.datetime.now()
train_path = '/public/home/hpc182212046/dataset/result/'
train_name = 'train_2_poker_schema.csv_10len_32p_3it_4samp_0025samp/part-00000-f7849d74-8c0a-4964-9cfb-7d33d8342743-c000.csv'
test_path = '/public/home/hpc182212046/dataset/classification/processed/10-fold-dataset/'
test_name = 'test_2_poker_schema.csv'
if world_rank == 0:
  print('训练集：'+train_name)
  print('测试集：'+test_name)

train_lines = []
test_lines = []
validation = None
train_cnt = 0
test_cnt = 0
dimension_num = 0

# read file
train_f = open(train_path + train_name)
test_f = open(test_path + test_name)
for line in train_f:
  if train_cnt > 0:
    temp_arr = line.split(',')
    dimension_num = len(temp_arr)
    temp_line = []
    for ele in temp_arr:
      temp_line.append(float(ele))
    train_lines.append(temp_line)
  train_cnt += 1

for line in test_f:
  if test_cnt > 0:
    temp_arr = line.split(',')
    temp_line = []
    for ele in temp_arr:
      temp_line.append(float(ele))
    test_lines.append(temp_line)
  test_cnt += 1
print('数据读取完毕')
valid_start = int(world_rank * len(test_lines)/world_size)
valid_end = int((world_rank+1) * len(test_lines)/world_size-1)
if world_rank == world_size-1:
  valid_end = len(test_lines)-1
#print('rank'+str(world_rank)+'的起始测试集样本id与终止id:'+str(valid_start)+','+str(valid_end))
all_cur_vali = []
correct_ins_num = 0
for i in range(valid_start, valid_end+1):
  all_cur_vali.append(test_lines[i])
for v_ins in all_cur_vali:
  min_dis = float('inf')
  predict_res = False
  for train_ins in train_lines:
    cur_dis = 0
    for i in range(0, dimension_num-1):
      cur_dis += (train_ins[i]-v_ins[i]) ** 2
    if cur_dis < min_dis:
      predict_res = (v_ins[dimension_num-1] == train_ins[dimension_num-1])
      min_dis = cur_dis
  if predict_res:
    correct_ins_num += 1
#print('rank'+str(world_rank)+'已完成，'+'正确样本数：'+str(correct_ins_num))
correct_sum = world.reduce(correct_ins_num, root=0, op=MPI.SUM)
if world_rank == 0:
  print('训练集总样本数'+str(len(train_lines)))
  print('总正确样本数：'+str(correct_sum))
  print('测试集中总样本数：'+str(len(test_lines)))
  print('准确率：'+str(correct_sum/len(test_lines)))
  end_time = datetime.datetime.now()
  print('运行速度: '+ str(end_time-start_time))
