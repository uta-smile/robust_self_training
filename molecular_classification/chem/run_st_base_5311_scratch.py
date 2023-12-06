import os
import time


task_name = 'st_base_5311_scratch'
output_path = '/mnt/ssd4/hm/saved_model/pre-gin/output/logs/new111'
save_path = os.path.join(output_path, task_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
si = 3
se = 100

run_times = 3

for i in range(run_times):
    cmd = 'python st_base_5311.py --dataset hiv --filename {} --device 3 --split random --eval_train 1 --self_training 1 \
    --epochs 100 --use_unlabeled 1 --output_pseudo_dataset /mnt/ssd4/hm/saved_model/pre-gin/output/new111/ --s_epochs {} --st_iter {}\
    2>&1 | tee {}/{}_{}_runs{}.txt'.format(task_name, se, si, save_path, se, current_time, i)
    print(cmd)
    os.system(cmd)