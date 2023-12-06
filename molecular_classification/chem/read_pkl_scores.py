import pickle
import matplotlib.pyplot as plt
import math


with open('/mnt/ssd4/hm/saved_model/pre-gin/output/tmp_student/runs0/tmp_stud/2022_01_02_02_04_31/scores.txt', 'rb') as f:
    a = pickle.load(f)

print('a', a)

train_list = a['']
def get_parameters_list(*args):
    train_list, val_list, test_list = [[] for _ in range(3)]
    for i in args:
        train_list.append(['train'])
        val_list.append(['val'])
        test_list.append(['val'])
    return [train_list, train_list, train_list]

tmp_list = get_parameters_list(a)
print('tmp_list', tmp_list)
