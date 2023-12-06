# Robust Self-training

Implementation of "Towards Robust Self-training for Molecular Biology Prediction Tasks". Three tasks are conducted to evaluate the performance of our method, molecular classification, molecular regression and protein secondary structure prediction. The details information about the backbone models can be found within each task folder. 

### Molecular classification
train teacher
```
python st_train_teacher.py --dataset hiv --filename teacher_model --device 2 --split random --eval_train 1 --epochs 100 --output_pseudo_dataset *path* --self_training 1 --num_workers 6
```

train student
```
python st_train_student.py --dataset hiv --filename student_model --device 2 --split random --eval_train 1 --self_training 1 --output_pseudo_dataset *your path* --s_epochs 100 --st_iter 3 --load_teacher 1 --use_unlabeled 1 --load_s_best 1 --loss GCE --q 0.3
```

### Molecular regression
```
CUDA_VISIBLE_DEVICES=1 python -u main_qm9.py --num_workers 6 --lr 5e-4 --property alpha --exp_name exp_1_alpha --outf *your path*
```

### Protein Secondary Structure Prediction
See details in the corresponding folder. 




