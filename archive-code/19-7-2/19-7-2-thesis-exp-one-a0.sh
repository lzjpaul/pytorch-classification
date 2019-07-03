# each block 24 hours + after LSTM-MIMIC-III-baseline/reg: 48 hours each
# LSTM-MIMIC-III-baseline/reg
# ncra0
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_gmreg_real_wlm_tune_hyperparam_baseline.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname lstm -blocks 1 -lr 1.0 -weightdecay 0.0001 -batchsize 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 --batch_first --nhid 128 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-12.log
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_gmreg_real_wlm_tune_hyperparam_tune_mimic_a_0_alpha_0.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname reglstm -blocks 1 -lr 1.0 -weightdecay 0.0001 -batchsize 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 --batch_first --nhid 128 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-16.log
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 0 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-26.log
python gm_prior_cifar_tune_resnet_cifar_a_0_alpha_0_b_0.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/regresnet-110 --gmnum 4 --gmuptfreq 100 --paramuptfreq 50 --gpu-id 0 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-30.log
# ncrb0