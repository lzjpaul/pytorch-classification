# each block 24 hours + after LSTM-MIMIC-III-baseline/reg: 48 hours each
# ncrd2
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 2 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-27.log
python gm_prior_cifar_tune_resnet_cifar_a_1_alpha_0_b_1.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/regresnet-110 --gmnum 4 --gmuptfreq 100 --paramuptfreq 50 --gpu-id 2 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-35.log
# ncrf1