# each block 24 hours + after LSTM-MIMIC-III-baseline/reg: 48 hours each
# Resnet-110-baseline/reg
# ncrd1
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 1 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-26.log
python gm_prior_cifar_tune_resnet_cifar_a_1_alpha_0_b_0.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/regresnet-110 --gmnum 4 --gmuptfreq 100 --paramuptfreq 50 --gpu-id 1 | tee -a 19-7-2-results/19-7-2-thesis-exp-one-34.log
# ncrd2