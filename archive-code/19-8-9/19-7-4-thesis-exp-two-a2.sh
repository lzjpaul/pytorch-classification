# each block 24 hours
# ncra2
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 0.0 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 2 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-1.log
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 2 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-2.log
python gm_prior_cifar_tune_resnet_cifar_a_0_alpha_0_b_0.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/regresnet-110 --gmnum 4 --gmuptfreq 100 --paramuptfreq 50 --gpu-id 2 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-3.log
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 0.0 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 2 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-4.log
# ncrb2