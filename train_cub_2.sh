python train_origin.py --cfg configs/cub200/sdd_kd/vgg13_mv2.yaml --gpu 1 --M [1] 52.23
python train_origin.py --cfg configs/cub200/sdd_kd/vgg13_mv2.yaml --gpu 1 --M [1,2,4] 60.2

python train_origin.py --cfg configs/cub200/sdd_kd/res32x4_mv2.yaml --gpu 1 --M [1] 56.0
#python train_origin.py --cfg configs/cub200/sdd_kd/res32x4_mv2.yaml --gpu 1 --M [1,2,4] 61.18


python train_origin.py --cfg configs/cub200/sdd_dkd/vgg13_mv2.yaml --gpu 1 --M [1] 58.4
#python train_origin.py --cfg configs/cub200/sdd_dkd/vgg13_mv2.yaml --gpu 1 --M [1,2,4] 65.58

#python train_origin.py --cfg configs/cub200/sdd_dkd/res32x4_mv2.yaml --gpu 1 --M [1]
#python train_origin.py --cfg configs/cub200/sdd_dkd/res32x4_mv2.yaml --gpu 1 --M [1,2,4]
#
#
#python train_origin.py --cfg configs/cub200/sdd_nkd/vgg13_mv2.yaml --gpu 1 --M [1]
#python train_origin.py --cfg configs/cub200/sdd_nkd/vgg13_mv2.yaml --gpu 1 --M [1,2,4]
#
##python train_origin.py --cfg configs/cub200/sdd_nkd/res32x4_mv2.yaml --gpu 1 --M [1] 57.38
#python train_origin.py --cfg configs/cub200/sdd_nkd/res32x4_mv2.yaml --gpu 1 --M [1,2,4]