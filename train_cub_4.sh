python train_origin.py --cfg configs/cub200/sdd_dkd/res32x4_mv2.yaml --gpu 3 --M [1] 60.3
python train_origin.py --cfg configs/cub200/sdd_dkd/res32x4_mv2.yaml --gpu 3 --M [1,2,4] 63.4


python train_origin.py --cfg configs/cub200/sdd_nkd/vgg13_mv2.yaml --gpu 3 --M [1] 58.9
python train_origin.py --cfg configs/cub200/sdd_nkd/vgg13_mv2.yaml --gpu 3 --M [1,2,4] 63.3

#python train_origin.py --cfg configs/cub200/sdd_nkd/res32x4_mv2.yaml --gpu 1 --M [1] 57.38
python train_origin.py --cfg configs/cub200/sdd_nkd/res32x4_mv2.yaml --gpu 3 --M [1,2,4] 63.7