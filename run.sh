

python -m torch.distributed.launch --nproc_per_node=2 train.py --cfg ./configs/imagenet/r34_r18/sdd_dkd.yaml --M [1,2]  72.02
python -m torch.distributed.launch --nproc_per_node=2 train.py --cfg ./configs/imagenet/r50_mv2/sdd_dkd.yaml --M [1,2,4] 73.08


python -m torch.distributed.launch --nproc_per_node=2 train.py --cfg ./configs/imagenet/r50_mv2/sdd_nkd.yaml --M [1,2,4] 73.12