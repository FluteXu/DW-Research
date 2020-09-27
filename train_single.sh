CUDA_VISIBLE_DEVICES=4 python ./tools/train_net.py \
--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
--num-gpus 1 SOLVER.IMS_PER_BATCH 2 DATALOADER.NUM_WORKERS 1