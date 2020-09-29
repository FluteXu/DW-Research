CUDA_VISIBLE_DEVICES=1 python ./tools/train_net_vov.py \
--config-file configs/vovnet/mask_rcnn_V_39_FPN_3x.yaml \
--num-gpus 1 SOLVER.IMS_PER_BATCH 2 DATALOADER.NUM_WORKERS 1