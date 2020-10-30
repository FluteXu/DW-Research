CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./tools/train_net_vov.py --num-gpus 8 \
--config-file configs/vovnet/mask_rcnn_V_39_FPN_2x.yaml \
--eval-only MODEL.WEIGHTS /home/wangcheng/ms-project/MRCN-V2-39-2x/model_final.pth