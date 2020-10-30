python pred_to_eval/pred_to_eval_format.py \
--config-file configs/vovnet/mask_rcnn_V_39_FPN_2x.yaml \
--image-root /data/ms_data/npz/test \
--png-root /data/ms_data/3d_slice_origin/test \
--mask-dir /data/ms_data/segment/test/lung_mask
