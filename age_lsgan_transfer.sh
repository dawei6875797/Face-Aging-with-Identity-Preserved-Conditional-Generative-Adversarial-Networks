python mp_test.py \
  --gan_loss_weight=75 \
  --fea_loss_weight=0.5e-4 \
  --age_loss_weight=30 \
  --fea_layer_name=conv5 \
  --checkpoint_dir=/new_disk2/wangzw/tensorflow_models/292000/resnet_generator/age/0_conv5_lsgan_transfer_g75_0.5f-4_a30/ \
  --sample_dir=age/0_conv5_lsgan_transfer_g75_0.5f-4_a30 \
  --source_checkpoint_dir=/new_disk2/wangzw/tensorflow_models/292000/resnet_generator/age/0_conv5_lsgan_transfer_g75_0.5f-4_a20/