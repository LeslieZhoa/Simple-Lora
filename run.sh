export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO


# python -m torch.distributed.launch train.py --batch_size 1 --train_text_encoder
python  train.py  --batch_size 1 --dist --train_text_encoder 
