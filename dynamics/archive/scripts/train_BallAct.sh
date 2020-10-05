CUDA_VISIBLE_DEVICES=1 python train.py \
	--env BallAct \
	--gauss_std 5e-2 \
	--n_kp 5 \
	--nf_hidden 16 \
	--n_his 5 \
	--n_roll 10 \
	--node_attr_dim 0 \
	--edge_attr_dim 1 \
	--edge_type_num 3 \
	--batch_size 24 \
	--lr 1e-3 \
	--gen_data 0 \
	--num_workers 10 \
	--dy_epoch -1 \
	--dy_iter -1 \
	# --log_per_iter 1 \
	# --eval 1
