# Official Code of GUNets

### Dependencies

* Python >= 3.6.8
* PyTorch >= 1.3.0
* numpy
* scipy
* pickle
* ipdb
* klearn



### Directory

Datasets can be downloaded from the [Google Drive Link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) provided in GraphSAINT (ICLR2020). After downloading the dataset, the directory structure should be as below: 

Note that the floders such as ,`pre/`, `out_res/`, and `model_save/`, need to be created manually. 



```
MainFolder/
│   README.md
│
└───GUNets/
│   │   models.py
│   │   train.py
│   │   pre/
│   │   out_res/
│   │   model_save/
│   │   ...
│   
└───data/
│   └───ppi/
│   │   │    adj_train.npz
│   │   │    adj_full.npz
│   │   │    ...
│   │   
│   └───reddit/
│   │   │    ...
│   │
│   └───...
│
```

### Run the Code

For the unfolding process, we use ''multiprocessing'' in Python to achieve a 40-thread parallelization. Please set the value of `n_jobs` for each dataset according to the thread count of you CPU(s).



PPI

```
python train.py --warm_batch_num 100 --if_output --lr 0.001 --dataset ppi --mlp_layer 6 --mlp_size 768 --emb_size 768 --drop_rate 0.1 --run_times 3 --patience 10 --output_batch 15 --batch_size 10240 --weight same --max_degree 256 --if_trans_bn --if_mlp_bn --if_bn_share --mlp_init kaiming --if_trans_share --n_jobs 40
```

Flickr

```
python train.py --warm_batch_num 0 --if_output --lr 0.01 --dataset flickr --mlp_layer 2 --mlp_size 256 --emb_size 256 --drop_rate 0.5 --run_times 3 --patience 10 --output_batch 15 --batch_size 8192 --weight same --max_degree 100 --if_trans_bn --if_mlp_bn --if_trans_share --bn_mom 0.01 --trans_init kaiming --n_jobs 40
```

Reddit

```
python train.py --warm_batch_num 100 --if_output --lr 0.001 --dataset reddit --mlp_layer 2 --mlp_size 256 --emb_size 256 --drop_rate 0.4 --run_times 3 --patience 50 --output_batch 30 --batch_size 4096 --weight same --max_degree 100 --if_trans_bn --if_mlp_bn --if_trans_share --n_jobs 40
```

Yelp

```
python train.py --warm_batch_num 100 --batch_size 20480 --lr 0.001 --dataset yelp --mlp_layer 2 --mlp_size 768 --emb_size 768 --drop_rate 0.1 --run_times 3 --patience 100 --output_batch 25 --weight same --max_degree 256 --if_trans_bn --mlp_act relu --if_trans_share --if_mlp_bn --if_bn_share --mlp_act relu --weight_decay 0 --n_jobs 40
```

Amazon

```
python train.py --warm_batch_num 100 --batch_size 20480 --lr 0.001 --dataset amazon --mlp_layer 6 --mlp_size 512 --emb_size 512 --drop_rate 0.1 --run_times 3 --patience 100 --output_batch 25 --weight same --max_degree 256 --if_trans_bn --if_trans_share --if_mlp_bn --if_bn_share --weight_decay 0 --n_jobs 40
```

