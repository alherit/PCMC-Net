# PCMC-Net
This repository contains the Python code to reproduce the experiments of my paper:

	PCMC-Net: Feature-based Pairwise Choice Markov Chains
    Alix Lhéritier
    ICLR 2020

Everything runs on Python 3 except the original PCMC model requiring Python 2.

## Experiment 1

Requirements:
* pandas
* matplotlib
* numpy
* scikit-learn
* PyTorch >= 1.1

Synthetic data is included in the mlba folder.
For regenerating it, you need mlba.hpp and mlba.cpp (in the mlba folder) from https://github.com/tkngch/choice-models/tree/master/mlba and GSL, Eigen and C++11 to be installed. Then do: 
```
cd mlba
git clone https://github.com/pybind/pybind11.git
cmake .
make 
#training set
python data_generation.py --seed 1234 -N 20000
#test set
python data_generation.py --seed 5678 -N 10000
cd ..
```

Figure 2 plot:
```
python figure2i.py
``` 

MNL: requires larch (https://pypi.org/project/larch/)
```
pip install larch
mkdir exp1-mnl
cd exp1-mnl
python ../mnl/mnl_synthetic.py --trainset ../mlba/generated_N20000_seed1234.csv --testset ../mlba/generated_N10000_seed5678.bz2
cd ..
```

To run the original PCMC (requires pcmc_utils.py from https://github.com/sragain/pcmc-nips in PYTHONPATH and Python 2):
```
mkdir exp1-pcmc
cd exp1-pcmc
bash ../pcmc/launch.sh
cd ..
```

PCMC-Net:
```
mkdir exp1-pcmcnet
cd exp1-pcmcnet
# training
# KL evaluation
# figure generation
for h in {1..3}; do \
	python ../train_synthetic.py --batch_size 1 --activation 3 --lr .001 --hidden_layers $h --nodes_per_layer 16 --dropout 0. --max_epochs 100 --final  --fname ../mlba/generated_N20000_seed1234.bz2 ;\
	python ../eval_synthetic.py --testset ../mlba/generated_N10000_seed5678.bz2 --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~1_max_epochs~100_hidden_layers~${h}_lr~0.001_index~0_nodes_per_layer~16_dropout~0.0.pth  --batch_size 1024 ;\
	python ../eval_synth_fig.py --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~1_max_epochs~100_hidden_layers~${h}_lr~0.001_index~0_nodes_per_layer~16_dropout~0.0.pth  --batch_size 1024 ;\
done 

cd ..
```

## Experiment 2

Data was obtained from: Mottini, Alejandro, and Rodrigo Acuna-Agost. "Deep choice model using pointer networks for airline itinerary prediction." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017. 
```
mkdir exp2-pcmcnet
cd exp2-pcmcnet
python train.py  --data_folder <YOUR_FOLDER>  --batch_size  16 --activation 3  --lr .001 --hidden_layers 2 --nodes_per_layer 512 --final  --max_epochs 66  
python eval_test.py --batch_size 16 --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~16_max_epochs~66_hidden_layers~2_lr~0.001_index~0_nodes_per_layer~512_dropout~0.5.pth 
cd ..
```

## License
[MIT license](https://github.com/alherit/PCMC-Net/blob/master/LICENSE).

If you have questions or comments about anything regarding this work, please see the paper for contact information.
