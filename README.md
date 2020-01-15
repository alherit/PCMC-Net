# PCMC-Net
PCMC-Net: Feature-based Pairwise Choice Markov Chains

Everything runs on Python 3 except the original PCMC model on Python 2.

## Experiment 1

Data generation requires mlba.hpp and mlba.cpp (in mlba folder) from https://github.com/tkngch/choice-models/tree/master/mlba and GSL, Eigen and C++11 to be installed. Pybind11 files should be obtained from Then, do: 
cd mlba
git clone https://github.com/pybind/pybind11.git
cmake .
make 
#training set
python data_generation.py --seed 1234 -N 20000
#test set
python data_generation.py --seed 5678 -N 10000

Figure 2 plot:
python figure2i.py

MNL:
cd mnl
pip install larch
python mnl_synthetic.py --trainset ../mlba/generated_N20000_seed1234.csv --testset ../mlba/generated_N10000_seed5678.bz2

To run the original PCMC (requires pcmc_utils.py from https://github.com/sragain/pcmc-nips in PYTHONPATH and Python 2):
python pcmc-orig-synthetic.py --trainset generated_N20000_seed1234.bz2 --testset generated_N10000_seed5678.bz2

PCMC-Net:
# training
python train_synthetic.py --batch_size 16 --activation 3 --lr .001 --hidden_layers 1 --nodes_per_layer 512 --dropout 0.2 --fname ../mlba/generated_N20000_seed1234.bz2 --max_epochs 11 --final 
# KL evaluation
python eval_synthetic.py --testset ../mlba/generated_N10000_seed5678.bz2 --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~16_max_epochs~11_hidden_layers~1_lr~0.001_index~0_nodes_per_layer~512_dropout~0.2.pth --batch_size 1024
# figure generation
python eval_synth_fig.py  --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~16_max_epochs~11_hidden_layers~1_lr~0.001_index~0_nodes_per_layer~512_dropout~0.2.pth --batch_size 1024



## Experiment 2

Data was obtained from: Mottini, Alejandro, and Rodrigo Acuna-Agost. "Deep choice model using pointer networks for airline itinerary prediction." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017. 

python train.py  --batch_size  16 --activation 3  --lr .001 --hidden_layers 2 --nodes_per_layer 512 --final  --max_epochs 66  
python eval_test.py --batch_size 16 --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~16_max_epochs~66_hidden_layers~2_lr~0.001_index~0_nodes_per_layer~512_dropout~0.5.pth 

