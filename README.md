# PCMC-Net
PCMC-Net: Feature-based Pairwise Choice Markov Chains

Everything runs on Python 3 except the original PCMC model on Python 2.

## Experiment 1

Data generation (requires GSL library) and Figure 2 plot: 
cd mlba
cmake .
make 
python data_generation.py
python figure2i.py

MNL is unintesting on this example, but if you want to run it anyway:
cd mnl
pip install larch
python mnl_synthetic.py

To run the original PCMC (requires code from ... and Python 2):
python Pcmc-orig-synthetic.py

PCMC-Net:
python train_synthetic.py --batch_size 16 --activation 3 --lr .001 --hidden_layers 1 --nodes_per_layer 512 --dropout 0.2 --fname ../mlba/generated_N20000_seed1234.bz2 --max_epochs 11 --final 
python eval_synthetic.py
python eval_synth_fig.py


## Experiment 2

Data was obtained from: Mottini, Alejandro, and Rodrigo Acuna-Agost. "Deep choice model using pointer networks for airline itinerary prediction." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017. 

python train.py  --batch_size  16 --activation 3  --lr .001 --hidden_layers 2 --nodes_per_layer 512 --final  --max_epochs 66  
python eval_test.py --batch_size 16 --model models/pcmcNet_device~cuda_patience~5_sig_imp~0.01_dev_batch_size~8_activation~3_train_batch_size~16_max_epochs~66_hidden_layers~2_lr~0.001_index~0_nodes_per_layer~512_dropout~0.5.pth 

