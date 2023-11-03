# ML_Engineer - Sairub Naaz

The provided code defines a custom dataset class called `SequenceDataset` for use with PyTorch's DataLoader. This dataset is designed to handle sequences and their corresponding labels for a machine learning task. 

Files:
1. Utils.py - contains the util functions that support the main code
2. Dataset.py - contains the SequenceDataset class, which makes relevant dictionaries of sequences and the IDs (labels)
3. Plotter.py - is a supporting script that is used to plot the three graphs if the user asks for it
4. train.py - main file to be executed. It contains default arguments and hyperparameters, but if one wants to test it with different hyperparameters, then follow the given sequence to train the network:

   python train.py --data_dir "path to dataset" --batch_size 64 --learning_rate 0.01 --momentum 0.9 --weight_decay 0.01 --num_epochs 20 --num_workers 7 --train True --evaluate True
