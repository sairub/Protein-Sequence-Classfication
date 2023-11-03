# ML_Engineer - Sairub Naaz

The provided code defines a custom dataset class called `SequenceDataset` for use with PyTorch's DataLoader. This dataset is designed to handle sequences and their corresponding labels for a machine learning task. 

Files:
** `requirements.txt` contains all the libraries you need to run the code. To install the libraries, type the following in your terminal-
   `pip install . -r requirements.txt`
   
1. `Utils.py` - contains the util functions that support the main code
   
2. `Dataset.py` - contains the SequenceDataset class, which makes relevant dictionaries of sequences and the IDs (labels)
   
3. `Plotter.py` - is a supporting script that is used to plot the three graphs if the user asks for it
   
4. `train.py` - main file to be executed. It contains default arguments and hyperparameters, but if one wants to test it with different hyperparameters, then follow the given sequence to train the network:

      `python train.py --data_dir "path to dataset" --batch_size 64 --learning_rate 0.01 --momentum 0.9 --weight_decay 0.01 --num_epochs 20 --num_workers 7 --train True --evaluate True`

5. `Dockerfile` is the docker setup file. The next step is to build the docker image -
      `docker build -t protein_classifier .`
   This will create a runnable image protein classifier in your docker container. When you build with docker, it will see if all libraries have been installed from requirements.txt, and it might fail if your dependencies are not compatible with the version of Python you are using.

6. Next step is to run this docker image -
     `docker run protein_classifier`
