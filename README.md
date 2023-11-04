# ML_Engineer - Sairub Naaz

The provided code defines a custom dataset class called `SequenceDataset` for use with PyTorch's DataLoader. This dataset is designed to handle sequences and their corresponding labels for a machine learning task. 

## MODEL:

The model comprises of an embedding layer (or a self attention layer) to make dense vector representations before going into the convolution network. This is thus followed by a convolution layer and a sequential layer of two residual nets. A max pooling of kernel_size=3 is applied to this, followed by flattening the layer so that we have 128*60 as the input size for the fully connected layer.

With an embedding layer, the model can learn task-specific representations during training. The embedding weights are updated through backpropagation to minimize the loss of the model on the task, adapting the embeddings to the specific characteristics of the data and the task. We can also change this embedding layer to self attention layer in the `prot_cnn.py` file if we do not want to use pre-trained embeddings. The self-attention layer will allow the model to learn the representations directly from the input data. This can be beneficial when you have a domain-specific dataset. You can uncomment the self-attention layer in the forward pass in `prot_cnn.py`.


## DESCRIPTION OF FILES:

** `requirements.txt` contains all the libraries you need to run the code. To install the libraries, type the following in your terminal-
   `pip install . -r requirements.txt`
   
1. `Utils.py` - contains the util functions that support the main code
   
2. `Dataset.py` - contains the SequenceDataset class, which makes relevant dictionaries of sequences and the IDs (labels)
   
3. `Plotter.py` - is a supporting script that is used to plot the three graphs if the user asks for it
   
4. `train.py` - main file to be executed. It contains default arguments and hyperparameters, but if one wants to test it with different hyperparameters, then follow the given sequence to train the network:

      `python train.py --data_dir "path to dataset dir" --batch_size 32 --learning_rate 0.01 --weight_decay 0.01 --num_epochs 20 --num_workers 7

5. `Dockerfile` is the docker setup file. The next step is to build the docker image -

   `docker build -t protein_classifier .`

   This will create a runnable image protein classifier in your docker container. When you build with docker, it will see if all libraries have been installed from requirements.txt, and it might fail if your dependencies are not compatible with the version of Python you are using.

7. Next step is to run this docker image -
   
     `docker run protein_classifier`

8. To run the tests, open a terminal, navigate to the project_dir->tests, and run:

   `pytest`