# ML_Engineer - Sairub Naaz

The provided code defines a custom dataset class called `SequenceDataset` for use with PyTorch's DataLoader. This dataset is designed to handle sequences and their corresponding labels for a machine learning task. Here's a breakdown of what happens in the code:

1. Import necessary libraries:
   - `numpy` is imported as `np` for numerical operations.
   - `torch` is imported to work with PyTorch.

2. Define the `SequenceDataset` class:
   - This class extends the `torch.utils.data.Dataset` class, indicating that it's intended for use with PyTorch's data loading utilities.

3. Constructor (`__init__` method):
   - The constructor takes several parameters:
     - `word2id`: A dictionary mapping words (or tokens) to unique IDs.
     - `fam2label`: A dictionary mapping family labels to their corresponding IDs.
     - `max_len`: The maximum sequence length for preprocessing.
     - `data_path`: The path to the data.
     - `split`: The data split (e.g., "train" or "test").
     - `u`: An unspecified object, possibly containing a `reader` method for reading data.

4. Inside the constructor:
   - The constructor initializes instance variables:
     - `self.word2id`: Stores the provided `word2id` dictionary.
     - `self.fam2label`: Stores the provided `fam2label` dictionary.
     - `self.max_len`: Stores the provided `max_len`.
     - `self.data` and `self.label`: Call the `u.reader` method to read data and labels from the specified `split` and `data_path`. The specifics of the `u.reader` method are not provided in this code snippet.

5. `__len__` method:
   - This method returns the length of the dataset, which is the number of data samples.

6. `__getitem__` method:
   - This method is called when you access an element by index using `dataset[index]`. It takes an `index` as input.
   - It performs the following steps:
     - Retrieve the `index`-th sequence from the data and preprocess it using the `preprocess` method.
     - Get the corresponding label (family ID) from the `self.fam2label` dictionary. If the label is not found, it defaults to 0 (corresponding to '<unk>').
     - Return a dictionary containing two items:
       - 'sequence': The preprocessed sequence in one-hot encoded form.
       - 'target': The label (family ID).

7. `preprocess` method:
   - This method takes a text sequence as input and preprocesses it for model input.
   - It initializes an empty list `seq` to store the encoded sequence.
   - It iterates through each word in the input sequence (up to `max_len`) and looks up its ID in the `word2id` dictionary.
   - It pads the sequence with '<pad>' (with ID 0) if its length is less than `max_len`.
   - It converts the list of IDs into a PyTorch tensor.
   - It one-hot encodes the tensor using `torch.nn.functional.one_hot`.
   - It permutes the dimensions of the one-hot tensor to make it suitable for a convolutional neural network (CNN).
   - Finally, it returns the preprocessed sequence as a tensor.

This dataset class is designed to handle sequences and labels for training a model, particularly for tasks like text classification. It takes care of preprocessing the text data and converting it into the required format for training a neural network.
