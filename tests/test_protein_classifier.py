import pandas as pd
import pytest
import os
import sys
main_code_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(main_code_dir)

from utils import Utils
from Dataset import SequenceDataset

@pytest.fixture
def sample_data():
    data = ["ABCDEF", "XYZABC"]
    labels = ["Family1", "Family2"]
    return data, labels

def test_build_labels(sample_data):
    data, labels = sample_data
    targets = pd.Series(labels)
    u = Utils()
    fam2label = u.build_labels(targets)
    assert fam2label["Family2"] == 1
    assert fam2label['<unk>'] == 0

def test_build_vocab(sample_data):
    data, _ = sample_data
    u = Utils()
    word2id = u.build_vocab(data)
    assert word2id["A"] == 2
    assert word2id["C"] == 3
    assert word2id["<unk>"] == 1