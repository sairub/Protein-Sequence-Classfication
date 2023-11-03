import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    def __init__(self,sorted_targets,sequence_lengths,mean,median,amino_acid_counter) -> None:
        self.sorted_targets = sorted_targets
        self.sequence_lengths = sequence_lengths
        self.mean = mean
        self.median = median
        self.amino_acid_counter = amino_acid_counter

    def plot_family_sizes(self):
        # Plot the distribution of family sizes

        f, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.sorted_targets.values, kde=True, log_scale=True, ax=ax)
        plt.title("Distribution of family sizes for the 'train' split")
        plt.xlabel("Family size (log scale)")
        plt.ylabel("# Families")
        plt.show()

    def plot_dist_sequences_lengths(self):
        # Plot the distribution of sequences' lengths

        f, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.sequence_lengths.values, kde=True, log_scale=True, bins=60, ax=ax)

        ax.axvline(self.mean, color='r', linestyle='-', label=f"Mean = {self.mean:.1f}")
        ax.axvline(self.median, color='g', linestyle='-', label=f"Median = {self.median:.1f}")
            
        plt.title("Distribution of sequence lengths")
        plt.xlabel("Sequence' length (log scale)")
        plt.ylabel("# Sequences")
        plt.legend(loc="best")
        plt.show()

    def plot_AA_frequencies(self):
        # Plot the distribution of AA frequencies
        f, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='AA', y='Frequency', data=self.amino_acid_counter.sort_values(by=['Frequency'], ascending=False), ax=ax)

        plt.title("Distribution of AAs' frequencies in the 'train' split")
        plt.xlabel("Amino acid codes")
        plt.ylabel("Frequency (log scale)")
        plt.yscale("log")
        plt.show()
