# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      sampler
   Author :         zmfy
   DateTime :       2023/12/17 15:13
   Description :    
-------------------------------------------------
"""
import numpy as np
import torch


class FewShotBatchSampler:
    """Define a batch sampler for few shot task_type"""

    def __init__(self, labels, n_ways, k_shots, iterations) -> None:
        self.labels = labels
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.iterations = iterations

        self.classes, self.count = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: (self.classes, max(self.count))
        # fill it with zeros.
        self.idxs = range(len(self.labels))
        self.indexes = torch.zeros(size=(len(self.classes), max(self.count)), dtype=int)
        self.numel_per_class = torch.zeros_like(self.classes)  # store the number of samples of each class\row
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(self.indexes[label_idx] == 0)[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        """yield a batch of samples"""
        n, k = self.n_ways, self.k_shots

        for _ in range(self.iterations):
            b_size = n * k
            batch = []
            c_idxs = torch.randperm(len(self.classes))[:n]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * k, (i + 1) * k)
                label_idx = torch.arange(len(self.classes)).long()[c_idxs[i]].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:k]
                batch.append(self.indexes[label_idx][sample_idxs])
            # batch = np.random.shuffle(batch)
            batch = torch.stack(batch).t().reshape(-1)  # abcdabcdabcd form

            yield batch

    def __len__(self):
        """returns the number of iterations (episodes) per step"""
        return self.iterations
