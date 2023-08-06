import torch
import torch.nn as nn
import ot
class SlicedWassersteinDist(nn.Module):
    def __init__(self, n_projections=100):
        super(SlicedWassersteinDist, self).__init__()
        self.n_projections = n_projections

    def forward(self, P_batch, Q_batch):
      batch_size = P_batch.shape[0]
      loss = 0
      for i in range(batch_size):
        loss += ot.sliced_wasserstein_distance(P_batch[i], Q_batch[i], n_projections=self.n_projections)
      return loss / batch_size

class SphericalSlicedWassersteinDist(nn.Module):
    def __init__(self):
        super(SphericalSlicedWassersteinDist, self).__init__()

    def push_to_sphere(self, X):
        return X / torch.sqrt(torch.sum(X**2, -1, keepdims=True))

    def forward(self, P_batch, Q_batch):
        batch_size = P_batch.shape[0]
        loss = 0
        for i in range(batch_size):
            P = self.push_to_sphere(P_batch[i])
            Q = self.push_to_sphere(Q_batch[i])
            loss += ot.sliced_wasserstein_sphere(P, Q)
        return loss / batch_size
