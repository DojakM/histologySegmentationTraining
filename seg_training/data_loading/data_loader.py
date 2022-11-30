import torch.utils.data as td
from seg_training.data_loading.conic_data import ConicData

class ConicDataLoader(td.DataLoader):
    def __init__(self):
        super(ConicDataLoader, self).__init__(dataset=ConicData(), batch_size=12)

