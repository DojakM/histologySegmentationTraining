import pytorch_lightning as pt
from torch.utils.data import DataLoader
from seg_training.data_loading.conic_data import ConicData

class ConicDataModule(pt.LightningDataModule):
    def __init__(self, **kwargs):
        super(ConicDataModule, self).__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs
        self.setup()
        self.prepare_data()
        self.train_ids = []
        self.test_ids = []

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        self.df_train = ConicData(self.train_ids, download=False)
        self.df_test = ConicData(self.test_ids, download=False)

    def train_dataloader(self):
        return DataLoader(self.df_train, batch_size=self.args['training_batch_size'], num_workers=self.args['num_workers'], shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.df_test, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'], shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.df_test, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'], shuffle=False)

    def transfer_batch_to_device(self, batch, device):
        pass
