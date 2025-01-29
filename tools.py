import pytorch_lightning as pl
from keras import ops
import torch

def compile(model, optimizer, loss, metrics, device="cpu"):
  class LightningModel(pl.LightningModule):
    def __init__(self, model, optimizer, loss, metrics):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss
        self.device = device
        self.metrics_name = []
        self.metrics_func = []
        for i in range(len(metrics)):
            name, func = metrics[i].items()
            self.metrics_name.append(name)
            self.metrics_func.append(func)

    def forward(self, x):
        return self.model(x)

    def prepare_input(self, x, y):
        x = ops.convert_to_tensor(x).type(torch.LongTensor).to(self.device)
        y = ops.convert_to_tensor(y).type(torch.LongTensor).to(self.device)
        return x, y

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = self.prepare_input(inputs, labels)
        predictions = self(inputs)
        loss = self.criterion(predictions, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        for i in range(len(self.metrics_func)):
            self.log(f'train_{self.metrics_name[i]}', self.metrics_func[i](predictions, labels), prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def summary(self):
        return self.model.summary()
        
  return LightningModel(model,optimizer,loss,metrics)


def load(model,checkpoint_path):
  checkpoint = torch.load(checkpoint_path, weights_only=True)
  model.model.load_state_dict(checkpoint['state_dict'])
  return model