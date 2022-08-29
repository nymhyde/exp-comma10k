#!/usr/bin/env python3

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import os, random
from pathlib import Path
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import _logger as log
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger

from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.core.composition import Compose

pl.seed_everything(1997)


parser = ArgumentParser()
parser.add_argument('--backbone',
                    default='efficientnet-b0',
                    type=str,
                    metavar='BK',
                    help='Name as in segmentation_models_pytorch')
parser.add_argument('--data-path',
                    default='/home/patel4db/comma10k-exp/data/',
                    type=str,
                    metavar='DP',
                    help='data_path')
parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    metavar='N',
                    help='total number of epochs')
parser.add_argument('--batch-size',
                    default=32,
                    type=int,
                    metavar='B',
                    help='batch size',
                    dest='batch_size')
parser.add_argument('--gpus',
                    type=int,
                    default=1,
                    help='number of gpus to use')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-4,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='LR')
parser.add_argument('--eps',
                    default=1e-7,
                    type=float,
                    help='eps for adaptive optimizers',
                    dest='eps')
parser.add_argument('--height',
                    default=14*32,
                    type=int,
                    help='image height')
parser.add_argument('--width',
                    default=18*32,
                    type=int,
                    help='image width')
parser.add_argument('--num-workers',
                    default=40,
                    type=int,
                    metavar='W',
                    help='number of CPU workers',
                    dest='num_workers')
parser.add_argument('--weight-decay',
                    default=1e-3,
                    type=float,
                    metavar='WD',
                    help='Optimizer weight decay')
parser.add_argument('--version',
                    default=None,
                    type=str,
                    metavar='V',
                    help='version or id of the net')
parser.add_argument('--resume-from-checkpoint',
                    default=None,
                    type=str,
                    metavar='RFC',
                    help='path to checkpoint')
parser.add_argument('--seed-from-checkpoint',
                    default=None,
                    type=str,
                    metavar='SFC',
                    help='path to checkpoint seed')

args = parser.parse_args()



## Dataset and Dataloader
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def pad_to_multiple(x, k=32):
    return int(k*(np.ceil(x/k)))

def get_train_transforms(height = 437, 
                         width = 582): 
    return A.Compose([
            A.Resize(height=height, width=width, p=1.0),
            A.PadIfNeeded(pad_to_multiple(height), 
                          pad_to_multiple(width), 
                          border_mode=cv2.BORDER_CONSTANT, 
                          value=0, 
                          mask_value=0)
        ], p=1.0)

def get_valid_transforms(height = 437, 
                         width = 582): 
    return A.Compose([
            A.Resize(height=height, width=width, p=1.0),
            A.PadIfNeeded(pad_to_multiple(height), 
                          pad_to_multiple(width), 
                          border_mode=cv2.BORDER_CONSTANT, 
                          value=0, 
                          mask_value=0)
        ], p=1.0)

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


class CommaLoader(Dataset):
    
    def __init__(self, data_path, images_path, preprocess_fn, transforms, class_values):
        super().__init__()
        
        self.data_path = data_path
        self.images_path = images_path
        self.transforms = transforms
        self.preprocess = get_preprocessing(preprocess_fn)
        self.class_values = class_values
        self.images_folder = 'imgs'
        self.masks_folder = 'masks'
        
    def __getitem__(self, idx):
        image = self.images_path[idx]
        img = cv2.imread(str(self.data_path/self.images_folder/image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.data_path/self.masks_folder/image), 0).astype('uint8')
        
        if self.transforms:
            sample = self.transforms(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']
        
        mask = np.stack([(mask == v) for v in self.class_values], axis=-1).astype('uint8')
        
        if self.preprocess:
            sample = self.preprocess(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']
            
        return img, mask
    
    def __len__(self):
        return len(self.images_path)



# | Model Class |
'''
- __init__         :: set up params for Trainer
-  model           :: build model Unet 
-  foward          :: forward pass; return logits
-  loss            :: loss function
-  training step   :: Forward pass -> loss
-  validation_step :: Forward pass -> loss
-  configure optim :: configure optimizers
-  check data      :: check whether or not we have the files
-  setup           :: setup dataset for the dataloader
-  dataloaders      :: train / validation loaders
'''
class SegNet(pl.LightningModule):
    # Transfer Learning
    
    def __init__(self,
                 data_path='/home/patel4db/comma10k-exp/data/',
                 backbone='efficientnet-b0',
                 batch_size=8,
                 lr=1e-4,
                 eps=1e-7,
                 height=14*32,
                 width=18*32,
                 num_workers=40,
                 epochs=30,
                 gpus=1,
                 weight_decay=1e-3,
                 class_values=[41, 76, 90, 124, 161, 0], **kwargs):

        super().__init__()
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.height = height
        self.width = width
        self.num_workers = num_workers
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.eps = eps
        self.class_values = class_values

        self.save_hyperparameters()
        self.preprocess_fn = smp.encoders.get_preprocessing_fn(self.backbone, pretrained='imagenet')

        self.__build_model()


    def __build_model(self):
        # Define model layers & loss 

        self.net = smp.Unet(self.backbone, classes=len(self.class_values),
                            activation=None, encoder_weights='imagenet')

        self.loss_func = lambda x, y: torch.nn.CrossEntropyLoss()(x, torch.argmax(y, axis=1))


    def forward(self, x):
        # Forward pass. Returns logits
        return self.net(x)


    def loss(self, logits, labels):
        # use the loss_function
        return self.loss_func(logits, labels)


    def training_step(self, batch, batch_idx):

        # Forward Pass 
        x, y = batch
        y_logits = self.forward(x)

        # compute loss and accuracy
        train_loss = self.loss(y_logits, y)
        result = pl.TrainResult(train_loss)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return result 


    def validation_step(self, batch, batch_idx):

        # Forward pass
        x, y = batch
        y_logits = self.forward(x)

        # Compute loss and accuracy
        val_loss = self.loss(y_logits, y)
        result = pl.EvalResult(checkpoint_on=val_loss)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        result.log('val_loss', val_loss)

        return result


    def configure_optimizers(self):

        optimizer = torch.optim.Adam
        optimizer_kwargs = {'eps': self.eps}

        optimizer = optimizer(self.parameters(),
                              lr=self.lr,
                              weight_decay=self.weight_decay,
                              **optimizer_kwargs)

        scheduler_kwargs = {'T_max': self.epochs*len(self.train_dataset)//self.gpus//self.batch_size,
                            'eta_min':self.lr/50} 

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        interval = 'step'
        scheduler = scheduler(optimizer, **scheduler_kwargs)

        return [optimizer], [{'scheduler':scheduler, 'interval': interval, 'name': 'lr'}]


    def check_data(self):
        assert (self.data_path/'imgs').is_dir(), "Images folder not found"
        assert (self.data_path/'masks').is_dir(), "Masks folder not found"
        assert (self.data_path/'files_trainable').exists(), "Files trainable file not found"

        print('Data Tree is found and ready for setup')


    def setup(self, stage):
        images_path = np.loadtxt(self.data_path/'files_trainable', dtype='str').tolist()
        random.shuffle(images_path)

        '''
        For now, we are validating on images ending with "9.png" and are seeing a categorical cross entropy loss of 0.051.
        '''

        self.train_dataset = CommaLoader(data_path=self.data_path,
                                         images_path=[x.split('masks/')[-1] for x in images_path if not x.endswith('9.png')],
                                         preprocess_fn=self.preprocess_fn,
                                         transforms=get_train_transforms(self.height, self.width),
                                         class_values=self.class_values
                                        )

        self.valid_dataset = CommaLoader(data_path=self.data_path,
                                         images_path=[x.split('masks/')[-1] for x in images_path if x.endswith('9.png')],
                                         preprocess_fn=self.preprocess_fn,
                                         transforms=get_valid_transforms(self.height, self.width),
                                         class_values=self.class_values
                                        )

    def __dataloader(self, train):
        # Train / Validation loaders
        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(dataset =_dataset,
                            batch_size = self.batch_size,
                            # num_workers = self.num_workers,
                            shuffle=True if train else False)
        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)



# Model training

## Setting up loggers for Callbacks
log_path = Path('/home/patel4db/comma10k-exp/loggerd')
name = args.backbone
version = args.version
tb_logger = TensorBoardLogger(log_path, name=name, version=version)
lr_logger = LearningRateLogger(logging_interval='epoch')
ckpt_callback = ModelCheckpoint(filepath=Path(tb_logger.log_dir)/'checkpoints/{epoch:02d}_{val_loss:.4f}', 
                                monitor='val_loss', save_top_k=10, save_last=True)

if args.seed_from_checkpoint:
    model = SegNet.load_from_checkpoint(args.seed_from_checkpoint)
    print('Model is Seeded.')
else:
    model = SegNet()

trainer = pl.Trainer(checkpoint_callback=ckpt_callback,
                     logger=tb_logger,
                     callbacks=[lr_logger],
                     gpus=args.gpus,
                     min_epochs=args.epochs,
                     max_epochs=args.epochs,
                     #limit_train_batches=100,
                     #limit_val_batches=100,
                     precision=16,
                     amp_backend='native',
                     row_log_interval=100,
                     log_save_interval=100,
                     distributed_backend='ddp',
                     benchmark=True,
                     sync_batchnorm=True,
                     resume_from_checkpoint=args.resume_from_checkpoint
                    )

trainer.logger.log_hyperparams(model.hparams)

trainer.fit(model)

print('Model Training Successful.')
