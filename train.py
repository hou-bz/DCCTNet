import os

import imageio
import torch
import argparse

from matplotlib import pyplot as plt

from networks import get_model
from utils.base_pl_model import BasePLModel
from datasets.midataset import SliceDataset
from utils.loss_functions import calc_loss, region_affinity_distillation
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import seed
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

seed.seed_everything(123)
parser = argparse.ArgumentParser('train')
parser.add_argument('--train_data_path', type=str,
                    default='/media/zgm/c3c40470-006a-4c53-aa4a-9f50621c7edb/zgm/students/hbz/final_code/kits/preprocess_tumor/train')
parser.add_argument('--test_data_path', type=str,
                    default='/media/zgm/c3c40470-006a-4c53-aa4a-9f50621c7edb/zgm/students/hbz/final_code/kits/preprocess_tumor/test')
parser.add_argument('--checkpoint_path', type=str,
                    default='/media/zgm/c3c40470-006a-4c53-aa4a-9f50621c7edb/zgm/students/hbz/final_code/net2/checkpoints')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--model', type=str, default='mynet2')
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-4)


class SegPL(BasePLModel):
    def __init__(self, params):
        super(SegPL, self).__init__()
        self.save_hyperparameters(params)
        self.net = get_model(self.hparams.model, channels=2)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        output, a5, a4, a3, x4, x3, x2 = self.forward(ct)
        loss1 = calc_loss(output, mask)  # Dice_loss Used
        loss2 = region_affinity_distillation(a5, x4, mask)
        loss3 = region_affinity_distillation(a4, x3, mask)
        loss4 = region_affinity_distillation(a3, x2, mask)
        loss = loss1 + 0.4*(loss2+loss3+loss4)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output, _, _, _, _, _, _ = self.forward(ct)

        self.measure(batch, output)
        # print(ct.shape)
        for out in output:
            # print(output.shape)
            # print(name)
            output = torch.argmax(torch.softmax(  # torch.argmax()函数中dim表示该维度会消失。
                out, dim=0), dim=0).squeeze(0)  # squeeze()表示维度为１的维度消失  #  0,0,0不可变
            out = output.cpu().numpy()  # 数据类型转换
            # print(out.shape)

            save_image = os.path.join(self.hparams.model, "pred")
            if not os.path.exists(save_image):
                os.makedirs(save_image)
            save_image_path = os.path.join(save_image, f'{batch_idx}.jpg')
            # print(save_image_path)
            plt.imsave(save_image_path, out)
        ################################################################################################
        for m in mask:
            # print(ct.shape)
            # print(name)
            output = torch.argmax(torch.softmax(  # torch.argmax()函数中dim表示该维度会消失。
                m, dim=0), dim=0).squeeze(0)  # squeeze()表示维度为１的维度消失  #  0,0,0不可变
            out = output.cpu().numpy()  # 数据类型转换
            # print(out.shape)

            save_image = os.path.join(self.hparams.model, "mask")
            if not os.path.exists(save_image):
                os.makedirs(save_image)
            save_image_path = os.path.join(save_image, f'{batch_idx}.jpg')
            # print(save_image_path)
            plt.imsave(save_image_path, out)
        ################################################################################
        for c in ct:
            # print(c.shape)
            # print(name)
            output = c.squeeze(0)  # squeeze()表示维度为１的维度消失  #  0,0,0不可变
            out = output.cpu().numpy()  # 数据类型转换
            # print(out.shape)

            save_image = os.path.join(self.hparams.model, "ct")
            if not os.path.exists(save_image):
                os.makedirs(save_image)
            save_image_path = os.path.join(save_image, f'{batch_idx}.jpg')
            # print(save_image_path)
            # plt.imsave(save_image_path, out)
            imageio.imwrite(save_image_path, out)
            # cv2.imwrite(save_image_path, out)
            # plt.imshow(out)
            # plt.show()

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.train_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.test_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task,
            train=False
        )
        return DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1}
        return [opt], [scheduler]


def main():
    args = parser.parse_args()
    model = SegPL(args)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path),
        filename='checkpoint_%s_%s_%s_{epoch}' % (args.dataset, args.task, args.model),
        save_last=True,
        save_top_k=-1,
    )

    logger = TensorBoardLogger('log', name='%s_%s_%s' % (args.dataset, args.task, args.model))
    trainer = Trainer.from_argparse_args(args, max_epochs=args.epochs, gpus=[1], callbacks=checkpoint_callback,
                                         logger=logger)
    trainer.fit(model)


def test():
    args = parser.parse_args()
    model = SegPL.load_from_checkpoint(checkpoint_path=os.path.join(args.checkpoint_path, 'last.ckpt'))
    trainer = Trainer(gpus=[1])
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()
