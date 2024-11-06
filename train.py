"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import pandas as pd
from collections import OrderedDict

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

  # 创建一个DataFrame来存储损失信息
    loss_df1 = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss_Name', 'Loss_Value'])
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        if epoch % opt.save_epoch_freq == 0 or epoch == opt.niter + opt.niter_decay: 
            # 初始化一个空的累加字典
            lossesTrain = OrderedDict()
            for name in model.loss_names:
                lossesTrain[name] = 0.0  # 初始化每个损失项的累加值为 0

        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if epoch % opt.save_epoch_freq == 0 or epoch == opt.niter + opt.niter_decay: 
                current_losses = model.get_current_losses()
                # 累加每个损失项
                for name, value in current_losses.items():
                    lossesTrain[name] += value

        if epoch % opt.save_epoch_freq == 0 or epoch == opt.niter + opt.niter_decay:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            for name, loss_value in lossesTrain.items():
                loss_value =loss_value/dataset_size
                print(f"{name}: {loss_value}")
                new_row = {'Epoch': epoch, 'Iteration': total_iters, 'Loss_Name': name, 'Loss_Value': loss_value}
                loss_df1 = pd.concat([loss_df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        print('End of epoch %d / %d \t ' % (epoch, opt.niter + opt.niter_decay))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    training_loss_file = 'training_losses.xlsx'
    loss_df1.to_excel(training_loss_file, index=False)
    print(f"Training losses saved to {training_loss_file}")
