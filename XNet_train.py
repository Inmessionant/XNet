import argparse
import glob
import logging
import os
import time

import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Model.XNet import XNet
from Model.data_loader import (RescaleT, RandomCrop, ToTensorLab, SODDataset)
from Model.torch_utils import (init_seeds, model_info, check_file)

logging.getLogger().setLevel(logging.INFO)


def main(opt):
    init_seeds(2 + opt.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define Model
    model = XNet(3, 1)  # input channels and output channels
    model_info(model, verbose=True)  # logging.info(summary(model, (3, 320, 320)))

    model.to(device)

    # optimizer
    # if opt.SGD:
    #     optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    # else:
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_image_dir = os.path.join(os.getcwd(), 'TrainData', 'TR-Image')
    train_label_dir = os.path.join(os.getcwd(), 'TrainData', 'TR-Mask')
    saved_model_dir = os.path.join(os.getcwd(), 'SavedModels' + os.sep)
    checkfile = os.path.join(os.getcwd(), 'SavedModels', opt.model_name + '_Temp.pt')

    if not os.path.exists(saved_model_dir): os.makedirs(saved_model_dir, exist_ok=True)

    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    train_image_files = sorted(glob.glob(os.path.join(train_image_dir, '*.*')))
    train_label_files = sorted(glob.glob(os.path.join(train_label_dir, '*.*')))

    train_image_list = [x for x in train_image_files if os.path.splitext(x)[-1].lower() in img_formats]
    train_label_list = [x for x in train_label_files if os.path.splitext(x)[-1].lower() in img_formats]

    logging.info('Train Images Numbers: %g' % len(train_image_list))
    logging.info('Train Labels Numbers: %g' % len(train_label_list))

    assert len(train_image_list) == len(
        train_label_list), 'The number of training images: %g the number of training labels: %g .' % (
        len(train_image_list), len(train_label_list))

    train_dataset = SODDataset(img_name_list=train_image_list, lbl_name_list=train_label_list,
                               transform=transforms.Compose(
                                   [RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                              pin_memory=True, prefetch_factor=2)  # prefetch works when pin_memory > 0

    start_epoch = 0

    if opt.resume:
        ckpt = torch.load(check_file(checkfile), map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']

    ite_num = 0
    running_loss = 0.0  # total_loss = fusion_loss
    t0 = time.time()

    for epoch in range(start_epoch, opt.epochs):

        model.train()

        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader))

        for i, data in pbar:
            ite_num = ite_num + 1

            input, label = data['image'].type(torch.FloatTensor).to(device, non_blocking=True), data['label'].type(
                torch.FloatTensor).to(device, non_blocking=True)

            # forward + backward + optimize
            optimizer.zero_grad()

            fusion_loss = model(input)
            loss = nn.BCELoss(reduction='mean')(fusion_loss, label).cuda()
            
            # scheduler.step(loss)
            loss.backward()
            optimizer.step()    

            running_loss += loss.item()

            s = ('%15s' + '%-15s' + '%15s' + '%-15s' + '%15s' + '%-15d' + '%15s' + '%-15.4f') % (
                'Epoch: ',
                '%g/%g' % (epoch + 1, opt.epochs),
                'Batch: ',
                '%g/%g' % ((i + 1) * opt.batch_size, len(train_image_list)),
                'Iteration: ',
                ite_num,
                'Loss: ',
                running_loss / ite_num)
            pbar.set_description(s)

        # The model is saved every 50 epoch
        if (epoch + 1) % 50 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, saved_model_dir + opt.model_name + "_Temp.pt")

    torch.save(model.state_dict(), saved_model_dir + opt.model_name + ".pt")
    logging.info('%g epochs completed in %.3f hours.\n' % (opt.epochs - start_epoch + 1, (time.time() - t0) / 3600))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='total epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--model-name', type=str, default='XNet', help='define model name')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # parser.add_argument('--SGD', nargs='?', const=True, default=True, help='SGD/ Adam optimizer, default SGD')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    opt = parser.parse_args()
    print(opt)

    main(opt)
