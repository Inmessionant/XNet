import argparse
import glob
import logging
import os
import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Model.Net import Net
from Model.data_loader import (RescaleT, RandomCrop, ToTensorLab, SODDataset)
from Model.torch_utils import (init_seeds, model_info, check_file)

logging.getLogger().setLevel(logging.INFO)


def main(opt):
    init_seeds(2 + opt.batch_size)

    # Define Model
    model = Net(3, 1)  # input channels and output channels
    model_info(model, verbose=True)  # logging.info(summary(model, (3, 320, 320)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer
    if opt.SGD:
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    TRImage = os.path.join(os.getcwd(), 'TrainData', 'TR-Image')
    TRMask = os.path.join(os.getcwd(), 'TrainData', 'TR-Mask')
    SavedModels = os.path.join(os.getcwd(), 'SavedModels' + os.sep)
    Ckpt = os.path.join(os.getcwd(), 'SavedModels', 'XNet_Temp.pt')

    if not os.path.exists(SavedModels):
        os.makedirs(SavedModels, exist_ok=True)

    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    Images_Files = sorted(glob.glob(os.path.join(TRImage, '*.*')))
    Labels_Files = sorted(glob.glob(os.path.join(TRMask, '*.*')))

    TRImage_list = [x for x in Images_Files if os.path.splitext(x)[-1].lower() in img_formats]
    TRMask_list = [x for x in Labels_Files if os.path.splitext(x)[-1].lower() in img_formats]

    logging.info('Train Images Numbers: %g' % len(TRImage_list))
    logging.info('Train Labels Numbers: %g' % len(TRMask_list))

    assert len(TRImage_list) == len(
        TRMask_list), 'The number of training images: %g the number of training labels: %g .' % (
        len(TRImage_list), len(TRMask_list))

    start_epoch = 0

    salobj_dataset = SODDataset(img_name_list=TRImage_list, lbl_name_list=TRMask_list,
                                transform=transforms.Compose(
                                    [RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                   pin_memory=True)

    if opt.resume:
        ckpt = torch.load(Ckpt, map_location=device) if check_file(Ckpt) else None
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']

    ite_num = 0
    running_loss = 0.0  # total_loss = fusion_loss
    t0 = time.time()

    for epoch in range(start_epoch, opt.epochs):

        model.train()

        pbar = enumerate(salobj_dataloader)
        pbar = tqdm(pbar, total=len(salobj_dataloader))

        for i, data in pbar:
            ite_num = ite_num + 1

            input, label = data['image'].type(torch.FloatTensor).to(device, non_blocking=True), data['label'].type(
                torch.FloatTensor).to(device, non_blocking=True)

            # forward + backward + optimize
            fusion_loss = model(input)
            loss = nn.BCELoss(reduction='mean')(fusion_loss, label).cuda()

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            s = ('%15s' + '%-15s' + '%15s' + '%-15s' + '%15s' + '%-15d' + '%15s' + '%-15.4f') % (
                'Epoch: ',
                '%g/%g' % (epoch + 1, opt.epochs),
                'Batch: ',
                '%g/%g' % ((i + 1) * opt.batch_size, len(TRImage_list)),
                'Iteration: ',
                ite_num,
                'Loss: ',
                running_loss / ite_num)
            pbar.set_description(s)

        # The model is saved every 50 epoch
        if (epoch + 1) % 10 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, SavedModels + "XNet_Temp.pt")

    torch.save(model.state_dict(), SavedModels + "XNet.pt")

    logging.info('%g epochs completed in %.3f hours.\n' % (opt.epochs - start_epoch + 1, (time.time() - t0) / 3600))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--SGD', nargs='?', const=True, default=True, help='SGD/ Adam optimizer, default SGD')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    opt = parser.parse_args()
    print(opt)

    main(opt)
