import argparse
import glob
import logging
import os
import time
import torch.distributed as dist
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Model.XNet import XNet
from Model.data_loader import (RescaleT, RandomCrop, ToTensorLab, SalObjDataset)
from Model.torch_utils import (init_seeds, model_info, check_file, select_device, strip_optimizer)

logging.getLogger().setLevel(logging.INFO)


def main(opt):
    init_seeds(2 + opt.batch_size)

    # Define Model
    model = XNet(3, 1)  # input channels and output channels
    model_info(model, verbose=True)  # logging.info(summary(model, (3, 320, 320)))

    # optimizer
    if opt.SGD:
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # Single GPU or DDP model
    if not opt.DDP:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = select_device(opt.device)
        print("Using apex synced BN.")
        model = amp.parallel.convert_syncbn_model(model)
        model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    tra_image_dir = os.path.abspath(str(Path('TrainData/TR-Image')))
    tra_label_dir = os.path.abspath(str(Path('TrainData/TR-Mask')))
    saved_model_dir = os.path.join(os.getcwd(), 'SavedModels' + os.sep)
    log_dir = os.path.join(os.getcwd(), 'SavedModels', opt.model_name + '_Temp.pth')

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)

    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    images_files = sorted(glob.glob(os.path.join(tra_image_dir, '*.*')))
    labels_files = sorted(glob.glob(os.path.join(tra_label_dir, '*.*')))

    tra_img_name_list = [x for x in images_files if os.path.splitext(x)[-1].lower() in img_formats]
    tra_lbl_name_list = [x for x in labels_files if os.path.splitext(x)[-1].lower() in img_formats]

    logging.info('================================================================')
    logging.info('train images numbers: %g' % len(tra_img_name_list))
    logging.info('train labels numbers: %g' % len(tra_lbl_name_list))

    assert len(tra_img_name_list) == len(
        tra_lbl_name_list), 'The number of training images: %g the number of training labels: %g .' % (
        len(tra_img_name_list), len(tra_lbl_name_list))

    start_epoch = 0

    if not opt.DDP:  # Single GPU
        salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                       transform=transforms.Compose(
                                           [RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))
        salobj_dataloader = DataLoader(salobj_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                       pin_memory=True)

    else:  # DDP Model
        salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                       transform=transforms.Compose(
                                           [RescaleT(400), RandomCrop(300), ToTensorLab(flag=0)]))
        train_sampler = torch.utils.data.distributed.DistributedSampler(salobj_dataset)
        salobj_dataloader = torch.utils.data.DataLoader(salobj_dataset, batch_size=opt.batch_size,
                                                        sampler=train_sampler,
                                                        shuffle=False, num_workers=opt.workers, pin_memory=True)
    if opt.resume:
        ckpt = torch.load(log_dir, map_location=device) if check_file(log_dir) else None
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

            input, label = data['image'].type(torch.FloatTensor).to(device, non_blocking=True), data['label'].type(torch.FloatTensor).to(device, non_blocking=True)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            fusion_loss = model(input)
            loss = nn.BCELoss(reduction='mean')(fusion_loss, label).cuda()

            if not opt.DDP:
                loss.backward()
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # del temporary outputs and loss
            # del final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6, final_fusion_loss_mblf, total_loss

            s = ('%15s' + '%-15s' + '%15s' + '%-15s' + '%15s' + '%-15d' + '%15s' + '%-15.4f' + '%15s' + '%-15.4f') % (
                'Epoch: ',
                '%g/%g' % (epoch + 1, opt.epochs),
                'Batch: ',
                '%g/%g' % ((i + 1) * opt.batch_size, len(tra_img_name_list)),
                'Iteration: ',
                ite_num,
                'Total_loss: ',
                running_loss / ite_num)
            pbar.set_description(s)

        # The model is saved every 50 epoch
        if (epoch + 1) % 50 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, saved_model_dir + opt.model_name + "_Temp.pt")

    file = saved_model_dir + opt.model_name + ".pt"
    torch.save(model.state_dict(), file)

    # Strip optimizers
    if os.path.exists(file) and str(file).endswith('.pt'):
        strip_optimizer(file)

    logging.info('%g epochs completed in %.3f hours.\n' % (opt.epochs - start_epoch + 1, (time.time() - t0) / 3600))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--model-name', type=str, default='XNet', help='model')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--SGD', nargs='?', const=True, default=True, help='SGD/ Adam optimizer, default SGD')
    parser.add_argument('--device', default='0, 1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    opt = parser.parse_args()
    print(opt)

    main(opt)
