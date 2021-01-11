import argparse
import glob
import logging
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Model.XNet import XNet
from Model.data_loader import (Rescale, RescaleT, RandomCrop, ToTensor, ToTensorLab, SalObjDataset)
from Model.torch_utils import (init_seeds, time_synchronized, XBCELoss, model_info, check_file, select_device,
                               strip_optimizer)

logging.getLogger().setLevel(logging.INFO)


def main(opt):
    init_seeds(2 + opt.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.model_name == 'XNet':
        model = XNet(3, 1)  # input channels and output channels
    else:
        return

    model_info(model, verbose=True)

    model.to(device)
    # logging.info(summary(model, (3, 320, 320)))

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)

    tra_image_dir = os.path.abspath(str(Path('train_data/TR-Image')))
    tra_label_dir = os.path.abspath(str(Path('train_data/TR-Mask')))
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models' + os.sep)
    log_dir = os.path.join(os.getcwd(), 'saved_models', opt.model_name + '_Temp.pth')

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

    salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                   transform=transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                   pin_memory=True)

    start_epoch = 0
    # If there is a saved model, load the model and continue training based on it
    if opt.resume:
        check_file(log_dir)
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    ite_num = 0
    running_loss = 0.0  # total_loss = final_fusion_loss +sup1 +sup2 + sup3 + sup4 +sup5 +sup6
    running_tar_loss = 0.0  # final_fusion_loss
    t0 = time.time()

    for epoch in range(start_epoch, opt.epochs):

        model.train()

        pbar = enumerate(salobj_dataloader)
        pbar = tqdm(pbar, total=len(salobj_dataloader))

        for i, data in pbar:
            ite_num = ite_num + 1

            input, label = data['image'].type(torch.FloatTensor), data['label'].type(torch.FloatTensor)
            inputs, labels = input.to(device, non_blocking=True), label.to(device, non_blocking=True)

            # forward + backward + optimize
            final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6 = model(inputs)
            final_fusion_loss_mblf, total_loss = XBCELoss(final_fusion_loss, sup1, sup2, sup3, sup4, sup5,
                                                          sup6, labels)

            # y zero the parameter gradients
            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += total_loss.item()
            running_tar_loss += final_fusion_loss_mblf.item()

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
                running_loss / ite_num,
                'Final_fusion_loss: ',
                running_tar_loss / ite_num)
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
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')

    opt = parser.parse_args()

    main(opt)
