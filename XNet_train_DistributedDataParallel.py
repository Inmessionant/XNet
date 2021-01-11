import argparse
import logging
from pathlib import Path

import torch.distributed as dist
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Model.XNet import XNet
from Model.data_loader import (Rescale, RescaleT, RandomCrop, ToTensor, ToTensorLab, SalObjDataset)
from Model.torch_utils import (init_seeds, time_synchronized, XBCELoss, model_info, check_file, select_device)

logging.getLogger().setLevel(logging.INFO)


# change model_name，epoch_num，batch_size，resume, model, num_workers
def main(opt):
    init_seeds(2 + batch_size_train)
    device = select_device(opt.device)

    if opt.model-name == 'XNet':
        model = XNet(3, 1)    # input channels and output channels
    else:
        model = X2Net(3, 1)
    
    model_info(model, verbose=True)

    print("Using apex synced BN.")
    model = amp.parallel.convert_syncbn_model(model)
    model.to(device)
    # logging.info(summary(model, (3, 320, 320)))

    # optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)

    tra_image_dir = os.path.abspath(str(Path('train_data/TR-Image')))
    tra_label_dir = os.path.abspath(str(Path('train_data/TR-Mask')))
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models' + os.sep)
    log_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '_Temp.pth')

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    dist.init_process_group(backend='nccl',  # 'distributed backend'
                            init_method='tcp://127.0.0.1:9999',  # distributed training init method
                            world_size=1,  # number of nodes for distributed training
                            rank=0)  # distributed training node rank

    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    images_files = sorted(glob.glob(os.path.join(tra_image_dir, '*.*')))
    labels_files = sorted(glob.glob(os.path.join(tra_label_dir, '*.*')))

    tra_img_name_list = [x for x in images_files if os.path.splitext(x)[-1].lower() in img_formats]
    tra_lbl_name_list = [x for x in labels_files if os.path.splitext(x)[-1].lower() in img_formats]

    logging.info('================================================================')
    logging.info('train images numbers: %g' % len(tra_img_name_list))
    logging.info('train labels numbers: %g' % len(tra_lbl_name_list))

    assert len(tra_img_name_list) == len(
        tra_lbl_name_list), 'The number of training images: %g  , the number of training labels: %g .' % (
        len(tra_img_name_list), len(tra_lbl_name_list))

    salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                   transform=transforms.Compose([RescaleT(400), RandomCrop(300), ToTensorLab(flag=0)]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(salobj_dataset)
    salobj_dataloader = torch.utils.data.DataLoader(salobj_dataset, batch_size=opt.batch-size, sampler=train_sampler,
                                                    shuffle=False, num_workers=opt.workers, pin_memory=True)
    
    start_epoch = 0

    # If there is a saved model, load the model and continue training based on it
    if opt.resume:
        check_file(log_dir)
        checkpoint = torch.load(log_dir, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    # training parameter
    ite_num = 0
    running_loss = 0.0  # total_loss = final_fusion_loss +sup1 +sup2 + sup3 + sup4 +sup5 +sup6
    running_tar_loss = 0.0  # final_fusion_loss

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
            final_fusion_loss_mblf, total_loss = muti_bce_loss_fusion(final_fusion_loss, sup1, sup2, sup3, sup4, sup5,
                                                                      sup6, labels)

            optimizer.zero_grad()

            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            # # print statistics
            running_loss += total_loss.item()
            running_tar_loss += final_fusion_loss_mblf.item()

            # del temporary outputs and loss
            del final_fusion_loss, sup1, sup2, sup3, sup4, sup5, sup6, final_fusion_loss_mblf, total_loss

            s = ('%10s' + '%-15s' + '%10s' + '%-15s' + '%10s' + '%-10d' + '%20s' + '%-10.4f' + '%20s' + '%-10.4f') % (
                'Epoch: ',
                '%g/%g' % (epoch + 1, opt.epochs),
                'Batch: ',
                '%g/%g' % ((i + 1) * opt.batch-size, len(tra_img_name_list)),
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
            torch.save(state, saved_model_dir + opt.model-name + ".pth")

    torch.save(model.state_dict(), saved_model_dir + opt.model-name + ".pth")
    torch.cuda.empty_cache()


# change device
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--model-name', type=str, default='XNet', help='model')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0, 1', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
