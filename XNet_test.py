import argparse
import glob
import logging
import os
import time

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from Model.Net import Net
from Model.data_loader import (RescaleT, ToTensorLab, SODDataset)
from Model.torch_utils import (time_synchronized, check_file, normPRED, save_output)

logging.getLogger().setLevel(logging.INFO)


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(3, 1)
    model.to(device).eval()

    datasets = os.path.join(os.getcwd(), 'TestData', opt.dataset)
    inferencedir = os.path.join(os.getcwd(), opt.save_dir, opt.dataset + '_Results', os.sep)
    weights = os.path.join(os.getcwd(), 'SavedModels', 'XNet.pt')
    Ckpt = os.path.join(os.getcwd(), 'SavedModels', 'XNet_Temp.pt')
    datalist = sorted(glob.glob(os.path.join(datasets, '*.*')))

    if not os.path.exists(inferencedir):
        os.makedirs(inferencedir, exist_ok=True)

    # optimizer
    if opt.SGD:
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    if opt.resume:
        ckpt = torch.load(Ckpt, map_location=device) if check_file(Ckpt) else None
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        model.load_state_dict(torch.load(weights), strict=False)

    # dataloader
    TestSODDataSet = SODDataset(img_name_list=datalist, lbl_name_list=[],
                                transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    TestSODDataLoader = DataLoader(TestSODDataSet, batch_size=1, shuffle=False, num_workers=opt.workers,
                                   pin_memory=True)
    time_sum = 0
    logging.info('Start inference!')

    t0 = time.time()
    # inference for each image
    for i_test, data_test in enumerate(TestSODDataLoader):
        logging.info('testing: %s' % datalist[i_test].split(os.sep)[-1])

        input = data_test['image'].type(torch.FloatTensor).to(device, non_blocking=True)

        with torch.no_grad():
            start = time_synchronized()
            fusion_loss = model(input)
            time_sum += time_synchronized() - start

            # normalization
            pred = fusion_loss[:, 0, :, :]
            pred = normPRED(pred)

            save_output(datalist[i_test], pred, inferencedir)

    logging.info('\n' + '%s is %f fps in the %s DataSet.' % (model, len(datalist) / time_sum, inferencedir))
    print('Done. (%.3fs)' % (time.time() - t0))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SOD', help='TUDS-TE PASCAL HKU')
    parser.add_argument('--save-dir', type=str, default='Inference', help='directory to save results')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='most recent training Model')
    parser.add_argument('--SGD', nargs='?', const=True, default=True, help='SGD/ Adam optimizer, default SGD')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    opt = parser.parse_args()
    print(opt)

    main(opt)
