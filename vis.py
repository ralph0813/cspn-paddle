import argparse
import os

import cv2
import paddle
from paddle.io import DataLoader
from tqdm import tqdm

import utils
from dataloader.nyu_loader import NyuDepth
from loss import Wighted_L1_Loss
from model.cspn_model import get_model_cspn_resnet
from utils import get_vis_depth_img


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--root', type=str, default='./data/nyudepth_hdf5', help='data root')
    parser.add_argument('--device', type=str, default='cpu', help='specify gpu device')
    parser.add_argument('--out_path', type=str, default='out/', help='path to save the images')
    parser.add_argument('--pretrain', type=str, default='None', help='path to load the pretrain model')
    parser.add_argument('--log_dir', type=str, default=None, help='path to save the log')

    return parser.parse_args()


@paddle.no_grad()
def test_vis_epoch(model, data_loader, loss_fn, epoch):
    error_sum = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    model.eval()
    for i, data in tqdm(enumerate(data_loader), desc='Val Epoch: {}'.format(epoch), total=len(data_loader)):
        inputs = data['rgbd']
        targets = data['depth']
        outputs = model(inputs)
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        outputs = outputs.numpy()
        targets = targets.numpy()
        pred_img = outputs[0, 0, :, :]
        gt_img = targets[0, 0, :, :]

        pred_img, grad_colored = get_vis_depth_img(pred_img)
        gt_img, _ = get_vis_depth_img(gt_img)

        # save result
        cv2.imwrite('out/{}_pred_colorized.png'.format(i), pred_img)
        # cv2.imwrite('out/{}_pred_lut.png'.format(i), grad_colored)
        cv2.imwrite('out/{}_gt_colorized.png'.format(i), gt_img)

        for key in error_sum.keys():
            error_sum[key] += error_result[key]

        logger.write_log(epoch * len(data_loader) + i, error_result, "test")

    for key in error_sum.keys():
        error_sum[key] /= len(data_loader)
    return error_sum


def main(args):
    paddle.device.set_device(args.device)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    val_set = NyuDepth(args.root, 'test', 'val.csv')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    model = get_model_cspn_resnet()
    if args.pretrain and os.path.exists(args.pretrain):
        model.load_state_dict(paddle.load(args.pretrain)['model'])

    lose_fn = Wighted_L1_Loss()
    val_metrics = test_vis_epoch(model, val_loader, lose_fn, 0)


if __name__ == '__main__':
    args = parse_args()
    with utils.Logger(args.log_dir) as logger:
        main(args)