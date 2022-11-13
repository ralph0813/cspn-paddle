import argparse
import os

import paddle
from paddle import optimizer
from paddle.io import DataLoader
from tqdm import tqdm

import utils
from dataloader.nyu_loader import NyuDepth
from loss import Wighted_L1_Loss
from model.cspn_model import get_model_cspn_resnet


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--root', type=str, default='./data/nyudepth_hdf5', help='data root')
    parser.add_argument('--device', type=str, default='gpu', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--interval', default=3, type=float, help='interval of save model')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')
    parser.add_argument('--save_path', type=str, default='checkpoints/', help='path to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default=None, help='path to save the log')
    parser.add_argument('--pretrain', type=str, default='checkpoints/model_best.pdparams',
                        help='path to load the pretrain model')
    return parser.parse_args()


def train_epoch(model, data_loader, loss_fn, optimizer, epoch):
    error_sum_train = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    model.train()
    for i, data in tqdm(enumerate(data_loader), desc='Train Epoch: {}'.format(epoch), total=len(data_loader)):
        optimizer.clear_grad()
        inputs = data['rgbd']
        targets = data['depth']
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # print('Epoch: [{0}][{1}/{2}]\t'
        #       'Loss {loss:.4f}\t'.format(epoch, i, len(data_loader), loss=loss.item()))
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        for key in error_sum_train.keys():
            error_sum_train[key] += error_result[key]

        logger.write_log(epoch * len(data_loader) + i, error_result, "train")

    for key in error_sum_train.keys():
        error_sum_train[key] /= len(data_loader)
    return error_sum_train


@paddle.no_grad()
def val_epoch(model, data_loader, loss_fn, epoch):
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
        # loss = loss_fn(outputs, targets)
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)

        for key in error_sum.keys():
            error_sum[key] += error_result[key]

        logger.write_log(epoch * len(data_loader) + i, error_result, "val")

    for key in error_sum.keys():
        error_sum[key] /= len(data_loader)
    return error_sum


def train(args):
    paddle.device.set_device(args.device)

    train_set = NyuDepth(args.root, 'train', 'train.csv')
    val_set = NyuDepth(args.root, 'test', 'val.csv')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model_cspn_resnet()
    model_named_params = [p for _, p in model.named_parameters() if not p.stop_gradient]
    optim = optimizer.Adam(learning_rate=args.lr, parameters=model_named_params, weight_decay=args.weight_decay)
    lose_fn = Wighted_L1_Loss()

    if args.pretrain and os.path.exists(args.pretrain):
        checkpoints = paddle.load(args.pretrain)
        model.set_state_dict(checkpoints['model'])
        optim.set_state_dict(checkpoints['optimizer'])
        start_epoch = checkpoints['epoch']
        best_error = checkpoints['val_metrics']
    else:
        start_epoch = 0
        best_error = {
            'MSE': float('inf'), 'RMSE': float('inf'), 'ABS_REL': float('inf'), 'LG10': float('inf'),
            'MAE': float('inf'),
            'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
            'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
        }

    for epoch in range(start_epoch, args.epoch):
        train_metrics = train_epoch(model, train_loader, lose_fn, optim, epoch)
        val_metrics = val_epoch(model, val_loader, lose_fn, epoch)
        logger.write_log(epoch, train_metrics, "train_epoch")
        logger.write_log(epoch, val_metrics, "val_epoch")

        is_best = False
        if val_metrics['ABS_REL'] < best_error['ABS_REL']:
            best_error = val_metrics
            is_best = True
        if epoch % args.interval == 0 or is_best:
            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, is_best, epoch, args.save_path)


if __name__ == '__main__':
    args = parse_args()
    with utils.Logger(args.log_dir) as logger:
        train(args)
