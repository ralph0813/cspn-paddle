import argparse

import paddle
from paddle import optimizer
from paddle.io import DataLoader

import utils
from dataloader.nyu_loader import NyuDepth
from loss import Wighted_L1_Loss
from model.cspn_model import get_model_cspn_resnet


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--root', type=str, default='./data/nyudepth_hdf5', help='data root')
    parser.add_argument('--device', type=str, default='cuda', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')
    parser.add_argument('--save_path', type=str, default='checkpoints/', help='path to save the checkpoints')
    parser.add_argument('--pretrain', type=str, default='checkpoints/best_model.pth',
                        help='path to save the checkpoints')
    return parser.parse_args()


def train_epoch(model, data_loader, loss_fn, optimizer, epoch):
    error_sum_train = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    model.train()
    for i, data in enumerate(data_loader):
        optimizer.clear_grad()
        inputs = data['rgbd']
        targets = data['depth']
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss:.4f}\t'.format(epoch, i, len(data_loader), loss=loss.item()))

        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        for key in error_sum_train.keys():
            error_sum_train[key] += error_result[key]

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
    for i, data in enumerate(data_loader):
        inputs = data['rgbd']
        targets = data['depth']
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss:.4f}\t'.format(epoch, i, len(data_loader), loss=loss.item()))

        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)
        for key in error_sum.keys():
            error_sum[key] += error_result[key]

    for key in error_sum.keys():
        error_sum[key] /= len(data_loader)
    return error_sum


def train(args):
    train_set = NyuDepth(args.root, 'train', 'train.csv')
    val_set = NyuDepth(args.root, 'test', 'val.csv')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model_cspn_resnet()
    model_named_params = [p for _, p in model.named_parameters() if not p.stop_gradient]
    optim = optimizer.Adam(learning_rate=args.lr, parameters=model_named_params, weight_decay=args.weight_decay)
    lose_fn = Wighted_L1_Loss()
    for epoch in range(args.epoch):
        train_epoch(model, train_loader, lose_fn, optim, epoch)
        val_epoch(model, val_loader, lose_fn, epoch)
        utils.save_checkpoint({
            # save checkpoint
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'args': args,
        }, False, epoch, args.save_path)


if __name__ == '__main__':
    args = parse_args()
    train(args)
