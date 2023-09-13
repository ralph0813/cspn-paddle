import argparse
import os

import paddle
from paddle import optimizer
from paddle.io import DataLoader
from tqdm import tqdm

import utils
from dataloader import NyuDepth
from loss import Wighted_L1_Loss
from lr_wrappers import WarmupLR
from model import resnet50 as CSPN


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--root', type=str, default='./data/nyudepth_hdf5', help='data root')
    parser.add_argument('--device', type=str, default='gpu', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epoch', default=40, type=int, help='number of epoch in training')
    parser.add_argument('--interval', default=1, type=float, help='interval of save model')
    parser.add_argument('--n_sample', default=500, type=float, help='learning rate in training')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate in training')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum in training')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening for momentum')
    parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')
    parser.add_argument('--save_path', type=str, default='weights/', help='path to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default=None, help='path to save the log')
    parser.add_argument('--pretrain', type=str, default='weights/model_best.pdparams',
                        help='path to load the pretrain model')
    parser.add_argument('--resnet_pretrain', '-r', action='store_true', help='use resnet pretrain model')
    return parser.parse_args()


def train_epoch(model, data_loader, loss_fn, optim, epoch, lr_scheduler):
    error_sum_train = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    model.train()
    loss_sum = 0
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, data in tbar:
        step = epoch * len(data_loader) + i

        optim.clear_grad()
        inputs = data['rgbd']
        targets = data['depth']
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        loss_sum += loss.item()
        lr_scheduler.warmup_step()
        optim.step()

        # print('Epoch: [{0}][{1}/{2}]\tLoss {loss:.4f}\t'.format(epoch, i, len(data_loader), loss=loss.item()))
        error_result = utils.evaluate_error(gt_depth=targets.clone(), pred_depth=outputs.clone())
        for key in error_sum_train.keys():
            error_sum_train[key] += error_result[key]

        logger.write_log(step, error_result, "train")
        logger.add_scalar('train/learning_rate', optim.get_lr(), step)

        if i % 100 == 0:
            pred_img = outputs[0]  # [1,h,w]
            gt_img = targets[0]  # [1,h,w]
            out_img = utils.get_out_img(pred_img[0], gt_img[0])
            logger.write_image("train", out_img, epoch * len(data_loader) + i)
        RMSE = float(error_sum_train['RMSE'] / (i + 1))
        error_str = f'Epoch: {epoch}, RMSE={RMSE:.4f}, lr={optim.get_lr():.4f}'
        tbar.set_description(error_str)

    for key in error_sum_train.keys():
        error_sum_train[key] /= len(data_loader)
    return error_sum_train, float(error_sum_train['MAE'] / len(data_loader))


@paddle.no_grad()
def val_epoch(model, data_loader, loss_fn, epoch):
    error_sum = {
        'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'MAE': 0,
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    loss_sum = 0
    model.eval()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, data in tbar:
        step = epoch * len(data_loader) + i
        inputs = data['rgbd']
        targets = data['depth']
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_sum += loss.item()
        error_result = utils.evaluate_error(gt_depth=targets, pred_depth=outputs)

        pred_img = outputs[0]  # [1,h,w]
        gt_img = targets[0]  # [1,h,w]
        out_img = utils.get_out_img(pred_img[0], gt_img[0])
        logger.write_image("val", out_img, step)
        logger.write_log(step, error_result, "val")

        for key in error_sum.keys():
            error_sum[key] += error_result[key]
        RMSE = float(error_sum['RMSE'] / (i + 1))
        error_str = f'Epoch: {epoch}, loss={RMSE:.4f}'
        tbar.set_description(error_str)

    for key in error_sum.keys():
        error_sum[key] /= len(data_loader)
    return error_sum


def train(args):
    paddle.device.set_device(args.device)
    # load data
    train_set = NyuDepth(args.root, 'train', 'train.csv', args.n_sample)
    val_set = NyuDepth(args.root, 'test', 'val.csv', args.n_sample)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # define model
    model = CSPN(pretrained=args.resnet_pretrain)
    model_named_params = [p for _, p in model.named_parameters() if not p.stop_gradient]
    # define loss
    lose_fn = Wighted_L1_Loss()
    # define lr_scheduler
    lr_scheduler = optimizer.lr.ReduceOnPlateau(
        learning_rate=args.lr,
        mode='min',
        factor=0.1,
        patience=3,
        min_lr=0.000001,
        epsilon=1e-4,
        threshold=1e-2,
        threshold_mode='rel',
    )
    # add warmup
    lr_scheduler = WarmupLR(lr_scheduler, init_lr=0., num_warmup=100, warmup_strategy='cos')
    # define optimizer
    optim = optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=model_named_params,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        use_nesterov=args.nesterov,
        # dampening=args.dampening ###paddle not support
    )
    # load pretrain model
    start_epoch = 0
    best_error = {
        'MSE': float('inf'), 'RMSE': float('inf'), 'ABS_REL': float('inf'),
        'MAE': float('inf'),
        'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
        'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    }
    if args.pretrain and os.path.exists(args.pretrain):
        try:
            checkpoints = paddle.load(args.pretrain, return_numpy=True)
            model.set_state_dict(checkpoints['model'])
            optim.set_state_dict(checkpoints['optimizer'])
            lr_scheduler.set_state_dict(checkpoints['lr_scheduler'])
            start_epoch = checkpoints['epoch'] + 1
            best_error = checkpoints['val_metrics']
            print(f'load pretrain model from {args.pretrain}')
        except Exception as e:
            print(f"{e} load pretrain model failed")

    # train
    for epoch in range(start_epoch, args.epoch):
        train_metrics, train_mae = train_epoch(model, train_loader, lose_fn, optim, epoch, lr_scheduler)
        val_metrics = val_epoch(model, val_loader, lose_fn, epoch)

        lr_scheduler.step(train_mae)

        logger.add_scalar('train_epoch/learning_rate', optim.get_lr(), epoch)
        logger.write_log(epoch, train_metrics, "train_epoch")
        logger.write_log(epoch, val_metrics, "val_epoch")

        state = {
            'args': args,
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }

        if val_metrics['MSE'] < best_error['MSE']:
            best_error = val_metrics
            is_best = True
        else:
            is_best = False

        if epoch % args.interval == 0:
            paddle.save(state, os.path.join(args.save_path, f"checkpoint_{epoch}.pdparams"))
            print(f"save model at epoch {epoch}")
        if is_best:
            paddle.save(state, os.path.join(args.save_path, "model_best.pdparams"))
            print(f"save best model at epoch {epoch} with val_metrics\n{val_metrics}")


if __name__ == '__main__':
    args = parse_args()
    with utils.Logger(args.log_dir) as logger:
        train(args)
