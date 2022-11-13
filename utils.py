import os
import shutil

import paddle


@paddle.no_grad()
def max_of_two(y_over_z, z_over_y):
    return paddle.max((y_over_z, z_over_y))


@paddle.no_grad()
def evaluate_error(gt_depth, pred_depth):
    # for numerical stability
    depth_mask = gt_depth > 0.0001
    error = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0, 'MAE': 0,
             'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
             'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0,
             }
    _pred_depth = pred_depth[depth_mask]
    _gt_depth = gt_depth[depth_mask]
    n_valid_element = _gt_depth.shape[0]

    if n_valid_element > 0:
        diff_mat = paddle.abs(_gt_depth - _pred_depth)
        rel_mat = paddle.divide(diff_mat, _gt_depth)
        error['MSE'] = paddle.sum(paddle.pow(diff_mat, 2)) / n_valid_element
        error['RMSE'] = paddle.sqrt(error['MSE'])
        error['MAE'] = paddle.sum(diff_mat) / n_valid_element
        error['ABS_REL'] = paddle.sum(rel_mat) / n_valid_element
        y_over_z = paddle.divide(_gt_depth, _pred_depth)
        z_over_y = paddle.divide(_pred_depth, _gt_depth)
        max_ratio = max_of_two(y_over_z, z_over_y)
        error['DELTA1.02'] = paddle.sum(max_ratio < 1.02).numpy() / float(n_valid_element)
        error['DELTA1.05'] = paddle.sum(max_ratio < 1.05).numpy() / float(n_valid_element)
        error['DELTA1.10'] = paddle.sum(max_ratio < 1.10).numpy() / float(n_valid_element)
        error['DELTA1.25'] = paddle.sum(max_ratio < 1.25).numpy() / float(n_valid_element)
        error['DELTA1.25^2'] = paddle.sum(max_ratio < 1.25 ** 2).numpy() / float(n_valid_element)
        error['DELTA1.25^3'] = paddle.sum(max_ratio < 1.25 ** 3).numpy() / float(n_valid_element)
    return error


def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, f'checkpoint-{epoch}.pdparams')
    paddle.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pdparams')
        shutil.copyfile(checkpoint_filename, best_filename)
