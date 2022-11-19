import paddle

from model.cspn_model import resnet50
import torch


def convert():
    paddle.device.set_device('cpu')

    torch_path = "weights/pytorch/best_model.pth"
    paddle_path = "weights/model_pytorch.pdparams"

    torch_ckpt = torch.load(torch_path, map_location='cpu')
    pd_model = resnet50()

    pd_state_dict = pd_model.state_dict()
    pd_list = []

    for key in pd_state_dict.keys():
        pd_list.append((pd_state_dict[key].numpy(), key))

    torch_list = []
    for i, key in enumerate(torch_ckpt.keys()):
        if "num_batches_tracked" in key:
            continue
        torch_list.append((torch_ckpt[key].numpy(), key))

    for i in range(len(pd_list)):
        pd_weight, pd_key = pd_list[i]
        torch_weight, torch_key = torch_list[i]
        assert pd_weight.shape == torch_weight.shape, \
            f"shape not match: {pd_weight.shape} vs {torch_weight.shape}, {pd_key} vs {torch_key}"
        pd_state_dict[pd_key] = torch_ckpt[torch_key].detach().cpu().numpy()

    state = {
        'model': pd_state_dict,
    }
    print("convert success")
    paddle.save(state, paddle_path)


if __name__ == '__main__':
    convert()
