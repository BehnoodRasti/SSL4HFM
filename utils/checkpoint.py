import shutil
import torch


def load_checkpoint_eval(checkpoint_path, net):
    print("Loading", checkpoint_path)
    checkpoint = torch.load(checkpoint_path,weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()


def load_checkpoint_train(checkpoint_path, net, optimizer, device):
    print("Loading", checkpoint_path, "to continue training.")
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=True)
    last_epoch = checkpoint["epoch"] + 1
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return last_epoch


def save_checkpoint(state, is_best, save_dir="./results/", filename="last.pth.tar"):
    save_path_last = f"{save_dir}{filename}"
    torch.save(state, save_path_last)
    if is_best:
        save_path_best = f"{save_dir}best.pth.tar"
        shutil.copyfile(save_path_last, save_path_best)


def strip_checkpoint(checkpoint_path, save_dir="./results/", filename="final.pth.tar"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu",weights_only=True)
    state_dict = {
        "state_dict": checkpoint["state_dict"]
    }
    torch.save(state_dict, f"{save_dir}{filename}")

# def get_pretrained_weights(model, path, device):
#     print('load weights from {}'.format(path))
#     pretrained_state_dict = torch.load('weights/' + path, map_location=torch.device(device))
#     model_state_dict = model.state_dict()  # Your model's state dictionary
#     pre_filter = {k: v for k, v in pretrained_state_dict.items() if k.split('.')[0] != 'Projector'}
#     print(len(pretrained_state_dict.keys()))
#     print(len(pre_filter.keys()))
#     filtered_state_dict = {k: v for k, v in pre_filter.items() if k in model_state_dict}
#     print(len(filtered_state_dict.keys()))
def get_pretrained_weights(model, path, device):
    print('load weights from {}'.format(path))
    pretrained_state_dict = torch.load(path, map_location=torch.device(device))
    model_state_dict = model.state_dict()  # Your model's state dictionary
    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

    # Update the model's state dictionary with the filtered one
    model_state_dict.update(filtered_state_dict)

    # Load the updated state dictionary into your model
    model.load_state_dict(model_state_dict)