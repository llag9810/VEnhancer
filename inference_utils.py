import os
import subprocess
import tempfile
from typing import Mapping

from PIL import Image
import cv2
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transforms_F

from video_to_video.utils.logger import get_logger

logger = get_logger()


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    video = video * 255.0
    images = rearrange(video, "b c f h w -> b f h w c")[0]
    return images


def preprocess(input_frames):
    out_frame_list = []
    for pointer in range(len(input_frames)):
        frame = input_frames[pointer]
        frame = frame[:, :, ::-1]
        frame = Image.fromarray(frame.astype("uint8")).convert("RGB")
        frame = transforms_F.to_tensor(frame)
        out_frame_list.append(frame)
    out_frames = torch.stack(out_frame_list, dim=0)
    out_frames.clamp_(0, 1)
    mean = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    std = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    out_frames.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
    return out_frames


def adjust_resolution(h, w):
    if w < h:
        target_w = 1080
        target_h = int(h * (1080 / w))
    else:
        target_h = 1080
        target_w = int(w * (1080 / h))
    return target_h, target_w


def make_mask_cond(in_f_num, interp_f_num):
    mask_cond = []
    interp_cond = [-1 for _ in range(interp_f_num)]
    for i in range(in_f_num):
        mask_cond.append(i)
        if i != in_f_num - 1:
            mask_cond += interp_cond
    return mask_cond


def load_prompt_list(file_path):
    files = []
    with open(file_path, "r") as fin:
        for line in fin:
            path = line.strip()
            if path:
                files.append(path)
    return files


def load_video(vid_path):
    capture = cv2.VideoCapture(vid_path)
    _fps = capture.get(cv2.CAP_PROP_FPS)
    _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    pointer = 0
    frame_list = []
    stride = 1
    while len(frame_list) < _total_frame_num:
        ret, frame = capture.read()
        pointer += 1
        if (not ret) or (frame is None):
            break
        if pointer >= _total_frame_num + 1:
            break
        if pointer % stride == 0:
            frame_list.append(frame)
    capture.release()
    return frame_list, _fps


def save_video(video, save_dir, file_name, fps=16.0):
    output_path = os.path.join(save_dir, file_name)
    images = [(img.numpy()).astype("uint8") for img in video]
    temp_dir = tempfile.mkdtemp()
    for fid, frame in enumerate(images):
        tpth = os.path.join(temp_dir, "%06d.png" % (fid + 1))
        cv2.imwrite(tpth, frame[:, :, ::-1])
    tmp_path = os.path.join(save_dir, "tmp.mp4")
    cmd = f"ffmpeg -y -f image2 -framerate {fps} -i {temp_dir}/%06d.png \
     -vcodec libx264 -crf 17 -pix_fmt yuv420p {tmp_path}"
    status, output = subprocess.getstatusoutput(cmd)
    if status != 0:
        logger.error(f"Save Video Error with {output}")
    os.system(f"rm -rf {temp_dir}")
    os.rename(tmp_path, output_path)


def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        return type(data)({k: collate_fn(v, device) if k != "img_metas" else v for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    else:
        raise ValueError(f"Unsupported data type {type(data)}")
