# datasets/dataset.py
import os
import random
import PIL
import torch
from torchvision import transforms
import json
from PIL import Image
import numpy as np

from medsyn.tasks import (
    CutPastePatchBlender,
    SmoothIntensityChangeTask,
    GaussIntensityChangeTask,
    SinkDeformationTask,
    SourceDeformationTask,
    IdentityTask,
)


# =========================
# A 模式：全幅扩散样本任务
# =========================
class DiffusionFullTask:
    """
    从 generated_image_dir 下读取全幅扩散生成的图像与同名掩码，直接作为异常样本返回。

    目录结构要求：
      generated_image_dir/
        images/
          xxx_0001.png
          ...
        masks/
          xxx_0001.png  # 与 images 同名
          ...

    返回：
      image_np: HxWx3, uint8
      mask_np:  HxW, float32 in {0,1}
    """

    def __init__(self, gen_dir, image_size):
        self.gen_img_dir = os.path.join(gen_dir, "images")
        self.gen_msk_dir = os.path.join(gen_dir, "masks")
        assert os.path.isdir(self.gen_img_dir), f"Not found: {self.gen_img_dir}"
        assert os.path.isdir(self.gen_msk_dir), f"Not found: {self.gen_msk_dir}"

        # 仅保留常见图片文件
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.names = sorted(
            [n for n in os.listdir(self.gen_img_dir) if os.path.splitext(n.lower())[1] in valid_ext]
        )
        assert len(self.names) > 0, f"No files in {self.gen_img_dir}"
        self.image_size = image_size

    def __call__(self, _base_img_ignored):
        # 随机抽一对
        name = random.choice(self.names)
        img_path = os.path.join(self.gen_img_dir, name)
        msk_path = os.path.join(self.gen_msk_dir, name)

        # 读图（RGB）与掩码（L）
        img = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), PIL.Image.Resampling.BILINEAR
        )
        if not os.path.exists(msk_path):
            raise FileNotFoundError(f"Mask not found for generated image: {msk_path}")
        msk = Image.open(msk_path).convert("L").resize(
            (self.image_size, self.image_size), PIL.Image.Resampling.NEAREST
        )

        img_np = np.array(img).astype(np.uint8)  # HxWx3 uint8
        msk_np = (np.array(msk).astype(np.float32) / 255.0)  # HxW float32
        msk_np = (msk_np > 0.5).astype(np.float32)  # binarize to {0,1}

        return img_np, msk_np


class TrainDataset(torch.utils.data.Dataset):
    """
    训练集：
      - 从 samples/train.json 读取基图
      - 根据 anomaly_tasks 采样一种任务生成 (image, mask)
      - 如果配置了 DiffusionTask + use_generated_images=true，则可从 generated_image_dir 读取扩散对
    """

    def __init__(
        self,
        args,
        source,
        preprocess,
        k_shot=-1,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.k_shot = k_shot
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()
        self.augs, self.augs_pro = self.load_anomaly_syn()

        prob_sum = float(sum(self.augs_pro))
        assert abs(prob_sum - 1.0) < 1e-6, f"anomaly_tasks probs must sum to 1.0, got {prob_sum}"

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, "images", info["filename"])
        base_gray = self.read_image(image_path)  # HxW, uint8

        # 随机选择一种任务
        choice_aug = np.random.choice(
            a=[aug for aug in self.augs],
            p=[pro for pro in self.augs_pro],
            size=(1,),
            replace=False,
        )[0]

        # 任务签名：fn(image_np_gray) -> (image_np_rgb_uint8, mask_np_float32[0/1])
        image_np_rgb, mask_np = choice_aug(base_gray)

        image_pil = Image.fromarray(image_np_rgb.astype(np.uint8)).convert("RGB")
        image_tensor = self.transform_img(image_pil)

        mask_tensor = torch.from_numpy(mask_np.astype(np.float32))  # HxW

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def read_image(self, path):
        image = Image.open(path).resize(
            (self.args.image_size, self.args.image_size),
            PIL.Image.Resampling.BILINEAR,
        ).convert("L")
        return np.array(image).astype(np.uint8)  # HxW

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source, "samples", "train.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        if self.k_shot != -1 and self.k_shot < len(data_to_iterate):
            data_to_iterate = random.sample(data_to_iterate, self.k_shot)
        return data_to_iterate

    def load_anomaly_syn(self):
        tasks = []
        task_probability = []

        # 是否可以注册扩散任务（A 模式）
        can_use_diffusion = bool(getattr(self.args, "use_generated_images", False)) and \
            hasattr(self.args, "generated_image_dir") and \
            os.path.isdir(getattr(self.args, "generated_image_dir"))

        for task_name in self.args.anomaly_tasks.keys():
            if task_name == "CutpasteTask":
                support_images = [
                    self.read_image(os.path.join(self.source, "images", data["filename"]))
                    for data in self.data_to_iterate
                ]
                task = CutPastePatchBlender(support_images)
            elif task_name == "SmoothIntensityTask":
                task = SmoothIntensityChangeTask(30.0)
            elif task_name == "GaussIntensityChangeTask":
                task = GaussIntensityChangeTask()
            elif task_name == "SinkTask":
                task = SinkDeformationTask()
            elif task_name == "SourceTask":
                task = SourceDeformationTask()
            elif task_name == "IdentityTask":
                task = IdentityTask()
            elif task_name == "DiffusionTask":
                # A 模式（全幅扩散样本）
                assert can_use_diffusion, (
                    "Config 启用了 DiffusionTask，但未提供有效的 generated_image_dir "
                    "或 use_generated_images=false"
                )
                task = DiffusionFullTask(
                    gen_dir=self.args.generated_image_dir,
                    image_size=self.args.image_size,
                )
            else:
                raise NotImplementedError(
                    "task must be in [CutpasteTask, SmoothIntensityTask, "
                    "GaussIntensityChangeTask, SinkTask, SourceTask, IdentityTask, DiffusionTask]"
                )

            tasks.append(task)
            task_probability.append(float(self.args.anomaly_tasks[task_name]))

        return tasks, task_probability


class ChexpertTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        use_generated=False,
        generated_dir=None,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.use_generated = use_generated
        self.generated_dir = generated_dir
        self.data_to_iterate = self.get_image_data()

        if self.use_generated:
            assert os.path.exists(self.generated_dir), "Generated image directory does not exist."
            # 如需严格一一对应，可启用下面这行：
            # assert len(os.listdir(self.generated_dir)) == len(self.data_to_iterate), "Generated images and test.json not aligned!"

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        if self.use_generated:
            image_path = os.path.join(self.generated_dir, f"{idx:05d}.jpg")
        else:
            image_path = os.path.join(self.source, "images", info["filename"])

        image = Image.open(image_path).convert("RGB").resize(
            (self.args.image_size, self.args.image_size), PIL.Image.Resampling.BILINEAR
        )
        mask = np.zeros((self.args.image_size, self.args.image_size), dtype=np.float32)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "classname": info["clsname"],
            "is_anomaly": info["label"],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        with open(os.path.join(self.source, "samples", "test.json"), "r") as f_r:
            return [json.loads(line) for line in f_r]


class BrainMRITestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, "images", info["filename"])
        image = Image.open(image_path).convert("RGB").resize(
            (self.args.image_size, self.args.image_size), PIL.Image.Resampling.BILINEAR
        )
        mask = np.zeros((self.args.image_size, self.args.image_size), dtype=np.float32)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "classname": info["clsname"],
            "is_anomaly": info["label"],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source, "samples", "test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate


class BusiTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, "images", info["filename"])
        image = Image.open(image_path).convert("RGB").resize(
            (self.args.image_size, self.args.image_size), PIL.Image.Resampling.BILINEAR
        )

        if info.get("mask", None):
            mask_path = os.path.join(self.source, "images", info["mask"])
            mask_img = Image.open(mask_path).convert("L").resize(
                (self.args.image_size, self.args.image_size), PIL.Image.Resampling.NEAREST
            )
            mask = np.array(mask_img).astype(np.float32) / 255.0
            mask[mask != 0.0] = 1.0
        else:
            mask = np.zeros((self.args.image_size, self.args.image_size), dtype=np.float32)

        image = self.transform_img(image)
        mask = torch.from_numpy(mask.astype(np.float32))

        return {
            "image": image,
            "mask": mask,
            "classname": info["clsname"],
            "is_anomaly": info["label"],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source, "samples", "test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate