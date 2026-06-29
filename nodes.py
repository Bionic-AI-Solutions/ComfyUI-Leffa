# -*- coding: utf-8 -*-
"""
ComfyUI-Leffa: full-pipeline virtual try-on + pose-transfer nodes.

Unlike the upstream StartHua/Comfyui_leffa node (which only runs the diffusion
step and requires the caller to supply a pre-computed DensePose image + agnostic
mask), these nodes run the *entire* Leffa pipeline inside ComfyUI:

    person/garment image -> human-parsing (onnx, CPU) -> openpose (gpu)
        -> agnostic mask (AutoMasker) -> densepose (detectron2) -> Leffa diffusion

mirroring franciszzj/Leffa's `LeffaPredictor.leffa_predict`.

Heavy models are cached at module scope so they survive between prompts. To keep
peak VRAM under the RTX 5090's budget, only one diffusion model (try-on HD,
try-on DC, or pose-transfer) is resident at a time; loading a different task
evicts the previous diffusion model. The small preprocessing predictors
(parsing/openpose/densepose/automask) are loaded once and kept.
"""

import gc
import os

import numpy as np
import torch
from PIL import Image

import folder_paths  # provided by ComfyUI at runtime

from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

from .lib.ximg import tensor2pil, pil2tensor


def _ckpts_root():
    """Leffa weights live under ComfyUI/models/Leffa (bind-mounted from
    /mnt/ai-models/comfyui/models/Leffa)."""
    return os.path.join(folder_paths.models_dir, "Leffa")


# Map a task to (sd_config_subdir, checkpoint_filename).
_TASK_SPEC = {
    "viton_hd": ("stable-diffusion-inpainting", "virtual_tryon.pth"),
    "viton_dc": ("stable-diffusion-inpainting", "virtual_tryon_dc.pth"),
    "pose_transfer": ("stable-diffusion-xl-1.0-inpainting-0.1", "pose_transfer.pth"),
}

# Module-level caches (persist across prompts within the ComfyUI process).
_PRE_CACHE = {}   # preprocessing predictors, loaded once
_DIFF_CACHE = {"task": None, "inference": None}  # one diffusion model at a time


def _get_preprocessors():
    if _PRE_CACHE:
        return _PRE_CACHE
    root = _ckpts_root()
    _PRE_CACHE["mask_predictor"] = AutoMasker(
        densepose_path=os.path.join(root, "densepose"),
        schp_path=os.path.join(root, "schp"),
    )
    _PRE_CACHE["densepose_predictor"] = DensePosePredictor(
        config_path=os.path.join(root, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml"),
        weights_path=os.path.join(root, "densepose", "model_final_162be9.pkl"),
    )
    _PRE_CACHE["parsing"] = Parsing(
        atr_path=os.path.join(root, "humanparsing", "parsing_atr.onnx"),
        lip_path=os.path.join(root, "humanparsing", "parsing_lip.onnx"),
    )
    _PRE_CACHE["openpose"] = OpenPose(
        body_model_path=os.path.join(root, "openpose", "body_pose_model.pth"),
    )
    return _PRE_CACHE


def _get_inference(task):
    """Return a LeffaInference for the task, evicting any other resident
    diffusion model to bound VRAM."""
    if _DIFF_CACHE["task"] == task and _DIFF_CACHE["inference"] is not None:
        return _DIFF_CACHE["inference"]
    # Evict previous diffusion model.
    _DIFF_CACHE["inference"] = None
    _DIFF_CACHE["task"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sd_subdir, ckpt_name = _TASK_SPEC[task]
    root = _ckpts_root()
    model = LeffaModel(
        pretrained_model_name_or_path=os.path.join(root, sd_subdir),
        pretrained_model=os.path.join(root, ckpt_name),
        dtype="float16",
    )
    inference = LeffaInference(model=model)
    _DIFF_CACHE["task"] = task
    _DIFF_CACHE["inference"] = inference
    return inference


class LeffaPipe:
    """Opaque handle passed between the loader and run nodes."""

    def __init__(self, task):
        self.task = task


class LeffaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (["viton_hd", "viton_dc", "pose_transfer"],),
            }
        }

    RETURN_TYPES = ("LEFFA_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "Leffa"

    def load(self, task):
        # Warm both the preprocessing models and the task diffusion model so the
        # run node is pure compute.
        _get_preprocessors()
        _get_inference(task)
        return (LeffaPipe(task),)


class LeffaTryOn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("LEFFA_PIPE",),
                "person": ("IMAGE",),
                "garment": ("IMAGE",),
                "garment_type": (["upper_body", "lower_body", "dresses"],),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 50.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "ref_acceleration": ("BOOLEAN", {"default": False}),
                "repaint": ("BOOLEAN", {"default": False}),
                "restore_size": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "densepose", "mask")
    FUNCTION = "run"
    CATEGORY = "Leffa"

    def run(self, pipe, person, garment, garment_type, steps, cfg, seed,
            ref_acceleration=False, repaint=False, restore_size=True):
        if pipe.task not in ("viton_hd", "viton_dc"):
            raise ValueError(
                f"LeffaTryOn requires a viton_hd/viton_dc pipe, got task={pipe.task}"
            )
        vt_model_type = "viton_hd" if pipe.task == "viton_hd" else "dress_code"

        pre = _get_preprocessors()
        inference = _get_inference(pipe.task)

        src_image = tensor2pil(person).convert("RGB")
        original_size = src_image.size
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(tensor2pil(garment).convert("RGB"), 768, 1024)
        src_array = np.array(src_image)

        # Agnostic mask from human-parsing + openpose keypoints.
        model_parse, _ = pre["parsing"](src_image.resize((384, 512)))
        keypoints = pre["openpose"](src_image.resize((384, 512)))
        if vt_model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, garment_type)
        else:
            mask = get_agnostic_mask_dc(model_parse, keypoints, garment_type)
        mask = mask.resize((768, 1024))

        # DensePose conditioning.
        if vt_model_type == "viton_hd":
            seg = pre["densepose_predictor"].predict_seg(src_array)[:, :, ::-1]
            densepose = Image.fromarray(seg)
        else:
            iuv = pre["densepose_predictor"].predict_iuv(src_array)
            seg = np.concatenate([iuv[:, :, 0:1]] * 3, axis=-1)
            densepose = Image.fromarray(seg)

        transform = LeffaTransform()
        data = transform({
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        })
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=steps,
            guidance_scale=cfg,
            seed=seed,
            repaint=repaint,
        )
        gen_image = output["generated_image"][0]
        if restore_size:
            gen_image = gen_image.resize(original_size, Image.LANCZOS)

        mask_t = pil2tensor(mask.convert("L"))
        return (pil2tensor(gen_image), pil2tensor(densepose), mask_t)


class LeffaPoseTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("LEFFA_PIPE",),
                "person": ("IMAGE",),
                "target_pose": ("IMAGE",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 50.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "ref_acceleration": ("BOOLEAN", {"default": False}),
                "restore_size": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "densepose")
    FUNCTION = "run"
    CATEGORY = "Leffa"

    def run(self, pipe, person, target_pose, steps, cfg, seed,
            ref_acceleration=False, restore_size=True):
        if pipe.task != "pose_transfer":
            raise ValueError(
                f"LeffaPoseTransfer requires a pose_transfer pipe, got task={pipe.task}"
            )
        pre = _get_preprocessors()
        inference = _get_inference(pipe.task)

        # src = target pose person; ref = person whose appearance is preserved.
        ref_image = resize_and_center(tensor2pil(person).convert("RGB"), 768, 1024)
        src_image = tensor2pil(target_pose).convert("RGB")
        original_size = src_image.size
        src_image = resize_and_center(src_image, 768, 1024)
        src_array = np.array(src_image)

        mask = Image.fromarray(np.ones_like(src_array) * 255)
        iuv = pre["densepose_predictor"].predict_iuv(src_array)[:, :, ::-1]
        densepose = Image.fromarray(iuv)

        transform = LeffaTransform()
        data = transform({
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        })
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=steps,
            guidance_scale=cfg,
            seed=seed,
        )
        gen_image = output["generated_image"][0]
        if restore_size:
            gen_image = gen_image.resize(original_size, Image.LANCZOS)
        return (pil2tensor(gen_image), pil2tensor(densepose))


NODE_CLASS_MAPPINGS = {
    "LeffaLoader": LeffaLoader,
    "LeffaTryOn": LeffaTryOn,
    "LeffaPoseTransfer": LeffaPoseTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LeffaLoader": "Leffa Loader",
    "LeffaTryOn": "Leffa Virtual Try-On",
    "LeffaPoseTransfer": "Leffa Pose Transfer",
}
