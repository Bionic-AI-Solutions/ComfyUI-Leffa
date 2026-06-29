# ComfyUI-Leffa

Full-pipeline **virtual try-on** and **pose transfer** ComfyUI custom node, built on
Meta's [Leffa](https://github.com/franciszzj/Leffa) (MIT). Powers the
`mcp.baisoln.com/leffa/*` API.

Unlike [StartHua/Comfyui_leffa](https://github.com/StartHua/Comfyui_leffa) — which
only runs the diffusion step and requires the caller to pre-compute a DensePose
image and an agnostic mask — these nodes run the **entire** pipeline inside
ComfyUI (human parsing → openpose → agnostic mask → densepose → Leffa diffusion),
mirroring upstream `LeffaPredictor.leffa_predict`, and they support pose transfer.

## Nodes

| Node | Purpose |
|------|---------|
| **Leffa Loader** (`LeffaLoader`) | Loads one task model: `viton_hd`, `viton_dc`, or `pose_transfer`. |
| **Leffa Virtual Try-On** (`LeffaTryOn`) | person + garment IMAGE → tried-on image. `garment_type` ∈ {upper_body, lower_body, dresses}. |
| **Leffa Pose Transfer** (`LeffaPoseTransfer`) | person + target_pose IMAGE → reposed image. |

A typical graph: `LoadImage`(person) + `LoadImage`(garment) → `LeffaLoader` →
`LeffaTryOn` → `SaveImage`.

## Models

Weights are expected under `ComfyUI/models/Leffa/` (the whole
[`franciszzj/Leffa`](https://huggingface.co/franciszzj/Leffa) HF repo, ~34 GB):

```
hf download franciszzj/Leffa --local-dir ComfyUI/models/Leffa
```

Subtree used: `virtual_tryon.pth`, `virtual_tryon_dc.pth`, `pose_transfer.pth`,
`stable-diffusion-inpainting/` (+ SDXL inpaint configs),
`densepose/`, `humanparsing/`, `schp/`, `openpose/`.

In the cluster these are bind-mounted from `/mnt/ai-models/comfyui/models/Leffa`.

## Notes

- The vendored `detectron2`/`densepose` route ROIAlign/NMS through
  `torchvision.ops`, so **no detectron2 compilation is required** for the
  `densepose_rcnn_R_50_FPN_s1x` model used here.
- Human parsing runs on CPU via onnxruntime; openpose/densepose/diffusion on GPU.
- Only one diffusion model is kept resident at a time (VRAM bound for the RTX
  5090); switching task evicts the previous one.
- `torch`/`torchvision`/`transformers` come from the ComfyUI env and must not be
  upgraded; see `requirements.txt`.

## License

MIT (inherits franciszzj/Leffa). Vendored upstream source under `leffa/`,
`leffa_utils/`, `preprocess/`, `detectron2/`, `densepose/`, `SCHP/`.
