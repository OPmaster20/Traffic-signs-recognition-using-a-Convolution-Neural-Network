# predictor.py
import io
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.ResNet18 import ResNet, BasicBlock
from labels import class_labels


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrafficSignPredictor:
    """
    GTSRB ResNet-18 inference wrapper.

    Two usage patterns:
      1. Simple interface: predict(image, top_k=5) / predict_bytes(bytes, top_k=5)
      2. NiceGUI parameterized interface: predict_with_params(image, params) -> (predictions, elapsed_ms)
    """

    def __init__(self, weight_path="gtsrb_resnet18.pth", num_classes=43, use_imagefolder_order=True):
        self.weight_path = weight_path
        self.num_classes = num_classes
        self.use_imagefolder_order = use_imagefolder_order
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load checkpoint (compatible with PyTorch 2.6+ default weights_only=True and older versions) ----
        try:
            ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(weight_path, map_location=self.device)

        # Compatible with two save formats: raw state_dict / {"model_state_dict": ...}
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            self.input_size = int(ckpt.get("input_size", 64))
            self.num_classes = int(ckpt.get("num_classes", num_classes))
            self.normalize_mean = ckpt.get("normalize_mean", IMAGENET_MEAN)
            self.normalize_std = ckpt.get("normalize_std", IMAGENET_STD)
        else:
            sd = ckpt
            ckpt = {}
            self.input_size = 64
            self.normalize_mean = IMAGENET_MEAN
            self.normalize_std = IMAGENET_STD

        # Build mappings: "model output index -> true integer ClassId" and "model output index -> class name"
        self.idx_to_class_id, self.idx_to_name = self._build_label_maps(ckpt)
        self.class_labels = dict(class_labels)

        # ---- Primary model (FP32 resident) ----
        self._models = {}
        self.model = self._build_model(sd, torch.float32)

    # ---------------- Internal utilities ----------------
    def _build_label_maps(self, ckpt: dict):
        """
        Unified handling of the mapping between ImageFolder lexicographical order
        ('0', '1', '10', ..., '2', ...) and the true ClassId, linking to the actual
        traffic sign names in labels.py.
        """
        # 1. If checkpoint contains numeric folder mapping (class_to_idx / class_labels)
        for key in ("class_to_idx", "class_labels"):
            raw = ckpt.get(key)
            if isinstance(raw, dict) and len(raw) == self.num_classes:
                first_k = next(iter(raw.keys()))
                first_v = next(iter(raw.values()))
                # Format: {'0': 0, '1': 1, '10': 2, ...}
                if isinstance(first_k, str) and first_k.isdigit():
                    idx_to_cid = {int(v): int(k) for k, v in raw.items()}
                    idx_to_name = {
                        idx: class_labels.get(cid, f"class_{cid}")
                        for idx, cid in idx_to_cid.items()
                    }
                    return idx_to_cid, idx_to_name
                # Format: {0: '0', 1: '1', 2: '10', ...}
                if isinstance(first_v, str) and first_v.isdigit():
                    idx_to_cid = {int(k): int(v) for k, v in raw.items()}
                    idx_to_name = {
                        idx: class_labels.get(cid, f"class_{cid}")
                        for idx, cid in idx_to_cid.items()
                    }
                    return idx_to_cid, idx_to_name
                # Format: {"Speed limit (20km/h)": 0, ...}
                if isinstance(first_k, str):
                    idx_to_name = {int(v): k for k, v in raw.items()}
                    idx_to_cid = {idx: idx for idx in idx_to_name}
                    return idx_to_cid, idx_to_name

        # 2. Rebuild mapping based on ImageFolder string lexicographical order ('0', '1', '10', ..., '2', ...)
        if self.use_imagefolder_order:
            folder_names = sorted(str(i) for i in range(self.num_classes))
            idx_to_cid = {idx: int(name) for idx, name in enumerate(folder_names)}
            idx_to_name = {
                idx: class_labels.get(cid, f"class_{cid}")
                for idx, cid in idx_to_cid.items()
            }
            return idx_to_cid, idx_to_name

        # 3. Already aligned to true integer ClassId (0~42) order during training
        idx_to_cid = {i: i for i in range(self.num_classes)}
        idx_to_name = {i: class_labels.get(i, f"class_{i}") for i in range(self.num_classes)}
        return idx_to_cid, idx_to_name

    def _build_model(self, state_dict, dtype: torch.dtype):
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.num_classes)
        model.load_state_dict(state_dict)
        model.to(self.device).to(dtype)
        model.eval()
        return model

    def _get_model(self, dtype: torch.dtype):
        """Lazily load/cache model by dtype (FP32 is built in __init__)."""
        if dtype == torch.float32:
            return self.model
        key = str(dtype)
        if key not in self._models:
            self._models[key] = self._build_model(self.model.state_dict(), dtype)
        return self._models[key]

    def warmup(self):
        """Warm up the FP32 model at startup, initialized with training input size (self.input_size x self.input_size)."""
        _ = self._get_model(torch.float32)
        dummy = torch.zeros(
            1, 3, self.input_size, self.input_size, device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            self.model(dummy)

    def _build_transform(self, normalize: bool = True):
        """Inference preprocessing: consistent with training Resize((64, 64)) + ToTensor + Normalize, no crop or random augmentation."""
        ops = [
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
        ]
        if normalize:
            ops.append(transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std))
        return transforms.Compose(ops)

    def _default_transform(self):
        return self._build_transform(normalize=True)

    # ---------------- Simple interface (backward compatible) ----------------
    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5):
        """Original interface: returns dict(label_id, label_name, confidence, top_k)."""
        img = image.convert("RGB")
        tensor = self._default_transform()(img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        conf, raw_idx = torch.max(probs, dim=0)
        raw_idx = int(raw_idx.item())
        pred_id = self.idx_to_class_id.get(raw_idx, raw_idx)
        pred_name = self.idx_to_name.get(raw_idx, "Unknown")

        topk_conf, topk_idx = torch.topk(probs, k=min(top_k, probs.numel()))
        topk_list = [
            (
                self.idx_to_class_id.get(int(i.item()), int(i.item())),
                self.idx_to_name.get(int(i.item()), "Unknown"),
                float(c.item()),
            )
            for c, i in zip(topk_conf, topk_idx)
        ]

        return {
            "label_id": pred_id,
            "label_name": pred_name,
            "confidence": float(conf.item()),
            "top_k": topk_list,
        }

    @torch.no_grad()
    def predict_bytes(self, file_bytes: bytes, top_k: int = 5):
        return self.predict(Image.open(io.BytesIO(file_bytes)), top_k=top_k)

    # ---------------- NiceGUI parameterized interface ----------------
    @torch.no_grad()
    def predict_with_params(self, image: Image.Image, params: dict):
        """
        Called by NiceGUI run_local_resnet18_hook.

        Returns: (predictions, elapsed_ms)
          predictions = [{"label": str, "prob": float, "logit": float}, ...]
        """
        want_fp16 = "FP16" in str(params.get("weights", "FP32"))
        # Conv2d on CPU does not support FP16 (Half), automatically fallback to FP32 to avoid runtime errors
        dtype = torch.float16 if (want_fp16 and self.device.type == "cuda") else torch.float32
        model = self._get_model(dtype)

        tfm = self._build_transform(
            normalize=bool(params.get("normalize_imagenet", True)),
        )
        x = tfm(image.convert("RGB")).unsqueeze(0).to(self.device, dtype=dtype)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if params.get("tta_horizontal_flip", False):
            logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
        else:
            logits = model(x)
        logits = logits.float().squeeze(0)

        tau = max(float(params.get("softmax_temperature", 1.0)), 1e-6)
        probs = F.softmax(logits / tau, dim=0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        k = max(1, min(int(params.get("top_k", 5)), probs.numel()))
        top_probs, top_idx = torch.topk(probs, k)

        predictions = [
            {
                "label": self.idx_to_name.get(int(idx), f"class_{idx}"),
                "prob": float(prob),
                "logit": float(logits[int(idx)].item()),
            }
            for prob, idx in zip(top_probs.tolist(), top_idx.tolist())
        ]
        return predictions, elapsed_ms