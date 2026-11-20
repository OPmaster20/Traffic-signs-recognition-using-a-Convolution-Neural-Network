# infer_cli.py
import argparse
import os
import sys
from PIL import Image

from predictor import TrafficSignPredictor


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clean_path(raw_path: str) -> str:
    """Strip whitespace and surrounding quotes (e.g. from terminal drag-and-drop)."""
    return raw_path.strip().strip('"').strip("'")


def infer_single_image(predictor: TrafficSignPredictor, image_path: str, params: dict) -> bool:
    image_path = clean_path(image_path)
    if not image_path:
        return False

    if not os.path.isfile(image_path):
        print(f"[Error] File not found: {image_path}")
        return False

    ext = os.path.splitext(image_path)[1].lower()
    if ext and ext not in SUPPORTED_EXTS:
        print(f"[Warn] Unusual image extension '{ext}', attempting to open anyway...")

    try:
        with Image.open(image_path) as im:
            img = im.convert("RGB")
            width, height = img.size
    except Exception as exc:
        print(f"[Error] Failed to open image '{image_path}': {exc}")
        return False

    preds, elapsed_ms = predictor.predict_with_params(img, params)
    threshold = float(params["confidence_threshold"])
    passing = [p for p in preds if p["prob"] >= threshold]

    print("-" * 72)
    print(f"File       : {os.path.basename(image_path)} ({width}x{height} px)")
    print(f"Path       : {os.path.abspath(image_path)}")
    print(f"Device     : {predictor.device} | Precision: {params['weights']} | Latency: {elapsed_ms:.2f} ms")
    print("-" * 72)
    print(f"{'Rank':<6}{'Class Name':<46}{'Prob':>9}{'Logit':>11}")
    print("-" * 72)
    for rank, item in enumerate(preds, 1):
        mark = "*" if item["prob"] >= threshold else " "
        print(
            f"{rank:<2}{mark:<4}{item['label']:<46}"
            f"{item['prob'] * 100:>8.2f}%{item['logit']:>+11.2f}"
        )
    print("-" * 72)
    if passing:
        top = passing[0]
        print(f"Result     : {top['label']} ({top['prob'] * 100:.2f}% >= {threshold * 100:.0f}% threshold)")
    else:
        print(f"Result     : Below confidence threshold ({threshold * 100:.0f}%)")
    print()
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="GTSRB ResNet-18 Headless CLI Inference (based on predictor.py)"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to input image. Omit to enter interactive path-input mode.",
    )
    parser.add_argument(
        "--weights",
        default="gtsrb_resnet18.pth",
        help="Path to trained checkpoint file (default: gtsrb_resnet18.pth)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (default: 1.0)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision when running on CUDA (auto-falls back to FP32 on CPU)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable horizontal flip Test-Time Augmentation",
    )
    parser.add_argument(
        "--no-norm",
        action="store_true",
        help="Disable ImageNet RGB normalization",
    )
    parser.add_argument(
        "--no-imagefolder-order",
        action="store_true",
        help="Disable ImageFolder lexicographical label mapping (use when trained with integer ClassId 0..42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.weights):
        print(f"[Error] Checkpoint file not found: {args.weights}")
        sys.exit(1)

    print(f"Loading model from '{args.weights}'...")
    predictor = TrafficSignPredictor(
        weight_path=args.weights,
        use_imagefolder_order=not args.no_imagefolder_order,
    )
    predictor.warmup()
    print(f"Model ready on {predictor.device} (input_size={predictor.input_size}x{predictor.input_size}).\n")

    params = {
        "weights": "FP16" if args.fp16 else "FP32",
        "confidence_threshold": args.threshold,
        "softmax_temperature": args.temperature,
        "top_k": max(1, args.top_k),
        "normalize_imagenet": not args.no_norm,
        "tta_horizontal_flip": args.tta,
    }

    # 单次命令行传参模式：python infer_cli.py test.png
    if args.image_path is not None:
        ok = infer_single_image(predictor, args.image_path, params)
        sys.exit(0 if ok else 1)

    # 交互式手动输入路径模式：python infer_cli.py
    print("Interactive Mode — enter an image path (or 'q' to quit):")
    while True:
        try:
            user_input = input("Image path > ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        cleaned = clean_path(user_input)
        if cleaned.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if not cleaned:
            continue

        infer_single_image(predictor, cleaned, params)


if __name__ == "__main__":
    main()