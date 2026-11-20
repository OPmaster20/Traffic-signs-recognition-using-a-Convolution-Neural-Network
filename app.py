import asyncio
import base64
import os
from io import BytesIO

import torch
from PIL import Image
from nicegui import events, ui
from torchvision import transforms

import predictor
from model.ResNet18 import ResNet, BasicBlock

predictor = predictor.TrafficSignPredictor(weight_path="gtsrb_resnet18.pth")

try:
    predictor.warmup()
except FileNotFoundError:
    print("[warn] gtsrb_resnet18.pth not found; training first is required.")


def pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def run_local_resnet18_hook(img: Image.Image, params: dict):
    return predictor.predict_with_params(img, params)


DEFAULT_PARAMS = {
    "weights": "IMAGENET1K_V1 (FP32)",
    "target_layer": "layer4",
    "resize_shorter_edge": 256,
    "confidence_threshold": 0.25,
    "softmax_temperature": 1.0,
    "top_k": 5,
    "normalize_imagenet": True,
    "tta_horizontal_flip": False,
}


@ui.page("/")
def index():
    ui.query("body").style("background-color: #f1f5f9; margin: 0;")

    state = {
        "image": None,
        "filename": "",
        "width": 0,
        "height": 0,
        "size_kb": 0.0,
        "running": False,
    }

    def render_empty_results(message: str = "Waiting for input image..."):
        results_container.clear()
        with results_container:
            with ui.column().classes("w-full items-center justify-center py-10 text-center gap-2"):
                ui.icon("analytics", size="40px").classes("text-slate-300")
                ui.label(message).classes("text-sm text-slate-400 font-medium")
                ui.label("Upload an image and click Start Recognition to inspect Top-K predictions.").classes(
                    "text-xs text-slate-400 max-w-xs"
                )

    def render_predictions(preds: list, elapsed_ms: float, params: dict):
        threshold = params["confidence_threshold"]
        passing = [p for p in preds if p["prob"] >= threshold]

        results_container.clear()
        with results_container:
            # 1. Primary Detection Summary Card
            if passing:
                top = passing[0]
                with ui.card().classes("w-full bg-emerald-50 border border-emerald-200 shadow-none p-3"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.row().classes("items-center gap-1.5"):
                            ui.icon("check_circle", size="18px").classes("text-emerald-600")
                            ui.label("Detected Class").classes("text-xs font-semibold uppercase tracking-wider text-emerald-700")
                        ui.badge(f"{elapsed_ms:.1f} ms", color="green").props("outline")
                    ui.label(top["label"]).classes("text-lg font-bold text-slate-900 mt-1 leading-tight")
                    with ui.row().classes("w-full items-center justify-between mt-1"):
                        ui.label(f"Confidence: {top['prob'] * 100:.2f}%").classes("text-xs font-semibold text-emerald-700")
                        ui.label(f"Logit: {top['logit']:+.2f}").classes("text-xs font-mono text-slate-500")
            else:
                with ui.card().classes("w-full bg-amber-50 border border-amber-200 shadow-none p-3"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.row().classes("items-center gap-1.5"):
                            ui.icon("warning_amber", size="18px").classes("text-amber-600")
                            ui.label("Below Threshold").classes("text-xs font-semibold uppercase tracking-wider text-amber-700")
                        ui.badge(f"{elapsed_ms:.1f} ms", color="amber").props("outline")
                    ui.label(f"No predictions >= {threshold * 100:.0f}% threshold").classes("text-sm font-medium text-amber-900 mt-1")

            # 2. Top-K Distribution Bars
            ui.label(f"Top-{len(preds)} Predictions").classes("text-xs font-bold uppercase tracking-wider text-slate-500 mt-2")
            with ui.column().classes("w-full gap-2"):
                for idx, item in enumerate(preds, 1):
                    prob_pct = item["prob"] * 100
                    passed = item["prob"] >= threshold
                    bar_color = "primary" if passed else "grey-5"
                    card_cls = (
                        "w-full p-2.5 bg-white border border-slate-200 shadow-none rounded-lg"
                        if passed
                        else "w-full p-2.5 bg-slate-50 border border-slate-200/70 shadow-none rounded-lg opacity-80"
                    )
                    with ui.card().classes(card_cls):
                        with ui.row().classes("w-full items-center justify-between no-wrap gap-2"):
                            with ui.row().classes("items-center gap-2 min-w-0"):
                                ui.label(f"#{idx}").classes("text-xs font-mono font-bold text-slate-400 w-5 shrink-0")
                                ui.label(item["label"]).classes("text-sm font-semibold text-slate-800 truncate")
                            with ui.row().classes("items-center gap-2 shrink-0"):
                                ui.label(f"{prob_pct:5.2f}%").classes(
                                    "text-xs font-mono font-bold text-indigo-600" if passed else "text-xs font-mono text-slate-500"
                                )
                                ui.label(f"({item['logit']:+.2f})").classes("text-[11px] font-mono text-slate-400")
                        ui.linear_progress(value=min(max(item["prob"], 0.0), 1.0), show_value=False, color=bar_color).classes(
                            "mt-1.5 h-1.5 rounded-full"
                        )

            # 3. Collapsible Raw Log
            raw_lines = [
                f"File: {state['filename']} ({state['width']}x{state['height']} px)",
                f"Weights: {params['weights']} | Layer: {params['target_layer']} | Latency: {elapsed_ms:.1f} ms",
                "--- Top-K Predictions ---",
                *[
                    f"{i}. {p['label']:<24} {p['prob']*100:6.2f}%  (logit {p['logit']:+.2f})"
                    for i, p in enumerate(preds, 1)
                ],
            ]
            with ui.expansion("Raw Inference Log", icon="terminal").classes("w-full border border-slate-200 rounded-lg bg-slate-50 text-xs mt-1"):
                ui.label("\n".join(raw_lines)).classes("text-[11px] font-mono text-slate-600 whitespace-pre-wrap p-1")

    async def start_recognition():
        if state["running"]:
            return
        if state["image"] is None:
            ui.notify("Please select an image first.", type="warning")
            return

        params = {
            "weights": weights_select.value,
            "resize_shorter_edge": int(resize_slider.value),
            "confidence_threshold": float(conf_slider.value),
            "softmax_temperature": float(temp_slider.value),
            "top_k": int(topk_slider.value),
            "target_layer": layer_select.value,
            "normalize_imagenet": norm_switch.value,
            "tta_horizontal_flip": tta_switch.value,
        }

        state["running"] = True
        run_btn.disable()
        run_btn.props(add="loading")
        status_label.set_text("Running ResNet-18 inference...")
        status_badge.set_text("Running")
        status_badge.props("color=amber")

        try:
            preds, elapsed_ms = await asyncio.to_thread(
                run_local_resnet18_hook, state["image"], params
            )
            render_predictions(preds, elapsed_ms, params)
            latency_badge.set_text(f"{elapsed_ms:.1f} ms")
            latency_badge.set_visibility(True)
            status_label.set_text(f"Completed in {elapsed_ms:.1f} ms")
            status_badge.set_text("Completed")
            status_badge.props("color=positive")
        except NotImplementedError as hook_err:
            results_container.clear()
            with results_container:
                with ui.card().classes("w-full bg-slate-50 border border-slate-200 shadow-none p-3"):
                    ui.label("Model Hook Not Connected").classes("text-sm font-bold text-slate-800")
                    ui.label(f"{hook_err}\n\nPrepared Parameters:\n{params}").classes(
                        "text-xs font-mono text-slate-600 whitespace-pre-wrap mt-2"
                    )
            status_label.set_text("Frontend ready: connect local model in run_local_resnet18_hook()")
            status_badge.set_text("Hook Pending")
            status_badge.props("color=info")
            ui.notify("Connect your local model in run_local_resnet18_hook()", type="info")
        except Exception as exc:
            status_label.set_text(f"Error: {exc}")
            status_badge.set_text("Error")
            status_badge.props("color=negative")
            ui.notify(f"Inference failed: {exc}", type="negative")
        finally:
            state["running"] = False
            run_btn.props(remove="loading")
            run_btn.enable()

    def clear_all():
        state.update({
            "image": None,
            "filename": "",
            "width": 0,
            "height": 0,
            "size_kb": 0.0,
            "running": False,
        })
        preview_image.set_source("")
        preview_image.set_visibility(False)
        placeholder.set_visibility(True)
        image_meta_badge.set_visibility(False)
        latency_badge.set_visibility(False)
        clear_btn.disable()
        render_empty_results()
        status_label.set_text("Select an image to begin")
        status_badge.set_text("Idle")
        status_badge.props("color=grey-6")
        file_info_label.set_text("No image loaded")

    def reset_params():
        weights_select.set_value(DEFAULT_PARAMS["weights"])
        layer_select.set_value(DEFAULT_PARAMS["target_layer"])
        resize_slider.set_value(DEFAULT_PARAMS["resize_shorter_edge"])
        conf_slider.set_value(DEFAULT_PARAMS["confidence_threshold"])
        temp_slider.set_value(DEFAULT_PARAMS["softmax_temperature"])
        topk_slider.set_value(DEFAULT_PARAMS["top_k"])
        norm_switch.set_value(DEFAULT_PARAMS["normalize_imagenet"])
        tta_switch.set_value(DEFAULT_PARAMS["tta_horizontal_flip"])
        ui.notify("Parameters reset to defaults", type="info")

    async def on_upload(e: events.UploadEventArguments):
        try:
            file_bytes = await e.file.read() if hasattr(e, "file") else e.content.read()
            raw_name = getattr(getattr(e, "file", None), "name", None) or getattr(e, "name", "upload.png")
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            state.update({
                "image": img,
                "filename": os.path.basename(raw_name),
                "width": img.size[0],
                "height": img.size[1],
                "size_kb": round(len(file_bytes) / 1024, 1),
            })
            preview_image.set_source(pil_to_data_url(img))
            preview_image.set_visibility(True)
            placeholder.set_visibility(False)
            clear_btn.enable()

            meta_text = f"{state['width']}x{state['height']} px · {state['size_kb']} KB"
            image_meta_badge.set_text(meta_text)
            image_meta_badge.set_visibility(True)

            file_info_label.set_text(f"{state['filename']} | {meta_text}")
            status_label.set_text(f"Loaded {state['filename']}. Adjust parameters and run.")
            status_badge.set_text("Ready")
            status_badge.props("color=primary")

            render_empty_results(f"Ready: {state['filename']}")
            upload_dialog.close()
        except Exception as exc:
            ui.notify(f"Failed to open image: {exc}", type="negative")

    # ==========================================================================
    # Upload Dialog
    # ==========================================================================
    with ui.dialog() as upload_dialog, ui.card().classes("w-[420px] p-5"):
        with ui.row().classes("w-full items-center justify-between mb-1"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("cloud_upload", size="22px").classes("text-indigo-600")
                ui.label("Select Input Image").classes("text-base font-bold text-slate-800")
            ui.button(icon="close", on_click=upload_dialog.close).props("flat round dense color=grey-7")
        ui.label("Supports JPG, PNG, BMP, and WEBP formats").classes("text-xs text-slate-500 mb-3")
        ui.upload(on_upload=on_upload, auto_upload=True).classes("w-full").props(
            'accept=".jpg,.jpeg,.png,.bmp,.webp" flat bordered'
        )

    # ==========================================================================
    # Top Header Toolbar
    # ==========================================================================
    with ui.header().classes("items-center justify-between px-5 py-2.5 bg-white border-b border-slate-200 text-slate-800 shadow-none"):
        with ui.row().classes("items-center gap-3"):
            with ui.element("div").classes("w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white"):
                ui.icon("hub", size="20px")
            with ui.column().classes("gap-0"):
                ui.label("ResNet-18 Vision Workbench").classes("text-base font-bold leading-tight text-slate-900")
                ui.label("GTSRB Traffic Sign Classification & Parameter Inspector").classes("text-[11px] text-slate-500")
            status_badge = ui.badge("Idle", color="grey-6").classes("ml-2 text-xs")

        with ui.row().classes("items-center gap-2"):
            ui.button("Select Image", icon="add_photo_alternate", on_click=upload_dialog.open).props("outline color=primary")
            clear_btn = ui.button("Clear", icon="delete_outline", on_click=clear_all).props("flat color=grey-7")
            clear_btn.disable()
            run_btn = ui.button("Start Recognition", icon="play_arrow", on_click=start_recognition).props("unelevated color=primary")

    # ==========================================================================
    # Main 3-Column Workbench Layout
    # ==========================================================================
    with ui.element("div").classes("w-full grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 lg:h-[calc(100vh-104px)]"):

        # ----------------------------------------------------------------------
        # Left Column: Hyperparameters & Preprocessing Controls (3/12)
        # ----------------------------------------------------------------------
        with ui.card().classes("lg:col-span-3 h-full flex flex-col p-0 overflow-hidden border border-slate-200 shadow-sm"):
            with ui.row().classes("w-full items-center justify-between px-4 py-3 border-b border-slate-200 bg-slate-50/70 shrink-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("tune", size="18px").classes("text-indigo-600")
                    ui.label("Parameters").classes("text-sm font-bold text-slate-800")
                ui.button("Reset", icon="restart_alt", on_click=reset_params).props("flat dense size=sm color=grey-7")

            with ui.column().classes("w-full flex-1 overflow-y-auto p-4 gap-4"):
                # Group 1: Model Configuration
                with ui.column().classes("w-full gap-2.5"):
                    ui.label("MODEL & ACTIVATION").classes("text-[11px] font-bold tracking-wider text-slate-400")
                    weights_select = ui.select(
                        ["IMAGENET1K_V1 (FP32)", "IMAGENET1K_V1 (FP16)"],
                        value=DEFAULT_PARAMS["weights"],
                        label="Checkpoint Weights",
                    ).classes("w-full").props("outlined dense")
                    layer_select = ui.select(
                        ["layer1", "layer2", "layer3", "layer4"],
                        value=DEFAULT_PARAMS["target_layer"],
                        label="Activation Layer",
                    ).classes("w-full").props("outlined dense")

                ui.separator()

                # Group 2: Image Preprocessing
                with ui.column().classes("w-full gap-2.5"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("PREPROCESSING").classes("text-[11px] font-bold tracking-wider text-slate-400")
                        ui.badge("64x64 Fixed", color="indigo-1", text_color="indigo-9").classes("text-[10px] font-mono")

                    with ui.column().classes("w-full gap-0"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Resize Shorter Edge").classes("text-xs font-medium text-slate-700")
                            resize_val_label = ui.label().classes("text-xs font-mono font-semibold text-indigo-600")
                        resize_slider = ui.slider(
                            min=224, max=384, step=8, value=DEFAULT_PARAMS["resize_shorter_edge"]
                        ).props("dense")
                        resize_val_label.bind_text_from(resize_slider, "value", backward=lambda v: f"{int(v)} px")

                    norm_switch = ui.switch("ImageNet RGB Normalization", value=DEFAULT_PARAMS["normalize_imagenet"]).props("dense")

                ui.separator()

                # Group 3: Inference & Post-processing
                with ui.column().classes("w-full gap-3"):
                    ui.label("POST-PROCESSING & TTA").classes("text-[11px] font-bold tracking-wider text-slate-400")

                    with ui.column().classes("w-full gap-0"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Confidence Threshold").classes("text-xs font-medium text-slate-700")
                            conf_val_label = ui.label().classes("text-xs font-mono font-semibold text-indigo-600")
                        conf_slider = ui.slider(min=0.0, max=1.0, step=0.01, value=DEFAULT_PARAMS["confidence_threshold"]).props("dense")
                        conf_val_label.bind_text_from(conf_slider, "value", backward=lambda v: f"{v:.2f}")

                    with ui.column().classes("w-full gap-0"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Softmax Temperature").classes("text-xs font-medium text-slate-700")
                            temp_val_label = ui.label().classes("text-xs font-mono font-semibold text-indigo-600")
                        temp_slider = ui.slider(min=0.1, max=3.0, step=0.05, value=DEFAULT_PARAMS["softmax_temperature"]).props("dense")
                        temp_val_label.bind_text_from(temp_slider, "value", backward=lambda v: f"{v:.2f}")

                    with ui.column().classes("w-full gap-0"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Top-K Predictions").classes("text-xs font-medium text-slate-700")
                            topk_val_label = ui.label().classes("text-xs font-mono font-semibold text-indigo-600")
                        topk_slider = ui.slider(min=1, max=10, step=1, value=DEFAULT_PARAMS["top_k"]).props("dense")
                        topk_val_label.bind_text_from(topk_slider, "value", backward=lambda v: f"{int(v)}")

                    tta_switch = ui.switch("Horizontal Flip TTA", value=DEFAULT_PARAMS["tta_horizontal_flip"]).props("dense")

        # ----------------------------------------------------------------------
        # Center Column: Image Viewport (5/12)
        # ----------------------------------------------------------------------
        with ui.card().classes("lg:col-span-5 h-full flex flex-col p-0 overflow-hidden border border-slate-200 shadow-sm"):
            with ui.row().classes("w-full items-center justify-between px-4 py-3 border-b border-slate-200 bg-slate-50/70 shrink-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("image", size="18px").classes("text-indigo-600")
                    ui.label("Image Viewport").classes("text-sm font-bold text-slate-800")
                image_meta_badge = ui.badge("", color="indigo-1", text_color="indigo-9").classes("text-xs font-mono")
                image_meta_badge.set_visibility(False)

            with ui.column().classes("w-full flex-1 min-h-[320px] p-4 items-center justify-center bg-slate-100/80 relative overflow-hidden"):
                preview_image = ui.image().classes("w-full h-full rounded-lg").props("fit=contain")
                preview_image.set_visibility(False)

                with ui.column().classes(
                    "w-full h-full items-center justify-center border-2 border-dashed border-slate-300 rounded-xl bg-white/60 p-6 gap-3"
                ) as placeholder:
                    with ui.element("div").classes("w-14 h-14 rounded-full bg-indigo-50 flex items-center justify-center"):
                        ui.icon("add_photo_alternate", size="30px").classes("text-indigo-500")
                    ui.label("No Image Loaded").classes("text-base font-semibold text-slate-700")
                    ui.label("Supports JPG / PNG / BMP / WEBP").classes("text-xs text-slate-400")
                    ui.button("Browse Image", icon="upload_file", on_click=upload_dialog.open).props("unelevated color=primary size=sm").classes("mt-1")

        # ----------------------------------------------------------------------
        # Right Column: Recognition Results & Analytics (4/12)
        # ----------------------------------------------------------------------
        with ui.card().classes("lg:col-span-4 h-full flex flex-col p-0 overflow-hidden border border-slate-200 shadow-sm"):
            with ui.row().classes("w-full items-center justify-between px-4 py-3 border-b border-slate-200 bg-slate-50/70 shrink-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("insights", size="18px").classes("text-indigo-600")
                    ui.label("Recognition Results").classes("text-sm font-bold text-slate-800")
                latency_badge = ui.badge("", color="green-1", text_color="green-9").classes("text-xs font-mono")
                latency_badge.set_visibility(False)

            results_container = ui.column().classes("w-full flex-1 overflow-y-auto p-4 gap-3")
            render_empty_results()

    # ==========================================================================
    # Bottom Status Footer
    # ==========================================================================
    with ui.footer().classes("bg-white border-t border-slate-200 px-5 py-1.5 text-xs text-slate-500 justify-between items-center"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("circle", size="8px").classes("text-indigo-500")
            status_label = ui.label("Select an image to begin").classes("font-medium text-slate-600")
        file_info_label = ui.label("No image loaded").classes("font-mono text-slate-400")


ui.run(title="ResNet-18 Vision Workbench", reload=False)