import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

# A very simple user interface for testing

# Enable High DPI scaling support in Windows to ensure fonts and controls remain sharp.
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

class ImageRecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Software Engine GUI")
        self.root.geometry("860x600")
        self.root.minsize(720, 500)
        self.root.configure(bg="#f8fafc")

        # Data state variables
        self.image_path = None
        self.original_image = None
        self.tk_photo = None
        self.status_var = tk.StringVar(value="Please click the [Select Image] button above to begin")

        # Initialize modern ttk widget styles
        self._init_style()

        # Construct the interface layout
        self._build_ui()

    def _init_style(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure(".", font=("Microsoft YaHei", 9), background="#f8fafc")

        # Primary button (highlighted/emphasized)
        self.style.configure(
            "Primary.TButton",
            font=("Microsoft YaHei", 9, "bold"),
            background="#2563eb",
            foreground="#ffffff",
            padding=(14, 6)
        )
        self.style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])

        # Secondary button
        self.style.configure(
            "Secondary.TButton",
            font=("Microsoft YaHei", 9),
            padding=(10, 6)
        )

    def _build_ui(self):
        # Top Action Toolbar
        top_bar = tk.Frame(self.root, bg="#ffffff", padx=16, pady=12, bd=1, relief=tk.SOLID)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            top_bar,
            text="Image Recognition",
            font=("Microsoft YaHei", 12, "bold"),
            bg="#ffffff",
            fg="#0f172a"
        ).pack(side=tk.LEFT)

        # Right-side action button group
        btn_box = tk.Frame(top_bar, bg="#ffffff")
        btn_box.pack(side=tk.RIGHT)

        self.btn_upload = ttk.Button(
            btn_box,
            text="Select your image",
            style="Secondary.TButton",
            command=self.select_image
        )
        self.btn_upload.pack(side=tk.LEFT, padx=4)

        self.btn_recognize = ttk.Button(
            btn_box,
            text="Start recognition",
            style="Primary.TButton",
            command=self.start_recognition
        )
        self.btn_recognize.pack(side=tk.LEFT, padx=4)

        self.btn_clear = ttk.Button(
            btn_box,
            text="Clear",
            style="Secondary.TButton",
            command=self.clear_all
        )
        self.btn_clear.pack(side=tk.LEFT, padx=4)

        # Central Main Workspace (Split-pane Layout)
        main_body = tk.Frame(self.root, bg="#f8fafc", padx=16, pady=12)
        main_body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left side: Image preview area
        left_frame = tk.LabelFrame(
            main_body,
            text="Image Preview Area",
            font=("Microsoft YaHei", 10, "bold"),
            bg="#ffffff",
            fg="#1e293b",
            bd=1,
            relief=tk.SOLID,
            padx=10,
            pady=10
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.preview_canvas = tk.Canvas(
            left_frame,
            bg="#f1f5f9",
            highlightthickness=0,
            cursor="hand2"
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)
        self.preview_canvas.bind("<Button-1>", lambda e: self.select_image())

        self.placeholder_text = self.preview_canvas.create_text(
            200, 180,
            text="Click or upload an image to preview\n(Support JPG / PNG / BMP / WEBP)",
            font=("Microsoft YaHei", 10),
            fill="#94a3b8",
            justify=tk.CENTER
        )

        # Right side: Recognition results display area
        right_frame = tk.LabelFrame(
            main_body,
            text="The recognition results show",
            font=("Microsoft YaHei", 10, "bold"),
            bg="#ffffff",
            fg="#1e293b",
            bd=1,
            relief=tk.SOLID,
            padx=10,
            pady=10
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # Text output box with a scroll bar
        self.result_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#0f172a",
            relief=tk.FLAT,
            bd=0
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self._reset_result_tip()

        # Bottom status bar
        status_bar = tk.Frame(self.root, bg="#f1f5f9", bd=1, relief=tk.SOLID, padx=12, pady=6)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = tk.Label(
            status_bar,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 9),
            bg="#f1f5f9",
            fg="#64748b"
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_file_info = tk.Label(
            status_bar,
            text="Image not loaded",
            font=("Microsoft YaHei", 9),
            bg="#f1f5f9",
            fg="#94a3b8"
        )
        self.lbl_file_info.pack(side=tk.RIGHT)

    def _reset_result_tip(self):
        # Restore the initial prompt text in the text box
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(
            tk.END,
            "Waiting for action...\n"
        )

    def select_image(self):
        # Select and load an image file
        file_path = filedialog.askopenfilename(
            title="Select an image for identification",
            filetypes=[
                ("Common images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path)

            # Retrieve image metadata
            w, h = self.original_image.size
            size_kb = os.path.getsize(file_path) / 1024
            filename = os.path.basename(file_path)

            self.lbl_file_info.config(text=f"{filename} | {w}x{h} px | {size_kb:.1f} KB")
            self.status_var.set(f"Image loaded：{filename}，Please click [Start Recognition]")

            # Centered Preview of Render
            self._render_preview()

            # Prepare text box
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(
                tk.END,
                f"Ready files: {filename}\n"
                f"Path: {file_path}\n"
                f"Image resolution: {w} × {h} pixel\n"
                f"File size: {size_kb:.1f} KB\n\n"
            )
        except Exception as e:
            messagebox.showerror("Failed to open image", f"Unable to parse the selected image：\n{e}")

    def _on_canvas_resize(self, event):
        # Automatically redraw and center the image when the window size changes
        if self.original_image:
            self._render_preview()
        else:
            self.preview_canvas.coords(self.placeholder_text, event.width // 2, event.height // 2)

    def _render_preview(self):
        # Scale the image proportionally and apply anti-aliasing using the Lanczos algorithm.
        if not self.original_image:
            return

        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw < 30 or ch < 30:
            return

        img_w, img_h = self.original_image.size
        scale = min((cw - 20) / img_w, (ch - 20) / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_photo = ImageTk.PhotoImage(resized)

        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(cw // 2, ch // 2, image=self.tk_photo, anchor=tk.CENTER)

    def start_recognition(self):
        # Click the "Start Recognition" button
        if not self.image_path or not self.original_image:
            messagebox.showwarning("prompt", "Please click [Select Image] to upload an image！")
            return

        self.btn_recognize.config(state="disabled")
        self.btn_upload.config(state="disabled")
        self.status_var.set("Calling the recognition engine to analyze the image; please wait....")

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, ">>> Extracting high-dimensional image features and text regions...\n")

        # Launch a background thread to prevent the UI from freezing
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def _recognition_worker(self):
        # Background recognition task
        try:
            time.sleep(1.0)
            filename = os.path.basename(self.image_path)
            w, h = self.original_image.size

            result = (
                f"Target filename: {filename}\n"
                f"Native resolution: {w} × {h} Pixel\n"
            )

            self.root.after(0, self._on_complete, result)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _on_complete(self, result_text: str):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, result_text)
        self.status_var.set("Recognition complete! The results are displayed in the text box on the right.")
        self.btn_recognize.config(state="normal")
        self.btn_upload.config(state="normal")

    def _on_error(self, err_msg: str):
        messagebox.showerror("Recognition error", f"Error：\n{err_msg}")
        self.status_var.set("Recognition failed; please try again.")
        self.btn_recognize.config(state="normal")
        self.btn_upload.config(state="normal")

    def clear_all(self):
        # Reset all interface elements and variables
        self.image_path = None
        self.original_image = None
        self.tk_photo = None
        self.preview_canvas.delete("all")

        cw = self.preview_canvas.winfo_width() or 200
        ch = self.preview_canvas.winfo_height() or 200
        self.placeholder_text = self.preview_canvas.create_text(
            cw // 2, ch // 2,
            text="Click or upload an image to preview\n(Support JPG / PNG / BMP / WEBP)",
            font=("Microsoft YaHei", 10),
            fill="#94a3b8",
            justify=tk.CENTER
        )
        self._reset_result_tip()
        self.lbl_file_info.config(text="Image not loaded")
        self.status_var.set("Reset: Please select an image again")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRecognitionApp(root)
    root.mainloop()