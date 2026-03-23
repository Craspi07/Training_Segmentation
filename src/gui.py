#!/usr/bin/env python3
"""
Tkinter GUI for the Cellpose Segmentation Training Pipeline.

Provides tabs for:
  1. File Renaming  — Rename images/masks to pipeline convention
  2. Configuration  — Edit training parameters from YAML configs
  3. Training       — Launch and monitor model training
  4. Evaluation     — Run inference and view results
"""

import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

import yaml

# Ensure src/ modules are importable
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# Default path to the project inside the Docker container
_CONTAINER_DEFAULT_ROOT = "/workspace/Training/Training_Segmentation"

from rename_files import (
    list_image_files,
    rename_files,
    rename_image_mask_pair,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging handler that writes to a tkinter Text widget via a thread-safe queue
# ---------------------------------------------------------------------------
class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class SegmentationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cellpose Segmentation Pipeline")
        self.geometry("960x720")
        self.minsize(800, 600)

        # Thread-safe log queue
        self.log_queue: queue.Queue[str] = queue.Queue()

        # Currently loaded YAML config (named to avoid shadowing tk.Tk.config)
        self.yaml_config: dict = {}
        self.yaml_config_path: str = ""

        # Docker settings for running ML on Windows
        self.docker_container = tk.StringVar(value=os.environ.get("DOCKER_CONTAINER", ""))
        self.container_root = tk.StringVar(value=os.environ.get("CONTAINER_PROJECT_ROOT", _CONTAINER_DEFAULT_ROOT))
        # Python interpreter inside the Detectron2 venv in Docker
        self.d2_docker_python = tk.StringVar(value=os.environ.get("D2_DOCKER_PYTHON", "/opt/conda/bin/python3"))
        # Bind-mount mapping: Windows host folder → container folder
        # e.g. host = C:\Users\Windows\Documents\Segmentation  container = /workspace
        self.host_mount_path = tk.StringVar(value=os.environ.get("HOST_MOUNT_PATH", ""))
        self.container_mount_path = tk.StringVar(value=os.environ.get("CONTAINER_MOUNT_PATH", "/workspace"))

        self._build_menu()
        self._build_tabs()
        self._setup_logging()
        self._poll_log_queue()

    # ----- Docker helpers -----
    def _to_container_path(self, path: str) -> str:
        """Translate a local filesystem path to the equivalent container path.

        Translation order:
        1. If path is under the bind-mount host folder, map it to the container mount folder.
        2. If path is under PROJECT_ROOT, map it to container_root.
        3. Fall back to replacing backslashes (may still fail if not mounted).
        """
        host_mount = self.host_mount_path.get().strip()
        container_mount = self.container_mount_path.get().strip()
        if host_mount and container_mount:
            try:
                rel = Path(path).relative_to(Path(host_mount))
                return (Path(container_mount) / rel).as_posix()
            except ValueError:
                pass
        try:
            rel = Path(path).relative_to(PROJECT_ROOT)
            return (Path(self.container_root.get()) / rel).as_posix()
        except ValueError:
            return str(path).replace("\\", "/")

    def _run_docker_cmd(self, container: str, cmd: list) -> int:
        """Run a command inside the Docker container, streaming output to logger."""
        full_cmd = ["docker", "exec", container] + cmd
        logger.info(f"docker exec: {' '.join(full_cmd)}")
        proc = subprocess.Popen(
            full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            logger.info(line.rstrip())
        proc.wait()
        return proc.returncode

    # ----- Menu bar -----
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Config...", command=self._open_config)
        file_menu.add_command(label="Save Config", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # ----- Notebook tabs -----
    def _build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_preprocess = ttk.Frame(self.notebook)
        self.tab_rename = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)
        self.tab_detectron2 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_preprocess, text="  Preprocessing  ")
        self.notebook.add(self.tab_rename, text="  File Renaming  ")
        self.notebook.add(self.tab_config, text="  Configuration  ")
        self.notebook.add(self.tab_train, text="  Training  ")
        self.notebook.add(self.tab_eval, text="  Evaluation  ")
        self.notebook.add(self.tab_detectron2, text="  Detectron2  ")

        self._build_preprocess_tab()
        self._build_rename_tab()
        self._build_config_tab()
        self._build_train_tab()
        self._build_eval_tab()
        self._build_detectron2_tab()

    # ==================================================================
    # TAB 0: PREPROCESSING
    # ==================================================================
    def _build_preprocess_tab(self):
        tab = self.tab_preprocess

        # ---- Input / Output ----
        io_frame = ttk.LabelFrame(tab, text="Directories", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(io_frame, text="Image Directory:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.pp_img_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.pp_img_dir, width=55).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=self._pp_browse_img).grid(
            row=0, column=2, pady=2
        )

        ttk.Label(io_frame, text="Mask Output Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.pp_out_dir = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "draft_masks")
        )
        ttk.Entry(io_frame, textvariable=self.pp_out_dir, width=55).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=self._pp_browse_out).grid(
            row=1, column=2, pady=2
        )

        # ---- Channel & Normalization Settings ----
        settings_frame = ttk.LabelFrame(tab, text="Channel & Normalization", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # Segment channel
        ttk.Label(settings_frame, text="Segment Channel:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.pp_seg_channel = tk.IntVar(value=0)
        seg_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.pp_seg_channel,
            values=[0, 1, 2, 3],
            width=5,
            state="readonly",
        )
        seg_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.pp_seg_label = tk.StringVar(value="DIC")
        ttk.Label(settings_frame, textvariable=self.pp_seg_label, foreground="gray").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )
        seg_combo.bind("<<ComboboxSelected>>", self._pp_update_channel_labels)

        # Nuclear channel
        ttk.Label(settings_frame, text="Nuclear Channel:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.pp_nuc_channel = tk.IntVar(value=0)
        nuc_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.pp_nuc_channel,
            values=[0, 1, 2, 3],
            width=5,
            state="readonly",
        )
        nuc_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.pp_nuc_label = tk.StringVar(value="None (grayscale)")
        ttk.Label(settings_frame, textvariable=self.pp_nuc_label, foreground="gray").grid(
            row=1, column=2, sticky=tk.W, padx=5
        )
        nuc_combo.bind("<<ComboboxSelected>>", self._pp_update_channel_labels)

        # Percentile range
        ttk.Label(settings_frame, text="Normalization Percentile:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        pct_frame = ttk.Frame(settings_frame)
        pct_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(pct_frame, text="Low:").pack(side=tk.LEFT)
        self.pp_lower_pct = tk.DoubleVar(value=1.0)
        ttk.Entry(pct_frame, textvariable=self.pp_lower_pct, width=6).pack(
            side=tk.LEFT, padx=(2, 10)
        )
        ttk.Label(pct_frame, text="High:").pack(side=tk.LEFT)
        self.pp_upper_pct = tk.DoubleVar(value=99.0)
        ttk.Entry(pct_frame, textvariable=self.pp_upper_pct, width=6).pack(
            side=tk.LEFT, padx=2
        )

        # Tile normalization
        ttk.Label(settings_frame, text="DIC Tile Blocksize:").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.pp_tile_bs = tk.IntVar(value=128)
        ttk.Entry(settings_frame, textvariable=self.pp_tile_bs, width=8).grid(
            row=3, column=1, padx=5, pady=2, sticky=tk.W
        )
        ttk.Label(
            settings_frame,
            text="(0 = global normalization, >0 = tile-based for uneven illumination)",
            foreground="gray",
        ).grid(row=3, column=2, sticky=tk.W, padx=5)

        # Z-slice for ND2 Z-stacks
        ttk.Label(settings_frame, text="Z-Slice (ND2):").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        z_frame = ttk.Frame(settings_frame)
        z_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5)
        self.pp_z_slice = tk.StringVar(value="0")
        ttk.Entry(z_frame, textvariable=self.pp_z_slice, width=6).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(
            z_frame,
            text="(0-indexed Z-slice for Z-stack .nd2 files)",
            foreground="gray",
        ).pack(side=tk.LEFT)

        # Invert DIC
        self.pp_invert_dic = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings_frame,
            text="Invert DIC (cells dark on bright background)",
            variable=self.pp_invert_dic,
        ).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=2)

        # ---- Cellpose Settings ----
        cp_frame = ttk.LabelFrame(tab, text="Cellpose Settings", padding=10)
        cp_frame.pack(fill=tk.X, padx=10, pady=5)

        # Model
        ttk.Label(cp_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pp_model_var = tk.StringVar(value="cpsam (default)")
        pp_model_combo = ttk.Combobox(
            cp_frame,
            textvariable=self.pp_model_var,
            values=["cpsam (default)", "Custom model...", "BioImage.io model..."],
            width=22,
            state="readonly",
        )
        pp_model_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        pp_model_combo.bind("<<ComboboxSelected>>", self._pp_on_model_changed)

        self.pp_custom_model = tk.StringVar()
        self.pp_model_entry = ttk.Entry(
            cp_frame, textvariable=self.pp_custom_model, width=35
        )
        self.pp_model_entry.grid(row=0, column=2, padx=5, pady=2)
        self.pp_model_entry.config(state=tk.DISABLED)
        self.pp_model_btn = ttk.Button(
            cp_frame, text="Browse...", command=self._pp_browse_model
        )
        self.pp_model_btn.grid(row=0, column=3, pady=2)
        self.pp_model_btn.config(state=tk.DISABLED)

        # Diameter
        ttk.Label(cp_frame, text="Diameter:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pp_diameter = tk.StringVar(value="auto")
        ttk.Entry(cp_frame, textvariable=self.pp_diameter, width=10).grid(
            row=1, column=1, padx=5, pady=2, sticky=tk.W
        )
        ttk.Label(cp_frame, text="(pixels, or 'auto')", foreground="gray").grid(
            row=1, column=2, sticky=tk.W, padx=5
        )

        # Flow threshold
        ttk.Label(cp_frame, text="Flow Threshold:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.pp_flow_thr = tk.DoubleVar(value=0.4)
        ttk.Entry(cp_frame, textvariable=self.pp_flow_thr, width=10).grid(
            row=2, column=1, padx=5, pady=2, sticky=tk.W
        )

        # Cell prob threshold
        ttk.Label(cp_frame, text="Cell Prob Threshold:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.pp_cellprob_thr = tk.DoubleVar(value=0.0)
        ttk.Entry(cp_frame, textvariable=self.pp_cellprob_thr, width=10).grid(
            row=3, column=1, padx=5, pady=2, sticky=tk.W
        )

        # ---- Buttons ----
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame, text="Preview Channels", command=self._pp_preview_channels
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame, text="Expand ND2 Positions", command=self._pp_expand_nd2
        ).pack(side=tk.LEFT, padx=5)

        self.btn_pp_run = ttk.Button(
            btn_frame, text="Generate Masks", command=self._pp_generate
        )
        self.btn_pp_run.pack(side=tk.LEFT, padx=5)

        # Progress
        self.pp_progress = ttk.Progressbar(tab, mode="determinate")
        self.pp_progress.pack(fill=tk.X, padx=10, pady=5)

        self.pp_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.pp_status).pack(padx=10, anchor=tk.W)

        # Results table
        result_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        columns = ("filename", "objects", "status")
        self.pp_tree = ttk.Treeview(
            result_frame, columns=columns, show="headings", height=8
        )
        self.pp_tree.heading("filename", text="Filename")
        self.pp_tree.heading("objects", text="Objects Found")
        self.pp_tree.heading("status", text="Status")
        self.pp_tree.column("filename", width=350)
        self.pp_tree.column("objects", width=120, anchor=tk.CENTER)
        self.pp_tree.column("status", width=120, anchor=tk.CENTER)

        pp_scroll = ttk.Scrollbar(
            result_frame, orient=tk.VERTICAL, command=self.pp_tree.yview
        )
        self.pp_tree.configure(yscrollcommand=pp_scroll.set)
        self.pp_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pp_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _pp_browse_img(self):
        d = filedialog.askdirectory(title="Select Image Directory")
        if d:
            self.pp_img_dir.set(d)

    def _pp_browse_out(self):
        d = filedialog.askdirectory(title="Select Mask Output Directory")
        if d:
            self.pp_out_dir.set(d)

    def _pp_on_model_changed(self, event=None):
        choice = self.pp_model_var.get()
        if choice == "Custom model...":
            self.pp_model_entry.config(state=tk.NORMAL)
            self.pp_model_btn.config(state=tk.NORMAL)
            self.pp_custom_model.set("")
        elif choice == "BioImage.io model...":
            self.pp_model_entry.config(state=tk.NORMAL)
            self.pp_model_btn.config(state=tk.NORMAL)
            self.pp_custom_model.set("bioimage.io:")
        else:
            self.pp_model_entry.config(state=tk.DISABLED)
            self.pp_model_btn.config(state=tk.DISABLED)
            self.pp_custom_model.set("")

    def _pp_browse_model(self):
        if self.pp_model_var.get() == "BioImage.io model...":
            path = filedialog.askopenfilename(
                title="Select BioImage.io Model (rdf.yaml / .zip)",
                initialdir=str(PROJECT_ROOT / "models"),
                filetypes=[
                    ("BioImage.io", "*.yaml *.yml *.zip"),
                    ("All", "*.*"),
                ],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select Cellpose Model",
                initialdir=str(PROJECT_ROOT / "models"),
            )
        if path:
            self.pp_custom_model.set(path)

    _CHANNEL_NAMES = {0: "DIC", 1: "mEGFP", 2: "mScarlet", 3: "miRFPnano3"}

    def _pp_update_channel_labels(self, event=None):
        seg = self.pp_seg_channel.get()
        nuc = self.pp_nuc_channel.get()
        self.pp_seg_label.set(self._CHANNEL_NAMES.get(seg, f"Channel {seg}"))
        if nuc == 0:
            self.pp_nuc_label.set("None (grayscale)")
        else:
            self.pp_nuc_label.set(self._CHANNEL_NAMES.get(nuc, f"Channel {nuc}"))

    def _pp_preview_channels(self):
        """Save and open a channel preview for the first image."""
        img_dir = self.pp_img_dir.get()
        if not img_dir:
            messagebox.showwarning("Missing", "Select an image directory first.")
            return

        from preprocess import save_channel_preview
        import glob as _glob

        extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.nd2")
        files = []
        for ext in extensions:
            files.extend(_glob.glob(os.path.join(img_dir, ext)))
        files = sorted(files)

        if not files:
            messagebox.showinfo("Empty", "No images found in the directory.")
            return

        preview_dir = str(PROJECT_ROOT / "results" / "channel_previews")
        z_val = self.pp_z_slice.get().strip()
        z_slice = int(z_val) if z_val else None

        try:
            path = save_channel_preview(
                files[0],
                preview_dir,
                lower_percentile=self.pp_lower_pct.get(),
                upper_percentile=self.pp_upper_pct.get(),
                tile_blocksize_dic=self.pp_tile_bs.get(),
                z_slice=z_slice,
            )
            self.pp_status.set(f"Channel preview saved: {path}")
            logger.info(f"Channel preview saved to {path}")
            messagebox.showinfo(
                "Preview Saved",
                f"Channel preview for {Path(files[0]).name} saved to:\n{path}\n\n"
                "Open this file to inspect channel quality and normalization.",
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate preview:\n{e}")
            logger.exception("Channel preview failed")

    def _pp_expand_nd2(self):
        """Expand multi-position .nd2 files in the image directory into TIFFs."""
        img_dir = self.pp_img_dir.get()
        if not img_dir:
            messagebox.showwarning("Missing", "Select an image directory first.")
            return

        import glob as _glob
        nd2_files = sorted(_glob.glob(os.path.join(img_dir, "*.nd2")))
        if not nd2_files:
            messagebox.showinfo("No ND2", "No .nd2 files found in the directory.")
            return

        from data_preparation import expand_nd2_positions, get_nd2_info

        output_dir = os.path.join(img_dir, "_nd2_expanded")
        total_saved = []
        multi_pos_count = 0

        for nd2_file in nd2_files:
            try:
                info = get_nd2_info(nd2_file)
                n_pos = info["n_positions"]
                if n_pos > 1:
                    multi_pos_count += 1
                    logger.info(
                        f"Expanding {Path(nd2_file).name}: "
                        f"{n_pos} positions, shape={info['shape']}"
                    )
                saved = expand_nd2_positions(nd2_file, output_dir)
                total_saved.extend(saved)
            except Exception as e:
                logger.error(f"Failed to expand {Path(nd2_file).name}: {e}")
                messagebox.showerror(
                    "ND2 Error",
                    f"Failed to expand {Path(nd2_file).name}:\n{e}",
                )

        if total_saved:
            messagebox.showinfo(
                "ND2 Expanded",
                f"Processed {len(nd2_files)} ND2 file(s) "
                f"({multi_pos_count} multi-position).\n"
                f"Saved {len(total_saved)} individual TIFFs to:\n{output_dir}\n\n"
                "You can now use this directory as the image input, or the "
                "pipeline will auto-expand during mask generation.",
            )
            self.pp_status.set(
                f"Expanded {len(nd2_files)} ND2 -> {len(total_saved)} TIFFs"
            )

    def _pp_generate(self):
        """Generate draft masks in a background thread."""
        img_dir = self.pp_img_dir.get()
        out_dir = self.pp_out_dir.get()
        if not img_dir:
            messagebox.showwarning("Missing", "Select an image directory.")
            return

        self.btn_pp_run.config(state=tk.DISABLED)
        self.pp_tree.delete(*self.pp_tree.get_children())
        self.pp_progress.config(mode="determinate", value=0)
        self.pp_status.set("Generating masks...")

        # Parse diameter
        diam_str = self.pp_diameter.get().strip().lower()
        diameter = None if diam_str in ("auto", "none", "") else float(diam_str)

        # Parse Z-slice
        z_val = self.pp_z_slice.get().strip()
        z_slice = int(z_val) if z_val else None

        # Parse model (supports local paths and BioImage.io identifiers)
        model_path = None
        if self.pp_model_var.get() in ("Custom model...", "BioImage.io model..."):
            model_path = self.pp_custom_model.get() or None

        def progress_cb(current, total, filename):
            if total > 0:
                pct = int(100 * current / total)
                # Use thread-safe queue to update GUI
                self.log_queue.put(f"__PP_PROGRESS__{pct}||{current}/{total}: {filename}")

        def worker():
            try:
                from preprocess import generate_masks
                results = generate_masks(
                    image_dir=img_dir,
                    output_dir=out_dir,
                    segment_channel=self.pp_seg_channel.get(),
                    nuclear_channel=self.pp_nuc_channel.get(),
                    model_path=model_path,
                    diameter=diameter,
                    flow_threshold=self.pp_flow_thr.get(),
                    cellprob_threshold=self.pp_cellprob_thr.get(),
                    lower_percentile=self.pp_lower_pct.get(),
                    upper_percentile=self.pp_upper_pct.get(),
                    tile_blocksize_dic=self.pp_tile_bs.get(),
                    invert_dic=self.pp_invert_dic.get(),
                    use_gpu=True,
                    z_slice=z_slice,
                    progress_callback=progress_cb,
                )
                # Encode results for the GUI thread
                encoded = "|".join(f"{name},{n}" for name, n in results)
                self.log_queue.put(f"__PP_DONE__{encoded}")
            except Exception as e:
                logger.exception("Mask generation failed")
                self.log_queue.put(f"__PP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_pp_finished(self, results_str: str | None = None, error: str | None = None):
        self.btn_pp_run.config(state=tk.NORMAL)
        self.pp_progress.config(value=100)

        if error:
            self.pp_status.set(f"Error: {error}")
            messagebox.showerror("Preprocessing Error", str(error))
            return

        if results_str:
            entries = results_str.split("|")
            total_objects = 0
            for entry in entries:
                if not entry:
                    continue
                name, n_str = entry.rsplit(",", 1)
                n = int(n_str)
                if n < 0:
                    status = "FAILED"
                elif n == 0:
                    status = "No signal"
                else:
                    status = "OK"
                    total_objects += n
                self.pp_tree.insert("", tk.END, values=(name, n if n >= 0 else "-", status))

            self.pp_status.set(
                f"Done: {len(entries)} images processed, {total_objects} total objects. "
                f"Masks saved to: {self.pp_out_dir.get()}"
            )
            messagebox.showinfo(
                "Preprocessing Complete",
                f"Generated draft masks for {len(entries)} images.\n"
                f"Total objects detected: {total_objects}\n\n"
                f"Masks saved to:\n{self.pp_out_dir.get()}\n\n"
                "Review and curate the masks manually, then use the\n"
                "File Renaming tab to prepare them for training.",
            )

    # ==================================================================
    # TAB 1: FILE RENAMING
    # ==================================================================
    def _build_rename_tab(self):
        tab = self.tab_rename

        # ---- Top: directory selectors ----
        dir_frame = ttk.LabelFrame(tab, text="Directories", padding=10)
        dir_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Image directory
        ttk.Label(dir_frame, text="Image Directory:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.rename_img_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.rename_img_dir, width=60).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(dir_frame, text="Browse...", command=self._browse_img_dir).grid(
            row=0, column=2, pady=2
        )

        # Mask directory
        ttk.Label(dir_frame, text="Mask Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.rename_mask_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.rename_mask_dir, width=60).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(dir_frame, text="Browse...", command=self._browse_mask_dir).grid(
            row=1, column=2, pady=2
        )

        # ---- Options ----
        opts_frame = ttk.LabelFrame(tab, text="Rename Options", padding=10)
        opts_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(opts_frame, text="Prefix:").grid(row=0, column=0, sticky=tk.W)
        self.rename_prefix = tk.StringVar(value="dic_")
        prefix_combo = ttk.Combobox(
            opts_frame,
            textvariable=self.rename_prefix,
            values=["dic_", "fluor_", "cell_", "nuc_", ""],
            width=15,
        )
        prefix_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(opts_frame, text="Target Extension:").grid(
            row=0, column=2, sticky=tk.W, padx=(20, 0)
        )
        self.rename_ext = tk.StringVar(value=".tif")
        ext_combo = ttk.Combobox(
            opts_frame,
            textvariable=self.rename_ext,
            values=[".tif", ".png", "(keep original)"],
            width=15,
        )
        ext_combo.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)

        self.rename_copy_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts_frame, text="Copy (don't rename in place)", variable=self.rename_copy_mode
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Output dirs (for copy mode)
        ttk.Label(opts_frame, text="Image Output Dir:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.rename_img_out = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "train" / "dic_raw")
        )
        ttk.Entry(opts_frame, textvariable=self.rename_img_out, width=50).grid(
            row=2, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W
        )
        ttk.Button(
            opts_frame, text="Browse...", command=self._browse_img_out
        ).grid(row=2, column=3, pady=2)

        ttk.Label(opts_frame, text="Mask Output Dir:").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.rename_mask_out = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "train" / "dic_labels")
        )
        ttk.Entry(opts_frame, textvariable=self.rename_mask_out, width=50).grid(
            row=3, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W
        )
        ttk.Button(
            opts_frame, text="Browse...", command=self._browse_mask_out
        ).grid(row=3, column=3, pady=2)

        # ---- Buttons ----
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="Preview", command=self._rename_preview).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Rename / Copy", command=self._rename_execute).pack(
            side=tk.LEFT, padx=5
        )

        # ---- Preview table ----
        table_frame = ttk.LabelFrame(tab, text="Preview", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        columns = ("type", "original", "new_name")
        self.rename_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=12
        )
        self.rename_tree.heading("type", text="Type")
        self.rename_tree.heading("original", text="Original Name")
        self.rename_tree.heading("new_name", text="New Name")
        self.rename_tree.column("type", width=80, anchor=tk.CENTER)
        self.rename_tree.column("original", width=300)
        self.rename_tree.column("new_name", width=300)

        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.rename_tree.yview
        )
        self.rename_tree.configure(yscrollcommand=scrollbar.set)
        self.rename_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _browse_img_dir(self):
        d = filedialog.askdirectory(title="Select Image Directory")
        if d:
            self.rename_img_dir.set(d)

    def _browse_mask_dir(self):
        d = filedialog.askdirectory(title="Select Mask Directory")
        if d:
            self.rename_mask_dir.set(d)

    def _browse_img_out(self):
        d = filedialog.askdirectory(title="Select Image Output Directory")
        if d:
            self.rename_img_out.set(d)

    def _browse_mask_out(self):
        d = filedialog.askdirectory(title="Select Mask Output Directory")
        if d:
            self.rename_mask_out.set(d)

    def _get_rename_ext(self) -> str | None:
        ext = self.rename_ext.get()
        if ext == "(keep original)":
            return None
        return ext

    def _rename_preview(self):
        """Dry-run rename and populate the preview table."""
        self.rename_tree.delete(*self.rename_tree.get_children())

        img_dir = self.rename_img_dir.get()
        mask_dir = self.rename_mask_dir.get()
        if not img_dir or not mask_dir:
            messagebox.showwarning("Missing", "Select both image and mask directories.")
            return

        prefix = self.rename_prefix.get()
        ext = self._get_rename_ext()

        img_renames, mask_renames = rename_image_mask_pair(
            image_dir=img_dir,
            mask_dir=mask_dir,
            prefix=prefix,
            target_extension=ext,
            dry_run=True,
            copy_mode=self.rename_copy_mode.get(),
            image_output_dir=self.rename_img_out.get() if self.rename_copy_mode.get() else None,
            mask_output_dir=self.rename_mask_out.get() if self.rename_copy_mode.get() else None,
        )

        for old, new in img_renames:
            self.rename_tree.insert("", tk.END, values=("Image", old, new))
        for old, new in mask_renames:
            self.rename_tree.insert("", tk.END, values=("Mask", old, new))

        total = len(img_renames) + len(mask_renames)
        if total == 0:
            messagebox.showinfo("Empty", "No image files found in the selected directories.")
        else:
            logger.info(f"Preview: {len(img_renames)} images, {len(mask_renames)} masks")

    def _rename_execute(self):
        """Execute the rename/copy operation."""
        img_dir = self.rename_img_dir.get()
        mask_dir = self.rename_mask_dir.get()
        if not img_dir or not mask_dir:
            messagebox.showwarning("Missing", "Select both image and mask directories.")
            return

        action = "copy" if self.rename_copy_mode.get() else "rename"
        count_img = len(list_image_files(img_dir))
        count_mask = len(list_image_files(mask_dir))

        if not messagebox.askyesno(
            "Confirm",
            f"This will {action} {count_img} images and {count_mask} masks.\n\nProceed?",
        ):
            return

        prefix = self.rename_prefix.get()
        ext = self._get_rename_ext()

        try:
            img_renames, mask_renames = rename_image_mask_pair(
                image_dir=img_dir,
                mask_dir=mask_dir,
                prefix=prefix,
                target_extension=ext,
                dry_run=False,
                copy_mode=self.rename_copy_mode.get(),
                image_output_dir=self.rename_img_out.get() if self.rename_copy_mode.get() else None,
                mask_output_dir=self.rename_mask_out.get() if self.rename_copy_mode.get() else None,
            )
            messagebox.showinfo(
                "Done",
                f"Successfully processed {len(img_renames)} images and "
                f"{len(mask_renames)} masks.",
            )
            # Refresh preview
            self._rename_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logger.exception("Rename failed")

    # ==================================================================
    # TAB 2: CONFIGURATION
    # ==================================================================
    def _build_config_tab(self):
        tab = self.tab_config

        # Config file selector
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Config File:").pack(side=tk.LEFT)
        self.cfg_path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.cfg_path_var, width=50).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top, text="Open...", command=self._open_config).pack(side=tk.LEFT)
        ttk.Button(top, text="Save", command=self._save_config).pack(
            side=tk.LEFT, padx=5
        )

        # Preset buttons
        preset_frame = ttk.Frame(tab)
        preset_frame.pack(fill=tk.X, padx=10)
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        ttk.Button(
            preset_frame,
            text="DIC Whole-Cell",
            command=lambda: self._load_preset("dic"),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            preset_frame,
            text="Fluorescence Nucleus",
            command=lambda: self._load_preset("fluor"),
        ).pack(side=tk.LEFT, padx=5)

        # ---- Model selection ----
        model_frame = ttk.LabelFrame(tab, text="Model", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        ttk.Label(model_frame, text="Pretrained Model:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.model_source_var = tk.StringVar(value="cpsam (default)")
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_source_var,
            values=["cpsam (default)", "Custom model...", "BioImage.io model..."],
            width=25,
            state="readonly",
        )
        model_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_source_changed)

        self.custom_model_path_var = tk.StringVar()
        self.custom_model_entry = ttk.Entry(
            model_frame, textvariable=self.custom_model_path_var, width=40
        )
        self.custom_model_entry.grid(row=0, column=2, padx=5, pady=2)
        self.custom_model_entry.config(state=tk.DISABLED)

        self.custom_model_btn = ttk.Button(
            model_frame, text="Browse...", command=self._browse_custom_model
        )
        self.custom_model_btn.grid(row=0, column=3, pady=2)
        self.custom_model_btn.config(state=tk.DISABLED)

        self.model_info_var = tk.StringVar(
            value="Using built-in Cellpose-SAM (cpsam) model"
        )
        ttk.Label(
            model_frame, textvariable=self.model_info_var, foreground="gray"
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(2, 0))

        # Scrolled parameter editor
        param_frame = ttk.LabelFrame(tab, text="Parameters", padding=10)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(param_frame)
        scrollbar = ttk.Scrollbar(param_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.param_inner = ttk.Frame(canvas)

        self.param_inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.param_inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store param widgets for reading back
        self.param_vars: dict[str, tk.Variable] = {}

    def _open_config(self):
        path = filedialog.askopenfilename(
            title="Open Configuration",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
            initialdir=str(PROJECT_ROOT / "configs"),
        )
        if path:
            self._load_config_file(path)

    def _load_preset(self, task: str):
        presets = {
            "dic": PROJECT_ROOT / "configs" / "dic_wholecell.yaml",
            "fluor": PROJECT_ROOT / "configs" / "fluor_nucleus.yaml",
        }
        path = presets.get(task)
        if path and path.exists():
            self._load_config_file(str(path))
        else:
            messagebox.showerror("Error", f"Preset config not found: {path}")

    def _load_config_file(self, path: str):
        try:
            with open(path) as f:
                self.yaml_config = yaml.safe_load(f)
            self.yaml_config_path = path
            self.cfg_path_var.set(path)
            self._populate_param_editor()
            self._sync_model_selector_from_config()
            logger.info(f"Loaded config: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def _on_model_source_changed(self, event=None):
        """Toggle custom model path entry based on dropdown selection."""
        choice = self.model_source_var.get()
        if choice == "Custom model...":
            self.custom_model_entry.config(state=tk.NORMAL)
            self.custom_model_btn.config(state=tk.NORMAL)
            self.custom_model_path_var.set("")
            self.model_info_var.set("Select a custom model file below")
        elif choice == "BioImage.io model...":
            self.custom_model_entry.config(state=tk.NORMAL)
            self.custom_model_btn.config(state=tk.NORMAL)
            self.custom_model_path_var.set("bioimage.io:")
            self.model_info_var.set(
                "Enter a BioImage.io resource ID (e.g. bioimage.io:affable-shark) "
                "or browse for a rdf.yaml / .zip file"
            )
        else:
            self.custom_model_entry.config(state=tk.DISABLED)
            self.custom_model_btn.config(state=tk.DISABLED)
            self.custom_model_path_var.set("")
            self.model_info_var.set("Using built-in Cellpose-SAM (cpsam) model")
            # Update config to use default
            if self.yaml_config and "MODEL" in self.yaml_config:
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = None

    def _browse_custom_model(self):
        """Browse for a custom Cellpose or BioImage.io model file."""
        if self.model_source_var.get() == "BioImage.io model...":
            path = filedialog.askopenfilename(
                title="Select BioImage.io Model (rdf.yaml / .zip)",
                initialdir=str(PROJECT_ROOT / "models"),
                filetypes=[
                    ("BioImage.io", "*.yaml *.yml *.zip"),
                    ("All", "*.*"),
                ],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select Cellpose Model",
                initialdir=str(PROJECT_ROOT / "models"),
            )
        if path:
            self.custom_model_path_var.set(path)
            model_name = Path(path).name
            self.model_info_var.set(f"Custom model: {model_name}")
            # Update config
            if self.yaml_config:
                if "MODEL" not in self.yaml_config:
                    self.yaml_config["MODEL"] = {}
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = path
            logger.info(f"Custom model selected: {path}")

    def _sync_model_selector_from_config(self):
        """Update model selector widgets to reflect the loaded config."""
        if not self.yaml_config:
            return
        model_cfg = self.yaml_config.get("MODEL", {})
        pretrained = model_cfg.get("PRETRAINED_MODEL")
        if pretrained:
            pretrained_str = str(pretrained)
            is_bioimage = (
                pretrained_str.startswith("bioimage.io:")
                or pretrained_str.endswith((".yaml", ".yml", ".zip"))
            )
            if is_bioimage:
                self.model_source_var.set("BioImage.io model...")
                self.model_info_var.set(f"BioImage.io model: {pretrained_str}")
            else:
                self.model_source_var.set("Custom model...")
                self.model_info_var.set(f"Custom model: {Path(pretrained_str).name}")
            self.custom_model_path_var.set(pretrained_str)
            self.custom_model_entry.config(state=tk.NORMAL)
            self.custom_model_btn.config(state=tk.NORMAL)
        else:
            self.model_source_var.set("cpsam (default)")
            self.custom_model_path_var.set("")
            self.custom_model_entry.config(state=tk.DISABLED)
            self.custom_model_btn.config(state=tk.DISABLED)
            self.model_info_var.set("Using built-in Cellpose-SAM (cpsam) model")

    def _populate_param_editor(self):
        """Build parameter widgets from the loaded config."""
        # Clear old widgets
        for w in self.param_inner.winfo_children():
            w.destroy()
        self.param_vars.clear()

        row = 0
        row = self._add_section("TRAIN", row, [
            ("EPOCHS", "Epochs", "int"),
            ("LEARNING_RATE", "Learning Rate", "float"),
            ("WEIGHT_DECAY", "Weight Decay", "float"),
            ("BATCH_SIZE", "Batch Size", "int"),
            ("SAVE_EVERY", "Save Every N Epochs", "int"),
            ("MIN_TRAIN_MASKS", "Min Train Masks", "int"),
        ])

        row = self._add_section("DATA", row, [
            ("CHANNELS", "Channels [segment, nuclear]  (0=DIC, 1=mEGFP, 2=mScarlet, 3=miRFPnano3)", "str"),
            ("Z_SLICE", "Z-Slice (null=first, or 0-indexed integer)", "str"),
        ])

        row = self._add_section("INFERENCE", row, [
            ("DIAMETER", "Diameter (null=auto)", "str"),
            ("FLOW_THRESHOLD", "Flow Threshold", "float"),
            ("CELLPROB_THRESHOLD", "Cell Prob Threshold", "float"),
        ])

        row = self._add_section("AUGMENTATION", row, [
            ("ENABLE", "Enable Augmentation", "bool"),
            ("RANDOM_FLIP", "Random Flip", "bool"),
        ])

        row = self._add_section("PATHS", row, [
            ("MODEL_DIR", "Model Directory", "str"),
            ("RESULT_DIR", "Results Directory", "str"),
            ("MODEL_NAME", "Model Name", "str"),
        ])

    def _add_section(
        self, section: str, start_row: int, fields: list[tuple[str, str, str]]
    ) -> int:
        """Add a section header and its parameter fields."""
        cfg_section = self.yaml_config.get(section, {})

        ttk.Label(
            self.param_inner,
            text=f"--- {section} ---",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=start_row, column=0, columnspan=2, sticky=tk.W, pady=(10, 2))
        row = start_row + 1

        for key, label, dtype in fields:
            val = cfg_section.get(key, "")
            ttk.Label(self.param_inner, text=f"  {label}:").grid(
                row=row, column=0, sticky=tk.W, padx=(10, 5), pady=1
            )

            full_key = f"{section}.{key}"

            if dtype == "bool":
                var = tk.BooleanVar(value=bool(val))
                ttk.Checkbutton(self.param_inner, variable=var).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )
            elif dtype == "int":
                var = tk.StringVar(value=str(val))
                ttk.Entry(self.param_inner, textvariable=var, width=20).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )
            elif dtype == "float":
                var = tk.StringVar(value=str(val))
                ttk.Entry(self.param_inner, textvariable=var, width=20).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(self.param_inner, textvariable=var, width=40).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )

            self.param_vars[full_key] = var
            row += 1

        return row

    def _read_params_into_config(self):
        """Read GUI parameter values back into self.yaml_config."""
        # Sync model selection
        if self.yaml_config:
            if "MODEL" not in self.yaml_config:
                self.yaml_config["MODEL"] = {}
            if self.model_source_var.get() in ("Custom model...", "BioImage.io model..."):
                custom_path = self.custom_model_path_var.get()
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = custom_path if custom_path else None
            else:
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = None

        type_map = {
            "TRAIN.EPOCHS": int,
            "TRAIN.BATCH_SIZE": int,
            "TRAIN.SAVE_EVERY": int,
            "TRAIN.MIN_TRAIN_MASKS": int,
            "TRAIN.LEARNING_RATE": float,
            "TRAIN.WEIGHT_DECAY": float,
            "INFERENCE.FLOW_THRESHOLD": float,
            "INFERENCE.CELLPROB_THRESHOLD": float,
        }

        for full_key, var in self.param_vars.items():
            section, key = full_key.split(".", 1)
            raw = var.get()

            # Parse value
            if full_key in type_map:
                try:
                    val = type_map[full_key](raw)
                except (ValueError, TypeError):
                    val = raw
            elif isinstance(var, tk.BooleanVar):
                val = var.get()
            elif raw.lower() in ("null", "none", ""):
                val = None
            elif raw.startswith("["):
                # Parse list like [0, 0]
                try:
                    val = yaml.safe_load(raw)
                except yaml.YAMLError:
                    val = raw
            else:
                val = raw

            if section not in self.yaml_config:
                self.yaml_config[section] = {}
            self.yaml_config[section][key] = val

    def _save_config(self):
        if not self.yaml_config:
            messagebox.showwarning("No Config", "Load a config first.")
            return

        self._read_params_into_config()

        path = self.yaml_config_path or filedialog.asksaveasfilename(
            title="Save Configuration",
            filetypes=[("YAML", "*.yaml")],
            initialdir=str(PROJECT_ROOT / "configs"),
            defaultextension=".yaml",
        )
        if not path:
            return

        try:
            with open(path, "w") as f:
                yaml.dump(self.yaml_config, f, default_flow_style=False, sort_keys=False)
            self.yaml_config_path = path
            self.cfg_path_var.set(path)
            logger.info(f"Config saved to {path}")
            messagebox.showinfo("Saved", f"Configuration saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    # ==================================================================
    # TAB 3: TRAINING
    # ==================================================================
    def _build_train_tab(self):
        tab = self.tab_train

        # Docker settings (used when running GUI on Windows)
        docker_frame = ttk.LabelFrame(tab, text="Docker (Windows only — leave blank if running inside container)", padding=6)
        docker_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        ttk.Label(docker_frame, text="Container name/ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(docker_frame, textvariable=self.docker_container, width=30).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(docker_frame, text="  Project root in container:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        ttk.Entry(docker_frame, textvariable=self.container_root, width=35).grid(row=0, column=3, sticky=tk.W)

        # Controls
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Task:").pack(side=tk.LEFT)
        self.train_task = tk.StringVar(value="dic")
        task_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.train_task,
            values=["dic", "fluor", "both", "custom"],
            width=12,
            state="readonly",
        )
        task_combo.pack(side=tk.LEFT, padx=5)

        self.btn_train = ttk.Button(
            ctrl_frame, text="Start Training", command=self._start_training
        )
        self.btn_train.pack(side=tk.LEFT, padx=10)

        self.btn_stop = ttk.Button(
            ctrl_frame, text="Stop", command=self._stop_training, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Progress
        self.train_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.train_progress.pack(fill=tk.X, padx=10, pady=5)

        self.train_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.train_status).pack(padx=10, anchor=tk.W)

        # Log output
        log_frame = ttk.LabelFrame(tab, text="Training Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.train_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED, font=("Courier", 9)
        )
        self.train_log.pack(fill=tk.BOTH, expand=True)

        self._training_thread = None
        self._stop_event = threading.Event()

    def _start_training(self):
        task = self.train_task.get()

        if task == "custom" and not self.yaml_config:
            messagebox.showwarning(
                "No Config", "Load a config in the Configuration tab first."
            )
            return

        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.train_progress.start(10)
        self._stop_event.clear()
        self.train_status.set("Training...")

        # Clear log
        self.train_log.config(state=tk.NORMAL)
        self.train_log.delete("1.0", tk.END)
        self.train_log.config(state=tk.DISABLED)

        self._training_thread = threading.Thread(
            target=self._training_worker, args=(task,), daemon=True
        )
        self._training_thread.start()

    def _training_worker(self, task: str):
        try:
            configs_to_run = []
            if task == "dic":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "dic_wholecell.yaml"))
            elif task == "fluor":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "fluor_nucleus.yaml"))
            elif task == "both":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "dic_wholecell.yaml"))
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "fluor_nucleus.yaml"))
            elif task == "custom":
                self._read_params_into_config()
                configs_to_run.append(self.yaml_config_path)

            container = self.docker_container.get().strip()

            for cfg_path in configs_to_run:
                if self._stop_event.is_set():
                    logger.info("Training stopped by user.")
                    break
                logger.info(f"Loading config: {cfg_path}")

                if container:
                    # Running GUI on Windows — delegate to Docker container
                    container_cfg = self._to_container_path(cfg_path)
                    container_script = self.container_root.get().rstrip("/") + "/src/train_cellpose.py"
                    rc = self._run_docker_cmd(container, ["python", container_script, "--config", container_cfg])
                    if rc != 0:
                        raise RuntimeError(f"docker exec exited with code {rc}")
                else:
                    # Running inside container — import directly
                    from train_cellpose import load_config, train_cellpose_model
                    cfg = load_config(cfg_path)
                    train_cellpose_model(cfg)

            self.log_queue.put("__TRAINING_DONE__")
        except Exception as e:
            logger.exception("Training failed")
            self.log_queue.put(f"__TRAINING_ERROR__{e}")

    def _stop_training(self):
        self._stop_event.set()
        self.train_status.set("Stopping...")

    def _on_training_finished(self, error: str | None = None):
        self.train_progress.stop()
        self.btn_train.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if error:
            self.train_status.set(f"Error: {error}")
            messagebox.showerror("Training Error", str(error))
        else:
            self.train_status.set("Training complete")
            messagebox.showinfo("Done", "Training finished successfully.")

    # ==================================================================
    # TAB 4: EVALUATION
    # ==================================================================
    def _build_eval_tab(self):
        tab = self.tab_eval

        ctrl_frame = ttk.LabelFrame(tab, text="Evaluation Settings", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        # Model path
        ttk.Label(ctrl_frame, text="Trained Model:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.eval_model_path = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.eval_model_path, width=50).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(
            ctrl_frame, text="Browse...", command=self._browse_eval_model
        ).grid(row=0, column=2, pady=2)

        # Image dir for inference
        ttk.Label(ctrl_frame, text="Image Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.eval_img_dir = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.eval_img_dir, width=50).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(
            ctrl_frame, text="Browse...", command=self._browse_eval_img_dir
        ).grid(row=1, column=2, pady=2)

        # Output dir
        ttk.Label(ctrl_frame, text="Output Directory:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.eval_output_dir = tk.StringVar(
            value=str(PROJECT_ROOT / "results" / "inference")
        )
        ttk.Entry(ctrl_frame, textvariable=self.eval_output_dir, width=50).grid(
            row=2, column=1, padx=5, pady=2
        )
        ttk.Button(
            ctrl_frame, text="Browse...", command=self._browse_eval_out_dir
        ).grid(row=2, column=2, pady=2)

        # Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_eval = ttk.Button(
            btn_frame, text="Run Inference", command=self._start_eval
        )
        self.btn_eval.pack(side=tk.LEFT, padx=5)

        self.btn_eval_full = ttk.Button(
            btn_frame,
            text="Full Evaluation (with metrics)",
            command=self._start_full_eval,
        )
        self.btn_eval_full.pack(side=tk.LEFT, padx=5)

        self.eval_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.eval_progress.pack(fill=tk.X, padx=10, pady=5)

        self.eval_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.eval_status).pack(padx=10, anchor=tk.W)

        # Results display
        results_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.eval_results = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=("Courier", 9)
        )
        self.eval_results.pack(fill=tk.BOTH, expand=True)

    def _browse_eval_model(self):
        path = filedialog.askopenfilename(
            title="Select Trained Model",
            initialdir=str(PROJECT_ROOT / "models"),
        )
        if path:
            self.eval_model_path.set(path)

    def _browse_eval_img_dir(self):
        d = filedialog.askdirectory(title="Select Image Directory")
        if d:
            self.eval_img_dir.set(d)

    def _browse_eval_out_dir(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.eval_output_dir.set(d)

    def _start_eval(self):
        model = self.eval_model_path.get()
        img_dir = self.eval_img_dir.get()
        output = self.eval_output_dir.get()

        if not model or not img_dir:
            messagebox.showwarning("Missing", "Select a model and image directory.")
            return

        self.btn_eval.config(state=tk.DISABLED)
        self.eval_progress.start(10)
        self.eval_status.set("Running inference...")

        def worker():
            try:
                container = self.docker_container.get().strip()
                if container:
                    container_script = self.container_root.get().rstrip("/") + "/src/evaluate.py"
                    rc = self._run_docker_cmd(container, [
                        "python", container_script,
                        "--model", self._to_container_path(model),
                        "--image-dir", self._to_container_path(img_dir),
                        "--output-dir", self._to_container_path(output),
                    ])
                    if rc != 0:
                        raise RuntimeError(f"docker exec exited with code {rc}")
                    self.log_queue.put("__EVAL_DONE__Inference complete")
                else:
                    from evaluate import run_inference
                    masks, names = run_inference(
                        model_path=model,
                        image_dir=img_dir,
                        output_dir=output,
                    )
                    self.log_queue.put(f"__EVAL_DONE__Inference complete: {len(masks)} images processed")
            except Exception as e:
                logger.exception("Inference failed")
                self.log_queue.put(f"__EVAL_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _start_full_eval(self):
        """Run full evaluation using the loaded config."""
        model = self.eval_model_path.get()
        if not model:
            messagebox.showwarning("Missing", "Select a trained model.")
            return
        if not self.yaml_config:
            messagebox.showwarning(
                "No Config", "Load a config in the Configuration tab first."
            )
            return

        self.btn_eval_full.config(state=tk.DISABLED)
        self.eval_progress.start(10)
        self.eval_status.set("Running full evaluation...")

        def worker():
            try:
                container = self.docker_container.get().strip()
                if container:
                    if not self.yaml_config_path:
                        raise RuntimeError("Save the config file first (File → Save Config) before running full evaluation via Docker.")
                    container_script = self.container_root.get().rstrip("/") + "/src/evaluate.py"
                    rc = self._run_docker_cmd(container, [
                        "python", container_script,
                        "--config", self._to_container_path(self.yaml_config_path),
                        "--model", self._to_container_path(model),
                    ])
                    if rc != 0:
                        raise RuntimeError(f"docker exec exited with code {rc}")
                    self.log_queue.put("__EVAL_DONE__Full evaluation complete (see log for metrics)")
                else:
                    from evaluate import evaluate_model
                    self._read_params_into_config()
                    results = evaluate_model(self.yaml_config, model)
                    msg = "Full Evaluation Results:\n\n"
                    for k, v in results.items():
                        if not isinstance(v, (list, dict)):
                            msg += f"  {k}: {v}\n"
                    self.log_queue.put(f"__EVAL_DONE__{msg}")
            except Exception as e:
                logger.exception("Evaluation failed")
                self.log_queue.put(f"__EVAL_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_eval_finished(self, message: str | None = None, error: str | None = None):
        self.eval_progress.stop()
        self.btn_eval.config(state=tk.NORMAL)
        self.btn_eval_full.config(state=tk.NORMAL)

        if error:
            self.eval_status.set(f"Error: {error}")
            messagebox.showerror("Evaluation Error", str(error))
        else:
            self.eval_status.set("Evaluation complete")
            if message:
                self.eval_results.config(state=tk.NORMAL)
                self.eval_results.delete("1.0", tk.END)
                self.eval_results.insert(tk.END, message)
                self.eval_results.config(state=tk.DISABLED)

    # ==================================================================
    # TAB 5: DETECTRON2 CELL SEGMENTATION TRAINER
    # ==================================================================
    def _build_detectron2_tab(self):
        """Build the Detectron2 cellseg_trainer tab with sub-sections for
        dataset conversion, training, inference, and active learning."""
        tab = self.tab_detectron2

        # Scrollable canvas so the tab content doesn't get clipped
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        PAD = {"padx": 8, "pady": 4}

        # ----------------------------------------------------------------
        # Section 0a: Docker connection
        # ----------------------------------------------------------------
        docker_frame = ttk.LabelFrame(scroll_frame, text="Docker Connection (required on Windows)", padding=8)
        docker_frame.pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(docker_frame, text="Container name/ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(docker_frame, textvariable=self.docker_container, width=25).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(docker_frame, text="  Python (venv):").grid(row=0, column=2, sticky=tk.W, padx=(12, 4))
        ttk.Entry(docker_frame, textvariable=self.d2_docker_python, width=32).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(docker_frame, text="  Container project root:").grid(row=0, column=4, sticky=tk.W, padx=(12, 4))
        ttk.Entry(docker_frame, textvariable=self.container_root, width=30).grid(row=0, column=5, sticky=tk.W)

        ttk.Label(docker_frame, text="Host data folder (Windows):").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(docker_frame, textvariable=self.host_mount_path, width=40).grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=(6, 0))
        ttk.Label(docker_frame, text="  → Container folder:").grid(row=1, column=3, sticky=tk.W, padx=(12, 4), pady=(6, 0))
        ttk.Entry(docker_frame, textvariable=self.container_mount_path, width=20).grid(row=1, column=4, sticky=tk.W, pady=(6, 0))

        ttk.Label(
            docker_frame,
            text="Bind mount: set Host data folder to the Windows path you mounted (e.g. C:\\Users\\Windows\\Documents\\Segmentation)  "
                 "and Container folder to where it appears in Docker (e.g. /workspace).  "
                 "Get container name with  docker ps.",
            foreground="gray",
            wraplength=900,
            justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=6, sticky=tk.W, pady=(4, 0))

        # ----------------------------------------------------------------
        # Section 0b: Dependency status
        # ----------------------------------------------------------------
        dep_frame = ttk.LabelFrame(scroll_frame, text="Dependencies", padding=8)
        dep_frame.pack(fill=tk.X, padx=10, pady=6)

        self._d2_dep_labels: dict[str, tk.Label] = {}
        deps = ["torch", "torchvision", "detectron2", "cv2", "pycocotools", "tifffile"]
        dep_row = ttk.Frame(dep_frame)
        dep_row.pack(fill=tk.X)
        for i, dep in enumerate(deps):
            ttk.Label(dep_row, text=f"{dep}:").grid(row=0, column=i * 2, padx=(8, 2), sticky=tk.E)
            lbl = tk.Label(dep_row, text="…", width=4, font=("Courier", 10, "bold"))
            lbl.grid(row=0, column=i * 2 + 1, padx=(0, 8))
            self._d2_dep_labels[dep] = lbl

        btn_row = ttk.Frame(dep_frame)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_row, text="Check Dependencies", command=self._d2_check_deps).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Show Install Guide", command=self._d2_show_install_guide).pack(side=tk.LEFT, padx=4)

        # Run check automatically after window is ready
        self.after(500, self._d2_check_deps)

        # ----------------------------------------------------------------
        # Section 1: Paths
        # ----------------------------------------------------------------
        path_frame = ttk.LabelFrame(scroll_frame, text="Paths", padding=10)
        path_frame.pack(fill=tk.X, padx=10, pady=6)

        self.d2_image_dir     = tk.StringVar()
        self.d2_mask_dir      = tk.StringVar()
        self.d2_dataset_dir   = tk.StringVar(value=str(PROJECT_ROOT / "dataset"))
        self.d2_config_path   = tk.StringVar()
        self.d2_weights_path  = tk.StringVar()
        self.d2_output_dir    = tk.StringVar(value=str(PROJECT_ROOT / "output_detectron2"))
        self.d2_infer_img_dir = tk.StringVar()
        self.d2_infer_out_dir = tk.StringVar(value=str(PROJECT_ROOT / "predictions"))
        self.d2_unlabelled_dir = tk.StringVar()

        rows = [
            ("Image Directory:",          self.d2_image_dir,     "dir"),
            ("Mask Directory:",           self.d2_mask_dir,      "dir"),
            ("Dataset Output Dir:",       self.d2_dataset_dir,   "dir"),
            ("Detectron2 Config (YAML):", self.d2_config_path,   "file"),
            ("Weights (.pt/.pth/.pkl):",   self.d2_weights_path,  "file"),
            ("Training Output Dir:",      self.d2_output_dir,    "dir"),
            ("Inference Image Dir:",      self.d2_infer_img_dir, "dir"),
            ("Inference Output Dir:",     self.d2_infer_out_dir, "dir"),
            ("Unlabelled Images Dir:",    self.d2_unlabelled_dir,"dir"),
        ]
        for r, (label, var, kind) in enumerate(rows):
            ttk.Label(path_frame, text=label).grid(row=r, column=0, sticky=tk.W, **PAD)
            ttk.Entry(path_frame, textvariable=var, width=48).grid(row=r, column=1, padx=4)
            cmd = (lambda v=var, k=kind: self._d2_browse(v, k))
            ttk.Button(path_frame, text="Browse…", command=cmd).grid(row=r, column=2, padx=4)

        # ----------------------------------------------------------------
        # Section 2: Conversion options
        # ----------------------------------------------------------------
        conv_frame = ttk.LabelFrame(scroll_frame, text="Dataset Conversion", padding=10)
        conv_frame.pack(fill=tk.X, padx=10, pady=6)

        self.d2_mask_suffix   = tk.StringVar(value="_seg.npy")
        self.d2_train_split   = tk.DoubleVar(value=0.85)
        self.d2_patch_size    = tk.IntVar(value=512)
        self.d2_patch_overlap = tk.IntVar(value=64)
        self.d2_conv_workers  = tk.IntVar(value=4)

        conv_opts = [
            ("Mask suffix:",    self.d2_mask_suffix,   "entry", 12),
            ("Train split:",    self.d2_train_split,   "entry", 6),
            ("Patch size (0=off):", self.d2_patch_size, "entry", 6),
            ("Patch overlap:",  self.d2_patch_overlap, "entry", 6),
            ("Workers:",        self.d2_conv_workers,  "entry", 4),
        ]
        for c, (label, var, _, w) in enumerate(conv_opts):
            ttk.Label(conv_frame, text=label).grid(row=0, column=c * 2, sticky=tk.E, padx=4)
            ttk.Entry(conv_frame, textvariable=var, width=w).grid(row=0, column=c * 2 + 1, padx=2)

        ttk.Button(
            conv_frame, text="Convert Dataset → COCO",
            command=self._d2_run_convert,
        ).grid(row=1, column=0, columnspan=10, pady=6)

        # ----------------------------------------------------------------
        # Section 3: Training options
        # ----------------------------------------------------------------
        train_frame = ttk.LabelFrame(scroll_frame, text="Training", padding=10)
        train_frame.pack(fill=tk.X, padx=10, pady=6)

        self.d2_max_iter       = tk.IntVar(value=5000)
        self.d2_base_lr        = tk.DoubleVar(value=0.00025)
        self.d2_num_classes    = tk.IntVar(value=1)
        self.d2_num_gpus       = tk.IntVar(value=2)
        self.d2_multi_gpu      = tk.BooleanVar(value=True)
        self.d2_amp            = tk.BooleanVar(value=True)
        self.d2_freeze_backbone= tk.BooleanVar(value=False)
        self.d2_fast_finetune  = tk.BooleanVar(value=False)
        self.d2_resume         = tk.BooleanVar(value=False)

        r0 = [
            ("Max iter:",    self.d2_max_iter,    5),
            ("Base LR:",     self.d2_base_lr,     8),
            ("Num classes:", self.d2_num_classes, 4),
            ("Num GPUs:",    self.d2_num_gpus,    3),
        ]
        for c, (lbl, var, w) in enumerate(r0):
            ttk.Label(train_frame, text=lbl).grid(row=0, column=c * 2, sticky=tk.E, padx=4)
            ttk.Entry(train_frame, textvariable=var, width=w).grid(row=0, column=c * 2 + 1, padx=2)

        chk_row = ttk.Frame(train_frame)
        chk_row.grid(row=1, column=0, columnspan=8, sticky=tk.W, pady=4)
        for lbl, var in [
            ("Multi-GPU", self.d2_multi_gpu),
            ("AMP",       self.d2_amp),
            ("Freeze backbone", self.d2_freeze_backbone),
            ("Fast fine-tune", self.d2_fast_finetune),
            ("Resume",    self.d2_resume),
        ]:
            ttk.Checkbutton(chk_row, text=lbl, variable=var).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            train_frame, text="Start Training",
            command=self._d2_run_train,
        ).grid(row=2, column=0, columnspan=8, pady=6)

        # ----------------------------------------------------------------
        # Section 4: Inference options
        # ----------------------------------------------------------------
        inf_frame = ttk.LabelFrame(scroll_frame, text="Inference", padding=10)
        inf_frame.pack(fill=tk.X, padx=10, pady=6)

        self.d2_score_thresh  = tk.DoubleVar(value=0.5)
        self.d2_device        = tk.StringVar(value="cuda:0")
        self.d2_out_masks     = tk.BooleanVar(value=True)
        self.d2_out_overlay   = tk.BooleanVar(value=True)
        self.d2_out_json      = tk.BooleanVar(value=True)

        inf_r0 = ttk.Frame(inf_frame)
        inf_r0.pack(fill=tk.X, pady=2)
        for lbl, var, w in [("Score thresh:", self.d2_score_thresh, 6), ("Device:", self.d2_device, 10)]:
            ttk.Label(inf_r0, text=lbl).pack(side=tk.LEFT, padx=4)
            ttk.Entry(inf_r0, textvariable=var, width=w).pack(side=tk.LEFT, padx=2)

        inf_r1 = ttk.Frame(inf_frame)
        inf_r1.pack(fill=tk.X, pady=2)
        for lbl, var in [("Save masks", self.d2_out_masks), ("Save overlays", self.d2_out_overlay), ("Save JSON", self.d2_out_json)]:
            ttk.Checkbutton(inf_r1, text=lbl, variable=var).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            inf_frame, text="Run Inference",
            command=self._d2_run_inference,
        ).pack(pady=6)

        # ----------------------------------------------------------------
        # Section 5: Active Learning / Self-Training
        # ----------------------------------------------------------------
        al_frame = ttk.LabelFrame(scroll_frame, text="Active Learning / Self-Training", padding=10)
        al_frame.pack(fill=tk.X, padx=10, pady=6)

        self.d2_pseudo_thresh   = tk.DoubleVar(value=0.7)
        self.d2_al_iterations   = tk.IntVar(value=3)
        self.d2_suggest_only    = tk.BooleanVar(value=False)

        al_r0 = ttk.Frame(al_frame)
        al_r0.pack(fill=tk.X, pady=2)
        for lbl, var, w in [
            ("Pseudo-label threshold:", self.d2_pseudo_thresh, 6),
            ("Iterations:",             self.d2_al_iterations, 4),
        ]:
            ttk.Label(al_r0, text=lbl).pack(side=tk.LEFT, padx=4)
            ttk.Entry(al_r0, textvariable=var, width=w).pack(side=tk.LEFT, padx=2)

        ttk.Checkbutton(al_frame, text="Suggest only (no retraining)", variable=self.d2_suggest_only).pack(anchor=tk.W, padx=4)

        ttk.Button(
            al_frame, text="Run Self-Training",
            command=self._d2_run_self_train,
        ).pack(pady=6)

        # ----------------------------------------------------------------
        # Section 6: Log / Progress
        # ----------------------------------------------------------------
        log_frame = ttk.LabelFrame(scroll_frame, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.d2_log = scrolledtext.ScrolledText(log_frame, height=14, state=tk.DISABLED,
                                                 font=("Courier", 9))
        self.d2_log.pack(fill=tk.BOTH, expand=True)

        self.d2_progress = ttk.Progressbar(log_frame, mode="indeterminate", length=400)
        self.d2_progress.pack(fill=tk.X, pady=4)

        self.d2_status = tk.StringVar(value="Ready")
        ttk.Label(log_frame, textvariable=self.d2_status, foreground="navy").pack(anchor=tk.W)

        ttk.Button(log_frame, text="Clear Log", command=self._d2_clear_log).pack(anchor=tk.E, pady=2)

    # ------------------------------------------------------------------
    # Detectron2 tab helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Dependency checking
    # ------------------------------------------------------------------
    def _d2_check_deps(self) -> dict[str, bool]:
        """Import-check each required package and update status labels.

        When a Docker container is configured the check runs inside it using
        the configured venv Python.  Otherwise it checks the local environment.

        Returns:
            Dict mapping package name to availability bool.
        """
        deps = ["torch", "torchvision", "detectron2", "cv2", "pycocotools", "tifffile"]
        status: dict[str, bool] = {}

        container = self.docker_container.get().strip()
        python = self.d2_docker_python.get().strip() if container else None

        first_error: str | None = None  # capture first docker error for diagnosis

        for dep in deps:
            if container and python:
                try:
                    result = subprocess.run(
                        ["docker", "exec", container, python, "-c", f"import {dep}"],
                        capture_output=True, timeout=15, text=True,
                    )
                    ok = result.returncode == 0
                    if not ok and first_error is None:
                        err = (result.stderr or result.stdout or "").strip()
                        first_error = err if err else f"exit code {result.returncode}"
                except FileNotFoundError:
                    ok = False
                    first_error = first_error or "'docker' command not found — is Docker Desktop running and in PATH?"
                except Exception as exc:
                    ok = False
                    first_error = first_error or str(exc)
            else:
                import importlib
                try:
                    importlib.import_module(dep)
                    ok = True
                except ImportError:
                    ok = False

            status[dep] = ok
            lbl = self._d2_dep_labels.get(dep)
            if lbl:
                lbl.config(text="✓" if ok else "✗", fg="green" if ok else "red")

        missing = [k for k, v in status.items() if not v]
        src = f"Docker ({container})" if container else "local environment"
        if missing:
            self._d2_log_msg(f"[DEPS] Failed in {src}: {', '.join(missing)}")
            if first_error:
                self._d2_log_msg(f"[DEPS] Error detail: {first_error}")
                self._d2_log_msg("[DEPS] → Check container name (docker ps) and venv Python path.")
        else:
            self._d2_log_msg(f"[DEPS] All dependencies found in {src} ✓")
        return status

    def _d2_fix_numpy(self, container: str, python: str) -> None:
        """Downgrade numpy to <2.0 inside the container if a 2.x version is found."""
        check = subprocess.run(
            ["docker", "exec", container, python, "-c",
             "import numpy as np, sys; sys.exit(0 if tuple(int(x) for x in np.__version__.split('.')[:2]) < (2, 0) else 1)"],
            capture_output=True, timeout=30,
        )
        if check.returncode == 0:
            return  # already <2.0, nothing to do
        self._d2_log_msg("[DEPS] NumPy ≥2.0 detected in container — downgrading to <2.0 …")
        pip = python.replace("python3", "pip3").replace("python", "pip")
        rc = self._run_docker_cmd(container, [pip, "install", "numpy<2.0", "--quiet"])
        if rc == 0:
            self._d2_log_msg("[DEPS] NumPy downgraded successfully ✓")
        else:
            self._d2_log_msg("[DEPS] WARNING: NumPy downgrade failed — training may crash")

    def _d2_show_install_guide(self) -> None:
        """Open a popup window with step-by-step installation instructions."""
        win = tk.Toplevel(self)
        win.title("Detectron2 Installation Guide")
        win.geometry("720x560")
        win.resizable(True, True)

        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Courier", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        guide = """\
DETECTRON2 INSTALLATION GUIDE
==============================

Detectron2 must be built from source — it is NOT available on PyPI.
Follow the steps below for your environment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Install PyTorch with CUDA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Visit https://pytorch.org/get-started/locally/ and select your OS / CUDA version.

Example for CUDA 11.8:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Example for CUDA 12.1:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Verify:
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Install detectron2 dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pip-based environments:
  pip install opencv-python pycocotools tifffile pyyaml scipy tqdm matplotlib

Conda environments (run inside the container):
  conda install -c conda-forge opencv pycocotools -y
  pip install tifffile pyyaml scipy tqdm matplotlib

If using Docker, run the above inside the container:
  docker exec <container> /opt/conda/bin/python3 -m pip install opencv-python-headless pycocotools tifffile pyyaml scipy tqdm matplotlib

Verify cv2:
  docker exec <container> /opt/conda/bin/python3 -c "import cv2; print(cv2.__version__)"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Install detectron2 from source
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Option A — Build from GitHub (recommended, always up-to-date):
  pip install 'git+https://github.com/facebookresearch/detectron2.git'

Option B — Pre-built wheels (faster, matches specific torch+CUDA):
  # Replace cu118 and torch2.1.0 with your actual versions
  pip install detectron2 -f \\
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1.0/index.html

  Wheel index: https://dl.fbaipublicfiles.com/detectron2/wheels/
  Available: cu117, cu118, cu121 × torch 1.x / 2.x

Option C — Conda (if using conda environment):
  conda install -c conda-forge detectron2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Install cellseg_trainer package
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
From the project root directory:
  pip install -e .

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Verify everything works
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python -c "import detectron2; print('detectron2', detectron2.__version__)"
  python -c "from detectron2.config import get_cfg; print('config OK')"
  python -c "from detectron2.engine import DefaultTrainer; print('trainer OK')"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON ERRORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ModuleNotFoundError: No module named 'detectron2'
  → Follow Step 3 above.

CUDA error / device mismatch
  → Make sure torch and detectron2 were built against the same CUDA version.
  → Run: python -c "import torch; print(torch.version.cuda)"

ImportError: libGL.so.1: cannot open shared object file
  → On headless Linux: sudo apt install libgl1-mesa-glx libglib2.0-0

error: command 'gcc' failed — build error from source
  → Install build tools: sudo apt install build-essential python3-dev
  → Make sure gcc and g++ are available: gcc --version

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOR YOUR 2-GPU (2 × 24 GB) SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommended install command for 2x NVIDIA 24 GB GPUs (check your CUDA version first):

  nvidia-smi   ← note the 'CUDA Version' in the top-right corner

Then install the matching PyTorch + detectron2 combination.

After installation, click 'Check Dependencies' to verify all packages are found.
"""
        text.insert(tk.END, guide)
        text.config(state=tk.DISABLED)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    def _d2_assert_deps_or_warn(self) -> bool:
        """Return True if all required deps are present; show error dialog if not."""
        status = self._d2_check_deps()
        missing = [k for k, v in status.items() if not v]
        if missing:
            messagebox.showerror(
                "Missing Dependencies",
                f"The following packages are not installed:\n\n"
                f"  {', '.join(missing)}\n\n"
                f"Click 'Show Install Guide' in the Detectron2 tab for\n"
                f"step-by-step installation instructions.",
            )
            return False
        return True

    def _d2_browse(self, var: tk.StringVar, kind: str) -> None:
        if kind == "file":
            path = filedialog.askopenfilename()
        else:
            path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _d2_log_msg(self, msg: str) -> None:
        self.d2_log.config(state=tk.NORMAL)
        self.d2_log.insert(tk.END, msg + "\n")
        self.d2_log.see(tk.END)
        self.d2_log.config(state=tk.DISABLED)

    def _d2_clear_log(self) -> None:
        self.d2_log.config(state=tk.NORMAL)
        self.d2_log.delete("1.0", tk.END)
        self.d2_log.config(state=tk.DISABLED)

    def _d2_run_convert(self) -> None:
        """Convert Cellpose masks to COCO dataset in a background thread."""
        image_dir = self.d2_image_dir.get().strip()
        mask_dir  = self.d2_mask_dir.get().strip()
        output_dir = self.d2_dataset_dir.get().strip()

        if not image_dir or not mask_dir or not output_dir:
            messagebox.showerror("Missing paths", "Please set Image, Mask, and Dataset directories.")
            return

        if not self._d2_assert_deps_or_warn():
            return

        self.d2_progress.start()
        self.d2_status.set("Converting dataset …")

        def worker():
            try:
                container = self.docker_container.get().strip()
                if container:
                    python = self.d2_docker_python.get().strip()
                    self._d2_fix_numpy(container, python)
                    cmd = [
                        python, "-m", "cellseg_trainer", "convert",
                        "--images",      self._to_container_path(image_dir),
                        "--masks",       self._to_container_path(mask_dir),
                        "--output",      self._to_container_path(output_dir),
                        "--mask-suffix", self.d2_mask_suffix.get().strip(),
                        "--train-split", str(self.d2_train_split.get()),
                        "--workers",     str(self.d2_conv_workers.get()),
                        "--patch-size",  str(self.d2_patch_size.get()),
                        "--overlap",     str(self.d2_patch_overlap.get()),
                    ]
                    rc = self._run_docker_cmd(container, cmd)
                    if rc != 0:
                        raise RuntimeError(f"docker exec exited with code {rc}")
                    self.log_queue.put("__D2_DONE__Dataset conversion complete")
                else:
                    sys.path.insert(0, str(PROJECT_ROOT))
                    from cellseg_trainer.coco_builder import build_coco_dataset
                    from cellseg_trainer.patch_extractor import extract_and_save_patches

                    img_dir_path = Path(image_dir)
                    msk_dir_path = Path(mask_dir)
                    out_path = Path(output_dir)
                    mask_suffix = self.d2_mask_suffix.get().strip()
                    patch_size = self.d2_patch_size.get()
                    overlap = self.d2_patch_overlap.get()

                    if patch_size > 0:
                        patch_dir = out_path / "patches"
                        n = extract_and_save_patches(
                            image_dir=img_dir_path, mask_dir=msk_dir_path,
                            output_dir=patch_dir, patch_size=patch_size,
                            overlap=overlap, mask_suffix=mask_suffix,
                        )
                        self.log_queue.put(f"Extracted {n} patches")
                        img_dir_path = patch_dir / "images"
                        msk_dir_path = patch_dir / "masks"
                        mask_suffix = "_mask.tif"

                    train_j, val_j = build_coco_dataset(
                        image_dir=img_dir_path, mask_dir=msk_dir_path,
                        output_dir=out_path,
                        train_split=self.d2_train_split.get(),
                        mask_suffix=mask_suffix,
                        n_workers=self.d2_conv_workers.get(),
                    )
                    self.log_queue.put(f"__D2_DONE__Dataset ready\n  Train: {train_j}\n  Val:   {val_j}")
            except Exception as exc:
                self.log_queue.put(f"__D2_ERROR__{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _d2_run_train(self) -> None:
        """Launch Detectron2 training in a background thread."""
        d2_config = self.d2_config_path.get().strip()
        dataset   = self.d2_dataset_dir.get().strip()
        output    = self.d2_output_dir.get().strip()

        if not d2_config or not dataset or not output:
            messagebox.showerror("Missing paths", "Detectron2 Config, Dataset Dir and Output Dir are required.")
            return

        # Validate that the config file is a real Detectron2 YAML, not a BioImage RDF
        try:
            from cellseg_trainer.training import _assert_detectron2_yaml
            _assert_detectron2_yaml(d2_config)
        except Exception as exc:
            messagebox.showerror("Invalid Detectron2 Config", str(exc))
            return

        if not self._d2_assert_deps_or_warn():
            return

        self.d2_progress.start()
        self.d2_status.set("Training …")

        def worker():
            try:
                container = self.docker_container.get().strip()
                weights = self.d2_weights_path.get().strip()
                if container:
                    python = self.d2_docker_python.get().strip()
                    self._d2_fix_numpy(container, python)
                    cmd = [
                        python, "-m", "cellseg_trainer", "train",
                        "--dataset",    self._to_container_path(dataset),
                        "--d2-config",  self._to_container_path(d2_config),
                        "--output",     self._to_container_path(output),
                        "--max-iter",   str(self.d2_max_iter.get()),
                        "--lr",         str(self.d2_base_lr.get()),
                        "--num-classes",str(self.d2_num_classes.get()),
                        "--num-gpus",   str(self.d2_num_gpus.get()),
                        "--patch-size", str(self.d2_patch_size.get() or 512),
                    ]
                    if weights:
                        cmd += ["--weights", self._to_container_path(weights)]
                    cmd.append("--multi-gpu" if self.d2_multi_gpu.get() else "--no-multi-gpu")
                    cmd.append("--amp" if self.d2_amp.get() else "--no-amp")
                    if self.d2_freeze_backbone.get():
                        cmd.append("--freeze-backbone")
                    if self.d2_fast_finetune.get():
                        cmd.append("--fast-finetune")
                    if self.d2_resume.get():
                        cmd.append("--resume")
                    rc = self._run_docker_cmd(container, cmd)
                    if rc != 0:
                        raise RuntimeError(f"docker exec exited with code {rc}")
                    self.log_queue.put(f"__D2_DONE__Training complete → {self._to_container_path(output)}")
                else:
                    sys.path.insert(0, str(PROJECT_ROOT))
                    from cellseg_trainer.config_loader import DEFAULTS, override_from_args
                    from cellseg_trainer.training import train

                    cfg = dict(DEFAULTS)
                    cfg = override_from_args(
                        cfg,
                        **{
                            "TRAIN.MAX_ITER":        self.d2_max_iter.get(),
                            "TRAIN.BASE_LR":         self.d2_base_lr.get(),
                            "TRAIN.MULTI_GPU":       self.d2_multi_gpu.get(),
                            "TRAIN.AMP":             self.d2_amp.get(),
                            "TRAIN.FAST_FINETUNE":   self.d2_fast_finetune.get(),
                            "MODEL.FREEZE_BACKBONE": self.d2_freeze_backbone.get(),
                            "MODEL.WEIGHTS":         weights if weights else None,
                            "MODEL.NUM_CLASSES":     self.d2_num_classes.get(),
                            "DATASET.PATCH_SIZE":    self.d2_patch_size.get() or 512,
                        },
                    )
                    model_path = train(
                        d2_config_path=d2_config,
                        cellseg_config=cfg,
                        dataset_dir=dataset,
                        output_dir=output,
                        multi_gpu=self.d2_multi_gpu.get(),
                        amp=self.d2_amp.get(),
                        num_gpus=self.d2_num_gpus.get() or None,
                        resume=self.d2_resume.get(),
                    )
                    self.log_queue.put(f"__D2_DONE__Training complete → {model_path}")
            except Exception as exc:
                self.log_queue.put(f"__D2_ERROR__{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _d2_run_inference(self) -> None:
        """Run Detectron2 inference in a background thread."""
        d2_config  = self.d2_config_path.get().strip()
        weights    = self.d2_weights_path.get().strip()
        image_dir  = self.d2_infer_img_dir.get().strip()
        output_dir = self.d2_infer_out_dir.get().strip()

        if not all([d2_config, weights, image_dir, output_dir]):
            messagebox.showerror("Missing paths", "Detectron2 Config, Weights, Image Dir, and Output Dir are required.")
            return

        if not self._d2_assert_deps_or_warn():
            return

        self.d2_progress.start()
        self.d2_status.set("Running inference …")

        def worker():
            try:
                container = self.docker_container.get().strip()
                if container:
                    python = self.d2_docker_python.get().strip()
                    cmd = [
                        python, "-m", "cellseg_trainer", "predict",
                        "--model",        self._to_container_path(weights),
                        "--d2-config",    self._to_container_path(d2_config),
                        "--images",       self._to_container_path(image_dir),
                        "--output",       self._to_container_path(output_dir),
                        "--score-thresh", str(self.d2_score_thresh.get()),
                        "--device",       self.d2_device.get(),
                    ]
                    if not self.d2_out_masks.get():
                        cmd.append("--no-masks")
                    if not self.d2_out_overlay.get():
                        cmd.append("--no-overlay")
                    if not self.d2_out_json.get():
                        cmd.append("--no-json")
                    rc = self._run_docker_cmd(container, cmd)
                    if rc != 0:
                        raise RuntimeError(f"docker exec exited with code {rc}")
                    self.log_queue.put(f"__D2_DONE__Inference complete → {self._to_container_path(output_dir)}")
                else:
                    sys.path.insert(0, str(PROJECT_ROOT))
                    from cellseg_trainer.inference import run_inference

                    results = run_inference(
                        model_path=weights,
                        d2_config_path=d2_config,
                        image_dir=image_dir,
                        output_dir=output_dir,
                        score_thresh=self.d2_score_thresh.get(),
                        device=self.d2_device.get(),
                        output_masks=self.d2_out_masks.get(),
                        output_overlay=self.d2_out_overlay.get(),
                        output_json=self.d2_out_json.get(),
                    )
                    msg = (
                        f"Inference complete\n"
                        f"  Images: {results['n_images']}\n"
                        f"  Total instances: {results['n_total_instances']}\n"
                        f"  Output: {output_dir}"
                    )
                    self.log_queue.put(f"__D2_DONE__{msg}")
            except Exception as exc:
                self.log_queue.put(f"__D2_ERROR__{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _d2_run_self_train(self) -> None:
        """Run the self-training / active learning loop."""
        d2_config     = self.d2_config_path.get().strip()
        weights       = self.d2_weights_path.get().strip()
        dataset       = self.d2_dataset_dir.get().strip()
        unlabelled    = self.d2_unlabelled_dir.get().strip()
        output_dir    = self.d2_output_dir.get().strip()

        if not all([d2_config, weights, dataset, unlabelled, output_dir]):
            messagebox.showerror(
                "Missing paths",
                "Detectron2 Config, Weights, Dataset Dir, Unlabelled Dir, and Output Dir are all required."
            )
            return

        if not self._d2_assert_deps_or_warn():
            return

        self.d2_progress.start()
        self.d2_status.set("Self-training …")

        def worker():
            try:
                container = self.docker_container.get().strip()
                if container:
                    python = self.d2_docker_python.get().strip()
                    cmd = [
                        python, "-m", "cellseg_trainer", "self-train",
                        "--dataset",      self._to_container_path(dataset),
                        "--model",        self._to_container_path(weights),
                        "--d2-config",    self._to_container_path(d2_config),
                        "--unlabelled",   self._to_container_path(unlabelled),
                        "--output",       self._to_container_path(output_dir),
                        "--iterations",   str(self.d2_al_iterations.get()),
                        "--pseudo-thresh",str(self.d2_pseudo_thresh.get()),
                    ]
                    if self.d2_suggest_only.get():
                        cmd.append("--suggest-only")
                    rc = self._run_docker_cmd(container, cmd)
                    if rc != 0:
                        raise RuntimeError(f"docker exec exited with code {rc}")
                    self.log_queue.put(f"__D2_DONE__Self-training complete → {self._to_container_path(output_dir)}")
                else:
                    sys.path.insert(0, str(PROJECT_ROOT))
                    from cellseg_trainer.config_loader import DEFAULTS
                    from cellseg_trainer.active_learning import self_train, find_uncertain_patches

                    cfg = dict(DEFAULTS)

                    if self.d2_suggest_only.get():
                        pred_json = Path(output_dir) / "predictions.json"
                        copied = find_uncertain_patches(
                            predictions_json=pred_json,
                            image_dir=unlabelled,
                            output_dir=Path(output_dir) / "for_review",
                            confidence_threshold=self.d2_pseudo_thresh.get(),
                        )
                        self.log_queue.put(f"__D2_DONE__Flagged {len(copied)} images for review → {Path(output_dir) / 'for_review'}")
                        return

                    final_model = self_train(
                        d2_config_path=d2_config,
                        cellseg_config=cfg,
                        initial_model=weights,
                        unlabelled_dir=unlabelled,
                        dataset_dir=dataset,
                        output_dir=output_dir,
                        n_iterations=self.d2_al_iterations.get(),
                        pseudo_label_threshold=self.d2_pseudo_thresh.get(),
                        progress_callback=lambda m: self.log_queue.put(m),
                    )
                    self.log_queue.put(f"__D2_DONE__Self-training complete → {final_model}")
            except Exception as exc:
                self.log_queue.put(f"__D2_ERROR__{exc}")

        threading.Thread(target=worker, daemon=True).start()

    # ==================================================================
    # LOGGING
    # ==================================================================
    def _setup_logging(self):
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    def _poll_log_queue(self):
        """Periodically drain the log queue and update text widgets."""
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break

            # Check for control messages
            if msg == "__TRAINING_DONE__":
                self._on_training_finished()
                continue
            if msg.startswith("__TRAINING_ERROR__"):
                self._on_training_finished(error=msg[len("__TRAINING_ERROR__"):])
                continue
            if msg.startswith("__EVAL_DONE__"):
                self._on_eval_finished(message=msg[len("__EVAL_DONE__"):])
                continue
            if msg.startswith("__EVAL_ERROR__"):
                self._on_eval_finished(error=msg[len("__EVAL_ERROR__"):])
                continue
            if msg.startswith("__PP_PROGRESS__"):
                payload = msg[len("__PP_PROGRESS__"):]
                pct_str, status_text = payload.split("||", 1)
                self.pp_progress.config(value=int(pct_str))
                self.pp_status.set(status_text)
                continue
            if msg.startswith("__PP_DONE__"):
                self._on_pp_finished(results_str=msg[len("__PP_DONE__"):])
                continue
            if msg.startswith("__PP_ERROR__"):
                self._on_pp_finished(error=msg[len("__PP_ERROR__"):])
                continue
            if msg.startswith("__D2_DONE__"):
                self._on_d2_finished(message=msg[len("__D2_DONE__"):])
                continue
            if msg.startswith("__D2_ERROR__"):
                self._on_d2_finished(error=msg[len("__D2_ERROR__"):])
                continue

            # Append to training log AND Detectron2 log
            self._d2_log_msg(msg)

            # Append to training log
            self.train_log.config(state=tk.NORMAL)
            self.train_log.insert(tk.END, msg + "\n")
            self.train_log.see(tk.END)
            self.train_log.config(state=tk.DISABLED)

        self.after(100, self._poll_log_queue)

    def _on_d2_finished(self, message: str | None = None, error: str | None = None) -> None:
        """Called when a Detectron2 background task completes."""
        self.d2_progress.stop()
        if error:
            self.d2_status.set(f"Error: {error}")
            self._d2_log_msg(f"[ERROR] {error}")
            messagebox.showerror("Detectron2 Error", str(error))
        else:
            self.d2_status.set("Done")
            if message:
                self._d2_log_msg(message)

    # ==================================================================
    # MISC
    # ==================================================================
    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Cellpose Segmentation Pipeline GUI\n\n"
            "BiaPy-inspired workflow for training Cellpose models.\n\n"
            "Tasks:\n"
            "  - DIC Brightfield Whole-Cell Segmentation\n"
            "  - Fluorescence Nucleus Segmentation\n\n"
            "Built with Cellpose 4 (cpsam) + tkinter",
        )


def main():
    app = SegmentationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
