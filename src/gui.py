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

        # Currently loaded config
        self.config: dict = {}
        self.config_path: str = ""

        self._build_menu()
        self._build_tabs()
        self._setup_logging()
        self._poll_log_queue()

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

        self.tab_rename = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_rename, text="  File Renaming  ")
        self.notebook.add(self.tab_config, text="  Configuration  ")
        self.notebook.add(self.tab_train, text="  Training  ")
        self.notebook.add(self.tab_eval, text="  Evaluation  ")

        self._build_rename_tab()
        self._build_config_tab()
        self._build_train_tab()
        self._build_eval_tab()

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
                self.config = yaml.safe_load(f)
            self.config_path = path
            self.cfg_path_var.set(path)
            self._populate_param_editor()
            logger.info(f"Loaded config: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

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
            ("CHANNELS", "Channels [chan1, chan2]", "str"),
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
        cfg_section = self.config.get(section, {})

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
        """Read GUI parameter values back into self.config."""
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

            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = val

    def _save_config(self):
        if not self.config:
            messagebox.showwarning("No Config", "Load a config first.")
            return

        self._read_params_into_config()

        path = self.config_path or filedialog.asksaveasfilename(
            title="Save Configuration",
            filetypes=[("YAML", "*.yaml")],
            initialdir=str(PROJECT_ROOT / "configs"),
            defaultextension=".yaml",
        )
        if not path:
            return

        try:
            with open(path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            self.config_path = path
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

        if task == "custom" and not self.config:
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
            from train_cellpose import load_config, train_cellpose_model

            configs_to_run = []
            if task == "dic":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "dic_wholecell.yaml"))
            elif task == "fluor":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "fluor_nucleus.yaml"))
            elif task == "both":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "dic_wholecell.yaml"))
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "fluor_nucleus.yaml"))
            elif task == "custom":
                # Use config from GUI editor
                self._read_params_into_config()
                configs_to_run.append(self.config_path)

            for cfg_path in configs_to_run:
                if self._stop_event.is_set():
                    logger.info("Training stopped by user.")
                    break
                logger.info(f"Loading config: {cfg_path}")
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
        if not self.config:
            messagebox.showwarning(
                "No Config", "Load a config in the Configuration tab first."
            )
            return

        self.btn_eval_full.config(state=tk.DISABLED)
        self.eval_progress.start(10)
        self.eval_status.set("Running full evaluation...")

        def worker():
            try:
                from evaluate import evaluate_model
                self._read_params_into_config()
                results = evaluate_model(self.config, model)
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

            # Append to training log
            self.train_log.config(state=tk.NORMAL)
            self.train_log.insert(tk.END, msg + "\n")
            self.train_log.see(tk.END)
            self.train_log.config(state=tk.DISABLED)

        self.after(100, self._poll_log_queue)

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
