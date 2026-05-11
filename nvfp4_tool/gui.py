import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CLI = ROOT / "nvfp4_tool" / "convert_cli.py"
ENV_CHECK = ROOT / "nvfp4_tool" / "env_check.py"

MODEL_TYPES = [
    "Z-Image-Turbo",
    "Z-Image-Turbo-Conservative",
    "Z-Image-Base",
    "Flux.1-dev",
    "Flux.1-Fill",
    "Flux.2-dev",
    "Flux.2-Klein-9b",
    "Qwen-Image-Edit-2511",
    "Qwen-Image-2512",
    "Wan2.2-i2v-high-low",
    "LTX-2-19b-dev-or-distilled",
]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Z-Image NVFP4 Kitchen Converter")
        self.geometry("1100x760")
        self.proc = None
        self.q = queue.Queue()
        self.start_time = None
        self.last_output = None
        self.heartbeat_after = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.model_type_var = tk.StringVar(value="Z-Image-Turbo")
        self.device_var = tk.StringVar(value="cuda")
        self.allow_fp8_var = tk.BooleanVar(value=True)
        self.continue_error_var = tk.BooleanVar(value=False)
        self.progress_every_var = tk.StringVar(value="5")
        self.status_var = tk.StringVar(value="Ready.")
        self.time_var = tk.StringVar(value="Elapsed: 0s")

        self._build()
        self.after(100, self._poll_queue)

    def _build(self):
        pad = {"padx": 8, "pady": 5}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Input .safetensors:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.input_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self.browse_input).grid(row=0, column=2)

        ttk.Label(top, text="Output .safetensors:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_var).grid(row=1, column=1, sticky="ew")
        ttk.Button(top, text="Save as", command=self.browse_output).grid(row=1, column=2)
        top.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(self, text="Conversion options")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Profile:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.model_type_var, values=MODEL_TYPES, state="readonly", width=28).grid(row=0, column=1, sticky="w")

        ttk.Label(opts, text="Device:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(opts, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(opts, text="Progress every N tensors:").grid(row=0, column=4, sticky="w")
        ttk.Entry(opts, textvariable=self.progress_every_var, width=6).grid(row=0, column=5, sticky="w")

        ttk.Checkbutton(opts, text="Allow/force FP8 source", variable=self.allow_fp8_var).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(opts, text="Continue on per-layer error", variable=self.continue_error_var).grid(row=1, column=2, columnspan=2, sticky="w")

        warn = (
            "Aviso: BF16/FP16 original é a fonte recomendada. FP8 → NVFP4 funciona como tentativa, "
            "mas pode perder qualidade porque você quantiza algo já quantizado."
        )
        ttk.Label(opts, text=warn, foreground="#a15c00").grid(row=2, column=0, columnspan=7, sticky="w", pady=(6, 0))

        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Check environment", command=self.check_env).pack(side="left", padx=4)
        ttk.Button(btns, text="Dry scan", command=self.scan).pack(side="left", padx=4)
        ttk.Button(btns, text="Convert", command=self.convert).pack(side="left", padx=4)
        ttk.Button(btns, text="Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Button(btns, text="Open output folder", command=self.open_output_folder).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=4)

        progbox = ttk.Frame(self)
        progbox.pack(fill="x", **pad)
        self.pbar = ttk.Progressbar(progbox, orient="horizontal", mode="determinate", maximum=100)
        self.pbar.pack(fill="x", side="left", expand=True)
        ttk.Label(progbox, textvariable=self.time_var, width=18).pack(side="left", padx=8)

        ttk.Label(self, textvariable=self.status_var).pack(fill="x", padx=10)

        logframe = ttk.LabelFrame(self, text="Log")
        logframe.pack(fill="both", expand=True, padx=8, pady=8)
        self.log = tk.Text(logframe, wrap="word", height=25)
        yscroll = ttk.Scrollbar(logframe, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self._write("Z-Image NVFP4 Kitchen Converter ready.\n")
        self._write(f"Python: {PYTHON}\nRoot: {ROOT}\n\n")

    def browse_input(self):
        p = filedialog.askopenfilename(filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")])
        if p:
            self.input_var.set(p)
            ip = Path(p)
            if not self.output_var.get():
                self.output_var.set(str(ip.with_name(ip.stem + "_nvfp4.safetensors")))

    def browse_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".safetensors", filetypes=[("SafeTensors", "*.safetensors")])
        if p:
            self.output_var.set(p)

    def _write(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _run(self, cmd):
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("Already running", "Já tem um processo rodando.")
            return
        self.pbar["value"] = 0
        self.start_time = time.time()
        self.last_output = time.time()
        self.status_var.set("Running...")
        self._write("\n$ " + " ".join(str(c) for c in cmd) + "\n")

        def worker():
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                for line in self.proc.stdout:
                    self.q.put(line)
                code = self.proc.wait()
                self.q.put(f"\nProcess finished with code {code}\n")
                self.q.put(("__DONE__", code))
            except Exception as e:
                self.q.put(f"ERROR launching process: {type(e).__name__}: {e}\n")
                self.q.put(("__DONE__", 999))

        threading.Thread(target=worker, daemon=True).start()
        self._heartbeat()

    def _heartbeat(self):
        if self.proc and self.proc.poll() is None:
            elapsed = int(time.time() - self.start_time) if self.start_time else 0
            idle = int(time.time() - self.last_output) if self.last_output else 0
            self.time_var.set(f"Elapsed: {elapsed}s")
            if idle >= 20:
                self._write(f"...[still running] elapsed={elapsed}s idle={idle}s\n")
                self.last_output = time.time()
            self.heartbeat_after = self.after(1000, self._heartbeat)

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    code = item[1]
                    self.status_var.set("Done." if code == 0 else f"Failed with code {code}.")
                    if self.heartbeat_after:
                        self.after_cancel(self.heartbeat_after)
                        self.heartbeat_after = None
                    continue
                self.last_output = time.time()
                line = str(item)
                self._write(line)
                if line.startswith("PROGRESS|"):
                    try:
                        parts = line.strip().split("|", 4)
                        i, total, pct = int(parts[1]), int(parts[2]), float(parts[3])
                        self.pbar["value"] = pct
                        self.status_var.set(f"Converting tensor {i}/{total} ({pct:.1f}%)")
                    except Exception:
                        pass
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def check_env(self):
        self._run([PYTHON, "-u", str(ENV_CHECK)])

    def scan(self):
        ip = self.input_var.get().strip()
        op = self.output_var.get().strip() or str(Path(ip).with_name(Path(ip).stem + "_nvfp4.safetensors")) if ip else "out.safetensors"
        if not ip:
            messagebox.showwarning("Missing input", "Escolhe um .safetensors primeiro.")
            return
        cmd = [PYTHON, "-u", str(CLI), "--input", ip, "--output", op, "--model-type", self.model_type_var.get(), "--scan-only"]
        if self.allow_fp8_var.get():
            cmd.append("--allow-fp8")
        self._run(cmd)

    def convert(self):
        ip = self.input_var.get().strip()
        op = self.output_var.get().strip()
        if not ip or not op:
            messagebox.showwarning("Missing paths", "Escolhe input e output.")
            return
        try:
            pe = int(self.progress_every_var.get().strip())
            if pe < 1:
                pe = 5
        except Exception:
            pe = 5
        cmd = [
            PYTHON, "-u", str(CLI),
            "--input", ip,
            "--output", op,
            "--model-type", self.model_type_var.get(),
            "--device", self.device_var.get(),
            "--progress-every", str(pe),
        ]
        if self.allow_fp8_var.get():
            cmd.append("--allow-fp8")
        if self.continue_error_var.get():
            cmd.append("--continue-on-error")
        self._run(cmd)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self._write("\nTerminate requested...\n")
            try:
                self.proc.terminate()
            except Exception as e:
                self._write(f"Terminate failed: {e}\n")

    def open_output_folder(self):
        op = self.output_var.get().strip()
        folder = Path(op).parent if op else ROOT
        if folder.exists():
            os.startfile(str(folder))

if __name__ == "__main__":
    App().mainloop()
