import math
import os
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import imageprep
import nets

display_size = 192
net_labels = {"raw": "raw index", "taylor": "taylor features", "fourier": "fourier features"}
net_colors = {"raw": "#d62728", "taylor": "#1f77b4", "fourier": "#2ca02c"}


def parse_int(text, lo, hi, default):
    try:
        value = int(float(text))
    except ValueError:
        return default
    return max(lo, min(hi, value))


def parse_float(text, lo, hi, default):
    try:
        value = float(text)
    except ValueError:
        return default
    return max(lo, min(hi, value))


def add_spin(parent, label, var, values=None, lo=None, hi=None, step=1, width=8):
    box = ttk.Frame(parent)
    box.pack(side=tk.LEFT, padx=6)
    ttk.Label(box, text=label).pack(side=tk.LEFT)
    if values is not None:
        ttk.Combobox(box, textvariable=var, values=values, width=width).pack(side=tk.LEFT, padx=4)
    else:
        ttk.Spinbox(box, textvariable=var, from_=lo, to=hi, increment=step, width=width).pack(side=tk.LEFT, padx=4)


class App:

    def __init__(self, root):
        self.root = root
        root.title("expanded features vis")
        root.geometry("1020x760")
        root.minsize(1000, 720)

        self.image = None
        self.file_name = None
        self.z, self.x, self.y = nets.coords()
        self.steps = 0
        self.running = False
        self.rebuild_job = None
        self.lr_job = None
        self.ref_photo = None

        self.states = {}
        self.knobs = {}
        self.image_labels = {}
        for kind in nets.kinds:
            self.states[kind] = {"model": None, "history": [], "photo": None}

        self.make_toolbar()
        self.make_status()
        self.make_columns()
        self.make_plot()

        for kind in nets.kinds:
            self.build_net(kind)
        root.after(50, self.tick)

    def make_toolbar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        self.upload_btn = ttk.Button(bar, text="upload image", command=self.upload)
        self.upload_btn.pack(side=tk.LEFT)
        self.start_btn = ttk.Button(bar, text="start", width=7, command=self.start, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(10, 4))
        self.pause_btn = ttk.Button(bar, text="pause", width=7, command=self.pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT)
        self.reset_btn = ttk.Button(bar, text="reset", width=7, command=self.reset, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=4)

        self.batch_var = tk.StringVar(value="512")
        self.render_var = tk.StringVar(value="5")
        self.width_var = tk.StringVar(value="64")
        self.width_var.trace_add("write", lambda a, b, c: self.schedule_rebuild(None))
        self.lr_var = tk.StringVar(value="0.001")
        self.lr_var.trace_add("write", lambda a, b, c: self.schedule_lr())

        side = ttk.Frame(bar)
        side.pack(side=tk.RIGHT)
        add_spin(side, "batch", self.batch_var, lo=32, hi=4096, step=32)
        add_spin(side, "render every", self.render_var, lo=1, hi=500, step=1)
        add_spin(side, "net width", self.width_var, lo=16, hi=256, step=8)
        add_spin(side, "learning rate", self.lr_var, values=["0.0003", "0.001", "0.003", "0.01"], width=9)

    def make_status(self):
        self.status = ttk.Label(self.root, text="pick an image to start", padding=(8, 0))
        self.status.pack(side=tk.TOP, fill=tk.X)

    def make_columns(self):
        row = ttk.Frame(self.root, padding=(6, 4))
        row.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.make_reference(row)
        for kind in nets.kinds:
            self.make_net_column(row, kind)

    def make_reference(self, parent):
        box = ttk.LabelFrame(parent, text="reference", padding=6)
        box.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        self.ref_label = ttk.Label(box)
        self.ref_label.pack()
        self.ref_text = ttk.Label(box, text="no image yet", wraplength=150, justify="center")
        self.ref_text.pack()
        self.show_ref(imageprep.blank_image(), "no image yet")

    def make_net_column(self, parent, kind):
        box = ttk.LabelFrame(parent, text=net_labels[kind], padding=6)
        box.pack(side=tk.LEFT, fill=tk.Y, padx=6)

        knobs = ttk.Frame(box)
        knobs.pack(side=tk.TOP, fill=tk.X)

        v = {}
        self.knobs[kind] = v

        ttk.Label(knobs, text="norm").grid(row=0, column=0, sticky="w", pady=2)
        v["norm"] = tk.StringVar(value="0..1")
        ttk.Combobox(knobs, textvariable=v["norm"], values=["0..1", "-1..1"], width=7, state="readonly").grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(knobs, text="scale").grid(row=1, column=0, sticky="w", pady=2)
        v["scale"] = tk.StringVar(value="pi" if kind == "fourier" else "1")
        ttk.Combobox(knobs, textvariable=v["scale"], values=["1", "pi"], width=7, state="readonly").grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(knobs, text="act").grid(row=2, column=0, sticky="w", pady=2)
        v["act"] = tk.StringVar(value="leaky_relu")
        ttk.Combobox(knobs, textvariable=v["act"], values=nets.activations, width=11, state="readonly").grid(row=2, column=1, sticky="w", pady=2)

        if kind != "raw":
            ttk.Label(knobs, text="order").grid(row=3, column=0, sticky="w", pady=2)
            v["order"] = tk.StringVar(value="4")
            ttk.Spinbox(knobs, textvariable=v["order"], from_=1, to=8, width=7).grid(row=3, column=1, sticky="w", pady=2)

        if kind == "fourier":
            v["use_xy"] = tk.BooleanVar(value=False)
            ttk.Checkbutton(knobs, text="use x, y", variable=v["use_xy"]).grid(row=4, column=0, columnspan=2, sticky="w")

        for key in v:
            v[key].trace_add("write", lambda a, b, c, k=kind: self.schedule_rebuild(k))

        label = ttk.Label(box)
        label.pack(side=tk.TOP, pady=4)
        self.image_labels[kind] = label

    def make_plot(self):
        self.fig = Figure(figsize=(8.9, 2.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

    def net_settings(self, kind):
        v = self.knobs[kind]
        return {
            "kind": kind,
            "order": parse_int(v["order"].get(), 1, 8, 4) if "order" in v else 4,
            "use_xy": bool(v["use_xy"].get()) if "use_xy" in v else False,
            "zero_centered": v["norm"].get() == "-1..1",
            "scale": math.pi if v["scale"].get() == "pi" else 1.0,
            "act": v["act"].get(),
        }

    def build_net(self, kind):
        s = self.net_settings(kind)
        width = parse_int(self.width_var.get(), 16, 256, 64)
        lr = parse_float(self.lr_var.get(), 0.00001, 1.0, 0.001)
        model = nets.build_model(s, width, lr)
        self.states[kind] = {"model": model, "history": [], "photo": None}
        self.show_net_image(kind, imageprep.blank_image())
        self.update_plot()

    def show_net_image(self, kind, img):
        photo = ImageTk.PhotoImage(img.resize((display_size, display_size), Image.NEAREST))
        self.image_labels[kind].config(image=photo)
        self.states[kind]["photo"] = photo  # tk drops the image without a reference

    def show_ref(self, img, text):
        photo = ImageTk.PhotoImage(img.resize((display_size, display_size), Image.NEAREST))
        self.ref_label.config(image=photo)
        self.ref_photo = photo
        self.ref_text.config(text=text)

    def upload(self):
        path = filedialog.askopenfilename(
            title="pick an image",
            filetypes=[("images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("all files", "*.*")],
        )
        if not path:
            return
        img, orig = imageprep.load_image(path)
        self.image = img.astype(np.float32)
        self.file_name = os.path.basename(path)
        self.steps = 0
        self.running = False
        for kind in nets.kinds:
            self.build_net(kind)
        self.show_ref(imageprep.to_image(self.image), "%s\n%dx%d" % (self.file_name, orig[0], orig[1]))
        self.status.config(text="image uploaded, press start")
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)
        self.update_plot()

    def start(self):
        if self.image is None:
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)

    def pause(self):
        self.running = False
        self.pause_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)

    def reset(self):
        self.steps = 0
        for kind in nets.kinds:
            self.build_net(kind)

    def schedule_rebuild(self, kind):
        if self.rebuild_job is not None:
            self.root.after_cancel(self.rebuild_job)
        self.rebuild_job = self.root.after(300, lambda: self.apply_rebuild(kind))

    def apply_rebuild(self, kind):
        self.rebuild_job = None
        kinds = nets.kinds if kind is None else [kind]
        for k in kinds:
            self.build_net(k)

    def schedule_lr(self):
        if self.lr_job is not None:
            self.root.after_cancel(self.lr_job)
        self.lr_job = self.root.after(300, self.apply_lr)

    def apply_lr(self):
        self.lr_job = None
        lr = parse_float(self.lr_var.get(), 0.00001, 1.0, 0.001)
        for kind in nets.kinds:
            model = self.states[kind]["model"]
            if model is not None:
                model.optimizer.learning_rate.assign(lr)

    def tick(self):
        if self.running and self.image is not None:
            self.train_pass()
        self.root.after(50, self.tick)

    def train_pass(self):
        # one pass = every net sees the whole image once, shuffled
        batch = parse_int(self.batch_var.get(), 32, 4096, 512)
        render_every = parse_int(self.render_var.get(), 1, 500, 5)
        for kind in nets.kinds:
            state = self.states[kind]
            feats = nets.make_features(self.net_settings(kind), self.z, self.x, self.y)
            hist = state["model"].fit(feats, self.image, epochs=1, batch_size=batch, verbose=0)
            state["history"].append(float(hist.history["loss"][0]))
        self.steps += 1
        self.update_plot()
        if self.steps % render_every == 0:
            self.render_outputs()

    def render_outputs(self):
        for kind in nets.kinds:
            state = self.states[kind]
            feats = nets.make_features(self.net_settings(kind), self.z, self.x, self.y)
            pred = state["model"].predict(feats, verbose=0).flatten()
            self.show_net_image(kind, imageprep.to_image(pred))

    def update_plot(self):
        self.ax.clear()
        self.ax.set_yscale("log")
        shown = 0
        for kind in nets.kinds:
            hist = self.states[kind]["history"]
            if len(hist) > 1:
                self.ax.plot(hist, color=net_colors[kind], linewidth=1, label=net_labels[kind])
                shown += 1
        self.ax.set_xlim(0, max(self.steps, 10))
        self.ax.grid(True, alpha=0.25)
        self.ax.set_ylabel("mse")
        if shown:
            self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
