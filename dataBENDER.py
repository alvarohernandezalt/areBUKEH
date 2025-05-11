import cv2
import numpy as np
import os
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AreBureBokeh by Alvaro Hernandez")
        self.input_path = ""
        self.output_path = ""
        self.video_info = {"fps": 0, "duration": 0, "width": 0, "height": 0}
        self.previous_params = {}  # To store parameters when switching to random mode
        
        # Create GUI components
        self.create_file_selection()
        self.create_parameter_controls()
        self.create_duration_controls()
        self.create_processing_controls()

    def create_file_selection(self):
        frame = ttk.LabelFrame(self.root, text="File Selection")
        frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        # Input file selection
        ttk.Label(frame, text="Input Video:").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(frame, width=40)
        self.input_entry.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        # Output folder selection
        ttk.Label(frame, text="Output Folder:").grid(row=1, column=0, sticky="w")
        self.output_folder_entry = ttk.Entry(frame, width=40)
        self.output_folder_entry.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Choose", command=self.choose_output_folder).grid(row=1, column=2)
        
        # Output filename
        ttk.Label(frame, text="Output Filename:").grid(row=2, column=0, sticky="w")
        self.output_name_entry = ttk.Entry(frame, width=40)
        self.output_name_entry.insert(0, "output.mp4")
        self.output_name_entry.grid(row=2, column=1, padx=5, pady=5)

    def create_parameter_controls(self):
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Random Settings Tab
        random_frame = ttk.Frame(notebook)
        self.totally_random_var = tk.BooleanVar()
        ttk.Checkbutton(random_frame, text="Totally Random", variable=self.totally_random_var,
                       command=self.toggle_random_mode).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(random_frame, text="Seed:").grid(row=0, column=1, padx=5)
        self.seed_entry = ttk.Entry(random_frame, width=15)
        self.seed_entry.grid(row=0, column=2, padx=5)
        notebook.add(random_frame, text="Random Settings")

        # Noise Tab
        noise_frame = ttk.Frame(notebook)
        self.create_slider(noise_frame, "NOISE_INTENSITY", 0.0, 1.0, row=0)
        self.create_combobox(noise_frame, "NOISE_TYPE", ['random', 'uniform', 'gaussian', 'saltpepper'], row=1)
        notebook.add(noise_frame, text="Noise")
        
        # Geometry Tab
        geom_frame = ttk.Frame(notebook)
        self.create_slider(geom_frame, "ROTATION_RANGE", 0, 45, row=0)
        self.create_slider(geom_frame, "SCALE_RANGE", 0.0, 1.0, row=1)
        self.create_slider(geom_frame, "FLIP_PROBABILITY", 0.0, 1.0, row=2)
        notebook.add(geom_frame, text="Geometry")
        
        # Color Tab
        color_frame = ttk.Frame(notebook)
        self.create_slider(color_frame, "BRIGHTNESS_RANGE", 0, 255, row=0)
        self.create_slider(color_frame, "CONTRAST_RANGE", 0.1, 3.0, row=1)
        self.create_slider(color_frame, "HUE_SHIFT", 0, 180, row=2)
        notebook.add(color_frame, text="Color")
        
        # Filter Tab
        filter_frame = ttk.Frame(notebook)
        self.create_slider(filter_frame, "BLUR_INTENSITY", 0, 15, is_int=True, row=0)
        self.create_slider(filter_frame, "EDGE_DETECTION_PROB", 0.0, 1.0, row=1)
        self.create_slider(filter_frame, "FRAME_SKIP_PROB", 0.0, 1.0, row=2)
        notebook.add(filter_frame, text="Filters")

    def create_duration_controls(self):
        frame = ttk.LabelFrame(self.root, text="Video Duration")
        frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        # Start time slider
        self.start_slider = self.create_time_slider(frame, "Start Time (s):", 0, 0, 0, row=0)
        
        # End time slider
        self.end_slider = self.create_time_slider(frame, "End Time (s):", 0, 0, 0, row=1)

    def create_time_slider(self, parent, label_text, min_val, max_val, init_val, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        
        label = ttk.Label(frame, text=label_text)
        label.grid(row=0, column=0, sticky="w")
        
        var = tk.DoubleVar(value=init_val)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var)
        slider.grid(row=0, column=1, padx=5, sticky="ew")
        
        value_label = ttk.Label(frame, text="0.00")
        value_label.grid(row=0, column=2, padx=5)
        
        var.trace_add("write", lambda *_: value_label.config(text=f"{var.get():.2f}"))
        return {"var": var, "slider": slider}

    def create_slider(self, parent, param, min_val, max_val, is_int=False, row=0):
        var = tk.DoubleVar(value=0)
        setattr(self, param, var)
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        
        label = ttk.Label(frame, text=f"{param.replace('_', ' ').title()}:")
        label.grid(row=0, column=0, sticky="w")
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var,
                          command=lambda v, p=param: self.update_slider_value(var, v, p, is_int))
        slider.grid(row=0, column=1, padx=5, sticky="ew")
        setattr(self, f"{param}_slider", slider)
        
        value_label = ttk.Label(frame, text="0.00")
        setattr(self, f"{param}_label", value_label)
        value_label.grid(row=0, column=2, padx=5)
        
        self.update_slider_value(var, var.get(), param, is_int)
        return slider

    def create_combobox(self, parent, param, options, row):
        var = tk.StringVar(value=options[0])
        setattr(self, param, var)
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        
        ttk.Label(frame, text=f"{param.replace('_', ' ').title()}:").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(frame, textvariable=var, values=options, state="readonly")
        cb.grid(row=0, column=1, padx=5)
        setattr(self, f"{param}_cb", cb)
        return cb

    def update_slider_value(self, var, value, param, is_int):
        value = float(value)
        if is_int:
            value = int(round(value))
            var.set(value)
        label = getattr(self, f"{param}_label")
        label.config(text=f"{value:.2f}" if not is_int else str(value))
        
    def create_processing_controls(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=3, column=0, pady=10)
        
        # Progress bar and status label
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.status_label = ttk.Label(frame, text="")
        
        # Process Video button
        ttk.Button(frame, text="Process Video", command=self.start_processing).grid(row=0, column=0, padx=5)
        self.progress.grid(row=0, column=1, padx=5)
        self.status_label.grid(row=0, column=2, padx=5)

    def toggle_random_mode(self):
        if self.totally_random_var.get():
            # Save current parameters
            self.previous_params = self.get_current_parameters()
            
            # Validate/generate seed
            seed = self.seed_entry.get().strip()
            if not seed:
                seed = random.randint(0, 1000000)
                self.seed_entry.delete(0, tk.END)
                self.seed_entry.insert(0, str(seed))
            
            try:
                seed = int(seed)
            except ValueError:
                messagebox.showerror("Error", "Seed must be an integer")
                self.totally_random_var.set(False)
                return
            
            # Set random seed and generate parameters
            random.seed(seed)
            np.random.seed(seed)
            self.randomize_parameters()
            self.disable_parameters(True)
        else:
            # Restore previous parameters
            self.set_parameters(self.previous_params)
            self.disable_parameters(False)

    def get_current_parameters(self):
        return {
            "NOISE_INTENSITY": self.NOISE_INTENSITY.get(),
            "NOISE_TYPE": self.NOISE_TYPE.get(),
            "ROTATION_RANGE": self.ROTATION_RANGE.get(),
            "SCALE_RANGE": self.SCALE_RANGE.get(),
            "FLIP_PROBABILITY": self.FLIP_PROBABILITY.get(),
            "BRIGHTNESS_RANGE": self.BRIGHTNESS_RANGE.get(),
            "CONTRAST_RANGE": self.CONTRAST_RANGE.get(),
            "HUE_SHIFT": self.HUE_SHIFT.get(),
            "BLUR_INTENSITY": self.BLUR_INTENSITY.get(),
            "EDGE_DETECTION_PROB": self.EDGE_DETECTION_PROB.get(),
            "FRAME_SKIP_PROB": self.FRAME_SKIP_PROB.get(),
            "start_time": self.start_slider["var"].get(),
            "end_time": self.end_slider["var"].get()
        }

    def get_parameters(self):
        return {
            "NOISE_INTENSITY": float(self.NOISE_INTENSITY.get()),
            "NOISE_TYPE": self.NOISE_TYPE.get(),
            "ROTATION_RANGE": float(self.ROTATION_RANGE.get()),
            "SCALE_RANGE": float(self.SCALE_RANGE.get()),
            "FLIP_PROBABILITY": float(self.FLIP_PROBABILITY.get()),
            "BRIGHTNESS_RANGE": int(self.BRIGHTNESS_RANGE.get()),  # Force integer
            "CONTRAST_RANGE": float(self.CONTRAST_RANGE.get()),
            "HUE_SHIFT": int(self.HUE_SHIFT.get()),  # Force integer
            "BLUR_INTENSITY": int(self.BLUR_INTENSITY.get()),  # Force integer
            "EDGE_DETECTION_PROB": float(self.EDGE_DETECTION_PROB.get()),
            "FRAME_SKIP_PROB": float(self.FRAME_SKIP_PROB.get()),
            "start_time": float(self.start_slider["var"].get()),
            "end_time": float(self.end_slider["var"].get())
        }
        
    def set_parameters(self, params):
        self.NOISE_INTENSITY.set(params["NOISE_INTENSITY"])
        self.NOISE_TYPE.set(params["NOISE_TYPE"])
        self.ROTATION_RANGE.set(params["ROTATION_RANGE"])
        self.SCALE_RANGE.set(params["SCALE_RANGE"])
        self.FLIP_PROBABILITY.set(params["FLIP_PROBABILITY"])
        self.BRIGHTNESS_RANGE.set(params["BRIGHTNESS_RANGE"])
        self.CONTRAST_RANGE.set(params["CONTRAST_RANGE"])
        self.HUE_SHIFT.set(params["HUE_SHIFT"])
        self.BLUR_INTENSITY.set(params["BLUR_INTENSITY"])
        self.EDGE_DETECTION_PROB.set(params["EDGE_DETECTION_PROB"])
        self.FRAME_SKIP_PROB.set(params["FRAME_SKIP_PROB"])

    def randomize_parameters(self):
        # Randomize all parameters within their valid ranges
        self.NOISE_INTENSITY.set(random.uniform(0.0, 1.0))
        self.NOISE_TYPE.set(random.choice(['random', 'uniform', 'gaussian', 'saltpepper']))
        self.ROTATION_RANGE.set(random.uniform(0, 45))
        self.SCALE_RANGE.set(random.uniform(0.0, 1.0))
        self.FLIP_PROBABILITY.set(random.uniform(0.0, 1.0))
        self.BRIGHTNESS_RANGE.set(random.randint(0, 255))
        self.CONTRAST_RANGE.set(random.uniform(0.1, 3.0))
        self.HUE_SHIFT.set(random.randint(0, 180))
        self.BLUR_INTENSITY.set(random.randint(0, 15))
        self.EDGE_DETECTION_PROB.set(random.uniform(0.0, 1.0))
        self.FRAME_SKIP_PROB.set(random.uniform(0.0, 1.0))

    def disable_parameters(self, disable=True):
        state = "disabled" if disable else "!disabled"
        for param in ["NOISE_INTENSITY", "ROTATION_RANGE", "SCALE_RANGE", 
                     "FLIP_PROBABILITY", "BRIGHTNESS_RANGE", "CONTRAST_RANGE",
                     "HUE_SHIFT", "BLUR_INTENSITY", "EDGE_DETECTION_PROB",
                     "FRAME_SKIP_PROB"]:
            getattr(self, f"{param}_slider").configure(state=state)
        self.NOISE_TYPE_cb.configure(state=state)

    def browse_input(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if filepath:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, filepath)
            self.load_video_info(filepath)

    def choose_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder)

    def load_video_info(self, filepath):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        
        self.video_info = {
            "fps": fps,
            "duration": duration,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        cap.release()
        
        # Update duration sliders
        for slider in [self.start_slider, self.end_slider]:
            slider["slider"].config(to=duration)
            slider["var"].set(0 if slider == self.start_slider else duration)

    def start_processing(self):
        if not self.validate_inputs():
            return
        
        params = self.get_parameters()
        output_path = os.path.join(
            self.output_folder_entry.get(),
            self.output_name_entry.get()
        )
        
        self.progress["value"] = 0
        self.progress.start()
        self.status_label.config(text="0%")
        
        # Start video processing in a separate thread
        Thread(target=self.process_video, args=(self.input_entry.get(), output_path, params)).start()

    def validate_inputs(self):
        if not os.path.exists(self.input_entry.get()):
            messagebox.showerror("Error", "Input file does not exist")
            return False
        if not os.path.isdir(self.output_folder_entry.get()):
            messagebox.showerror("Error", "Invalid output directory")
            return False
        if self.start_slider["var"].get() >= self.end_slider["var"].get():
            messagebox.showerror("Error", "End time must be after start time")
            return False
        return True

    def process_video(self, input_path, output_path, params):
        try:
            cap = cv2.VideoCapture(input_path)
            fps = self.video_info["fps"]
            start_frame = int(params["start_time"] * fps)
            end_frame = int(params["end_time"] * fps)
            total_frames = end_frame - start_frame
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (self.video_info["width"], self.video_info["height"]))
            
            processed_frames = 0
            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed = self.process_frame(frame, params)
                out.write(processed)
                
                processed_frames += 1
                percent = (processed_frames / total_frames) * 100
                self.root.after(0, self.update_progress, percent)
            
            cap.release()
            out.release()
            
            self.root.after(0, lambda: self.status_label.config(text="Done! 100%"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status_label.config(text="Error"))
        finally:
            self.root.after(0, self.progress.stop)

    def update_progress(self, percent):
        self.progress['value'] = percent
        self.status_label.config(text=f"Processing... {int(percent)}%")
        
    def process_frame(self, frame, params):
        if random.random() < params["FRAME_SKIP_PROB"]:
            return frame
        
        frame = self.apply_random_geometric(frame, params)
        frame = self.apply_color_effects(frame, params)
        frame = self.apply_noise(frame, params)
        frame = self.apply_filters(frame, params)
        return frame

    def apply_random_geometric(self, frame, params):
        h, w = frame.shape[:2]
        
        # Random rotation
        if params["ROTATION_RANGE"] > 0:
            angle = random.uniform(-params["ROTATION_RANGE"], params["ROTATION_RANGE"])
            # Use integer division for center
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            frame = cv2.warpAffine(frame, M, (w, h))
        
        if params["SCALE_RANGE"] > 0:
            scale = 1 + random.uniform(-params["SCALE_RANGE"], params["SCALE_RANGE"])
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            h_new, w_new = frame.shape[:2]
            # Integer crop coordinates
            y = (h_new - h) // 2
            x = (w_new - w) // 2
            frame = frame[y:y+h, x:x+w] if y >=0 and x >=0 else frame[:h, :w]
        
        # Random flip
        if random.random() < params["FLIP_PROBABILITY"]:
            flip_code = random.choice([-1, 0, 1])
            frame = cv2.flip(frame, flip_code)
        
        return frame

    def apply_color_effects(self, frame, params):
        # Brightness and contrast
        if params["BRIGHTNESS_RANGE"] != 0 or params["CONTRAST_RANGE"] != 1.0:
            brightness = random.randint(-int(params["BRIGHTNESS_RANGE"]), int(params["BRIGHTNESS_RANGE"]))
            contrast = random.uniform(1/params["CONTRAST_RANGE"], params["CONTRAST_RANGE"])
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        # Hue shift
        if params["HUE_SHIFT"] > 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = hsv[..., 0].astype(np.int16)
            shift = random.randint(-int(params["HUE_SHIFT"]), int(params["HUE_SHIFT"]))
            hue = (hue + shift) % 180
            hsv[..., 0] = hue.astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return frame

    def apply_noise(self, frame, params):
        if params["NOISE_INTENSITY"] <= 0:
            return frame
        
        noise_type = params["NOISE_TYPE"]
        if noise_type == 'random':
            noise_type = random.choice(['uniform', 'gaussian', 'saltpepper'])
        
        if noise_type == 'uniform':
            noise = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
        elif noise_type == 'gaussian':
            noise = np.random.normal(0, 128, frame.shape).astype(np.uint8)
        elif noise_type == 'saltpepper':
            noise = np.zeros_like(frame)
            prob = random.uniform(0.01, 0.2)
            noise[np.random.rand(*frame.shape[:2]) < prob/2] = 255
            noise[np.random.rand(*frame.shape[:2]) < prob/2] = 0
        
        return cv2.addWeighted(frame, 1-params["NOISE_INTENSITY"], noise, params["NOISE_INTENSITY"], 0)

    def apply_filters(self, frame, params):
        # Blur effect
        blur_value = int(params["BLUR_INTENSITY"])
        if blur_value > 0:
            blur_value = max(1, blur_value)
            blur_value = blur_value if blur_value % 2 == 1 else blur_value + 1
            frame = cv2.GaussianBlur(frame, (blur_value, blur_value), 0)
        
        # Random edge detection
        if random.random() < params["EDGE_DETECTION_PROB"]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return frame

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()