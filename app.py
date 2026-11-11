import os
import sys
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Thiết lập FFMPEG tối ưu kết nối cáp mạng RJ45 trực tiếp (Zero Delay, No Buffer)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)

# Bảng màu hiển thị cho từng class (BGR cho OpenCV)
CLASS_COLORS = {
    'Person': (255, 180, 0),        # Vàng cam
    'helmet': (0, 230, 0),          # Xanh lá cây sáng
    'Vest': (0, 215, 120),          # Xanh ngọc
    'gloves': (255, 0, 255),        # Hồng tím
    'Boots': (240, 130, 0),         # Xanh dương
    'shoes': (200, 100, 0),         # Xanh dương đậm
    'Mask': (0, 165, 255),          # Cam
    'glasses': (0, 255, 255),       # Vàng chanh
    'Ear-protection': (200, 160, 255) # Tím nhạt
}

CLASS_DISPLAY_TAG = {
    'Person': 'Nguoi',
    'helmet': 'Mu bao ho',
    'Vest': 'Ao bao ho',
    'gloves': 'Gang tay',
    'Boots': 'Ung bao ho',
    'shoes': 'Giay bao ho',
    'Mask': 'Khau trang',
    'glasses': 'Kinh bao ho',
    'Ear-protection': 'Bit tai'
}

CLASS_VIETNAMESE = {
    'Person': 'Người',
    'helmet': 'Mũ bảo hộ',
    'Vest': 'Áo bảo hộ',
    'gloves': 'Găng tay',
    'Boots': 'Ủng bảo hộ',
    'shoes': 'Giày bảo hộ',
    'Mask': 'Khẩu trang',
    'glasses': 'Kính bảo hộ',
    'Ear-protection': 'Bịt tai chống ồn'
}


class VideoStreamThread:
    """Thread chuyên lấy frame từ RTSP qua cổng RJ45 chuẩn testcam.py (cv2.CAP_FFMPEG, nobuffer, low_delay)"""
    def __init__(self, source):
        self.source = source
        # Sử dụng backend CAP_FFMPEG như testcam.py khi nguồn là URL RTSP
        if isinstance(source, str):
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source)
        # Giảm buffer size của OpenCV xuống 1 để luôn lấy frame mới nhất thời gian thực
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ok, f = self.cap.read()
                    if ok:
                        with self.lock:
                            self.ret = True
                            self.frame = f
                    else:
                        time.sleep(0.005)
                else:
                    time.sleep(0.02)
            except Exception:
                break

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


class PPEDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG GIÁM SÁT ĐỒ BẢO HỘ LAO ĐỘNG (PPE DETECTION AI)")
        self.root.geometry("1380x850")
        self.root.minsize(1150, 720)
        self.root.configure(bg="#181825")

        # Biến trạng thái
        self.is_running = False
        self.video_stream = None
        self.model = None
        self.person_model = None  # Model hỗ trợ nhận diện người siêu nhạy
        self.model_path = r"c:\Users\PC\Downloads\best.pt"
        self.person_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
        self.snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.last_frame = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.detected_stats = {k: 0 for k in CLASS_VIETNAMESE.keys()}

        # Tải model AI ban đầu
        self.load_ai_model(self.model_path)
        self.load_person_model(self.person_model_path)

        # Xây dựng giao diện
        self.setup_ui()

        # Xử lý sự kiện đóng app
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_ai_model(self, path):
        """Khởi tạo mô hình YOLO PPE"""
        if not os.path.exists(path):
            return False
        try:
            self.model = YOLO(path)
            return True
        except Exception as e:
            print(f"Lỗi tải model PPE: {e}")
            return False

    def load_person_model(self, path):
        """Khởi tạo mô hình YOLO nhận diện Người bổ trợ (chính xác >95%)"""
        if not os.path.exists(path):
            return False
        try:
            self.person_model = YOLO(path)
            return True
        except Exception as e:
            print(f"Lỗi tải model Người: {e}")
            return False

    def setup_ui(self):
        # 1. Header Bar
        header = tk.Frame(self.root, bg="#11111b", height=50)
        header.pack(fill="x", side="top")

        lbl_title = tk.Label(
            header,
            text="🛡️ HỆ THỐNG GIÁM SÁT ĐỒ BẢO HỘ LAO ĐỘNG (PPE) - IMOU AI",
            font=("Segoe UI", 14, "bold"),
            fg="#cdd6f4",
            bg="#11111b"
        )
        lbl_title.pack(side="left", padx=20, pady=10)

        self.lbl_model_status = tk.Label(
            header,
            text=f"Model: {os.path.basename(self.model_path)} (Đã sẵn sàng)" if self.model else "Model: Chưa tải",
            font=("Segoe UI", 10),
            fg="#a6e3a1" if self.model else "#f38ba8",
            bg="#11111b"
        )
        self.lbl_model_status.pack(side="right", padx=20)

        # 2. Main Content Split (Trái: Video, Phải: Sidebar cài đặt & thống kê)
        main_content = tk.Frame(self.root, bg="#181825")
        main_content.pack(fill="both", expand=True, padx=12, pady=12)

        # --- Khung Video (Bên trái) ---
        video_wrapper = tk.Frame(main_content, bg="#1e1e2e", bd=1, relief="ridge")
        video_wrapper.pack(side="left", fill="both", expand=True)

        # Thanh trạng thái phía trên video (FPS & Tình trạng kết nối)
        video_header = tk.Frame(video_wrapper, bg="#1e1e2e")
        video_header.pack(fill="x", padx=10, pady=6)

        self.lbl_stream_status = tk.Label(
            video_header,
            text="⚪ Trạng thái: Chưa kết nối",
            font=("Segoe UI", 10, "bold"),
            fg="#bac2de",
            bg="#1e1e2e"
        )
        self.lbl_stream_status.pack(side="left")

        self.lbl_fps = tk.Label(
            video_header,
            text="FPS: 0.0",
            font=("Consolas", 10, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e"
        )
        self.lbl_fps.pack(side="right")

        # Canvas hiển thị video
        self.canvas_video = tk.Canvas(video_wrapper, bg="#000000", highlightthickness=0)
        self.canvas_video.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas_video.bind("<Configure>", self.on_canvas_resize)

        # Thanh công cụ điều khiển dưới video
        control_bar = tk.Frame(video_wrapper, bg="#1e1e2e")
        control_bar.pack(fill="x", padx=10, pady=10)

        self.btn_toggle = tk.Button(
            control_bar,
            text="▶ BẮT ĐẦU GIÁM SÁT",
            font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1",
            fg="#11111b",
            activebackground="#94e2d5",
            cursor="hand2",
            padx=16,
            pady=6,
            command=self.toggle_stream
        )
        self.btn_toggle.pack(side="left", padx=5)

        self.btn_snapshot = tk.Button(
            control_bar,
            text="📸 Chụp Vi Phạm",
            font=("Segoe UI", 10, "bold"),
            bg="#89b4fa",
            fg="#11111b",
            activebackground="#b4befe",
            cursor="hand2",
            padx=12,
            pady=6,
            command=self.take_snapshot
        )
        self.btn_snapshot.pack(side="left", padx=5)

        self.btn_open_folder = tk.Button(
            control_bar,
            text="📁 Mở Thư Mục Ảnh",
            font=("Segoe UI", 10),
            bg="#313244",
            fg="#cdd6f4",
            activebackground="#45475a",
            cursor="hand2",
            padx=10,
            pady=6,
            command=self.open_snapshot_folder
        )
        self.btn_open_folder.pack(side="left", padx=5)

        # --- Sidebar Cấu hình & Thống kê (Bên phải) ---
        sidebar = tk.Frame(main_content, bg="#1e1e2e", width=420, bd=1, relief="ridge")
        sidebar.pack(side="right", fill="y", padx=(12, 0))
        sidebar.pack_propagate(False)

        # Sidebar Title
        tk.Label(
            sidebar,
            text="CẤU HÌNH & THỐNG KÊ",
            font=("Segoe UI", 12, "bold"),
            fg="#cdd6f4",
            bg="#1e1e2e"
        ).pack(fill="x", padx=14, pady=(12, 8))

        # 1. Cấu hình Camera IMOU
        box_cam = tk.LabelFrame(
            sidebar,
            text=" ⚙️ Cấu hình Camera Imou ",
            font=("Segoe UI", 10, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
            bd=1
        )
        box_cam.pack(fill="x", padx=12, pady=6)

        # Nguồn Video (RTSP Camera Imou / Webcam / Video)
        tk.Label(box_cam, text="Nguồn phát:", font=("Segoe UI", 9), fg="#bac2de", bg="#1e1e2e").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.var_source_type = tk.StringVar(value="imou")
        cb_source = ttk.Combobox(
            box_cam,
            textvariable=self.var_source_type,
            values=["imou", "webcam", "file"],
            state="readonly",
            width=18
        )
        cb_source.grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        cb_source.bind("<<ComboboxSelected>>", self.on_source_type_changed)

        # IP Camera Entry
        tk.Label(box_cam, text="IP Camera:", font=("Segoe UI", 9), fg="#bac2de", bg="#1e1e2e").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.entry_ip = tk.Entry(box_cam, font=("Consolas", 10), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.entry_ip.insert(0, "192.168.1.108")
        self.entry_ip.grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        # Safety Code Entry
        tk.Label(box_cam, text="Safety Code:", font=("Segoe UI", 9), fg="#bac2de", bg="#1e1e2e").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.entry_code = tk.Entry(box_cam, font=("Consolas", 10), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.entry_code.insert(0, "L2C2A89B")
        self.entry_code.grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        # Chọn Subtype (Luồng phụ / Luồng chính)
        tk.Label(box_cam, text="Loại luồng:", font=("Segoe UI", 9), fg="#bac2de", bg="#1e1e2e").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        self.var_subtype = tk.StringVar(value="1")  # Mặc định luồng phụ (1) cực mượt và ổn định cho AI
        cb_stream = ttk.Combobox(
            box_cam,
            textvariable=self.var_subtype,
            values=["1", "0"],
            state="readonly",
            width=18
        )
        cb_stream.grid(row=3, column=1, sticky="ew", padx=8, pady=4)

        lbl_hint = tk.Label(
            box_cam,
            text="*Gợi ý: Luồng 1 (Sub-stream) mượt & nhanh hơn cho AI",
            font=("Segoe UI", 8, "italic"),
            fg="#9399b2",
            bg="#1e1e2e"
        )
        lbl_hint.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        box_cam.columnconfigure(1, weight=1)

        # 2. Cấu hình AI Model & Ngưỡng
        box_ai = tk.LabelFrame(
            sidebar,
            text=" 🧠 Cấu hình AI Model ",
            font=("Segoe UI", 10, "bold"),
            fg="#fab387",
            bg="#1e1e2e",
            bd=1
        )
        box_ai.pack(fill="x", padx=12, pady=6)

        # Model file
        frame_model = tk.Frame(box_ai, bg="#1e1e2e")
        frame_model.pack(fill="x", padx=8, pady=4)

        self.lbl_model_name = tk.Label(
            frame_model,
            text=f"File: {os.path.basename(self.model_path)}",
            font=("Segoe UI", 9),
            fg="#cdd6f4",
            bg="#1e1e2e"
        )
        self.lbl_model_name.pack(side="left")

        btn_browse = tk.Button(
            frame_model,
            text="Đổi...",
            font=("Segoe UI", 8),
            bg="#313244",
            fg="#cdd6f4",
            cursor="hand2",
            command=self.browse_model
        )
        btn_browse.pack(side="right")

        # Confidence Threshold Slider
        tk.Label(box_ai, text="Độ tin cậy (Confidence):", font=("Segoe UI", 9), fg="#bac2de", bg="#1e1e2e").pack(anchor="w", padx=8, pady=(4, 0))
        self.var_conf = tk.DoubleVar(value=0.15)
        self.slider_conf = tk.Scale(
            box_ai,
            from_=0.05,
            to=0.80,
            resolution=0.05,
            orient="horizontal",
            variable=self.var_conf,
            bg="#1e1e2e",
            fg="#cdd6f4",
            highlightthickness=0,
            troughcolor="#313244",
            activebackground="#fab387"
        )
        self.slider_conf.pack(fill="x", padx=8, pady=(0, 4))

        # Checkbox Tăng cường nhận diện Người (YOLO Person Boost)
        self.var_person_boost = tk.BooleanVar(value=True)
        self.chk_boost = tk.Checkbutton(
            box_ai,
            text="Tăng cường phát hiện Người (YOLO Boost)",
            font=("Segoe UI", 9, "bold"),
            variable=self.var_person_boost,
            bg="#1e1e2e",
            fg="#a6e3a1",
            selectcolor="#313244",
            activebackground="#1e1e2e",
            activeforeground="#a6e3a1"
        )
        self.chk_boost.pack(anchor="w", padx=8, pady=(0, 6))

        # 3. Quy chuẩn giám sát trang bị bắt buộc
        box_rules = tk.LabelFrame(
            sidebar,
            text=" 📋 Quy Chuẩn Trang Bị Bắt Buộc ",
            font=("Segoe UI", 10, "bold"),
            fg="#89dceb",
            bg="#1e1e2e",
            bd=1
        )
        box_rules.pack(fill="x", padx=12, pady=4)

        rules_frame = tk.Frame(box_rules, bg="#1e1e2e")
        rules_frame.pack(fill="x", padx=6, pady=3)

        self.var_req_helmet = tk.BooleanVar(value=True)
        self.var_req_vest = tk.BooleanVar(value=True)
        self.var_req_gloves = tk.BooleanVar(value=True)

        chk_h = tk.Checkbutton(
            rules_frame,
            text="Mũ",
            font=("Segoe UI", 9),
            variable=self.var_req_helmet,
            bg="#1e1e2e",
            fg="#cdd6f4",
            selectcolor="#313244",
            activebackground="#1e1e2e",
            activeforeground="#cdd6f4"
        )
        chk_h.pack(side="left", padx=5)

        chk_v = tk.Checkbutton(
            rules_frame,
            text="Áo",
            font=("Segoe UI", 9),
            variable=self.var_req_vest,
            bg="#1e1e2e",
            fg="#cdd6f4",
            selectcolor="#313244",
            activebackground="#1e1e2e",
            activeforeground="#cdd6f4"
        )
        chk_v.pack(side="left", padx=5)

        chk_g = tk.Checkbutton(
            rules_frame,
            text="Găng tay",
            font=("Segoe UI", 9),
            variable=self.var_req_gloves,
            bg="#1e1e2e",
            fg="#cdd6f4",
            selectcolor="#313244",
            activebackground="#1e1e2e",
            activeforeground="#cdd6f4"
        )
        chk_g.pack(side="left", padx=5)

        # 4. Đánh Giá An Toàn Lao Động (PPE Safety Status Banner)
        self.banner_alert = tk.Label(
            sidebar,
            text="⚪ ĐANG CHỜ GIÁM SÁT",
            font=("Segoe UI", 10, "bold"),
            fg="#cdd6f4",
            bg="#313244",
            pady=10,
            wraplength=390,
            justify="center"
        )
        self.banner_alert.pack(fill="x", padx=12, pady=8)

        # 5. Bảng Thống kê phát hiện thời gian thực (Real-time PPE Counts)
        box_stats = tk.LabelFrame(
            sidebar,
            text=" 📊 Số Lượng Phát Hiện Thời Gian Thực ",
            font=("Segoe UI", 10, "bold"),
            fg="#a6e3a1",
            bg="#1e1e2e",
            bd=1
        )
        box_stats.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self.stat_labels = {}
        for idx, (eng_cls, vn_cls) in enumerate(CLASS_VIETNAMESE.items()):
            row_frame = tk.Frame(box_stats, bg="#1e1e2e")
            row_frame.pack(fill="x", padx=10, pady=3)

            # Icon màu nhận diện
            bgr = CLASS_COLORS.get(eng_cls, (255, 255, 255))
            hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
            tk.Label(row_frame, text="■", font=("Segoe UI", 10), fg=hex_color, bg="#1e1e2e").pack(side="left")

            tk.Label(row_frame, text=f"{vn_cls}:", font=("Segoe UI", 9), fg="#cdd6f4", bg="#1e1e2e").pack(side="left", padx=5)

            lbl_val = tk.Label(row_frame, text="0", font=("Consolas", 10, "bold"), fg="#a6e3a1", bg="#1e1e2e")
            lbl_val.pack(side="right")
            self.stat_labels[eng_cls] = lbl_val

    def on_source_type_changed(self, event=None):
        src = self.var_source_type.get()
        if src == "webcam":
            self.entry_ip.config(state="disabled")
            self.entry_code.config(state="disabled")
        elif src == "file":
            self.entry_ip.config(state="disabled")
            self.entry_code.config(state="disabled")
            file_path = filedialog.askopenfilename(
                title="Chọn file video",
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
            )
            if file_path:
                self.custom_file_path = file_path
        else:
            self.entry_ip.config(state="normal")
            self.entry_code.config(state="normal")

    def browse_model(self):
        f = filedialog.askopenfilename(
            title="Chọn file mô hình YOLO (.pt)",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if f:
            if self.load_ai_model(f):
                self.model_path = f
                self.lbl_model_name.config(text=f"File: {os.path.basename(f)}")
                self.lbl_model_status.config(text=f"Model: {os.path.basename(f)} (Đã sẵn sàng)", fg="#a6e3a1")
                messagebox.showinfo("Thành công", f"Đã nạp model: {os.path.basename(f)}")
            else:
                messagebox.showerror("Lỗi", "Không thể nạp file model này!")

    def get_rtsp_url(self):
        src_type = self.var_source_type.get()
        if src_type == "webcam":
            return 0
        elif src_type == "file":
            return getattr(self, 'custom_file_path', 0)
        
        # Camera Imou
        ip = self.entry_ip.get().strip()
        code = self.entry_code.get().strip()
        subtype = self.var_subtype.get().strip()

        if not ip:
            messagebox.showwarning("Thiếu IP", "Vui lòng nhập địa chỉ IP của Camera Imou!")
            return None
        if not code:
            messagebox.showwarning("Thiếu Safety Code", "Vui lòng nhập mã Safety Code (dưới đáy camera)!")
            return None

        # Cú pháp RTSP chuẩn cho dòng Imou / Dahua
        url = f"rtsp://admin:{code}@{ip}:554/cam/realmonitor?channel=1&subtype={subtype}"
        return url

    def toggle_stream(self):
        if not self.is_running:
            # Bắt đầu stream
            if not self.model:
                messagebox.showerror("Chưa có model", "Chưa nạp được model AI. Vui lòng kiểm tra lại file best.pt!")
                return

            rtsp_source = self.get_rtsp_url()
            if rtsp_source is None:
                return

            self.lbl_stream_status.config(text="🟡 Đang kết nối camera...", fg="#f9e2af")
            self.root.update_idletasks()

            try:
                self.video_stream = VideoStreamThread(rtsp_source)
            except Exception as e:
                messagebox.showerror("Lỗi kết nối", f"Không thể mở luồng video:\n{e}")
                self.lbl_stream_status.config(text="🔴 Kết nối thất bại", fg="#f38ba8")
                return

            self.is_running = True
            self.btn_toggle.config(text="⏹ DỪNG GIÁM SÁT", bg="#f38ba8")
            self.lbl_stream_status.config(text="🟢 Đang giám sát (Trực tiếp)", fg="#a6e3a1")

            # Bắt đầu vòng lặp xử lý video
            self.last_fps_time = time.time()
            self.frame_count = 0
            self.process_video_loop()
        else:
            # Dừng stream
            self.stop_stream()

    def stop_stream(self):
        self.is_running = False
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None

        self.btn_toggle.config(text="▶ BẮT ĐẦU GIÁM SÁT", bg="#a6e3a1")
        self.lbl_stream_status.config(text="⚪ Trạng thái: Đã dừng", fg="#bac2de")
        self.lbl_fps.config(text="FPS: 0.0")
        self.banner_alert.config(text="⚪ ĐÃ DỪNG GIÁM SÁT", bg="#313244", fg="#cdd6f4")
        self.canvas_video.delete("all")

    def process_video_loop(self):
        if not self.is_running:
            return

        ret, frame = self.video_stream.read() if self.video_stream else (False, None)
        if ret and frame is not None:
            # Tính FPS
            self.frame_count += 1
            now = time.time()
            dt = now - self.last_fps_time
            if dt >= 1.0:
                self.fps = self.frame_count / dt
                self.frame_count = 0
                self.last_fps_time = now
                self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")

            # AI Inference với YOLO
            annotated_frame, counts = self.detect_ppe(frame)
            self.last_frame = annotated_frame.copy()

            # Cập nhật thống kê và cảnh báo
            self.update_statistics_and_alert(counts)

            # Vẽ lên Canvas Tkinter
            self.display_frame_on_canvas(annotated_frame)
        else:
            # Đang đợi frame từ camera
            pass

        # Lặp lại sau 15-25ms
        self.root.after(20, self.process_video_loop)

    def detect_ppe(self, frame):
        """Chạy model YOLO dự đoán đồ bảo hộ và vẽ bounding box trực tiếp trên khung hình:
        - Nếu người thiếu đồ bảo hộ bắt buộc: Box viền ĐỎ và hiển thị nhãn [!] THIEU: Mu, Ao, Gang tay...
        - Nếu người đầy đủ đồ bảo hộ: Box viền XANH LÁ/Màu cũ và hiển thị nhãn [OK] DU BAO HO
        - Các món bảo hộ được phát hiện (Mũ, Áo, Găng tay...) hiển thị theo màu riêng biệt của món đó.
        """
        conf_thresh = self.var_conf.get()
        counts = {k: 0 for k in CLASS_VIETNAMESE.keys()}

        try:
            # 1. Chạy mô hình PPE
            results = self.model.predict(
                source=frame,
                conf=conf_thresh,
                iou=0.45,
                verbose=False,
                imgsz=640
            )

            raw_boxes = results[0].boxes
            all_persons = []
            all_ppe_items = []

            for box in raw_boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                if cls_name == "Person":
                    all_persons.append({'box': (x1, y1, x2, y2), 'conf': conf})
                else:
                    all_ppe_items.append({'cls': cls_name, 'conf': conf, 'box': (x1, y1, x2, y2)})
                    if cls_name in counts:
                        counts[cls_name] += 1

            # 2. Bổ trợ Person nếu bật Person Boost
            if getattr(self, 'var_person_boost', None) and self.var_person_boost.get() and self.person_model:
                person_res = self.person_model.predict(
                    source=frame,
                    classes=[0],  # class 0 trong COCO là Person
                    conf=0.35,
                    verbose=False,
                    imgsz=640
                )
                for p_box in person_res[0].boxes:
                    p_conf = float(p_box.conf[0].item())
                    px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)

                    # Tránh trùng lặp với Person đã có
                    is_dup = False
                    for p_exist in all_persons:
                        ex1, ey1, ex2, ey2 = p_exist['box']
                        ix1 = max(px1, ex1)
                        iy1 = max(py1, ey1)
                        ix2 = min(px2, ex2)
                        iy2 = min(py2, ey2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            p_area = (px2 - px1) * (py2 - py1)
                            if inter / p_area > 0.4:
                                is_dup = True
                                break

                    if not is_dup:
                        all_persons.append({'box': (px1, py1, px2, py2), 'conf': p_conf})

            counts['Person'] = len(all_persons)

            # 2.5. Quét chi tiết trên vùng người (Person Deep Scan) để bắt các vật thể nhỏ (Găng tay, Mũ, Kính)
            for p_info in all_persons[:3]:  # Quét tối đa 3 người gần nhất
                ppx1, ppy1, ppx2, ppy2 = p_info['box']
                h_f, w_f, _ = frame.shape
                c_y1 = max(0, ppy1)
                c_y2 = min(h_f, ppy2)
                c_x1 = max(0, ppx1)
                c_x2 = min(w_f, ppx2)
                if c_y2 > c_y1 + 40 and c_x2 > c_x1 + 40:
                    p_crop = frame[c_y1:c_y2, c_x1:c_x2]
                    # Chạy inference trên crop với ngưỡng nhạy để không bỏ sót găng tay
                    crop_res = self.model.predict(
                        source=p_crop,
                        conf=max(0.12, conf_thresh * 0.8),
                        verbose=False,
                        imgsz=640
                    )
                    for cb in crop_res[0].boxes:
                        c_cls_id = int(cb.cls[0].item())
                        c_cls_name = self.model.names.get(c_cls_id, str(c_cls_id))
                        if c_cls_name != "Person":  # Chỉ lấy trang bị bảo hộ
                            c_conf = float(cb.conf[0].item())
                            cx1, cy1, cx2, cy2 = cb.xyxy[0].cpu().numpy().astype(int)
                            fx1 = c_x1 + cx1
                            fy1 = c_y1 + cy1
                            fx2 = c_x1 + cx2
                            fy2 = c_y1 + cy2

                            # Kiểm tra tránh trùng lặp với item đã có
                            is_item_dup = False
                            for exist_it in all_ppe_items:
                                if exist_it['cls'] == c_cls_name:
                                    ex1, ey1, ex2, ey2 = exist_it['box']
                                    ix1 = max(fx1, ex1)
                                    iy1 = max(fy1, ey1)
                                    ix2 = min(fx2, ex2)
                                    iy2 = min(fy2, ey2)
                                    if ix2 > ix1 and iy2 > iy1:
                                        inter = (ix2 - ix1) * (iy2 - iy1)
                                        area = (fx2 - fx1) * (fy2 - fy1)
                                        if inter / area > 0.3:
                                            is_item_dup = True
                                            break

                            if not is_item_dup:
                                all_ppe_items.append({'cls': c_cls_name, 'conf': c_conf, 'box': (fx1, fy1, fx2, fy2)})
                                if c_cls_name in counts:
                                    counts[c_cls_name] += 1

            # 3. Vẽ các trang bị bảo hộ được phát hiện (Mũ, Áo, Găng tay...) theo màu riêng biệt
            for item in all_ppe_items:
                cls_name = item['cls']
                conf = item['conf']
                ix1, iy1, ix2, iy2 = item['box']
                color = CLASS_COLORS.get(cls_name, (0, 255, 0))
                display_label = CLASS_DISPLAY_TAG.get(cls_name, cls_name)
                label_text = f"{display_label} {conf*100:.0f}%"

                # Vẽ box trang bị nét mảnh (2px)
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (ix1, max(0, iy1 - 18)), (ix1 + w + 4, max(0, iy1)), color, -1)
                cv2.putText(
                    frame,
                    label_text,
                    (ix1 + 2, max(12, iy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

            # 4. Đánh giá từng người: Đủ hay thiếu trang bị bắt buộc
            req_helmet = self.var_req_helmet.get() if hasattr(self, 'var_req_helmet') else True
            req_vest = self.var_req_vest.get() if hasattr(self, 'var_req_vest') else True
            req_gloves = self.var_req_gloves.get() if hasattr(self, 'var_req_gloves') else True

            for person in all_persons:
                px1, py1, px2, py2 = person['box']
                p_conf = person['conf']

                # Tìm các món bảo hộ thuộc về người này (tâm nằm trong box người + mở rộng biên)
                p_items = []
                for it in all_ppe_items:
                    ibx1, iby1, ibx2, iby2 = it['box']
                    cx = (ibx1 + ibx2) / 2
                    cy = (iby1 + iby2) / 2
                    if (px1 - 30 <= cx <= px2 + 30) and (py1 - 45 <= cy <= py2 + 45):
                        p_items.append(it['cls'])

                has_h = ('helmet' in p_items)
                has_v = ('Vest' in p_items)
                has_g = ('gloves' in p_items)

                missing = []
                if req_helmet and not has_h:
                    missing.append("Mu")
                if req_vest and not has_v:
                    missing.append("Ao")
                if req_gloves and not has_g:
                    missing.append("Gang tay")

                # Xử lý màu sắc và nhãn theo yêu cầu:
                # - Nếu thiếu: Box viền MÀU ĐỎ và nhãn THIẾU
                # - Nếu đủ: Box viền MÀU CŨ (XANH LÁ/VÀNG) và nhãn ĐỦ BẢO HỘ
                if missing:
                    box_color = (0, 0, 255)       # MÀU ĐỎ VI PHẠM (BGR)
                    box_thickness = 3
                    badge_text = f"[!] THIEU: {', '.join(missing)}"
                else:
                    box_color = (0, 220, 0)       # MÀU XANH LÁ ĐỦ BẢO HỘ
                    box_thickness = 2
                    badge_text = "[OK] DU BAO HO"

                # Vẽ box người
                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, box_thickness)

                # Vẽ banner trạng thái vi phạm / an toàn phía trên đầu
                (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (px1, max(0, py1 - 25)), (px1 + bw + 8, max(0, py1)), box_color, -1)
                cv2.putText(
                    frame,
                    badge_text,
                    (px1 + 4, max(17, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Thêm nhãn phụ tên người + % ở góc dưới box
                sub_label = f"Nguoi {p_conf*100:.0f}%"
                (sw, sh), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (px1, py2), (px1 + sw + 6, min(frame.shape[0], py2 + 18)), box_color, -1)
                cv2.putText(
                    frame,
                    sub_label,
                    (px1 + 3, min(frame.shape[0] - 3, py2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        except Exception as e:
            print(f"Lỗi inference: {e}")

        return frame, counts

    def update_statistics_and_alert(self, counts):
        """Cập nhật các số liệu trên Sidebar và kích hoạt banner cảnh báo an toàn"""
        for cls_name, count in counts.items():
            if cls_name in self.stat_labels:
                lbl = self.stat_labels[cls_name]
                lbl.config(text=str(count))
                if count > 0:
                    lbl.config(fg="#fab387" if cls_name == "Person" else "#a6e3a1")
                else:
                    lbl.config(fg="#6c7086")

        # Logic đánh giá an toàn:
        # Nếu có Người (Person > 0)
        num_persons = counts.get('Person', 0)
        num_helmets = counts.get('helmet', 0)
        num_vests = counts.get('Vest', 0)
        num_gloves = counts.get('gloves', 0)

        if num_persons > 0:
            missing = []
            if getattr(self, 'var_req_helmet', None) and self.var_req_helmet.get() and num_helmets < num_persons:
                missing.append("MŨ")
            if getattr(self, 'var_req_vest', None) and self.var_req_vest.get() and num_vests < num_persons:
                missing.append("ÁO")
            if getattr(self, 'var_req_gloves', None) and self.var_req_gloves.get() and num_gloves < num_persons:
                missing.append("GĂNG TAY")

            if missing:
                msg = f"⚠️ CẢNH BÁO: PHÁT HIỆN THIẾU {', '.join(missing)}!"
                self.banner_alert.config(text=msg, bg="#f38ba8", fg="#11111b")
            else:
                self.banner_alert.config(text="✅ AN TOÀN: ĐẦY ĐỦ ĐỒ BẢO HỘ", bg="#a6e3a1", fg="#11111b")
        else:
            self.banner_alert.config(text="⚪ ĐANG GIÁM SÁT (KHÔNG CÓ NGƯỜI)", bg="#313244", fg="#cdd6f4")

    def display_frame_on_canvas(self, frame):
        """Resize frame phù hợp kích thước Canvas và hiển thị"""
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            return

        h, w, _ = frame.shape
        scale = min(canvas_w / w, canvas_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)

        self.canvas_video.delete("all")
        # Canh giữa màn hình
        x_center = canvas_w // 2
        y_center = canvas_h // 2
        self.canvas_video.create_image(x_center, y_center, image=img_tk, anchor="center")
        self.canvas_video.image = img_tk  # Giữ reference tránh bị garbage collector xóa

    def on_canvas_resize(self, event):
        if not self.is_running and self.last_frame is not None:
            self.display_frame_on_canvas(self.last_frame)

    def take_snapshot(self):
        """Chụp ảnh vi phạm/hiện trạng kèm timestamp lưu vào đĩa"""
        if self.last_frame is None:
            messagebox.showinfo("Thông báo", "Chưa có hình ảnh để chụp!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"PPE_Snapshot_{timestamp}.jpg"
        save_path = os.path.join(self.snapshot_dir, filename)

        cv2.imwrite(save_path, self.last_frame)
        messagebox.showinfo("Đã lưu ảnh", f"Đã lưu ảnh chụp vi phạm tại:\n{save_path}")

    def open_snapshot_folder(self):
        try:
            os.startfile(self.snapshot_dir)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở thư mục:\n{e}")

    def on_closing(self):
        self.stop_stream()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = PPEDetectionApp(root)
    root.mainloop()
