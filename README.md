# Hệ Thống Nhận Diện Đồ Bảo Hộ Lao Động (PPE AI) qua Camera Imou

Ứng dụng nhận diện và giám sát trang thiết bị bảo hộ lao động (Personal Protective Equipment - PPE) tự động trong thời gian thực bằng mô hình AI Deep Learning (YOLO), lấy dữ liệu trực tiếp từ camera Imou qua cổng mạng RJ45/LAN bằng giao thức RTSP.

---

## 📌 1. Các trang bị bảo hộ mô hình nhận diện
Mô hình `best.pt` phát hiện được 9 lớp đối tượng:
1. **Person** (Người lao động)
2. **helmet** (Mũ bảo hộ)
3. **Vest** (Áo bảo hộ / Áo phản quang)
4. **gloves** (Găng tay bảo hộ)
5. **Boots** (Ủng bảo hộ)
6. **shoes** (Giày bảo hộ)
7. **Mask** (Khẩu trang)
8. **glasses** (Kính bảo hộ)
9. **Ear-protection** (Bịt tai chống ồn)

---

## 🚀 2. Cách khởi động ứng dụng

### Cách 1: Chạy nhanh
*Tải full code.
* Nhấp đúp chuột vào file **`run.bat`** để khởi chạy ứng dụng.

### Cách 2: Chạy bằng dòng lệnh Terminal
```powershell
cd E:\vscode\cnpm
python app.py
```

---

## ⚙️ 3. Hướng dẫn sử dụng trên màn hình ứng dụng

1. **IP Camera:** Đã được điền sẵn mặc định là `192.168.1.108`.
2. **Safety Code:** Đã được điền sẵn mặc định là `L2C2A89B`.
3. **Chọn loại luồng (Khuyên dùng):**
   * **Luồng 1 (Sub-stream - Mặc định):** Luồng nén tối ưu băng thông mạng LAN, giúp mô hình AI phân tích mượt mà với FPS cao và không bị trễ hình.
   * **Luồng 0 (Main-stream):** Luồng phân giải cao full HD.
4. **Bấm "▶ BẮT ĐẦU GIÁM SÁT":**
   * Video từ camera sẽ hiển thị trực tiếp lên khung hình chính.
   * **Cảnh báo trực tiếp trên khung hình (On-Frame Visual):**
     * **Nếu thiếu trang bị:** Khung người chuyển sang **MÀU ĐỎ** viền đậm và gắn nhãn nổi bật `[!] THIEU: Mu, Ao, Gang tay...`.
     * **Nếu đầy đủ trang bị:** Khung người giữ **MÀU XANH LÁ** an toàn và gắn nhãn `[OK] DU BAO HO`.
     * Các trang bị phát hiện được (Mũ, Áo, Găng tay...) được đóng khung theo màu riêng biệt.
   * **Cột cảnh báo bên phải:** Tự động mở rộng và xuống dòng mềm mại, hiển thị rõ ràng không bị tràn viền hay mất chữ.
   * **Quy chuẩn trang bị bắt buộc:** Cho phép tích/bỏ tích chọn Mũ, Áo, Găng tay linh hoạt theo yêu cầu.
5. **Chụp ảnh vi phạm:**
   * Bấm nút **"📸 Chụp Vi Phạm"** để lưu lại khoảnh khắc vi phạm kèm bounding box. Ảnh sẽ được lưu tự động trong thư mục `E:\vscode\cnpm\snapshots`.
   * Bấm **"📁 Mở Thư Mục Ảnh"** để xem lại các ảnh đã chụp.
