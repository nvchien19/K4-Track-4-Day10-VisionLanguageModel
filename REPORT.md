# Báo cáo Lab Day 10 — Qwen3-VL road damage

Họ tên: Nguyễn Văn Chiến (2A202602926)

`dataset_version`: `d10-21e4070301`

> Cập nhật theo kết quả đã thực sự chạy trên Colab T4. E0 dùng kích thước ảnh mặc định 640×360; E1 và E2 dùng 320×180 để tiết kiệm VRAM/thời gian nên **E0 và E1 chưa phải phép so sánh công bằng**. E2 đã train đủ 300 bước nhưng chưa có điểm do runtime ngắt kết nối; E3/E4 chưa chạy. Không suy diễn số liệu còn thiếu.

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot, ảnh 640×360 | 0 | chưa đo | chưa đo | 0.051 | 0.029–0.078 | 0.80 | 0.124 | 0.008 | 0.097 |
| E1 | LoRA LLM r=16, ảnh 320×180 | 33,030,144 | 9.076 | 26.49 | 0.071 | 0.052–0.094 | 1.77 | 0.025 | 0.063 | 0.132 |
| E2 | E1 + projector, ảnh 320×180; đã train 300/300 bước, chưa chấm xong | 142,135,296 | 6.355 | 40.26 | — | — | — | — | — | — |
| E3 | E2 + LoRA ViT | chưa chạy | — | — | — | — | — | — | — | — |
| E4 | Chưa chạy | — | — | — | — | — | — | — | — | — |

Kaggle đã xác minh và cấp 2 GPU Tesla T4. Chạy lại E0 trên Kaggle (640×360, 311 ảnh) cho `valid=0.650`, P=0.082, R=0.038, F1=0.052, CI95 0.029–0.079, meanIoU=0.619, FP/ảnh=0.79; per-class F1: pothole 0.124, crack 0.008, manhole 0.097. Kết quả khớp với lần E0 trước trong sai số làm tròn. E1 Kaggle được thử ở 640×360 nhưng OOM ở batch 2 ngay forward đầu tiên; chưa có checkpoint E1 từ lần chạy này. E0 có `parse_valid_rate` 0.653; E1 Colab là 0.971. E1 loss cuối trung bình 100 lần cập nhật là 0.6051. Checkpoint E1 đã được đồng bộ lên Google Drive. E1 F1 chỉ cao hơn E0 0.020 trong khi thay đổi cả độ phân giải, vì vậy chưa thể kết luận LoRA cải thiện chất lượng.

## 10 câu hỏi

### Câu 1 — Kiến trúc và khối được cập nhật

Theo GUIDE, Qwen3-VL-4B có khoảng 4.0B tham số LLM, 415M ViT; projector gồm `visual.merger` và `visual.deepstack_merger_list`, dự kiến khoảng 109M. E1 cập nhật LoRA trên 252 module LLM, E2 mở thêm projector, E3 thêm LoRA trên các block ViT (tổng mục tiêu LoRA 348 module). E1 thực tế có 33,030,144 tham số huấn luyện. Sau khi giữ toàn bộ visual ở dense precision, projector đo được 109,105,152 tham số; E2 có tổng 142,135,296 tham số huấn luyện. Giá trị 54,579,200 ở lần thử cũ là phép đếm trên model visual bị lượng tử hóa 4-bit, không đại diện cho projector dense của E2.

### Câu 2 — E0

E0 có `parse_valid_rate = 0.653`. Trên test, precision 0.081, recall 0.038, F1 0.051; theo lớp: pothole F1 0.124, crack 0.008, manhole 0.097. Hai điểm yếu nhìn thấy qua số liệu là nhiều output không parse được (34.7%) và khả năng phát hiện thấp (recall 0.038). Notebook chưa lưu ví dụ raw output đại diện nên chưa trích dẫn một chuỗi cụ thể.

### Câu 3 — Thang tọa độ

Chưa chạy thí nghiệm đối chứng chấm cùng dự đoán theo pixel và theo thang 0–1000. Vì vậy hiện chưa có F1 để báo cáo. Các nhãn huấn luyện được chuyển từ YOLO chuẩn hoá sang số nguyên 0–1000; E4 cần bổ sung phép kiểm tra sai thang có kiểm soát.

### Câu 4 — Mask loss

Mask phần prompt bằng `-100` để loss chỉ tối ưu câu trả lời JSON. Nếu tính loss cả prompt, phần lớn mục tiêu sẽ là học lại prompt cố định và token cấu trúc/vision, làm tín hiệu cho tọa độ và lớp bị loãng. Notebook kiểm tra prompt là tiền tố của chuỗi đầy đủ và chỉ mask token prompt; mẫu đã encode có 250 token được giám sát. Bản kiểm tra ảnh cho `image_grid_thw = [1, 22, 40]` ở 640×360: 880 patch trên lưới trước khi gộp không gian 2×2, tương ứng 220 token vision sau khi gộp như README mô tả.

### Câu 5 — Projector lượng tử hoá

Projector lượng tử hoá 4-bit không thể được cập nhật đầy đủ như tensor tham số floating point thông thường. Bài yêu cầu giữ ViT/projector ở floating point; riêng projector được nâng lên float32 để cập nhật ổn định. Lần thử cũ có `Linear4bit` trong projector, làm bias bị ép về fp16 và tham số không hữu hạn sau optimizer. Lần chạy mới dùng `llm_int8_skip_modules=['model.visual','lm_head']` trong cấu hình NF4: kiểm tra thực tế thấy **0 `Linear4bit` trong visual**, projector là `Linear`. E2 qua thử forward/backward với loss 1.1067, gradient hữu hạn và VRAM đỉnh 8.133 GB ở bước thử; sau đó train đủ 300 bước, nhưng chưa có điểm test.

### Câu 6 — Chia theo phiên quay

Chia ngẫu nhiên có thể đưa các frame gần như trùng nhau của cùng video vào train và test, khiến kết quả test lạc quan giả. Split thực tế, không trùng session: train 1,422 ảnh (pothole 1,074; crack 1,750; manhole 691), validation 276 ảnh (66; 470; 103), test 311 ảnh (121; 299; 163). Tổng số box lần lượt là 3,515; 639; 583.

### Câu 7 — E1/E2/E3

Chưa thể so sánh đủ ba setup: E1 có số liệu ở 320×180; E2 đã train đủ 300 bước với projector dense ở cùng độ phân giải nhưng chưa chấm xong do mất runtime; E3 chưa chạy. E1 đạt F1 0.071, CI95 0.052–0.094, VRAM đỉnh 9.076 GB, 26.49 phút. Chênh lệch với E0 chưa công bằng do độ phân giải đầu vào khác nhau.

### Câu 8 — Giới hạn của test

Với 311 ảnh, CI95 của E0 là 0.029–0.078 (độ rộng 0.049), E1 là 0.052–0.094 (độ rộng 0.042). Hai khoảng chồng lấp và độ phân giải khác nhau, nên hiện không có bằng chứng đủ để khẳng định E1 tốt hơn E0. Cần so sánh trên cấu hình đầu vào thống nhất và hoàn tất các lần chạy còn thiếu.

### Câu 9 — Lỗi theo lớp

Ở E0, crack kém nhất (F1 0.008; P 0.011, R 0.007), tiếp theo manhole (F1 0.097) và pothole (F1 0.124). Ở E1, pothole thấp nhất (F1 0.025), rồi crack (0.063) và manhole (0.132). Số box train là crack 1,750, pothole 1,074, manhole 691; test lần lượt 299, 121, 163. Crack có nhiều ví dụ hơn nhưng E0 vẫn khó nhận ra; một giả thuyết là các vết nứt mảnh/nhỏ khó định vị chính xác. Thứ hạng thay đổi ở E1, nên chưa thể quy nguyên nhân cho tần suất hay hình dạng lớp nếu không xem các lỗi từng ảnh. Hình dưới minh hoạ F1 theo lớp; đây không phải ảnh dự đoán/ground-truth. E0 và E1 dùng độ phân giải khác nhau, vì vậy biểu đồ chỉ mô tả từng kết quả, không chứng minh E1 tốt hơn.

![F1 theo lớp trên test của E0 và E1](report/figures/class_f1.svg)

### Câu 10 — E4

Biến E4 được chọn là rank LoRA: so sánh r=4 với r=64 trên setup tốt nhất theo F1 validation, giữ dữ liệu, độ phân giải 320×180, số bước 300 và các siêu tham số khác cố định. Sau khi chọn setup/rank bằng validation mới chấm test một lần. Thí nghiệm chưa chạy: Kaggle đã cấp GPU (2×T4), nhưng E1 ở 640×360 OOM trước khi có checkpoint; vì vậy chưa thể chọn setup tốt nhất bằng validation để chạy rank ablation và chưa có F1/CI95. Hàng E4 trong bảng vẫn để trống có chủ đích, không thay bằng số ước đoán.

## Nhìn lại

Đã tải và chia dữ liệu theo session đúng phiên bản `d10-21e4070301`, kiểm tra không rò session, hoàn tất zero-shot E0 và huấn luyện E1 đủ 300 optimizer steps; adapter E1 được lưu trên Drive từ phiên trước. Khó khăn là notebook có nhiều cell sửa thử nên trạng thái hàm từng không nhất quán (`NameError: write_jsonl`), và lần E2 đầu dùng projector 4-bit. Trong phiên [Colab T4 hiện tại](https://colab.research.google.com/drive/1_ARbg2AKsMI9ihEgHBAmbz3CP0ym6JY9?authuser=1), đã tải lại dữ liệu/model, xác nhận visual dense và **hoàn tất E2 300/300 optimizer steps** ở 320×180, batch 1, gradient accumulation 8. `work/e2_r16/run.json` ghi 142,135,296 tham số train, VRAM đỉnh 6.355 GB trong vòng train, thời gian 40.26 phút; checkpoint adapter và projector đã được ghi trong runtime. Lượt suy luận E2 đầu tiên lỗi dtype Half/Float ở `generate`; đã sửa bằng autocast và chạy lại, nhưng Colab mất kết nối runtime trước khi có điểm. Drive của tài khoản Colab hiện tại chưa mount thành công, nên checkpoint E2 **chưa được xác nhận là còn truy cập được** sau ngắt kết nối. Notebook trên Drive vẫn giữ log huấn luyện. Kaggle cũ báo lỗi phiên tương tác; notebook Kaggle mới đang thử khởi động T4. E3/E4 sẽ chỉ được báo cáo sau khi thực sự chạy xong.
