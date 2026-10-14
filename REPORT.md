# Báo cáo Lab Day 10 — Qwen3-VL road damage

Họ tên: Nguyễn Văn Chiến (2A202602926)

`dataset_version`: `d10-21e4070301`

> Cập nhật theo kết quả đã thực sự chạy trên Colab T4. E0 dùng ảnh 640×360; E1 và E2 dùng 320×180, vì vậy so sánh E0 với E1/E2 chưa công bằng. E2 đã train 300 bước và chấm đủ 311 ảnh test. Đã thử khởi chạy E3 nhưng T4 hết VRAM ngay ở forward đầu tiên; chưa có checkpoint hay điểm E3. E4 chưa chạy; không suy diễn số liệu còn thiếu.

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot, ảnh 640×360 | 0 | chưa đo | chưa đo | 0.051 | 0.029–0.078 | 0.80 | 0.124 | 0.008 | 0.097 |
| E1 | LoRA LLM r=16, ảnh 320×180 | 33,030,144 | 9.076 | 26.49 | 0.071 | 0.052–0.094 | 1.77 | 0.025 | 0.063 | 0.132 |
| E2 | E1 + projector, ảnh 320×180; train 300/300 | 142,135,296 | 11.267 | 24.61 | 0.070 | 0.049–0.090 | 1.60 | 0.065 | 0.049 | 0.146 |
| E3 | E2 + LoRA ViT | 148,426,752 (đếm lúc setup) | OOM: 14.53/14.56 | — | — | — | — | — | — | — |
| E4 | Chưa chạy | — | — | — | — | — | — | — | — | — |

E0 có `parse_valid_rate` 0.653; E1 là 0.971; E2 là 0.990. E1 loss cuối trung bình 100 lần cập nhật là 0.6051; E2 là 0.6290. E2 đạt F1 0.070, CI95 0.049–0.090, FP/ảnh 1.60. E1 và E2 dùng cùng kích thước ảnh; CI95 chồng lấp, nên chưa có cơ sở khẳng định mở projector cải thiện F1. Theo yêu cầu của người học, phiên làm việc tiếp tục hoàn toàn trên Colab, không dùng Kaggle.

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

Projector lượng tử hoá 4-bit không thể được cập nhật đầy đủ như tensor tham số floating point thông thường. Bài yêu cầu giữ ViT/projector ở floating point; riêng projector được nâng lên float32 để cập nhật ổn định. Lần thử cũ có `Linear4bit` trong projector và phát sinh lỗi dtype/optimizer. Lần chạy mới dùng `llm_int8_skip_modules=['model.visual','lm_head']` trong cấu hình NF4: kiểm tra thực tế thấy **0 `Linear4bit` trong visual**, projector là `Linear`. E2 qua thử forward/backward với loss hữu hạn, sau đó train đủ 300 bước. Kết quả chính: 142,135,296 tham số trainable, peak VRAM 11.267 GB, 24.61 phút, loss cuối 0.6290; điểm test F1 0.070 (CI95 0.049–0.090).

### Câu 6 — Chia theo phiên quay

Chia ngẫu nhiên có thể đưa các frame gần như trùng nhau của cùng video vào train và test, khiến kết quả test lạc quan giả. Split thực tế, không trùng session: train 1,422 ảnh (pothole 1,074; crack 1,750; manhole 691), validation 276 ảnh (66; 470; 103), test 311 ảnh (121; 299; 163). Tổng số box lần lượt là 3,515; 639; 583.

### Câu 7 — E1/E2/E3

E1 đạt F1 0.071 (CI95 0.052–0.094), VRAM 9.076 GB, 26.49 phút. E2 đạt F1 0.070 (CI95 0.049–0.090), VRAM 11.267 GB, 24.61 phút. Hai thí nghiệm dùng cùng độ phân giải; khoảng tin cậy chồng lấp và E2 chưa cho thấy lợi ích F1 rõ rệt so với E1 dù thêm projector. E3 đã nạp đúng 348 mục tiêu LoRA nhưng forward đầu tiên vượt VRAM T4 (14.53/14.56 GB), nên không có bước tối ưu hay checkpoint E3. Chưa thể so sánh đủ ba setup. E0 dùng độ phân giải cao hơn nên không so trực tiếp với E1/E2.

### Câu 8 — Giới hạn của test

Với 311 ảnh, CI95 của E0 là 0.029–0.078 (độ rộng 0.049), E1 là 0.052–0.094 (0.042), E2 là 0.049–0.090 (0.042). E1/E2 có cùng độ phân giải nhưng khoảng tin cậy chồng lấp, nên khác biệt chưa thuyết phục. E0/E1 còn khác độ phân giải. Cần giữ cùng cấu hình đầu vào và hoàn tất E3/E4 trước khi kết luận.

### Câu 9 — Lỗi theo lớp

Ở E0, crack kém nhất (F1 0.008; P 0.011, R 0.007), tiếp theo manhole (0.097) và pothole (0.124). Ở E1, pothole thấp nhất (0.025), rồi crack (0.063), manhole (0.132). Ở E2, crack thấp nhất (0.049), tiếp theo pothole (0.065), manhole (0.146). Số box train là crack 1,750, pothole 1,074, manhole 691; test lần lượt 299, 121, 163. Crack có nhiều ví dụ hơn nhưng vẫn khó ở E0/E2; giả thuyết là các vết nứt mảnh/nhỏ khó định vị. Thứ hạng thay đổi theo setup, nên chưa thể quy nguyên nhân cho tần suất hay hình dạng nếu không xem lỗi từng ảnh. Hình dưới minh hoạ F1 theo lớp; đây không phải ảnh dự đoán/ground-truth. E0 dùng độ phân giải khác E1/E2 nên biểu đồ mô tả từng kết quả, không chứng minh setup nào tốt hơn.

![F1 theo lớp trên test của E0 và E1](report/figures/class_f1.svg)

### Câu 10 — E4

Chọn biến E4 là rank LoRA: dự kiến so sánh r=4 và r=64 trên E2, giữ dữ liệu, độ phân giải 320×180, số bước và các siêu tham số khác cố định; baseline là E2 r=16. E4 chưa được huấn luyện hoặc chấm, nên không có F1/CI95 và chưa tạo evidence. Có thể ước tính số tham số trainable theo tuyến tính với rank từ E2 r=16: projector giữ 109,105,152 tham số; LoRA r=16 có 33,030,144. Vì vậy E2 r=4 ước tính 117,362,688 tổng tham số trainable và E2 r=64 ước tính 241,225,728. Đây chỉ là phép tính quy mô adapter, không phải kết quả chạy hoặc file evidence hợp lệ. Cần chạy hai rank và chấm test theo hướng dẫn để hoàn tất E4.

## Nhìn lại

Đã chia dữ liệu theo session đúng phiên bản `d10-21e4070301`, kiểm tra không rò session, có kết quả E0, E1 và E2. E2 chạy trên Colab T4, giữ visual dense (0 `Linear4bit`), projector trainable FP32; train 300/300 ở 320×180, batch 1, gradient accumulation 8, đạt 142,135,296 tham số trainable, peak VRAM 11.267 GB, 24.61 phút, loss cuối 0.6290. Checkpoint, predictions và evidence E2 đã đồng bộ vào Drive `/MyDrive/road_damage_lab_2A202602926`; test E2 hoàn tất đủ 311 ảnh, F1 0.070 (CI95 0.049–0.090), parse-valid 0.990. Lần thử E3 hiện tại không qua được forward đầu tiên trên T4, chưa có checkpoint/evidence. E4 chưa chạy. Tài khoản Colab/Drive hiện tại chỉ thấy artifacts E2; evidence E0/E1 đang không có trong runtime/Drive này. Bộ nộp chưa hoàn chỉnh: còn thiếu evidence E0, E1, E3 và E4.
