# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Mô hình | f1 | f1_SUB |
| :--- | :---: | :---: |
| 1. BoW + LogReg (tự code) | 0.885 | 0.788 |
| 2. BoW + MLP (deep learning) | 0.957 | 0.889 |
| 3a. Encoder (MiniLM) + kNN | 0.873 | 0.974 |
| 3b. Encoder (MiniLM) + LogReg | 0.958 | 1.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.938 | 0.974 |
| 4a. LLM nhỏ zero-shot | 0.588 | 0.588 |
| 4b. LLM lớn zero-shot | 0.865 | 0.865 |
| 4c. LLM nhỏ few-shot | 0.500 | 0.500 |
## 3. Dự đoán và thực tế

**1. `emb_search` (tìm kiếm theo nghĩa)**
- Đoán: kết quả gần nhất là `ham`, vì query không chứa "free", "win", "prize".
- Ra: cả 6 tin gần nhất đều là spam (cosine 0.50–0.55), ví dụ tin "…won… Spanish holiday! CALL NOW…", "…collect your 4* Tenerife Holiday…".
- Vì sao lệch: tôi coi embedding như bộ so khớp từ khoá. Thực tế MiniLM so khớp theo nghĩa ("selected", "complimentary holiday", "claim" ≈ trúng thưởng/kỳ nghỉ), nên vẫn kéo về vùng spam dù không trùng từ.

**2. `llm_fewshot` (thêm 4 ví dụ vào prompt)**
- Đoán: F1 tăng, chi phí mỗi tin tăng khoảng ×10.
- Ra: F1 của LLM nhỏ (qwen2.5:0.5b) giảm 0.588 → 0.500 (recall 0.75 → 0.45, precision 0.484 → 0.562). Chi phí đo được là $0 vì chạy Ollama cục bộ. Theo số từ của prompt mẫu, prompt dài khoảng ×3 (ước lượng), không phải ×10.
- Vì sao lệch: model 0.5B học theo ngữ cảnh rất yếu. 4 ví dụ là 2 spam + 2 ham đầu tiên của train (không chọn theo độ đa dạng hay độ khó), nên không giúp mà còn làm model bỏ sót spam nhiều hơn. Tôi mặc định "thêm ví dụ = tốt hơn".

**3. `emb_tricky` (6 tin ngụy trang)**
- Đoán: 3b (Encoder + LogReg) đúng nhiều hơn BoW.
- Ra: hoà, mỗi bên đúng 5/6. Cả hai cùng sai tin "Last chance! Your account rewards expire tonight, confirm details to keep them" (đoán ham).
- Vì sao lệch: encoder không được train để nhận ra spam, phần "biết spam" nằm ở LogReg head. Head chỉ học từ 3.619 tin của bộ SMS cũ (chủ yếu trúng thưởng, nhạc chuông, số tính phí), nên kiểu "phần thưởng tài khoản sắp hết hạn" rơi vào vùng ham. 6 tin là mẫu quá nhỏ để kết luận ai hơn ai.

## 4. Phân tích 50 case khó

| Model | F1 test thường | F1 50 case khó | Tụt |
|---|---|---|---|
| 1. BoW + LogReg | 0.885 | 0.450 | −0.435 |
| 2. BoW + MLP | 0.957 | 0.476 | −0.481 |
| 3a. Encoder + kNN | 0.873 | 0.724 | −0.149 |
| 3b. Encoder + LogReg | 0.958 | 0.625 | −0.333 |
| 3c. Encoder + MLP head | 0.938 | 0.680 | −0.258 |

- Tụt nhiều nhất: **BoW + MLP** (−0.481), sau đó BoW + LogReg (−0.435). Accuracy của cả hai trên 50 case chỉ 56%.
- LLM nhỏ zero-shot (4a) gần như không đổi (0.588 → 0.567) vì vốn đã thấp. Với prompt phòng thủ (4d), F1 = 0.000 và accuracy 40%: model bỏ sót cả 29 tin spam.
- Nhóm khó nhất: **Spam giọng tự nhiên**. 4 case (#12, #13, #14, #16) làm cả 7 model cùng sai.
- [Bổ sung từ bảng "% đoán đúng theo nhóm": nhóm BoW sai nhiều nhất = …]

**Case #14** — "Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back" (spam, lừa việc làm)
- BoW cộng trọng số từng từ. Câu này gồm toàn từ đời thường (ví dụ "im", "me" nằm trong nhóm kéo về ham) và không có từ khoá spam đủ mạnh để kéo ngược lại.
- Embedding đọc câu như lời kể thân mật nên nằm gần cụm ham. Head chỉ học spam kiểu trúng thưởng/số tính phí, chưa từng thấy lừa việc làm.

**Case #12** — "hi there, sorry to bother you, my friend gave me your number, I'm new in town… check my profile at meetlocal.club" (spam, dụ vào trang hẹn hò)
- Mở đầu lịch sự, không có từ khoá. Tín hiệu spam duy nhất là tên miền "meetlocal.club", nằm ngoài vocab nên BoW bỏ qua.
- Encoder nén cả câu thành nghĩa "làm quen" nên gần tin ham hơn.

**Kết luận:** điểm trên test thường lạc quan vì test cùng phân phối với train (spam trúng thưởng/premium-rate lặp lại nhiều). Embedding chịu tốt hơn BoW (F1 0.625–0.724 so với 0.450–0.476), nhưng vẫn thua spam "nhẹ nhàng" không có từ khoá.

## 5. Thử thách cá nhân

- Đề: SEED = 6398, 6398 mod 6 = 2 → **đề #3**: đổi lớp ẩn MLP thành 4 và 512 nơ-ron; báo số tham số, F1 test, thời gian train; khi nào nên chọn mạng nhỏ.
- Cách làm: `build_mlp(2000, hidden)` + `train_mlp` (30 epoch, lr 1e-3, batch 64, `torch.manual_seed(SEED)`), cùng split train/val/test; đo tham số, thời gian và metrics trên test (775 tin, 97 spam). Mỗi cấu hình chạy 1 lần.
- Kết quả:

| Hidden | Tham số | Train (s) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 4 | 8,009 | 10.22 | 0.989 | 0.928 | 0.957 |
| 64 (đối chứng, Phần 2) | 128,129 | 5.1 | 0.989 | 0.928 | 0.957 |
| 512 | 1,025,025 | 32.24 | 0.978 | 0.928 | 0.952 |

- Giải thích:
  - Tham số tăng 128× (8,009 → 1,025,025) nhưng F1 không tăng, còn giảm 0.005. Recall giữ nguyên 0.928 (90/97 spam). Khác biệt duy nhất là thêm 1 tin ham bị chặn nhầm (FP từ 1 lên 2), nằm trong nhiễu của một lần chạy.
  - Đầu vào BoW thưa và gần tách tuyến tính (Thí nghiệm 2.B: bỏ ReLU vẫn F1 0.957), nên dung lượng mô hình không phải nút thắt. 1M tham số cho 3.619 mẫu train chỉ tăng nguy cơ overfit.
  - Thời gian: hidden 512 chậm khoảng 3× hidden 4. Hidden 4 (10.2 s) lại chậm hơn hidden 64 ở Phần 2 (5.1 s), nên số đo thời gian dao động theo tải Colab; chỉ xu hướng "512 chậm nhất" là đáng tin.
  - Nên chọn mạng nhỏ khi dữ liệu ít, khi cần độ trễ/bộ nhớ thấp (thiết bị biên, SLA chặt), và khi cần retrain thường xuyên. Chỉ tăng kích thước khi validation cho thấy lợi ích rõ ràng.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày ≈ 116 tin/s trung bình, p95 < 50 ms/tin.

- **Chọn:** 3b (MiniLM đóng băng + LogReg) làm bộ lọc chính. LLM chỉ xử lý phần nhỏ tin lưỡng lự (router), ngoài đường p95 hoặc chạy bất đồng bộ.
- Số liệu dẫn chứng:
  - Độ trễ p95: BoW+LogReg 0.105 ms, BoW+MLP 0.228 ms, 3c 36.9 ms, 3b 39.6 ms (đạt SLA). kNN 72.3 ms và LLM 2,470–10,326 ms không đạt (LLM đo bằng Ollama trên Colab, chưa đo qua API).
  - Chất lượng: F1 test 3b = 0.958 (BoW+MLP 0.957, BoW+LogReg 0.885). Trên 50 case khó, 3b F1 0.625 / accuracy 64% so với BoW+MLP 0.476 / 56%.
  - Chi phí LLM cho toàn bộ lưu lượng (gpt-6-luna, 100 token vào + 1 token ra, $0.10/$0.50 mỗi 1M token): $105/ngày ≈ $3,150/tháng. Nếu chỉ 3% lưu lượng qua LLM (giả định): ≈ $95/tháng. 3b không tốn phí API.
  - Thông lượng: 3b đạt 33 tin/s mỗi instance (đo theo lô trên Colab), cần ≥ 4 instance cho 116 tin/s; nếu đỉnh gấp 3 lần trung bình thì khoảng 11. BoW+MLP đạt 42,624 tin/s.
- Đánh đổi chấp nhận:
  - Trả khoảng 170× độ trễ so với BoW+MLP (39.6 ms so với 0.23 ms) và thêm hạ tầng, chỉ để F1 trên 50 case khó tăng từ 0.476 lên 0.625. Trên test thường hai model ngang nhau (0.958 so với 0.957).
  - Biên độ so với SLA chỉ khoảng 10 ms, đo trên 40 tin ở Colab, nên cần load test trên phần cứng thật.
  - Vẫn sai spam giọng tự nhiên (4 case cả 7 model cùng sai), nên cần cập nhật dữ liệu có nhãn định kỳ.
  - Router: nếu ≥ 5% tin đi qua LLM thì p95 chung bằng độ trễ LLM. Vì vậy cần vùng lưỡng lự hẹp hoặc xử lý bất đồng bộ (giữ tin, duyệt sau). Tỷ lệ tin lưỡng lự cần đo trên dữ liệu thật, lab chưa đo.

## 7. AI log và điều đã học

* **4.1 `call_openai`**: Yêu cầu AI viết hàm gọi chat completions trả về `(text, latency)` sử dụng `provider_kwargs`; tiến hành kiểm tra bằng API giả lập và đảm bảo truyền `**provider_kwargs` thay vì `max_tokens`.
* **4.2 prompt**: Yêu cầu AI viết prompt zero-shot/few-shot kết thúc bằng `Message: """…"""` và `Label:`; tiến hành kiểm tra định dạng kết thúc cùng các ví dụ few-shot, sau đó thực tế đo đạc cho thấy few-shot làm điểm F1 giảm từ 0.588 xuống 0.500.
* **4.3 `parse_label`**: Yêu cầu AI viết hàm đọc nhãn từ câu trả lời của LLM với khả năng chuẩn hóa chữ thường và bỏ dấu câu; tiến hành xử lý chuỗi "not spam" trước "spam", loại bỏ tiền tố "label:", đồng thời đối chiếu với các lỗi của hàm `parse_label_v2`.
* **4.4 `llm_classify_batch`**: Yêu cầu AI viết hàm phân loại hàng loạt kèm theo việc đếm token và tính chi phí thông qua vòng lặp gọi API sử dụng `retry_with_backoff`; tiến hành xử lý các tin không parse được thành ham và đếm vào `n_invalid`, chạy trên 40 tin SUB với lượng token ước lượng tính bằng số từ chia cho 0.75.|

**Điều hiểu ra hôm nay:** F1 cao trên test cùng phân phối chưa nói lên khả năng chống spam mới. BoW+MLP đạt 0.957, ngang 3b (0.958), nhưng chỉ còn 0.476 trên 50 case khó. Model lớn hơn hay prompt dài hơn không tự động tốt hơn: few-shot làm LLM 0.5B giảm F1 (0.588 → 0.500), và hidden 512 không hơn hidden 4. Embedding so khớp theo nghĩa chứ không theo từ khoá, nhưng "biết spam" vẫn phụ thuộc vào dữ liệu train của bộ phân loại phía trên.
