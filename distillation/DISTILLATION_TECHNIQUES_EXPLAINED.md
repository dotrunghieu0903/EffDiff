# Giải thích chi tiết các kỹ thuật Knowledge Distillation đã implement

> Dựa trên paper: **Yin et al., Improved Distribution Matching Distillation, NeurIPS 2024** và **SANA-Sprint LADD component (Chen et al., 2025)**
>
> File implement: `distillation/distilled.py`

---

## 1. DMD2 — Improved Distribution Matching Distillation (Yin et al., NeurIPS 2024)

### 1.1. Ý tưởng cốt lõi

DMD2 giải quyết bài toán: **làm sao để student model tạo ra ảnh chỉ trong 1–4 bước mà phân phối ảnh sinh ra vẫn khớp với phân phối của teacher (50 bước)?**

Thay vì ép student bắt chước từng cặp input→output của teacher (pointwise matching), DMD2 ép cho **phân phối toàn cục** (distribution) của student khớp teacher bằng cách so sánh hai "điểm số" (score):

- **Real score** $s_{\text{real}}$: Teacher ước lượng score trên dữ liệu sinh bởi student (= teacher biết dữ liệu thật "nên" như thế nào)
- **Fake score** $s_{\text{fake}}$: Một mạng riêng (Fake Score Network) ước lượng score trên chính dữ liệu sinh bởi student (= mạng này "nhớ" phân phối student đang tạo ra)

Hiệu giữa hai score cho ta hướng gradient để kéo phân phối student về phía phân phối thật.

### 1.2. `FakeScoreNetwork` — lines 82–131

```
FakeScoreNetwork = deep_copy(teacher_denoiser) + classification_head
```

- **`fake_denoiser`**: Bản sao sâu (deep copy) của teacher UNet/Transformer. Mạng này được huấn luyện riêng (requires_grad=True) để ước lượng score trên phân phối của student — tức là nó học cách denoise các ảnh **do student sinh ra** (không phải ảnh thật).

- **`cls_head`**: Một đầu phân loại nhỏ (classification head) gắn lên bottleneck features:
  ```
  AdaptiveAvgPool2d(1) → Flatten → Linear(bottleneck_dim, 256) → SiLU → Linear(256, 1)
  ```
  Đầu này đóng vai trò **discriminator GAN** — phân biệt latent thật vs latent do student sinh ra. Kích thước bottleneck phụ thuộc kiến trúc model:
  - SDXL: 1280 channels
  - Flux/SD3: 1536 channels
  - SANA: 2240 channels

- **Hai chức năng**:
  - `score_forward()`: Forward pass bình thường qua denoiser → dự đoán noise → dùng cho distribution matching
  - `classify()`: Forward qua cls_head → logit real/fake → dùng cho GAN loss

### 1.3. `DMD2Loss` — lines 134–267

#### A. Distribution Matching Loss (`compute_distribution_matching_loss`)

Quy trình tính:

1. **Lấy mẫu timestep** $t \sim \text{Uniform}[\text{min\_step}, \text{max\_step}]$, với min/max mặc định là 2%–98% tổng số timestep.

2. **Thêm noise** vào latent do student sinh ra: $x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1-\bar\alpha_t} \cdot \epsilon$

3. **Ước lượng real score**: Teacher dự đoán noise trên $x_t$, sau đó khôi phục $\hat{x}_{0,\text{real}}$ bằng công thức DDPM:
   $$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t} \cdot \hat\epsilon}{\sqrt{\bar\alpha_t}}$$

4. **Ước lượng fake score**: Tương tự nhưng dùng Fake Score Network.

5. **Tính gradient phân phối**:
   $$p_{\text{real}} = x_{\text{gen}} - \hat{x}_{0,\text{real}}$$
   $$p_{\text{fake}} = x_{\text{gen}} - \hat{x}_{0,\text{fake}}$$
   $$\text{grad} = \frac{p_{\text{real}} - p_{\text{fake}}}{\text{mean}(|p_{\text{real}}|)}$$

   Phép chia cho $\text{mean}(|p_{\text{real}}|)$ là **normalization trick** của DMD2 — giúp ổn định training bằng cách chuẩn hóa biên độ gradient theo scale thực tế.

6. **Pseudo-loss**: Vì gradient đã tính sẵn (stop-gradient), ta tạo MSE loss sao cho backward pass tạo đúng gradient mong muốn:
   $$\mathcal{L}_{\text{DM}} = \frac{1}{2} \|x_{\text{gen}} - (x_{\text{gen}} - \text{grad})\|^2$$

#### B. GAN Loss cho Generator (`compute_gan_loss_generator`)

Student cần đánh lừa discriminator (cls_head):
$$\mathcal{L}_{\text{GAN-G}} = \text{softplus}(-D(x_{\text{gen}}))$$

Softplus (thay vì log sigmoid) cho gradient mượt hơn khi D(x) rất lớn.

#### C. Fake Score Denoising Loss (`compute_fake_score_loss`)

Huấn luyện Fake Score Network bằng MSE denoising trên **dữ liệu do student sinh ra** (detach, không gradient ngược về student):
$$\mathcal{L}_{\text{fake}} = \|\hat\epsilon - \epsilon\|^2$$

#### D. GAN Loss cho Discriminator (`compute_gan_loss_discriminator`)

$$\mathcal{L}_{\text{GAN-D}} = \text{softplus}(D(x_{\text{fake}})) + \text{softplus}(-D(x_{\text{real}}))$$

### 1.4. Vòng lặp huấn luyện DMD2 (`_train_dmd2`) — lines 1017–1148

**Two-timescale update rule** — điểm mấu chốt của DMD2:

| Mỗi step | Mỗi `dfake_gen_update_ratio` step |
|---|---|
| Cập nhật Fake Score Network (denoising loss) | Cập nhật Student/Generator (DM + GAN loss) |
| Discriminator luôn được train | Generator chỉ train 1/N lần |

Mặc định `dfake_gen_update_ratio = 5` → Fake Score Network được cập nhật 5 lần cho mỗi 1 lần cập nhật generator. Lý do: mạng score cần hội tụ nhanh hơn generator để cung cấp gradient chính xác.

**Luồng thực thi mỗi step**:
```
noise → student(noise, t_max) → generated_latents (x_0)
    │
    ├── [Nếu generator step] → DM loss + GAN-G loss → backward → update student
    │
    └── [Mọi step] → fake score denoising loss → backward → update fake_score_net
```

---

## 2. CTCD — Continuous-Time Consistency Distillation (SANA-Sprint, Chen et al. 2025)

### 2.1. Ý tưởng cốt lõi

CTCD là phiên bản liên tục (continuous-time) của Consistency Distillation. Mục tiêu: huấn luyện student sao cho **mọi điểm trên quỹ đạo ODE** đều được ánh xạ về cùng $x_0$. Khi student hội tụ, chỉ cần 1 bước inference.

Điểm khác biệt so với Consistency Distillation rời rạc:
- Không cần chọn $\Delta t$ (khoảng cách giữa 2 timestep kề nhau)
- Dùng **JVP (Jacobian-Vector Product)** để ước lượng tangent trên quỹ đạo ODE liên tục
- Dùng **Trigonometric Flow** thay vì linear noise schedule

### 2.2. Trigonometric Flow — Parametrization

Thay vì $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ (DDPM), SANA-Sprint dùng:

$$x_t = \cos(t) \cdot x_0 + \sin(t) \cdot z \cdot \sigma_{\text{data}}$$

với $t \in [0, \pi/2]$:
- $t = 0$: $x_0$ sạch hoàn toàn (cos(0)=1, sin(0)=0)
- $t = \pi/2$: noise thuần túy (cos(π/2)=0, sin(π/2)=1)

Hàm `_sample_trigflow_timesteps()` (line 612) sample $t$ theo **logit-normal distribution**:
```python
u = sigmoid(randn() * logit_std + logit_mean)  # ∈ (0, 1)
t = u * π/2                                     # ∈ (0, π/2)
```
Logit-normal tập trung sample ở vùng mid-noise level — vùng mà teacher chênh lệch student nhiều nhất.

### 2.3. `CTCDLoss` — lines 277–440

Quy trình tính loss:

**Bước 1 — Tạo noisy latent theo trigonometric flow:**
$$x_t = \cos(t) \cdot x_0 + \sin(t) \cdot z \cdot \sigma_d$$

**Bước 2 — Teacher dự đoán velocity field:**
$$\frac{dx_t}{dt} = \sigma_d \cdot f_{\text{teacher}}\left(\frac{x_t}{\sigma_d}, t\right)$$

Teacher bị freeze (no_grad). Chia $x_t$ cho $\sigma_d$ trước khi đưa vào model là convention của SANA-Sprint.

**Bước 3 — Tính tangent directions cho JVP:**
$$v_x = \cos(t)\sin(t) \cdot \frac{dx_t/dt}{\sigma_d}, \quad v_t = \cos(t)\sin(t)$$

Đây là các vector tiếp tuyến dọc quỹ đạo ODE. Chúng cho biết "nếu ta dịch $(x_t, t)$ một chút dọc theo quỹ đạo, thì đầu ra student thay đổi như thế nào?"

**Bước 4 — Student forward + JVP bằng finite differences:**

Forward chuẩn:
$$F_\theta = f_{\text{student}}(x_t / \sigma_d, t)$$

JVP xấp xỉ bằng sai phân hữu hạn (vì native JVP của PyTorch khó dùng với model phức tạp):
$$F_{\theta,\text{jvp}} \approx \frac{f_{\text{student}}(x_t + \epsilon \cdot v_x, t + \epsilon \cdot v_t) - F_{\theta^-}}{\epsilon}$$
với $\epsilon = 10^{-3}$. $F_{\theta^-}$ là stop-gradient version (detach).

**Bước 5 — Tính tangent $g$:**
$$g = -\cos^2(t) \cdot (\sigma_d \cdot F_{\theta^-} - \frac{dx_t}{dt}) - r \cdot (\cos(t)\sin(t) \cdot x_t + \sigma_d \cdot F_{\theta,\text{jvp}})$$

Trong đó:
- Vế 1: "sai lệch" giữa student prediction và teacher velocity → ép student khớp teacher
- Vế 2: JVP tangent → ép consistency dọc quỹ đạo ODE
- $r = \min(1, \text{step} / \text{warmup})$: hệ số warmup — ban đầu chỉ ép khớp teacher, dần dần bật JVP tangent lên. Mặc định warmup 1000 step.

**Bước 6 — Tangent normalization:**
$$g \leftarrow \frac{g}{\|g\| + c}$$

với $c = 0.1$ (tangent_norm_constant). Ngăn gradient bùng nổ khi $g$ quá lớn ở early training.

**Bước 7 — Loss cuối cùng:**
$$\sigma = \tan(t) \cdot \sigma_d$$
$$w = 1/\sigma \quad \text{(weighting nghịch đảo noise level)}$$

Nếu model trả về **logvar** (log-variance, learnable):
$$\mathcal{L} = \frac{w}{\exp(\text{logvar})} \cdot \|F_\theta - F_{\theta^-} - g\|^2 + \text{logvar}$$

Logvar hoạt động như **adaptive weighting** — model tự học trọng số cho từng noise level. Khi model chưa tự tin → logvar lớn → loss bị down-weight. Khi model tự tin → logvar nhỏ → loss được up-weight.

Nếu không có logvar:
$$\mathcal{L} = w \cdot \|F_\theta - F_{\theta^-} - g\|^2$$

---

## 3. LADD — Latent Adversarial Diffusion Distillation (SANA-Sprint)

### 3.1. Ý tưởng

CTCD một mình thiếu "sharpness" — ảnh sinh ra có thể mờ (blurry) vì loss chỉ dựa trên MSE. LADD bổ sung **adversarial loss** trong không gian latent để tăng độ sắc nét và chi tiết.

### 3.2. `LatentDiscriminator` — lines 443–492

```
LatentDiscriminator = frozen_teacher_backbone + trainable_classification_head
```

- **Backbone**: Deep copy teacher, hoàn toàn freeze. Dùng như feature extractor — ý tưởng là teacher đã học các feature tốt, chỉ cần gắn thêm đầu phân loại.

- **Head**: Trainable MLP:
  ```
  AdaptiveAvgPool2d(4) → Flatten → Linear(ch*16, 512) → SiLU → Linear(512, 128) → SiLU → Linear(128, 1)
  ```
  Kích thước `latent_channels` phụ thuộc kiến trúc:
  - SDXL: 4 channels (VAE latent 4-dim)
  - Flux/SD3/SANA: 16 channels

- **Forward**: Đưa latent qua backbone (no_grad) → lấy feature → qua head → logit real/fake.

**Tại sao dùng noise-augmented latent?** Discriminator nhận latent đã thêm noise ở các mức khác nhau (noise augmentation). Điều này:
- Ngăn discriminator overfit vào artifact ở noise level cụ thể
- Cung cấp multi-scale feedback cho student

### 3.3. `LADDLoss` — lines 495–557

Hỗ trợ 2 kiểu loss:

**Hinge loss:**
- Generator: $\mathcal{L}_G = -\mathbb{E}[D(x_{\text{fake}})]$
- Discriminator: $\mathcal{L}_D = \frac{1}{2}[\mathbb{E}[\text{ReLU}(1 - D(x_{\text{real}}))] + \mathbb{E}[\text{ReLU}(1 + D(x_{\text{fake}}))]]$

**Cross-entropy loss:**
- Generator: $\mathcal{L}_G = \text{BCE}(D(x_{\text{fake}}), 1)$
- Discriminator: $\mathcal{L}_D = \text{BCE}(D(x_{\text{real}}), 1) + \text{BCE}(D(x_{\text{fake}}), 0)$

Mặc định dùng **hinge** (ổn định hơn cho diffusion model).

### 3.4. Vòng lặp CTCD + LADD (`_train_ctcd` với `use_ladd=True`) — lines 1150–1317

**Alternating G/D phases** — luân phiên huấn luyện generator và discriminator:

```
Phase G: Student + CTCD loss + LADD generator loss → update student
    ↓
Phase D: Discriminator real/fake classification loss → update discriminator head
    ↓
Phase G: ...
```

**Chi tiết Phase G:**
1. Tính CTCD loss (như mô tả ở phần 2)
2. Student sinh $\hat{x}_0$ từ $x_t$: $\hat{x}_0 = \cos(t) \cdot x_t - \sin(t) \cdot F_\theta \cdot \sigma_d$
3. Thêm noise augmentation cho $\hat{x}_0$: $x_{t_D} = \cos(t_D) \cdot \hat{x}_0 + \sin(t_D) \cdot z_D$
4. Đưa qua discriminator → lấy logit fake
5. Tính adversarial generator loss
6. Total loss: $\mathcal{L} = \lambda_{\text{scm}} \cdot \mathcal{L}_{\text{CTCD}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{LADD-G}}$

Mặc định $\lambda_{\text{scm}} = 1.0$, $\lambda_{\text{adv}} = 0.1$ — CTCD chiếm ưu thế, LADD chỉ bổ sung sharpness.

**Chi tiết Phase D:**
1. Student sinh $\hat{x}_0$ (detach, no grad)
2. Noise-augment cả $\hat{x}_0$ (fake) và $x_0$ thật (real) ở các noise level độc lập
3. Discriminator phân loại → hinge/CE loss → update chỉ discriminator head

---

## 4. Các Utility và Kỹ thuật phụ trợ

### 4.1. EMA (Exponential Moving Average) — line 1399

$$\theta_{\text{EMA}} \leftarrow \alpha \cdot \theta_{\text{EMA}} + (1-\alpha) \cdot \theta_{\text{student}}$$

với $\alpha = 0.9999$. EMA model được dùng khi evaluate và save checkpoint — cho kết quả mượt hơn.

### 4.2. Gradient Accumulation & Clipping

- Mặc định 4 bước accumulation → tương đương batch_size × 4
- Gradient clipping tại norm 10.0 cho cả student và discriminator
- Cosine annealing LR scheduler giảm dần LR xuống 1% giá trị ban đầu

### 4.3. Student Creation (`create_student`) — lines 883–927

Hai chiến lược:
- **Full copy**: Deep copy teacher → student có cùng kích thước (dùng cho distillation về tốc độ, không giảm kích thước)
- **Slim student** (khi `student_num_blocks` được chỉ định): Giữ lại mỗi N block (linspace sampling) → giảm kích thước mô hình

### 4.4. `_get_x0_from_noise` — line 566

Khôi phục $x_0$ từ noise prediction theo DDPM:
$$x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t} \cdot \hat\epsilon}{\sqrt{\bar\alpha_t}}$$

Dùng trong DMD2 (không dùng trong CTCD vì CTCD dùng trigonometric flow).

---

## 5. So sánh tổng quan 3 chế độ

| Thuộc tính | `dmd2` | `ctcd` | `ctcd_ladd` |
|---|---|---|---|
| **Paper gốc** | DMD2, NeurIPS 2024 | SANA-Sprint, 2025 | SANA-Sprint, 2025 |
| **Noise schedule** | DDPM ($\bar\alpha_t$) | Trigonometric Flow ($\cos/\sin$) | Trigonometric Flow |
| **Loss chính** | Distribution matching + GAN | Consistency (JVP tangent) | Consistency + Adversarial |
| **Mạng phụ** | FakeScoreNetwork (score + cls) | Không | LatentDiscriminator |
| **Cập nhật** | Two-timescale (D 5× > G) | Đơn thuần | Alternating G/D |
| **Hyperparams chính** | `dm_loss_weight`, `gan_loss_weight`, `dfake_gen_update_ratio` | `sigma_data`, `tangent_warmup_steps` | + `adv_lambda`, `scm_lambda` |
| **Ưu điểm** | Stable, proven | Memory-efficient (không mạng phụ) | Sharp + consistent |
| **Nhược điểm** | Tốn 3× memory (teacher + student + fake_score) | Có thể blurry | Phức tạp nhất, cần tune 2 LR |
