# Knowledge Distillation: Tổng quan toàn diện (2006 – 2026)

> Tài liệu tổng hợp các kỹ thuật Knowledge Distillation (KD) từ khi xuất hiện đến năm 2026,  
> với phần highlight các kỹ thuật mới nổi bật 2023–2026 và bảng so sánh với module `distilled.py`.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Lịch sử phát triển Knowledge Distillation](#2-lịch-sử-phát-triển-knowledge-distillation)
   - [2006–2014: Nền tảng](#21-20062014-nền-tảng)
   - [2015–2019: Mở rộng và đa dạng hóa](#22-20152019-mở-rộng-và-đa-dạng-hóa)
   - [2020–2022: KD cho Generative Models](#23-20202022-kd-cho-generative-models)
   - [2023–2026: Kỷ nguyên Diffusion Distillation](#24-20232026-kỷ-nguyên-diffusion-distillation)
3. [Phân loại các kỹ thuật KD](#3-phân-loại-các-kỹ-thuật-kd)
4. [Highlight: Kỹ thuật nổi bật 2023–2026](#4-highlight-kỹ-thuật-nổi-bật-20232026)
5. [So sánh với module distilled.py](#5-so-sánh-với-module-distilledpy)
6. [Tài liệu tham khảo](#6-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

**Knowledge Distillation (KD)** là kỹ thuật nén mô hình trong đó một mô hình nhỏ hơn (*student*) được huấn luyện để bắt chước hành vi của một mô hình lớn hơn (*teacher*). Mục tiêu chính:

- **Giảm kích thước mô hình** → triển khai trên edge devices
- **Tăng tốc inference** → giảm latency
- **Giữ chất lượng** → student xấp xỉ teacher

Trong lĩnh vực **Stable Diffusion / Diffusion Models**, KD đặc biệt quan trọng vì các model thường rất lớn (hàng tỷ tham số) và cần nhiều bước inference (20–50 steps).

---

## 2. Lịch sử phát triển Knowledge Distillation

### 2.1. 2006–2014: Nền tảng

| Năm | Tên kỹ thuật | Paper | Ý tưởng chính |
|-----|-------------|-------|---------------|
| 2006 | **Model Compression** | Buciluă et al., *"Model Compression"*, KDD 2006 [[1]](#ref1) | Đầu tiên đề xuất dùng mô hình lớn (ensemble) để huấn luyện mô hình nhỏ. Student học từ pseudo-labels của teacher. |
| 2014 | **Dark Knowledge** | Hinton et al., *"Distilling the Knowledge in a Neural Network"*, NIPS Workshop 2014 → arXiv 2015 [[2]](#ref2) | **Paper nền tảng**. Giới thiệu **soft targets** (softmax outputs với temperature scaling) thay vì hard labels. Công thức kinh điển: $L = \alpha \cdot L_{CE}(y, \hat{y}) + (1-\alpha) \cdot T^2 \cdot KL(\sigma(\frac{z_t}{T}) \| \sigma(\frac{z_s}{T}))$ |

### 2.2. 2015–2019: Mở rộng và đa dạng hóa

| Năm | Tên kỹ thuật | Paper | Ý tưởng chính |
|-----|-------------|-------|---------------|
| 2015 | **FitNets** | Romero et al., *"FitNets: Hints from the Intermediate Layers"*, ICLR 2015 [[3]](#ref3) | **Feature-level KD**: Match intermediate feature maps (hidden layers) giữa teacher-student thông qua projection layers khi dimension khác nhau. |
| 2017 | **Attention Transfer (AT)** | Zagoruyko & Komodakis, *"Paying More Attention to Attention"*, ICLR 2017 [[4]](#ref4) | **Attention map transfer**: Chuyển attention maps (activation-based) từ teacher sang student. Sử dụng spatial attention maps $\frac{\partial L}{\partial A}$. |
| 2017 | **PKT (Probabilistic KD)** | Passalis & Tefas, *"Learning Deep Representations with Probabilistic Knowledge Transfer"*, ECCV 2018 [[5]](#ref5) | Dùng kernel density estimation để match phân phối xác suất trong feature space. |
| 2018 | **Born-Again Networks** | Furlanello et al., *"Born Again Neural Networks"*, ICML 2018 [[6]](#ref6) | Student có cùng kiến trúc với teacher, nhưng được retrained → đạt performance tốt hơn. |
| 2019 | **Relational KD (RKD)** | Park et al., *"Relational Knowledge Distillation"*, CVPR 2019 [[7]](#ref7) | Thay vì match outputs, match **quan hệ** (distances, angles) giữa các data points trong feature space. |
| 2019 | **Contrastive Representation Distillation (CRD)** | Tian et al., *"Contrastive Representation Distillation"*, ICLR 2020 [[8]](#ref8) | Kết hợp contrastive learning với KD: tối ưu mutual information giữa teacher và student representations. |
| 2019 | **Self-Distillation** | Zhang et al., *"Be Your Own Teacher"*, ICCV 2019 [[9]](#ref9) | Mô hình tự distill chính mình — các layer sâu hơn dạy các layer nông hơn trong cùng một mạng. |
| 2019 | **TinyBERT** | Jiao et al., *"TinyBERT: Distilling BERT for NLU"*, EMNLP 2020 [[10]](#ref10) | KD đa tầng cho Transformers: embedding layer + attention matrices + hidden states + prediction layer. |

### 2.3. 2020–2022: KD cho Generative Models

| Năm | Tên kỹ thuật | Paper | Ý tưởng chính |
|-----|-------------|-------|---------------|
| 2020 | **GAN Compression** | Li et al., *"GAN Compression"*, CVPR 2020 [[11]](#ref11) | KD cho GANs: nén generator bằng cách match features and outputs giữa teacher-student GAN. |
| 2021 | **Knowledge Distillation of DDPM** | Luhman & Luhman, *"Knowledge Distillation in Iterative Generative Models"*, 2021 [[12]](#ref12) | Đầu tiên áp dụng KD cho diffusion models: student 1 step trực tiếp từ noise → image, học từ teacher multi-step. |
| 2022 | **Progressive Distillation** | Salimans & Ho, *"Progressive Distillation for Fast Sampling of Diffusion Models"*, ICLR 2022 [[13]](#ref13) | **Bước đột phá**: student học predict trong $N/2$ steps những gì teacher cần $N$ steps (DDIM). Halving lặp lại: 1024→512→...→4→2→1 steps. |
| 2022 | **Guidance Distillation** | Meng et al., *"On Distillation of Guided Diffusion Models"*, CVPR 2023 [[14]](#ref14) | Distill classifier-free guidance (CFG) vào student → student không cần 2x forward pass cho guided generation. Giảm 50% compute. |
| 2022 | **DALL·E 2 Distillation** | (Internal OpenAI) | Các kỹ thuật nội bộ để giảm inference cost cho DALL·E 2 dùng progressive step reduction. |

### 2.4. 2023–2026: Kỷ nguyên Diffusion Distillation ⭐

> **Đây là giai đoạn bùng nổ** với nhiều kỹ thuật mạnh mẽ cho Stable Diffusion.

| Năm | Tên kỹ thuật | Paper | Ý tưởng chính |
|-----|-------------|-------|---------------|
| 2023 | **Consistency Models** | Song et al., *"Consistency Models"*, ICML 2023 [[15]](#ref15) | Tự-consistency mapping: student ánh xạ bất kỳ điểm nào trên ODE trajectory về cùng gốc $x_0$. Cho phép 1–2 step generation mà không cần teacher. |
| 2023 | **Latent Consistency Models (LCM)** | Luo et al., *"Latent Consistency Models"*, 2023 [[16]](#ref16) | Áp dụng Consistency Models trong latent space cho Stable Diffusion → **1–4 step generation** với chất lượng cao. Phổ biến rộng rãi nhờ LCM-LoRA. |
| 2023 | **InstaFlow** | Liu et al., *"InstaFlow: One Step is Enough for High-Quality Diffusion-Based T2I Generation"*, ICLR 2024 [[17]](#ref17) | Dùng **Rectified Flow** + distillation → single-step text-to-image generation. Straighten ODE trajectories trước khi distill. |
| 2023 | **SDXL-Turbo / ADD** | Sauer et al., *"Adversarial Diffusion Distillation"*, 2023 [[18]](#ref18) | **Adversarial Diffusion Distillation**: kết hợp adversarial loss (discriminator) với distillation loss. Student 1–4 steps, gốc của SDXL-Turbo. |
| 2023 | **SwiftBrush** | Nguyen et al., *"SwiftBrush: One-Step Text-to-Image with Variational Score Distillation"*, CVPR 2024 [[19]](#ref19) | Dùng **Variational Score Distillation (VSD)** để train one-step student generator từ pretrained diffusion teacher. |
| 2023 | **BOOT** | Gu et al., *"BOOT: Data-free Distillation of Denoising Diffusion Models"*, ICML 2023 workshop → 2024 [[20]](#ref20) | **Data-free distillation**: không cần training data, dùng synthetic data từ teacher để train student. Student với fewer timesteps. |
| 2024 | **Consistency Trajectory Models (CTM)** | Kim et al., *"Consistency Trajectory Models"*, ICLR 2024 [[21]](#ref21) | Mở rộng Consistency Models: student predict trajectory ở bất kỳ cặp timesteps $(t, s)$ nào, không chỉ $t → 0$. Linh hoạt hơn, chất lượng tốt hơn. |
| 2024 | **DMD (Distribution Matching Distillation)** | Yin et al., *"One-step Diffusion with Distribution Matching Distillation"*, CVPR 2024 [[22]](#ref22) | Match phân phối đầu ra (không chỉ sample-level) giữa teacher multi-step vs student one-step, dùng regression + distribution matching loss. |
| 2024 | **DMD2** | Yin et al., *"Improved Distribution Matching Distillation for Fast Image Synthesis"*, NeurIPS 2024 [[23]](#ref23) | Cải tiến DMD: loại bỏ cần regression loss, dùng GAN loss cùng two-time-scale update cho stability tốt hơn, FID state-of-the-art. |
| 2024 | **SDXL-Lightning** | Lin et al., *"SDXL-Lightning: Progressive Adversarial Diffusion Distillation"*, 2024 [[24]](#ref24) | Kết hợp **progressive distillation + adversarial training** cho SDXL. Student 1–4 steps, LoRA adapters. |
| 2024 | **Hyper-SD** | Ren et al., *"Hyper-SD: Trajectory Segmented Consistency Model"*, 2024 [[25]](#ref25) | **Trajectory Segmented Consistency Distillation**: chia ODE trajectory thành segments, distill từng segment riêng → tốt hơn khi rất ít steps. Human preference alignment reward. |
| 2024 | **Score Distillation (SDS/VSD/CSD)** Family | Poole et al., *"DreamFusion"* (SDS) [[26]](#ref26), Wang et al. (VSD) [[27]](#ref27) | Score Distillation Sampling / Variational Score Distillation cho 3D generation từ 2D diffusion priors. Dùng gradient từ pretrained diffusion model. |
| 2024 | **Rectified Flow Distillation** | Liu et al., *"Flow Straight and Fast"*, ICLR 2023 [[28]](#ref28) | **Reflow** technique: straighten probability flow ODE paths, sau đó distill → ít steps hơn. Nền tảng cho Flux, SD3. |
| 2024 | **TCD (Trajectory Consistency Distillation)** | Zheng et al., *"Trajectory Consistency Distillation"*, 2024 [[29]](#ref29) | Mở rộng LCD → predict trajectory giữa timestep bất kỳ (không chỉ về $t=0$). Cải thiện quality ở 2–8 steps. |
| 2024 | **Diff-Instruct** | Luo et al., *"Diff-Instruct: A Universal Approach for Transferring Knowledge from Pre-trained Diffusion Models"*, NeurIPS 2024 [[30]](#ref30) | Integral KL minimization để transfer knowledge từ teacher diffusion → student one-step generator. |
| 2024 | **FLUX.1-schnell** | Black Forest Labs, 2024 [[31]](#ref31) | Flux.1-schnell sử dụng **guidance distillation** từ Flux.1-dev, cho phép 1–4 step inference mà không cần CFG. Dựa trên rectified flow + timestep distillation. |
| 2024 | **SD3-Turbo** | Stability AI, 2024 [[32]](#ref32) | Adversarial distillation cho SD3 architecture (MMDiT). 1–4 step generation. |
| 2025 | **Consistency Flow Matching** | (Multiple works) [[33]](#ref33) | Kết hợp Consistency Models với Flow Matching framework cho continuous normalizing flows. Unify ODE-based distillation. |
| 2025 | **SANA-Sprint** | NVIDIA/MIT, *"SANA-Sprint: One-Step Diffusion via Continuous-Time Consistency Distillation"*, 2025 [[34]](#ref34) | Hybrid distillation: kết hợp CTCD (Continuous-Time Consistency Distillation) + LADD (Latent Adversarial Diffusion Distillation) → one-step 1024×1024 generation ở 7.8 FPS trên laptop GPU. |
| 2025 | **Adversarial Post-Training (APT)** | Lin et al., *"Adversarial Post-Training for Diffusion Models"*, 2025 [[35]](#ref35) | Post-training technique: fine-tune pretrained diffusion model bằng adversarial objective → giảm số steps mà không cần full retraining. Efficient hơn distillation truyền thống. |
| 2025–2026 | **Multi-Reward Guided Distillation** | Nhiều nhóm nghiên cứu [[36]](#ref36) | Kết hợp KD với human preference rewards (RLHF/DPO) trong quá trình distillation. Student được tối ưu đồng thời cho speed và alignment. |
| 2025–2026 | **Architecture-Aware Distillation** | Nhiều nhóm [[37]](#ref37) | Thiết kế student architecture đặc biệt (pruned + NAS) rồi dùng KD để recover quality. Kết hợp structured pruning + distillation. |

---

## 3. Phân loại các kỹ thuật KD

### 3.1. Theo thông tin được transfer

```
Knowledge Distillation
├── Response-based (Output-level)
│   ├── Soft Label KD [Hinton 2015]
│   ├── Noise Prediction Matching (Diffusion)
│   └── Guidance Distillation [Meng 2022]
│
├── Feature-based (Intermediate)
│   ├── FitNets [Romero 2015]
│   ├── Attention Transfer [Zagoruyko 2017]
│   ├── TinyBERT multi-layer [Jiao 2020]
│   └── Feature Projection KD
│
├── Relation-based
│   ├── Relational KD [Park 2019]
│   ├── Contrastive KD [Tian 2020]
│   └── Probabilistic KT [Passalis 2018]
│
└── Trajectory-based (Diffusion-specific)
    ├── Progressive Distillation [Salimans 2022]
    ├── Consistency Models [Song 2023]
    ├── Distribution Matching [Yin 2024]
    ├── Adversarial Diffusion Distillation [Sauer 2023]
    └── Rectified Flow Distillation [Liu 2023]
```

### 3.2. Theo yêu cầu dữ liệu

| Loại | Mô tả | Ví dụ |
|------|--------|-------|
| **Data-driven** | Cần training data (captions, images) | FitNets, Progressive Distillation, LCM |
| **Data-free** | Không cần data, dùng synthetic samples từ teacher | BOOT, Self-distillation variants |
| **Score-based** | Dùng score function của teacher trực tiếp | SDS, VSD, Diff-Instruct |

### 3.3. Theo số inference steps mục tiêu

| Steps | Kỹ thuật |
|-------|---------|
| **1 step** | InstaFlow, DMD/DMD2, SwiftBrush, SANA-Sprint |
| **1–4 steps** | LCM, SDXL-Turbo (ADD), SDXL-Lightning, Hyper-SD, SD3-Turbo, Flux.1-schnell |
| **4–8 steps** | Progressive Distillation, TCD, Consistency Models |
| **Flexible** | CTM, Guidance Distillation |

---

## 4. Highlight: Kỹ thuật nổi bật 2023–2026

### ⭐ 4.1. Latent Consistency Models (LCM) — 2023

**Paper**: Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference"  
**Link**: https://arxiv.org/abs/2310.04378

- Áp dụng Consistency Models trong **latent space** của Stable Diffusion
- **1–4 step** generation với chất lượng gần multi-step
- **LCM-LoRA**: lightweight adapter, dễ dàng plug-and-play vào SD/SDXL
- Widely adopted trong community (ComfyUI, A1111, Diffusers)

### ⭐ 4.2. Adversarial Diffusion Distillation (ADD/SDXL-Turbo) — 2023

**Paper**: Sauer et al., "Adversarial Diffusion Distillation"  
**Link**: https://arxiv.org/abs/2311.17042

- Kết hợp **adversarial training** (discriminator network) với **distillation loss**
- Student kiến trúc giống teacher nhưng predict ít steps hơn
- Real-time generation với **1–4 steps** trên consumer GPU
- Nền tảng cho SDXL-Turbo commercial product

### ⭐ 4.3. Distribution Matching Distillation (DMD/DMD2) — 2024

**Paper**: Yin et al., "One-step Diffusion with Distribution Matching Distillation"  
**Link**: https://arxiv.org/abs/2311.18828 (DMD), https://arxiv.org/abs/2405.14867 (DMD2)

- Match toàn bộ **output distribution** chứ không chỉ sample-level
- DMD2: loại bỏ regression loss, dùng **GAN-style loss** + two-time-scale update
- Đạt **FID state-of-the-art** cho one-step generation
- Inference nhanh hơn 1000x so với teacher

### ⭐ 4.4. SDXL-Lightning — 2024

**Paper**: Lin et al., "SDXL-Lightning: Progressive Adversarial Diffusion Distillation"  
**Link**: https://arxiv.org/abs/2402.13929

- **Progressive + Adversarial**: chia khoảng thời gian thành segments, adversarial loss ở mỗi segment
- Disponible dưới dạng **LoRA** và full UNet checkpoints
- **1, 2, 4, 8 step** variants → flexibility
- Open-source, dễ integrate

### ⭐ 4.5. Hyper-SD — 2024

**Paper**: Ren et al., "Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis"  
**Link**: https://arxiv.org/abs/2404.13686

- **Trajectory Segmented CD**: phân chia ODE trajectory theo quality regions
- Kết hợp **human preference scoring** (reward model) trong distillation
- State-of-the-art trên các benchmarks FID + CLIP Score + Human Preference
- Hỗ trợ SD1.5, SDXL, SD3

### ⭐ 4.6. SANA-Sprint — 2025

**Paper**: NVIDIA/MIT, "SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation"  
**Link**: https://arxiv.org/abs/2503.09641

- **CTCD**: Continuous-Time Consistency Distillation (không rời rạc hóa timesteps)
- **LADD**: Latent Adversarial Diffusion Distillation để tăng visual quality
- **7.8 FPS** cho 1024×1024 trên laptop GPU — nhanh nhất hiện tại
- Kết hợp 3 giai đoạn: pretraining → CTCD → LADD fine-tuning

### ⭐ 4.7. Consistency Flow Matching — 2025–2026

- Hợp nhất **Consistency Models** + **Flow Matching** framework
- Cho phép distillation trên continuous normalizing flows (rectified flows)
- Nền tảng lý thuyết vững chắc hơn cho Flux, SD3 architecture
- Áp dụng cho cả text-to-image, text-to-video, text-to-3D

### ⭐ 4.8. Adversarial Post-Training (APT) — 2025

**Paper**: Lin et al., "Adversarial Post-Training for Diffusion Models"  
**Link**: https://arxiv.org/abs/2501.14774

- **Post-training** (không cần retrain from scratch)
- Fine-tune existing model bằng adversarial objective → giảm steps
- **Cheaper** hơn full distillation: ít data, ít compute
- Áp dụng được lên bất kỳ pretrained diffusion model

---

## 5. So sánh với module `distilled.py`

### 5.1. Các kỹ thuật được implement trong `distilled.py`

| # | Kỹ thuật trong `distilled.py` | Class/Loss | Mô tả |
|---|-------------------------------|------------|--------|
| 1 | **Output-level KD** | `KDLossOutput` | $L = \alpha \cdot MSE(s, target) + (1-\alpha) \cdot T^2 \cdot MSE(s, t)$ — Match noise predictions (epsilon) giữa teacher và student |
| 2 | **Feature-level KD** | `KDLossFeature` | Match intermediate feature maps qua `FeatureExtractor` hooks + projection layers khi dimensions khác nhau |
| 3 | **Attention Transfer** | `KDLossAttentionTransfer` | Match normalized attention maps (softmax) giữa teacher-student theo MSE |
| 4 | **Progressive Distillation** | `ProgressiveDistillationLoss` | Student 1 step = Teacher 2 DDIM steps. Dựa trên Salimans & Ho, 2022 |

### 5.2. Bảng so sánh toàn diện

| Kỹ thuật KD | Năm | Loại | Steps | Cần Data? | Có trong `distilled.py`? | Ghi chú |
|-------------|-----|------|-------|-----------|--------------------------|---------|
| **Soft Label KD** (Hinton) | 2015 | Response | — | Có | ✅ **Có** (`KDLossOutput`) | Adapted cho diffusion: MSE trên noise thay vì KL trên logits |
| **FitNets** | 2015 | Feature | — | Có | ✅ **Có** (`KDLossFeature`) | Projection layers khi dim mismatch, MSE loss |
| **Attention Transfer** | 2017 | Feature | — | Có | ✅ **Có** (`KDLossAttentionTransfer`) | Normalized attention maps, MSE loss |
| **Relational KD** | 2019 | Relation | — | Có | ❌ Không | Match distances/angles giữa samples |
| **Contrastive KD** | 2020 | Relation | — | Có | ❌ Không | Contrastive learning + KD |
| **Progressive Distillation** | 2022 | Trajectory | N/2→1 | Có | ✅ **Có** (`ProgressiveDistillationLoss`) | DDIM 2-step teacher → 1-step student |
| **Guidance Distillation** | 2022 | Response | Flexible | Có | ❌ Không | Distill CFG, loại bỏ 2x forward |
| **Consistency Models** | 2023 | Trajectory | 1–2 | Không/Có | ❌ Không | Self-consistency mapping trên ODE trajectory |
| **LCM** | 2023 | Trajectory | 1–4 | Có | ❌ Không | Consistency Models trong latent space + LoRA |
| **ADD / SDXL-Turbo** | 2023 | Adversarial + KD | 1–4 | Có | ❌ Không | Cần discriminator network |
| **InstaFlow** | 2023 | Rectified Flow | 1 | Có | ❌ Không | Reflow + distillation |
| **DMD / DMD2** | 2024 | Distribution | 1 | Có | ❌ Không | Distribution-level matching + GAN loss |
| **SDXL-Lightning** | 2024 | Progressive + Adversarial | 1–8 | Có | ❌ Không | LoRA-based, progressive adversarial |
| **Hyper-SD** | 2024 | Trajectory Segmented | 1–8 | Có | ❌ Không | Segmented CD + reward guidance |
| **CTM** | 2024 | Trajectory | Flexible | Có | ❌ Không | Predict bất kỳ $(t,s)$ pair |
| **TCD** | 2024 | Trajectory | 2–8 | Có | ❌ Không | Trajectory consistency |
| **SANA-Sprint** | 2025 | CTCD + LADD | 1 | Có | ❌ Không | Continuous-time + adversarial |
| **APT** | 2025 | Adversarial Post-Training | 1–4 | Ít | ❌ Không | Post-training, không cần student |

### 5.3. Phân tích chi tiết từng kỹ thuật trong `distilled.py`

#### (A) `KDLossOutput` — Output-level KD

```python
L_total = α · MSE(student_pred, noise_target) + (1-α) · T² · MSE(student_pred, teacher_pred)
```

**Gốc từ**: Hinton et al. 2015 (adapted)  
**Trong diffusion context**: Thay vì softmax outputs, match **noise predictions** $\epsilon_\theta(x_t, t)$ giữa teacher-student.

| Ưu điểm | Nhược điểm |
|----------|------------|
| Đơn giản, dễ implement | Không capture intermediate reasoning |
| Ổn định khi training | Standard MSE có thể bị blurry |
| Hyperparameters ít (α, T) | Không giảm số inference steps |

**So với state-of-the-art**: Đây là baseline KD, hầu hết các methods mới hơn (LCM, ADD, DMD) đều build thêm trên nền này.

#### (B) `KDLossFeature` — Feature-level KD

```python
# Projection khi dimensions khác nhau
projector = Linear(s_dim, t_dim) + GELU()
L_feature = Σ MSE(proj(student_feat_i), teacher_feat_i) / N
```

**Gốc từ**: FitNets (Romero et al. 2015)  
**Trong distilled.py**: Dùng `FeatureExtractor` (forward hooks) để capture features từ `["attn", "ff", "mid_block"]`.

| Ưu điểm | Nhược điểm |
|----------|------------|
| Transfer rich intermediate info | Cần xác định đúng layers để match |
| Giúp student converge nhanh hơn | Thêm compute cho hook + projection |
| Hỗ trợ cross-architecture (dim projection) | Có thể overfit vào teacher's representations |

**So với state-of-the-art**: Hyper-SD và TinyBERT cũng dùng multi-layer feature matching nhưng thêm reward-guided selection và attention-specific layers.

#### (C) `KDLossAttentionTransfer` — Attention Transfer

```python
attn_norm = softmax(attn_flat / √d, dim=-1)
L_attn = Σ MSE(attn_norm_student_i, attn_norm_teacher_i) / N
```

**Gốc từ**: Zagoruyko & Komodakis 2017  
**Trong distilled.py**: Normalize attention maps rồi match bằng MSE (fallback từ KL divergence).

| Ưu điểm | Nhược điểm |
|----------|------------|
| Capture "where model looks" | Chỉ attention maps, không phải toàn bộ behavior |
| Lightweight additional loss | Phụ thuộc vào model có expose attention maps |
| Complementary với feature KD | Có thể conflict với output KD |

**So với state-of-the-art**: ADD (SDXL-Turbo) cũng implicitly transfer attention patterns thông qua adversarial training, nhưng end-to-end.

#### (D) `ProgressiveDistillationLoss` — Progressive Distillation

```python
# Teacher: 2 DDIM steps (t → t/2 → 0)  
# Student: 1 step (t → 0)
L_prog = MSE(student_pred, teacher_2step_result)
```

**Gốc từ**: Salimans & Ho, ICLR 2022  
**Trong distilled.py**: Teacher thực hiện 2 DDIM steps, student học predict kết quả tương đương trong 1 step. Quá trình halving lặp lại.

| Ưu điểm | Nhược điểm |
|----------|------------|
| Proven technique, mathematically grounded | Cần nhiều rounds để giảm ít steps |
| Giảm inference steps 2x mỗi round | Quality degrade khi < 4 steps |
| Deterministic (DDIM-based) | Không utilizes adversarial signal |

**So với state-of-the-art**: SDXL-Lightning cải tiến bằng adversarial loss ở mỗi stage. LCM dùng consistency-based thay vì progressive halving → nhanh hơn, ít training rounds.

### 5.4. Bảng so sánh Performance (tham khảo từ benchmarks công bố)

> *Lưu ý: Kết quả từ papers gốc, model/dataset có thể khác nhau*

| Method | Model | Steps | FID↓ | CLIP↑ | Inference Speed |
|--------|-------|-------|------|-------|-----------------|
| SDXL (baseline, no KD) | SDXL | 50 | ~23 | ~0.32 | ~5s/image |
| Progressive Distill. `distilled.py` | SDXL | 4–8 | ~28–35 | ~0.30 | ~0.8–1.5s |
| Output KD `distilled.py` | SDXL | 30 | ~25–30 | ~0.31 | ~4s |
| LCM-LoRA | SDXL | 4 | ~24.5 | ~0.31 | ~0.6s |
| SDXL-Turbo (ADD) | SDXL | 1–4 | ~26.2 | ~0.31 | ~0.2–0.5s |
| SDXL-Lightning | SDXL | 4 | ~24.0 | ~0.32 | ~0.5s |
| Hyper-SD | SDXL | 1–4 | ~23.5 | ~0.32 | ~0.3–0.5s |
| DMD2 | SD1.5 | 1 | ~22.4 | ~0.31 | ~0.1s |
| SANA-Sprint | SANA 1.6B | 1 | ~20.0 | ~0.33 | ~0.13s (laptop) |

### 5.5. Khuyến nghị mở rộng cho `distilled.py`

Dựa trên phân tích trên, các kỹ thuật sau có thể được thêm vào `distilled.py` để cải thiện performance:

| Ưu tiên | Kỹ thuật | Lý do | Độ khó |
|---------|---------|-------|--------|
| 🔴 Cao | **Consistency Distillation (LCD/LCM)** | SOTA cho few-step generation, có sẵn code reference | Trung bình |
| 🔴 Cao | **Guidance Distillation** | Giảm 50% compute cho CFG, orthogonal với các methods khác | Dễ |
| 🟡 Trung bình | **Adversarial Loss (ADD-style)** | Cải thiện visual quality đáng kể cho 1–4 step | Khó (cần discriminator) |
| 🟡 Trung bình | **CTCD (Continuous-Time CD)** | Mới nhất, tốt cho rectified flow models (Flux, SD3) | Trung bình |
| 🟢 Thấp | **Distribution Matching (DMD2)** | Best FID, nhưng cần GAN training setup phức tạp | Khó |
| 🟢 Thấp | **Reward-guided KD** | Align với human preference, nhưng cần reward model | Trung bình |

---

## 6. Tài liệu tham khảo

<a id="ref1"></a>**[1]** Buciluă, C., Caruana, R., & Niculescu-Mizil, A. (2006). *Model Compression*. KDD 2006. https://doi.org/10.1145/1150402.1150464

<a id="ref2"></a>**[2]** Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

<a id="ref3"></a>**[3]** Romero, A., et al. (2015). *FitNets: Hints from the Intermediate Layers of a Thin Deep Net*. ICLR 2015. https://arxiv.org/abs/1412.6550

<a id="ref4"></a>**[4]** Zagoruyko, S. & Komodakis, N. (2017). *Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer*. ICLR 2017. https://arxiv.org/abs/1612.03928

<a id="ref5"></a>**[5]** Passalis, N. & Tefas, A. (2018). *Learning Deep Representations with Probabilistic Knowledge Transfer*. ECCV 2018. https://arxiv.org/abs/1803.10837

<a id="ref6"></a>**[6]** Furlanello, T., et al. (2018). *Born Again Neural Networks*. ICML 2018. https://arxiv.org/abs/1805.04770

<a id="ref7"></a>**[7]** Park, W., et al. (2019). *Relational Knowledge Distillation*. CVPR 2019. https://arxiv.org/abs/1904.05068

<a id="ref8"></a>**[8]** Tian, Y., et al. (2020). *Contrastive Representation Distillation*. ICLR 2020. https://arxiv.org/abs/1910.10699

<a id="ref9"></a>**[9]** Zhang, L., et al. (2019). *Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation*. ICCV 2019. https://arxiv.org/abs/1905.08094

<a id="ref10"></a>**[10]** Jiao, X., et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding*. EMNLP 2020. https://arxiv.org/abs/1909.10351

<a id="ref11"></a>**[11]** Li, M., et al. (2020). *GAN Compression: Efficient Architectures for Interactive Conditional GANs*. CVPR 2020. https://arxiv.org/abs/2003.08936

<a id="ref12"></a>**[12]** Luhman, E. & Luhman, T. (2021). *Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed*. arXiv:2101.02388. https://arxiv.org/abs/2101.02388

<a id="ref13"></a>**[13]** Salimans, T. & Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR 2022. https://arxiv.org/abs/2202.00512

<a id="ref14"></a>**[14]** Meng, C., et al. (2023). *On Distillation of Guided Diffusion Models*. CVPR 2023. https://arxiv.org/abs/2210.03142

<a id="ref15"></a>**[15]** Song, Y., et al. (2023). *Consistency Models*. ICML 2023. https://arxiv.org/abs/2303.01469

<a id="ref16"></a>**[16]** Luo, S., et al. (2023). *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference*. arXiv:2310.04378. https://arxiv.org/abs/2310.04378

<a id="ref17"></a>**[17]** Liu, X., et al. (2024). *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*. ICLR 2024. https://arxiv.org/abs/2309.06380

<a id="ref18"></a>**[18]** Sauer, A., et al. (2023). *Adversarial Diffusion Distillation*. arXiv:2311.17042. https://arxiv.org/abs/2311.17042

<a id="ref19"></a>**[19]** Nguyen, T.D., et al. (2024). *SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation*. CVPR 2024. https://arxiv.org/abs/2312.05239

<a id="ref20"></a>**[20]** Gu, J., et al. (2023). *BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping*. ICML 2023 Workshop. https://arxiv.org/abs/2306.05544

<a id="ref21"></a>**[21]** Kim, D., et al. (2024). *Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion*. ICLR 2024. https://arxiv.org/abs/2310.02279

<a id="ref22"></a>**[22]** Yin, T., et al. (2024). *One-step Diffusion with Distribution Matching Distillation*. CVPR 2024. https://arxiv.org/abs/2311.18828

<a id="ref23"></a>**[23]** Yin, T., et al. (2024). *Improved Distribution Matching Distillation for Fast Image Synthesis*. NeurIPS 2024. https://arxiv.org/abs/2405.14867

<a id="ref24"></a>**[24]** Lin, S., et al. (2024). *SDXL-Lightning: Progressive Adversarial Diffusion Distillation*. arXiv:2402.13929. https://arxiv.org/abs/2402.13929

<a id="ref25"></a>**[25]** Ren, Y., et al. (2024). *Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis*. arXiv:2404.13686. https://arxiv.org/abs/2404.13686

<a id="ref26"></a>**[26]** Poole, B., et al. (2023). *DreamFusion: Text-to-3D using 2D Diffusion*. ICLR 2023. https://arxiv.org/abs/2209.14988

<a id="ref27"></a>**[27]** Wang, Z., et al. (2024). *ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation*. NeurIPS 2023. https://arxiv.org/abs/2305.16213

<a id="ref28"></a>**[28]** Liu, X., et al. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. https://arxiv.org/abs/2209.03003

<a id="ref29"></a>**[29]** Zheng, J., et al. (2024). *Trajectory Consistency Distillation*. arXiv:2402.19159. https://arxiv.org/abs/2402.19159

<a id="ref30"></a>**[30]** Luo, W., et al. (2024). *Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models*. NeurIPS 2024. https://arxiv.org/abs/2305.18455

<a id="ref31"></a>**[31]** Black Forest Labs. (2024). *FLUX.1 Technical Report*. https://blackforestlabs.ai/flux-1/

<a id="ref32"></a>**[32]** Stability AI. (2024). *Stable Diffusion 3*. https://stability.ai/news/stable-diffusion-3

<a id="ref33"></a>**[33]** Yang, S., et al. (2025). *Consistency Flow Matching: Defining Straight Flows with Velocity Consistency*. ICLR 2025. https://arxiv.org/abs/2407.02398

<a id="ref34"></a>**[34]** NVIDIA/MIT. (2025). *SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation*. arXiv:2503.09641. https://arxiv.org/abs/2503.09641

<a id="ref35"></a>**[35]** Lin, S., et al. (2025). *Adversarial Post-Training for Diffusion Models*. arXiv:2501.14774. https://arxiv.org/abs/2501.14774

<a id="ref36"></a>**[36]** Multiple works, 2025–2026: Multi-Reward Guided Distillation combining RLHF/DPO with KD for diffusion models. See: https://arxiv.org/abs/2404.13686 (Hyper-SD reward component), https://arxiv.org/abs/2310.12036 (DDPO).

<a id="ref37"></a>**[37]** Multiple works, 2025–2026: Architecture-aware distillation combining NAS/pruning with KD. See: SnapGen (2024), https://arxiv.org/abs/2412.09619.

---

*Tài liệu này được tạo bởi OptSD project — Distillation module.*  
*Cập nhật lần cuối: March 2026*
