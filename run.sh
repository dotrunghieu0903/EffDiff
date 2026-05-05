# Run app.py
python app.py

python app.py --dataset_name "flickr8k" --num_images 5 --inference_steps 50 --guidance_scale 3.5 --metrics_subset 5
python app.py --dataset_name "coco" --num_images 10000 --inference_steps 50 --guidance_scale 3.5 --metrics_subset 5

# Run quantization + pruning module
python pruning/quant_pruning.py --pruning_amount 0.15 --num_images 100 --steps 25 --guidance_scale 3.5 --metrics_subset 100 --use_coco

# Run kv cache
python kvcache/sana_kvcache.py --use_coco --num_images 1000 --steps 50 --guidance_scale 3.5 --metrics_subset 5 --monitor_vram
python kvcache/sana_kvcache.py --use_flickr8k --num_images 1000 --steps 50 --guidance_scale 3.5 --metrics_subset 5 --monitor_vram

# Make script executable
chmod +x normal_coco.py

# Run directly from the normal directory
python normal_coco.py --num_images 100 --steps 25 --guidance_scale 7.5

# Run flash attention app with different settings from inside instead of app.py
python flash_attn/flash_attn_app.py --monitor-vram --metrics-subset 1000 --num-images 1000 --steps 25 --dataset coco --flash-attn --kv-cache --pruning 0.3 --precision int4
python flash_attn/flash_attn_app.py --monitor-vram --metrics-subset 1000 --num-images 1000 --steps 25 --dataset flickr8k --flash-attn --kv-cache --pruning 0.3 --precision int4

# Run pruning analysis script with different methods
python pruning/generate_synthetic_graph.py
python pruning/generate_synthetic_graph.py --method magnitude --num_images 50 --dataset coco

# Using SDXL base model
python app.py --model_name SDXL --num_images 10000 --inference_steps 50

# Using SDXL-Turbo for fast generation
python app.py --model_name SDXL-Turbo --num_images 100 --inference_steps 4

# Run SD3 with combined optimizations (INT4 quantization, 30% pruning, KV cache)
python sd3_optimized.py --dataset coco --num_images 10000 --precision int4 --pruning 0.3 --steps 50 --guidance 7.0 --no-flash-attn --metrics_subset 100

# Run SD3 with FP16 (no quantization) for comparison
python sd3_optimized.py --dataset coco --num_images 100 --precision fp16 --pruning 0.0 --steps 50 --guidance 7.0 --no-flash-attn --metrics_subset 100

# Run metrics only
python quantization/update_summary_simple.py --output_dir quantization/coco/20260125 --coco_dir coco/val2017

#SD3:
# SD3 với COCO val2017 only (5000 images max)
python sd3_optimized.py --dataset coco --num_images 5000 --precision int4 --pruning 0.3 --steps 30 --guidance 7.0 --no-flash-attn --metrics_subset 100

# SD3 với COCO 10000 images (val2017 + train2017)
python sd3_optimized.py --dataset coco --num_images 10000 --precision int4 --pruning 0.3 --steps 30 --guidance 7.0 --no-flash-attn --metrics_subset 100 --use_train

python sd3_optimized.py --precision fp16 --pruning 0.0 --no-kv-cache --no-flash-attn --dataset coco --num_images 10000 --output sd3_outputs/coco/baseline --use_train

#SDXL Baseline (no optimization)
# SDXL baseline - FP16, no pruning, no kv cache, no flash attention
python normal_coco.py --model_name SDXL --num_images 10000 --steps 50 --guidance_scale 7.5 --metrics_subset 100 --use_train --output normal_outputs/SDXL_baseline

# Standalone: distill SDXL với output-level KD
python -m distillation.distilled --teacher_model SDXL --kd_mode output --num_epochs 5

python -m distillation.distilled --teacher_model SDXL --kd_mode dmd2 --memory_efficient

# Progressive distillation cho Flux (giảm inference steps)
python -m distillation.distilled --teacher_model Flux.1-schnell --kd_mode progressive

# Feature-level KD với slim student (bỏ blocks)
python -m distillation.distilled --teacher_model SD3 --kd_mode feature --student_num_blocks 6

# Qua app.py
python app.py --use_distillation --kd_mode output --kd_epochs 5