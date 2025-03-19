# Introduction
Testing code for Finetuning Stable Diffusion Models with DDPO via TRL

# run command
python '.\tests.py'  --num_epochs=200   --train_gradient_accumulation_steps=3  --sample_num_steps=50   --sample_batch_size=1  --train_batch_size=1 --sample_num_batches_per_epoch=24   --per_prompt_stat_tracking=True   --per_prompt_stat_tracking_buffer_size=32  --tracker_project_name="stable_diffusion_training"  --log_with="wandb"
