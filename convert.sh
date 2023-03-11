python3 tools/all2diffusers.py \
    --checkpoint_path=./ckpt_models/v1-5-pruned.ckpt \
    --dump_path=./base_model \
    --original_config_file=./ckpt_models/model-v1.yaml \
    --scheduler_type="ddim"
#    --vae_path=./ckpt_models/animevae.pt
#    --from_safetensors