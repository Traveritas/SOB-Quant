from huggingface_hub import snapshot_download

model_id = "facebook/opt-125m"
local_dir = "./opt-125m"
snapshot_download(repo_id=model_id, local_dir=local_dir)