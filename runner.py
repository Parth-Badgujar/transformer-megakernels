from megakernel import megakernel
import subprocess
import time
import modal
import os

app = modal.App(name="megakernels")

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.12")
    .run_commands("apt update && apt install -y git")
    .uv_pip_install("torch", "nvidia-cutlass-dsl[cu13]")
    .workdir("/data")
)

vol = modal.Volume.from_name("megakernel_volume", create_if_missing=True)

@app.function(image=image, secrets=[modal.Secret.from_name("github_token")], volumes={"/data": vol}, gpu="RTX-PRO-6000", timeout=300)
def runner():
    import subprocess
    import os
    subprocess.run(["pip3", "install", "-e", "."])
    subprocess.run(["python3", "bench.py", "--n_iters", "1"])
    vol.commit()

@app.local_entrypoint()
def main():
    with vol.batch_upload(force = True) as batch:
        for file in os.listdir("."):
            if "__pycache__" in file:
                continue
            if os.path.isdir(file):
                batch.put_directory(file, file)
            else:
                batch.put_file(file, file)
    runner.remote()