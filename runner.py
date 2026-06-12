from asyncio import subprocess
import subprocess
import modal
import os

app = modal.App(name="megakernels")

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.12")
    .run_commands("apt update && apt install -y git")
    .uv_pip_install("torch", "nvidia-cutlass-dsl[cu13]")
    .pip_install("uv")
    .workdir("/data")
)

vol = modal.Volume.from_name("megakernel_volume", create_if_missing=True)

@app.function(image=image, secrets=[modal.Secret.from_name("github_token")], volumes={"/data": vol}, gpu="RTX-PRO-6000", timeout=300)
def runner():
    import subprocess
    import os
    if not os.path.exists(".venv"):
        subprocess.run(["uv", "venv", "--python", "3.12"])
    subprocess.run(["uv", "sync"])
    subprocess.run(["uv", "run", "bench.py", "--n_iters", "1000", "--num_rounds", "1000"])
    vol.commit()

blacklist = ["__pycache__", "ptx", "cubin", "ncu-rep", "png", "npy", ".venv", ".git"]

@app.local_entrypoint()
def main():
    with vol.batch_upload(force = True) as batch:
        for file in os.listdir("."):
            if any(item in file for item in blacklist):
                continue
            if os.path.isdir(file):
                print(f"Uploading Dir {file}")
                batch.put_directory(file, file)
            else:
                print(f"Uploading file {file}")
                batch.put_file(file, file)
    runner.remote()