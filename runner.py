import subprocess
import time
import modal

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
    PAT = os.getenv("GITHUB_PAT") 
    # subprocess.run(["python3", "bench_1_rms.py", "--n_iters", "1000"])
    # subprocess.run(["python3", "bench_2_qkv.py", "--n_iters", "1000"])
    # subprocess.run(["python3", "bench_3_attn.py", "--n_iters", "1000"])
    # subprocess.run(["python3", "bench_4_out.py"])
    # subprocess.run(["uv", "pip", "install", "-e", "."])
    subprocess.run(["python3", "bench.py", "--n_iters", "1"], cwd = "./src")
    # subprocess.run(["git", "clone", f"https://{PAT}@github.com/Parth-Badgujar/transformer-megakernels.git"])
    # subprocess.run(["uv", "pip", "install", "-e", "."], cwd="./transformer-megakernels")
    # subprocess.run(["git", "checkout", "dev"], cwd="./transformer-megakernels")
    # subprocess.run(["python3", "bench.py"], cwd="./transformer-megakernels")
    vol.commit()

@app.local_entrypoint()
def main():
    with vol.batch_upload(force = True) as batch:
        batch.put_directory("./src", "./src")
        batch.put_file("bench.py", "src/bench.py")
        batch.put_file("scheduler.py", "src/scheduler.py")
    #     # batch.put_file("model.py", "model.py")
        # batch.put_file("bench.py", "bench.py")
    runner.remote()