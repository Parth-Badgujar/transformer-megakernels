from asyncio import subprocess
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

@app.function(image=image, volumes={"/data": vol}, gpu="RTX-PRO-6000", timeout=300)
def runner():
    import subprocess
    import os
    os.system("nvidia-smi")

    # subprocess.run(["python3", "bench_1_rms.py", "--n_iters", "1000"])
    # subprocess.run(["python3", "bench_2_qkv.py", "--n_iters", "1000"])
    # subprocess.run(["python3", "bench_3_attn.py", "--n_iters", "1000"])
    # subprocess.run(["python3", "bench_4_out.py"])
    subprocess.run(["python3", "bench.py", "--n_iters", "1"])
    vol.commit()

@app.local_entrypoint()
def main():
    with vol.batch_upload(force = True) as batch:
        batch.put_directory("./operators", "operators")
        batch.put_file("bench.py", "bench.py")
        batch.put_file("scheduler.py", "scheduler.py")
        batch.put_file("model.py", "model.py")
        batch.put_file("bench.py", "bench.py")
    runner.remote()