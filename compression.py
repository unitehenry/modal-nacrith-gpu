import modal

app = modal.App("compression")
vol = modal.Volume.from_name("compression", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install(["git", "wget", "tar"])
    .run_commands(
        [
            "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
            "apt-get -y install cuda-toolkit-12-8",
        ]
    )
    .env({"PATH": "/usr/local/cuda/bin:$PATH"})
    .run_commands(["mkdir /scripts"])
    .add_local_file("scripts/smollm2", "/scripts/smollm2", copy=True)
    .add_local_file("scripts/nacrith", "/scripts/nacrith", copy=True)
    .run_commands(["chmod +x /scripts/smollm2", "/scripts/smollm2"])
    .run_commands(["chmod +x /scripts/nacrith", "/scripts/nacrith"])
)

def run(commands):
    import subprocess
    import tempfile
    import os

    c = "\n".join(commands)

    script = "\n".join(["#!/bin/bash", "set -e", c])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name

    os.chmod(path, 0o755)

    try:
        subprocess.run(["/bin/bash", path], check=False)
    finally:
        os.unlink(path)

@app.function(gpu="A10", image=image, volumes={ "/data": vol })
def compress(file_name):
    run([
        "source /Nacrith-GPU/venv/bin/activate",
        f'python /Nacrith-GPU/cli.py compress "/data/{file_name}" "/data/{file_name}.nc"',
    ])

@app.function(gpu="A10", image=image, volumes={ "/data": vol })
def decompress(file_name):
    outfile = "".join(file_name.rsplit(".nc", 1))

    run([
        "source /Nacrith-GPU/venv/bin/activate",
        f'python /Nacrith-GPU/cli.py decompress "/data/{file_name}" "/data/{outfile}"',
    ])
