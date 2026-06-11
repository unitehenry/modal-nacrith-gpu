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
    .env({"PATH": "/usr/local/cuda/bin:$PATH", "MODAL": "1"})
    .run_commands(["mkdir /scripts"])
    .add_local_file("scripts/smollm2", "/scripts/smollm2", copy=True)
    .add_local_file("scripts/nacrith", "/scripts/nacrith", copy=True)
    .run_commands(["chmod +x /scripts/smollm2", "/scripts/smollm2"])
    .run_commands(["chmod +x /scripts/nacrith", "/scripts/nacrith"])
    .pip_install("langchain-text-splitters")
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


@app.function(gpu="H100", image=image, volumes={"/data": vol})
def compress(file_name):
    run(
        [
            "source /Nacrith-GPU/venv/bin/activate",
            f'python /Nacrith-GPU/cli.py compress "/data/{file_name}" "/data/{file_name}.nc"',
        ]
    )


@app.function(gpu="H100", image=image, volumes={"/data": vol})
def decompress(file_name):
    outfile = "".join(file_name.rsplit(".nc", 1))

    run(
        [
            "source /Nacrith-GPU/venv/bin/activate",
            f'python /Nacrith-GPU/cli.py decompress "/data/{file_name}" "/data/{outfile}"',
        ]
    )


@app.function(image=image, volumes={"/data": vol})
def batch(file_name : str, chunks : int = 20):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import os
    from pathlib import Path

    input_file = f"/data/{file_name}"

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(500, len(text) // chunks),
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    text_chunks = text_splitter.split_text(text)

    input_path = Path(input_file)

    base_name = input_path.stem

    ext = input_path.suffix or ".txt"

    outputs = []

    for i, chunk in enumerate(text_chunks, 1):
        output_file_name = f"{base_name}_chunk_{i:03d}{ext}"

        output_file = f"/data/{output_file_name}"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(chunk)

        outputs.append(output_file_name)

    vol.commit()

    for output_file_name in outputs:
        compress.spawn(output_file_name)
