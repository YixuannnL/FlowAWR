"""CPU-only offline asset and import checks; never starts training."""
import argparse
import importlib
import json
from pathlib import Path
import struct


def check_weights(path):
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    assert path.stat().st_size == 8 + size + end, f"Incomplete weights: {path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--pickscore", action="store_true")
    args = parser.parse_args()
    for module in ("torch", "torchvision", "diffusers", "transformers", "peft", "swanlab", "ml_collections", "sentencepiece"):
        imported = importlib.import_module(module)
        print(module, getattr(imported, "__version__", "available"))
    index = json.loads((args.model / "model_index.json").read_text())
    for name, spec in index.items():
        if name.startswith("_") or spec[0] is None:
            continue
        component = args.model / name
        assert component.is_dir(), f"Missing component: {component}"
        for file in component.rglob("*"):
            assert file.exists(), f"Broken asset link: {file}"
        for file in component.glob("*.safetensors"):
            check_weights(file)
        for file in component.glob("*.index.json"):
            for shard in set(json.loads(file.read_text())["weight_map"].values()):
                assert (component / shard).is_file(), f"Missing shard: {shard}"
    from transformers import AutoProcessor
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig
    if args.pickscore:
        AutoProcessor.from_pretrained(args.reward / "pickscore/processor", local_files_only=True)
        check_weights(args.reward / "pickscore/model/model.safetensors")
    for name in ("train.txt", "test.txt"):
        assert (Path("dataset/pickscore") / name).is_file(), f"Missing prompts: {name}"
    print("PASS: offline assets, safetensors sizes, tokenizer and training imports (CPU only)")


if __name__ == "__main__":
    main()
