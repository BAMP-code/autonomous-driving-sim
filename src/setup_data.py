# Convert the CMSVG repo's combined JSON into our separate commands/ and proposals/ files, then enrich with obj_name from the official Talk2Car annotations.

import argparse
import json
from pathlib import Path


# Split CMSVG's bundled file into per-split commands.json and proposals.json files.
def convert(cmsvg_path: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    commands_dir = output_dir / "commands"
    proposals_dir = output_dir / "proposals"
    commands_dir.mkdir(parents=True, exist_ok=True)
    proposals_dir.mkdir(parents=True, exist_ok=True)

    with open(cmsvg_path) as f:
        data = json.load(f)

    for split in ("train", "val", "test"):
        if split not in data:
            continue

        commands = []
        proposals = {}

        for sample in data[split].values():
            img_key = sample["img"]

            if img_key not in proposals:
                proposals[img_key] = [
                    {"box": [d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]],
                     "class": d["class"], "score": d["score"]}
                    for d in sample["centernet"]
                ]

            if "referred_object" not in sample:
                continue

            commands.append({
                "command": sample["command"],
                "command_token": sample["command_token"],
                "2d_box": sample["referred_object"],
                "t2c_img": img_key,
                "obj_name": sample.get("obj_name", "unknown"),
            })

        with open(commands_dir / f"{split}.json", "w") as f:
            json.dump(commands, f, indent=2)
        with open(proposals_dir / f"{split}_proposals.json", "w") as f:
            json.dump(proposals, f)


def enrich_with_talk2car(output_dir: str, talk2car_commands_dir: str) -> None:
    output_dir = Path(output_dir)
    t2c_dir = Path(talk2car_commands_dir)

    for split in ("train", "val", "test"):
        our_path = output_dir / "commands" / f"{split}.json"
        t2c_path = t2c_dir / f"{split}_commands.json"
        if not t2c_path.exists() or not our_path.exists():
            continue

        with open(t2c_path) as f:
            t2c_data = json.load(f)
        t2c_commands = t2c_data.get("commands", t2c_data)
        t2c_lookup = {cmd.get("command_token", ""): cmd for cmd in t2c_commands} if isinstance(t2c_commands, list) else {}

        with open(our_path) as f:
            our_commands = json.load(f)

        for cmd in our_commands:
            t2c_cmd = t2c_lookup.get(cmd.get("command_token", ""))
            if t2c_cmd and "obj_name" in t2c_cmd:
                cmd["obj_name"] = t2c_cmd["obj_name"]

        with open(our_path, "w") as f:
            json.dump(our_commands, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmsvg", type=str, required=True)
    parser.add_argument("--talk2car", type=str, required=True)
    parser.add_argument("--output", type=str, default="../data")
    args = parser.parse_args()

    convert(args.cmsvg, args.output)
    enrich_with_talk2car(args.output, args.talk2car)
