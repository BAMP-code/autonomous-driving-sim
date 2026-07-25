"""Convert CMSVG data into our expected format.

The CMSVG repo's `talk2car_w_rpn_no_duplicates.json` contains everything:
- Commands with GT boxes
- 64 CenterNet proposals per sample
- Image filenames

This script splits it into our commands/ and proposals/ format.

Usage:
    python setup_data.py --cmsvg /tmp/CMSVG/data/talk2car_w_rpn_no_duplicates.json --output ../data
"""

import argparse
import json
from pathlib import Path


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
            print(f"  Skipping {split} (not in file)")
            continue

        split_data = data[split]
        commands = []
        proposals = {}

        for idx_str, sample in split_data.items():
            img_key = sample["img"]  # e.g. "img_train_0.jpg"

            # Test split has no GT box — skip it for our purposes
            if "referred_object" not in sample:
                # Still save proposals for test images if needed later
                if img_key not in proposals:
                    props = []
                    for det in sample["centernet"]:
                        bx, by, bw, bh = det["bbox"]
                        props.append({
                            "box": [bx, by, bx + bw, by + bh],
                            "class": det["class"],
                            "score": det["score"],
                        })
                    proposals[img_key] = props
                continue

            # Command annotation
            gt_box = sample["referred_object"]  # [x, y, w, h]
            commands.append({
                "command": sample["command"],
                "command_token": sample["command_token"],
                "2d_box": gt_box,
                "t2c_img": img_key,
                "obj_name": sample.get("obj_name", "unknown"),
            })

            # Proposals for this image (may have multiple commands per image)
            if img_key not in proposals:
                props = []
                for det in sample["centernet"]:
                    # CMSVG bbox format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
                    bx, by, bw, bh = det["bbox"]
                    props.append({
                        "box": [bx, by, bx + bw, by + bh],
                        "class": det["class"],
                        "score": det["score"],
                    })
                proposals[img_key] = props

        # Handle obj_name — get it from Talk2Car commands if available
        # The CMSVG file may not have obj_name, let's check
        # Actually looking at the data, centernet entries have 'class' but
        # the GT referred_object is just [x,y,w,h]. We need to figure out
        # obj_name from the class of the best-matching proposal.
        # For now, mark as "unknown" if not present — the dataset.py handles this.

        # Save commands
        commands_path = commands_dir / f"{split}.json"
        with open(commands_path, "w") as f:
            json.dump(commands, f, indent=2)
        print(f"  {split}: {len(commands)} commands -> {commands_path}")

        # Save proposals
        proposals_path = proposals_dir / f"{split}_proposals.json"
        with open(proposals_path, "w") as f:
            json.dump(proposals, f)
        print(f"  {split}: {len(proposals)} images with proposals -> {proposals_path}")


def enrich_with_talk2car(output_dir: str, talk2car_commands_dir: str) -> None:
    """Enrich our commands with obj_name from official Talk2Car annotations."""
    output_dir = Path(output_dir)
    t2c_dir = Path(talk2car_commands_dir)

    for split in ("train", "val", "test"):
        our_path = output_dir / "commands" / f"{split}.json"
        t2c_path = t2c_dir / f"{split}_commands.json"

        if not t2c_path.exists() or not our_path.exists():
            continue

        with open(t2c_path) as f:
            t2c_data = json.load(f)

        # Build lookup by command_token
        t2c_lookup = {}
        t2c_commands = t2c_data.get("commands", t2c_data)
        if isinstance(t2c_commands, list):
            for cmd in t2c_commands:
                token = cmd.get("command_token", "")
                t2c_lookup[token] = cmd

        with open(our_path) as f:
            our_commands = json.load(f)

        enriched = 0
        for cmd in our_commands:
            token = cmd.get("command_token", "")
            if token in t2c_lookup:
                t2c_cmd = t2c_lookup[token]
                if "obj_name" in t2c_cmd:
                    cmd["obj_name"] = t2c_cmd["obj_name"]
                    enriched += 1

        with open(our_path, "w") as f:
            json.dump(our_commands, f, indent=2)

        print(f"  {split}: enriched {enriched}/{len(our_commands)} with obj_name")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmsvg", type=str, default="/tmp/CMSVG/data/talk2car_w_rpn_no_duplicates.json")
    parser.add_argument("--talk2car", type=str, default="/tmp/Talk2Car/data/commands",
                        help="Path to Talk2Car commands dir (for obj_name enrichment)")
    parser.add_argument("--output", type=str, default="../data")
    args = parser.parse_args()

    print("Converting CMSVG data...")
    convert(args.cmsvg, args.output)

    print("\nEnriching with Talk2Car obj_name...")
    enrich_with_talk2car(args.output, args.talk2car)

    print("\nDone! Now download images from Talk2Car Google Drive link into data/images/")
