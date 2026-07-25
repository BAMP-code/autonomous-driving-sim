"""Command-line entry point: extract features, train a model, or evaluate baselines.

Examples:
    python -m gnd.cli extract --split train --backbone clip-b32 --image-tar data/images_train.tar
    python -m gnd.cli train --name clip_full --backbone clip-b32 --signals confidence class_match color_match spatial_match
    python -m gnd.cli eval-rules
"""

import argparse

from . import features
from .train import Config, train


def main():
    p = argparse.ArgumentParser(prog="gnd")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract and cache backbone features for a split")
    e.add_argument("--split", required=True, choices=["train", "val", "test"])
    e.add_argument("--backbone", default="clip-b32", choices=list(features.BACKBONES))
    e.add_argument("--crop-margin", type=float, default=0.0)
    e.add_argument("--image-tar", default=None)
    e.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("train", help="train the scoring MLP")
    t.add_argument("--name", default="model")
    t.add_argument("--backbone", default="clip-b32", choices=list(features.BACKBONES))
    t.add_argument("--crop-margin", type=float, default=0.0)
    t.add_argument("--signals", nargs="*", default=[])
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-3)

    r = sub.add_parser("eval-rules", help="evaluate the rule-based scorer on val")
    r.add_argument("--val-image-tar", default="data/images_val.tar")

    args = p.parse_args()

    if args.cmd == "extract":
        features.extract(args.split, args.backbone, image_tar=args.image_tar,
                         crop_margin=args.crop_margin, overwrite=args.overwrite)
    elif args.cmd == "train":
        cfg = Config(name=args.name, backbone=args.backbone, crop_margin=args.crop_margin,
                     signals=args.signals, epochs=args.epochs, lr=args.lr)
        train(cfg)
    elif args.cmd == "eval-rules":
        from .data import Talk2CarDataset, load_image_tar, decode_image
        from .rules import rule_based_score
        from .evaluate import ap50
        ds = Talk2CarDataset(split="val")
        images = load_image_tar(args.val_image_tar)
        preds = [rule_based_score(s, image=decode_image(images[s["img_key"]])) for s in ds.samples]
        print(f"rule-based val AP50: {ap50(preds, ds.samples):.4f}")


if __name__ == "__main__":
    main()
