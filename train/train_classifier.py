"""Stage 2: classify a cropped piece into its type. Compare a small CNN with a vision transformer.

Crops are stored one folder per piece id from configs/pieces.yaml:
    data/crops/{train,val,test}/<piece_id>/*.jpg
train/ can mix public single-piece images (B200C, Gdansk) with crops from your own photos.
test/ should be crops from your own held-out piles only, so the score reflects real photos.

    python train/train_classifier.py --arch resnet18
    python train/train_classifier.py --arch vit_b_16
    python train/train_classifier.py --arch resnet18 --train-fraction 0.25   # data-size comparison
    python train/train_classifier.py --arch resnet18 --pad --balance         # keep whole pieces, even out types
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data/crops"

ARCHS = {
    # name: (constructor, weights, function that swaps the final layer)
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT,
                 lambda m, n: setattr(m, "fc", nn.Linear(m.fc.in_features, n))),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT,
                 lambda m, n: setattr(m.heads, "head", nn.Linear(m.heads.head.in_features, n))),
}


class PadToSquare:
    """Pad the short side so a long piece (1x4, 2x4) is never cut down to its middle studs."""

    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        left, top = (s - w) // 2, (s - h) // 2
        return TF.pad(img, [left, top, s - w - left, s - h - top], fill=(124, 116, 104))


def make_transforms(train, pad=False):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if pad:
        # Whole piece always in view: no random crop in training, no center crop at test time.
        steps = [PadToSquare(), transforms.Resize((224, 224))]
        if train:
            steps += [transforms.RandomRotation(180), transforms.RandomHorizontalFlip(),
                      transforms.ColorJitter(0.3, 0.3, 0.2)]
        return transforms.Compose(steps + [transforms.ToTensor(), norm])
    if train:
        # Pieces can appear at any rotation; colors vary with lighting.
        return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                                   transforms.RandomRotation(180), transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(0.3, 0.3, 0.2), transforms.ToTensor(), norm])
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])


def evaluate(model, loader, device, n_classes):
    model.eval()
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, pred):
                confusion[t, p] += 1
    return confusion.diag().sum().item() / max(confusion.sum().item(), 1), confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-fraction", type=float, default=1.0, help="use part of the training set")
    ap.add_argument("--pad", action="store_true", help="pad crops to square instead of cropping them")
    ap.add_argument("--balance", action="store_true", help="sample every piece type equally often")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = datasets.ImageFolder(CROPS / "train", make_transforms(True, args.pad))
    val_ds = datasets.ImageFolder(CROPS / "val", make_transforms(False, args.pad))
    test_ds = datasets.ImageFolder(CROPS / "test", make_transforms(False, args.pad))
    assert train_ds.classes == val_ds.classes == test_ds.classes, "every split needs the same piece folders"
    if args.train_fraction < 1.0:
        g = torch.Generator().manual_seed(0)
        keep = torch.randperm(len(train_ds), generator=g)[: int(len(train_ds) * args.train_fraction)]
        train_ds = torch.utils.data.Subset(train_ds, keep.tolist())

    loader = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=2)
    train_loader = lambda: loader(train_ds, True)
    if args.balance:
        # Rare types get drawn more often, so brick_1x1 (twice the photos) stops being the safe guess.
        targets = torch.tensor([y for _, y in train_ds.samples] if hasattr(train_ds, "samples")
                               else [train_ds.dataset.targets[i] for i in train_ds.indices])
        weights_per_sample = (1.0 / torch.bincount(targets).float())[targets]
        sampler = torch.utils.data.WeightedRandomSampler(weights_per_sample, len(targets))
        train_loader = lambda: torch.utils.data.DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2)
    build, weights, set_head = ARCHS[args.arch]
    model = build(weights=weights)
    set_head(model, len(val_ds.classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    tag = "_pad" * args.pad + "_bal" * args.balance
    out = ROOT / f"runs/classifier_{args.arch}_{args.train_fraction}{tag}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader():
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        val_acc, _ = evaluate(model, loader(val_ds, False), device, len(val_ds.classes))
        print(f"epoch {epoch + 1}/{args.epochs}  val acc {val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save({"arch": args.arch, "model": model.state_dict(), "classes": val_ds.classes,
                        "pad": args.pad}, out)

    model.load_state_dict(torch.load(out)["model"])
    test_acc, confusion = evaluate(model, loader(test_ds, False), device, len(test_ds.classes))
    print(f"test acc on real crops: {test_acc:.3f}")
    out.with_suffix(".confusion.json").write_text(json.dumps(
        {"classes": test_ds.classes, "matrix": confusion.tolist()}, indent=1))


if __name__ == "__main__":
    main()
