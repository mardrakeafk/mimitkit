import os
import yaml

MOTION_DIR = "data/HitReactionWithoutStanding/LightClamp"
OUT_FILE = "data/envs/light_hit_set.yaml"
EXCLUDE = ["Leg", "Foot"]


def main():
    files = sorted(f for f in os.listdir(MOTION_DIR) if f.endswith(".pkl"))

    entries = []
    skipped = []
    for f in files:
        if any(kw.lower() in f.lower() for kw in EXCLUDE):
            skipped.append(f)
            continue
        entries.append({"file": os.path.join(MOTION_DIR, f), "weight": 1.0})

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as fs:
        yaml.safe_dump({"motions": entries}, fs, default_flow_style=False, sort_keys=False)

    print("wrote {:d} motions -> {:s}".format(len(entries), OUT_FILE))
    for i, e in enumerate(entries):
        print("  [{:d}] {:s}".format(i, os.path.basename(e["file"])))
    if skipped:
        print("skipped {:d}: {:s}".format(len(skipped), ", ".join(skipped)))
    return


if __name__ == "__main__":
    main()