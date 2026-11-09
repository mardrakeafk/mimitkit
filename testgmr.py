import pickle
import os
import numpy as np


# ============================================================
# 配置
# ============================================================

INPUT_FILE = (
    "data/HitReactionRetarget/Light/Front/"
    "AS_HitReaction_Light_Front_Abdomen.pkl"
)

OUTPUT_FILE = (
    "data/HitReactionRetarget/Light/Front/"
    "AS_HitReaction_Light_Front_Abdomen_down_1cm.pkl"
)

# 下降 1 cm
HEIGHT_OFFSET = 0.01


# ============================================================
# Load
# ============================================================

with open(INPUT_FILE, "rb") as f:
    data = pickle.load(f)

print("PKL type:", type(data))

if not isinstance(data, dict):
    raise TypeError(
        "Expected MimicKit motion pkl to contain a dict."
    )

print("Keys:", data.keys())

if "frames" not in data:
    raise KeyError(
        "Cannot find 'frames' in motion pkl."
    )


# ============================================================
# 修改 motion
# ============================================================

frames = np.asarray(data["frames"]).copy()

print("Frames shape:", frames.shape)

print()
print("Before:")
print("Root Z min :", frames[:, 2].min())
print("Root Z max :", frames[:, 2].max())
print("Root Z mean:", frames[:, 2].mean())


# ------------------------------------------------------------
# 所有帧的 root Z 整体下降 1 cm
# ------------------------------------------------------------

frames[:, 2] -= HEIGHT_OFFSET


print()
print("After:")
print("Root Z min :", frames[:, 2].min())
print("Root Z max :", frames[:, 2].max())
print("Root Z mean:", frames[:, 2].mean())


# 放回原来的dict
data["frames"] = frames


# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(
        data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print()
print("==============================")
print("Saved corrected motion")
print("==============================")
print("Input :", INPUT_FILE)
print("Output:", OUTPUT_FILE)
print("Root Z decreased by:", HEIGHT_OFFSET, "m")
print("Root Z decreased by:", HEIGHT_OFFSET * 100, "cm")