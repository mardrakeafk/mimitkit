import pickle
import numpy as np


INPUT_FILE = "data/HitReactionRetarget/Light/Front/AS_HitReaction_Light_Front_Abdomen.pkl"

OUTPUT_FILE = "stand_static.pkl"

# 选择作为站立姿势的帧
STAND_FRAME = 0

# 生成动作的时长
DURATION = 2.0


with open(INPUT_FILE, "rb") as f:
    motion = pickle.load(f)

print(motion)
fps = motion["fps"]
frames = motion["frames"]

# 取出一帧，并保留时间维度：[1, frame_dim]
stand_frame = frames[STAND_FRAME:STAND_FRAME + 1]

# MimicKit中动作长度通常为：(帧数 - 1) / fps
num_frames = int(DURATION * fps) + 1

# 重复同一个站立姿势
stand_frames = np.repeat(
    stand_frame,
    num_frames,
    axis=0,
)

stand_motion = {
    "loop_mode": motion["loop_mode"],
    "fps": fps,
    "frames": stand_frames,
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(stand_motion, f)

print("保存完成：", OUTPUT_FILE)
print("使用帧：", STAND_FRAME)
print("输出帧数：", stand_frames.shape[0])
print("frames shape：", stand_frames.shape)