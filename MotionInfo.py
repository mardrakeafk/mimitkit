import pickle

with open("data/HitReactionWithoutStanding/MediumClamp/AS_HitReaction_Medium_Front_LeftShoulder.pkl", "rb") as f:
    d = pickle.load(f)
num_frames = len(d["frames"])
fps = d["fps"]
length = float(num_frames - 1) / fps
file2="data/HitReactionWithoutStanding/MediumClamp/AS_HitReaction_Medium_Front_LeftShoulder.pkl"
filename="AS_HitReaction_Medium_Front_LeftShoulder.pkl"

with open("Light_Medium_start_frame.pkl", "rb") as f:
    motion_data = pickle.load(f)
start_frame = motion_data[filename]['frame_0based']

print(start_frame)

print("帧数:", num_frames)
print("fps:", fps)
print("时长: {:.3f}s".format(length))