import pickle
import os
import numpy as np


def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
        motion_fps = motion_data["fps"]
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"] # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        motion_local_body_pos = motion_data["local_body_pos"]
        motion_link_body_list = motion_data["link_body_list"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list

def switch_single_motion(motion_file, save_path):


    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list=load_robot_motion(motion_file)
    T=len(motion_root_pos)


    motion_pos_root=np.array(motion_root_pos,dtype=np.float32)
    motion_rot_root=np.array(motion_root_rot,dtype=np.float32)
    motion_pos_dof=np.array(motion_dof_pos,dtype=np.float32)


    motion_rot_root=motion_rot_root
    motion_pos_dof=motion_pos_dof
    motion_pos_root=motion_pos_root

    gmr = {
        "fps": motion_fps,
        "root_pos": motion_pos_root,

        # xyzw 格式
        "root_rot": motion_rot_root,

        "dof_pos": motion_pos_dof,

        # unused fields
        "local_body_pos": None,
        "link_body_list": None,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(gmr, f, protocol=4)

    print("Saved"+save_path)




folder = "data/HitReaction"
save_folder = "data/HitReactionConverted"

for dirpath, dirnames, filenames in os.walk(folder):
    print("当前目录：", dirpath)

    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        save_path = os.path.join(save_folder, file_path)
        switch_single_motion(file_path, save_path)


