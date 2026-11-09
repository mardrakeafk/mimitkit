"""
This module provides functionality to convert motion data from GMR format to MimicKit format. 

Usage:
    Command line:
        python tools/data_format/gmr_to_mimickit.py
    Required arguments:
        --input_file PATH       Path to the input GMR pickle file
        --output_file PATH      Path to save the output MimicKit pickle file
    Optional arguments:
        --loop {wrap,clamp}     Loop mode for the motion (default: wrap)
        --start_frame INT       Start frame for motion clipping (default: 0)
        --end_frame INT         End frame for motion clipping (default: -1, uses all frames)
        --output_fps INT        Frame rate for the output motion (default: same as input)
    
GMR Format:
    The input GMR format should be a pickle file containing a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4), format (x, y, z, w)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'local_body_pos': Currently unused (can be None)
    - 'link_body_list': Currently unused (can be None)

Output:
    Creates a dictionary containing MimicKit motion data saved as a pickle file, with loop mode stored as INT and motion data stored as
    concatenated arrays of [root_pos, root_rot_expmap, dof_pos] per frame.
"""
import sys
import numpy.core as core
import os
sys.modules['numpy._core'] = core

import argparse
import pickle
import numpy as np

import torch

sys.path.append(".")  # Ensure the repository root is on sys.path so we can use some utilities.

from mimickit.anim.motion import Motion, LoopMode
from mimickit.util.torch_util import quat_to_exp_map

def convert_gmr_to_mimickit(gmr_file_path, output_file_path, loop_mode, start_frame, end_frame, output_fps):
    """
    Convert a GMR compatible motion dataset to MimicKit compatible dataset.
    
    Args:
        gmr_file_path (str): Path to the GMR format pickle file
        output_file_path (str): Path to save the MimicKit format pickle file
        loop_mode (bool): Whether the motion should loop (Set to wrap as default)
    """
    if loop_mode == "wrap":
        loop_mode_out = LoopMode.WRAP # MimicKit LoopMode.WRAP
    elif loop_mode == "clamp":
        loop_mode_out = LoopMode.CLAMP # MimicKit LoopMode.CLAMP
    
    # Load GMR format data
    with open(gmr_file_path, 'rb') as f:
        gmr_data = pickle.load(f)

    # Extract data from GMR format
    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']  # Shape: (num_frames, 3)
    root_rot_quat = gmr_data['root_rot']  # Shape: (num_frames, 4), quaternion format
    dof_pos = gmr_data['dof_pos']    # Shape: (num_frames, num_dofs)

    # Log the type and shape of each extracted term
    print("\n" + "="*60)
    print("📥 LOADED GMR DATA")
    print("="*60)
    print(f"⏱️  FPS:           type={type(fps).__name__}, value={fps}")
    print(f"📍 Root Position: type={type(root_pos).__name__}, shape={root_pos.shape}")
    print(f"🔄 Root Rotation: type={type(root_rot_quat).__name__}, shape={root_rot_quat.shape}")
    print(f"🦴 DOF Position:  type={type(dof_pos).__name__}, shape={dof_pos.shape}")
    print("="*60 + "\n")

    # Verify shapes
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")

    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot_quat shape (num_frames, 4), got {root_rot_quat.shape}")

    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")

    # Convert quaternion to exponential map
    root_rot = quat_to_exp_map(torch.tensor(root_rot_quat)).numpy()

    # Stack all motion data along the last dimension
    # frames shape: (num_frames, 3 + 3 + num_dofs) = (num_frames, 6 + num_dofs)
    frames = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    # Chop frames
    if end_frame == -1:
        end_frame = frames.shape[0]
    assert 0 <= start_frame < end_frame <= frames.shape[0], "Invalid start_frame or end_frame."
    frames = frames[start_frame:end_frame, :]

    save_fps = fps if output_fps == -1 else output_fps

    out_data = Motion(loop_mode=loop_mode_out, fps=save_fps, frames=frames)

    # Save to MimicKit format
    out_data.save(output_file_path)

    print("\n" + "="*60)
    print("✅ CONVERSION SUCCESSFUL")
    print("="*60)
    print(f"📁 Input:  {gmr_file_path}")
    print(f"💾 Output: {output_file_path}")
    print("-"*60)
    print(f"📊 Frames Shape:  {frames.shape}")
    print(f"🎬 Total Frames: {frames.shape[0]}")
    print(f"⏱️  FPS:          {save_fps}")
    print(f"🔄 Loop Mode:    {loop_mode_out}")
    print("="*60 + "\n")

    return out_data

def main():
    folder = "/home/maxchen/PycharmProjects/MimicKit/data/HitReactionConverted/Medium"
    save_folder = "/home/maxchen/PycharmProjects/MimicKit/data/HitReactionWithoutStanding/MediumClamp"

    for dirpath, dirnames, filenames in os.walk(folder):
        print("当前目录：", dirpath)

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            save_path = os.path.join(save_folder, filename)
            with open("/home/maxchen/PycharmProjects/MimicKit/Light_Medium_start_frame.pkl", "rb") as f:
                motion_data = pickle.load(f)
            start_frame = motion_data[filename]['frame_0based']
            convert_gmr_to_mimickit(file_path, save_path, loop_mode="clamp", start_frame=start_frame, end_frame=-1, output_fps=-1)




if __name__ == "__main__":
    main()