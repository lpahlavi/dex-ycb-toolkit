# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of rendering a sequence."""

import argparse
import torch
import pyrender
import trimesh
import os
import numpy as np
import cv2

from dex_ycb_toolkit.sequence_loader import SequenceLoader
from dex_ycb_toolkit.mesh_intersection import intersection_eval

_MANO_SEG_VALUE = 1

_YCB_SEG_VALUES = {
     1:  2,  # 002_master_chef_can
     2:  3,  # 003_cracker_box
     3:  4,  # 004_sugar_box
     4:  5,  # 005_tomato_soup_can
     5:  6,  # 006_mustard_bottle
     6:  7,  # 007_tuna_fish_can
     7:  8,  # 008_pudding_box
     8:  9,  # 009_gelatin_box
     9: 10,  # 010_potted_meat_can
    10: 11,  # 011_banana
    11: 12,  # 019_pitcher_base
    12: 13,  # 021_bleach_cleanser
    13: 14,  # 024_bowl
    14: 15,  # 025_mug
    15: 15,  # 035_power_drill
    16: 17,  # 036_wood_block
    17: 18,  # 037_scissors
    18: 19,  # 040_large_marker
    19: 20,  # 051_large_clamp
    20: 21,  # 052_extra_large_clamp
    21: 22,  # 061_foam_brick
}


def parse_args():
  parser = argparse.ArgumentParser(
      description='Render hand & object poses in camera views.')
  parser.add_argument('--name',
                      help='Name of the sequence',
                      default=None,
                      type=str)
  parser.add_argument('--device',
                      help='Device for data loader computation',
                      default='cuda:0',
                      type=str)
  args = parser.parse_args()
  return args


class Renderer():
  """Renderer."""

  def __init__(self, name, device='cuda:0'):
    """Constructor.
    Args:
      name: Sequence name.
      device: A torch.device string argument. The specified device is used only
        for certain data loading computations, but not storing the loaded data.
        Currently the loaded data is always stored as numpy arrays on cpu.
    """
    assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
    self._name = name
    self._device = torch.device(device)

    self._loader = SequenceLoader(self._name,
                                  device=device,
                                  preload=False,
                                  app='renderer')

    # Create pyrender cameras.
    self._cameras = []
    for c in range(self._loader.num_cameras):
      K = self._loader.K[c].cpu().numpy()
      fx = K[0][0].item()
      fy = K[1][1].item()
      cx = K[0][2].item()
      cy = K[1][2].item()
      cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
      self._cameras.append(cam)

    # Create meshes for YCB objects.
    self._mesh_y = []
    self._trimesh_y = []
    for o in range(self._loader.num_ycb):
      obj_file = self._loader.ycb_group_layer.obj_file[o]
      mesh = trimesh.load(obj_file)
      self._trimesh_y.append(mesh)
      self._mesh_y.append(pyrender.Mesh.from_trimesh(mesh))

    self._faces = self._loader.mano_group_layer.f.cpu().numpy()

    w = self._loader.dimensions[0]
    h = self._loader.dimensions[1]
    self._r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    self._render_dir = [
        os.path.join(os.path.dirname(__file__), "..", "data", "render",
                     self._name, self._loader.serials[c])
        for c in range(self._loader.num_cameras)
    ]
    for d in self._render_dir:
      os.makedirs(d, exist_ok=True)

  
  def _render_mesh(self, mesh, camera_idx, pose=None):
    """
    Args:
        mesh (Pyrender mesh): Mesh to be rendered, with the correct pose already applied
        camera_idx (int)
    Returns:
        np.array: Binary segmentation mask of the given mesh
    """

    # Create pyrender scene.
    bg_color, ambient_light = np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

    # Add camera.
    scene.add(self._cameras[camera_idx], pose=np.eye(4))

    # Render segmentation map
    node = scene.add(mesh, pose=pose)
    mask, _ = self._r.render(scene, pyrender.RenderFlags.SEG, seg_node_map={node: 1})
    mask = mask.copy()[:, :, 0]

    return mask

  def _render_seg(self):
    """Renders segmentation masks."""
    print('Rendering segmentation masks')
    for i in range(self._loader.num_frames):
      print('{:03d}/{:03d}'.format(i + 1, self._loader.num_frames))

      self._loader.step()

      # DEBUGGING!
      if i < 20:
        continue
      if i > 25:
        break

      for c in range(self._loader.num_cameras):

        pose_y = self._loader.ycb_pose[c]
        vert_m = self._loader.mano_vert[c]

        # Render YCB meshes.
        if (self._loader.num_ycb == 0) or np.all(pose_y[0] == 0.0):
          has_object = False
        else:
          pose = pose_y[0].copy()
          pose[1] *= -1
          pose[2] *= -1

          object_mask = self._render_mesh(self._mesh_y[0], c, pose=pose)
          object_mask[object_mask.astype(bool)] = _YCB_SEG_VALUES[self._loader.ycb_ids[0]]

          object_trimesh = self._trimesh_y[0]
          object_trimesh.apply_transform(pose)

          has_object = True
          
        # Render MANO meshes.
        if (self._loader.num_mano == 0):
          has_hand = False
        else:
          w, h = self._loader.dimensions
          hand_mask = np.zeros((h, w), dtype=np.uint8)
          hand_trimesh = None

          for o in range(self._loader.num_mano):
            
            if np.all(vert_m[o] == 0.0):
              continue
            
            vert = vert_m[o].copy()
            vert[:, 1] *= -1
            vert[:, 2] *= -1
            
            this_hand_trimesh = trimesh.Trimesh(vertices=vert, faces=self._faces)
            
            mesh = pyrender.Mesh.from_trimesh(this_hand_trimesh)
            hand_mask += self._render_mesh(mesh, c)

            if hand_trimesh is None:
              hand_trimesh = this_hand_trimesh
            else:
              hand_trimesh = hand_trimesh.union(this_hand_trimesh)

          hand_mask[hand_mask.astype(bool)] = _MANO_SEG_VALUE
          has_hand = bool(np.any(hand_mask))

        # Determine if there is contact between hand and object
        if has_hand and has_object:
          _, mesh_mesh_distance = intersection_eval(hand_trimesh, object_trimesh, res=0.001)
          object_is_grasped = True
          print(f"hand-object distance: {mesh_mesh_distance}cm")
        else:
          object_is_grasped = False

        # Construct segmentation mask
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        if has_hand is True:
          np.putmask(seg_mask, hand_mask.astype(bool), hand_mask)
        if object_is_grasped is True:
          np.putmask(seg_mask, object_mask.astype(bool), object_mask)

        # # For some reason, there are often several pixels that have non-zero values in
        # # the background. Take care of these here.
        # mano_value = _MANO_SEG_VALUE[0]
        # grasped_ycb_value = _YCB_SEG_VALUES[self._loader.ycb_ids[0]][0]
        # is_garbage = (seg_masks != mano_value) & (seg_masks != grasped_ycb_value)
        # seg_masks[is_garbage] = 0

        seg_file = self._render_dir[c] + "/seg_{:06d}.png".format(i)
        cv2.imwrite(seg_file, seg_mask)

  def run(self):
    """Runs the renderer."""
    self._render_seg()


if __name__ == '__main__':
  args = parse_args()

  renderer = Renderer(args.name, args.device)
  renderer.run()