# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of rendering a sequence."""

import argparse
import os

import cv2
import glob
import igl
import numpy as np
import pandas as pd
import pyrender
import torch
import trimesh
from tqdm import tqdm
from tqdm import trange

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
    metainfo = []

    for i in trange(self._loader.num_frames, leave=False, desc="Frames"):
      self._loader.step()
      for c in range(self._loader.num_cameras):

        # Create pyrender scene.
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                               ambient_light=np.array([1.0, 1.0, 1.0]))

        # Add camera.
        scene.add(self._cameras[c], pose=np.eye(4))

        seg_node_map = {}

        pose_y = self._loader.ycb_pose[c]
        vert_m = self._loader.mano_vert[c]

        # Add YCB meshes.
        grasped_object_node = None
        grasped_object_pose = None

        for o in range(self._loader.num_ycb):
          if np.all(pose_y[o] == 0.0):
            continue
          pose = pose_y[o].copy()
          pose[1] *= -1
          pose[2] *= -1
          node = scene.add(self._mesh_y[o], pose=pose)
          seg_node_map.update({node: (_YCB_SEG_VALUES[self._loader.ycb_ids[o]],) * 3})

          if self._loader.ycb_ids[o] == self._loader._ycb_grasp_id:
            grasped_object_mesh = node.mesh
            grasped_object_pose = pose

        # Add MANO meshes.
        hand_meshs = []
        for o in range(self._loader.num_mano):
          if np.all(vert_m[o] == 0.0):
            continue
          vert = vert_m[o].copy()
          vert[:, 1] *= -1
          vert[:, 2] *= -1
          mesh = trimesh.Trimesh(vertices=vert, faces=self._faces)
          mesh = pyrender.Mesh.from_trimesh(mesh)
          node = scene.add(mesh)
          seg_node_map.update({node: (_MANO_SEG_VALUE,) * 3})
          hand_meshs.append(mesh)

        # Render raw mask (with all YCB objects)
        seg_mask, _ = self._r.render(
          scene, pyrender.RenderFlags.SEG, seg_node_map=seg_node_map
        )

        # Keep only one of the channels and copy since the rendered mask is immutable
        seg_mask = seg_mask[:, :, 0].copy()

        # Erase background objects and set grasped object pixel values to 2
        is_hand = (seg_mask == _MANO_SEG_VALUE)
        is_grasped_object = (seg_mask == _YCB_SEG_VALUES[self._loader._ycb_grasp_id])
        is_background = ~(is_hand | is_grasped_object)

        seg_mask[is_background] = 0
        seg_mask[is_hand] = 1
        seg_mask[is_grasped_object] = 2

        # Save segmentation mask as png (lossless compression!)
        seg_file = self._render_dir[c] + "/seg_{:06d}.png".format(i)
        cv2.imwrite(seg_file, seg_mask)

      # Compute hand-object distance (only for a single camera angle)
      def transform(matrix, vertices):
          """ Apply homogeneous transformation to array of 3D points"""
          n, _ = vertices.shape
          vertices = np.hstack([vertices, np.ones((n,1), vertices.dtype)])
          vertices = (matrix @ vertices.T).T
          return vertices[:,:3]

      if (grasped_object_mesh is not None) and (hand_meshs != []):
        mesh_mesh_distance = np.inf

        # Compute world coordinates of object mesh vertices
        object_primitive = grasped_object_mesh.primitives[0]
        object_faces = object_primitive.indices.astype(np.int32)
        object_vertices = transform(grasped_object_pose, object_primitive.positions)

        for hand_mesh in hand_meshs:

          # Compute distance between current hand and object
          hand_primitive = hand_mesh.primitives[0]
          hand_vertices = hand_primitive.positions + 1e-10

          # Compute smallest mesh-mesh distance
          smallest_signed_distances, _, _ = igl.signed_distance(
            hand_vertices, object_vertices, object_faces, return_normals=False
          )
          mesh_mesh_distance = min(mesh_mesh_distance, smallest_signed_distances.min())

      else:
        mesh_mesh_distance = np.nan

      # Save meta info for this sequence and camera angle
      subject, sequence = self._name.split("/")
      metainfo.append({
        "subject": subject,
        "sequence": sequence,
        "frame": i,
        "num_mano_hands": self._loader.num_mano,
        "grasped_object_ycb_index": self._loader._ycb_grasp_id,
        "hand_object_distance": mesh_mesh_distance,
      })

    # Save meta info for this sequence (in the 'data' directory, together with the
    # rendered segmentation masks)
    metainfo = pd.DataFrame(metainfo)
    metainfo.to_csv(os.path.join(os.path.dirname(renderer._render_dir[c]), "meta.csv"))
      

  def run(self):
    """Runs the renderer."""
    self._render_seg()


if __name__ == '__main__':
  args = parse_args()

  # Find all sequences
  sequence_paths = os.path.join(os.environ["DEX_YCB_DIR"], "*-subject-*", "*")
  sequence_paths = glob.glob(sequence_paths, recursive=True)
  sequence_names = [os.sep.join(path.split(os.sep)[-2:]) for path in sequence_paths]

  # Render each sequence sequentially since we only have a single GPU
  t = tqdm(sequence_names)
  for sequence_name in t:
    t.set_description(sequence_name, refresh=True)
    renderer = Renderer(sequence_name, args.device)
    renderer.run()

  print("All done!")

  