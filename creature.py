# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
import numpy as np

class Creature(legacy_base.Walker):
  """A fully configurable creature walker."""

  def _build(self, xml_config, name='creature', marker_rgba=None, initializer=None):
    """Build an creature.

    Args:
      xml_config: path to XML configuration of creature
      name: name of the walker.
      marker_rgba: (Optional) color the ant's front legs with marker_rgba.
      initializer: (Optional) A `WalkerInitializer` object.
    """
    
    super()._build(initializer=initializer)
    self._appendages_sensors = []
    self._bodies_pos_sensors = []
    self._mjcf_root = mjcf.from_path(xml_config)
    if name:
      self._mjcf_root.model = name
      
    # Set corresponding marker color if specified.
    
    if marker_rgba is not None:
      for geom in self.marker_geoms:
        geom.set_attributes(rgba=marker_rgba)

    # Initialize previous action.
    self._prev_action = np.zeros(shape=self.action_spec.shape,
                                 dtype=self.action_spec.dtype)

  def initialize_episode(self, physics, random_state):
    self._prev_action = np.zeros_like(self._prev_action)

  def apply_action(self, physics, action, random_state):
    super().apply_action(physics, action, random_state)

    # Updates previous action.
    self._prev_action[:] = action

  def _build_observables(self):
    return CreatureObservables(self)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def upright_pose(self):
    return base.WalkerPose()

  @property
  def marker_geoms(self):
    return self._mjcf_root.find_all('geom')

  @composer.cached_property
  def actuators(self):
    return self._mjcf_root.find_all('actuator')

  @composer.cached_property
  def root_body(self):
    return self._mjcf_root.find('body', 'seg0')

  @composer.cached_property
  def bodies(self):
    return tuple(self.mjcf_model.find_all('body'))

  @composer.cached_property
  def mocap_tracking_bodies(self):
    """Collection of bodies for mocap tracking."""
    return tuple(self.mjcf_model.find_all('body'))

  @property
  def mocap_joints(self):
    return self.mjcf_model.find_all('joint')

  @property
  def _foot_bodies(self):
    return tuple(self.mjcf_model.find_all('body'))

  @composer.cached_property
  def end_effectors(self):
    return self._foot_bodies

  @composer.cached_property
  def observable_joints(self):
    return [actuator.joint for actuator in self.actuators]  # pylint: disable=not-an-iterable

  @composer.cached_property
  def egocentric_camera(self):
    return self._mjcf_root.find('camera', 'egocentric')

  def aliveness(self, physics):
    return (physics.bind(self.root_body).xmat[-1] - 1.) / 2.

  @composer.cached_property
  def ground_contact_geoms(self):
    foot_geoms = []
    for foot in self._foot_bodies:
      foot_geoms.extend(foot.find_all('geom'))
    return tuple(foot_geoms)

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def appendages_sensors(self):
    return self._appendages_sensors

  @property
  def bodies_pos_sensors(self):
    return self._bodies_pos_sensors


class CreatureObservables(legacy_base.WalkerObservables):
  """Observables for the Creature."""

  # @composer.observable
  # def appendages_pos(self):
  #   """Equivalent to `end_effectors_pos` with the head's position appended."""
  #   appendages = self._entity.end_effectors
  #   self._entity.appendages_sensors[:] = []
  #   for body in appendages:
  #     self._entity.appendages_sensors.append(
  #         self._entity.mjcf_model.sensor.add(
  #             'framepos', name=body.name + '_appendage',
  #             objtype='xbody', objname=body,
  #             reftype='xbody', refname=self._entity.root_body))
  #   def appendages_ego_pos(physics):
  #     return np.reshape(
  #         physics.bind(self._entity.appendages_sensors).sensordata, -1)
  #   return observable.Generic(appendages_ego_pos)

  # @composer.observable
  # def bodies_zaxes(self):
  #   """Z-axes of all bodies in the world frame."""
  #   bodies = self._entity.bodies
  #   self._entity.bodies_zaxis_sensors = []
  #   
  #   for body in bodies:
  #     # Create sensors for z-axis in world frame by omitting reftype and refname
  #     self._entity.bodies_zaxis_sensors.append(
  #         self._entity.mjcf_model.sensor.add(
  #             'framezaxis', name=body.name + '_world_zaxis',
  #             objtype='xbody', objname=body))
  #             
  #   def bodies_ego_zaxes(physics):
  #     # Return z-axes directly
  #     return physics.bind(self._entity.bodies_zaxis_sensors).sensordata
  #   
  #   return observable.Generic(bodies_ego_zaxes)

  @composer.observable
  def bodies_pos(self):
    """Position of bodies relative to root, in the egocentric frame."""
    bodies = self._entity.bodies
    self._entity.bodies_pos_sensors[:] = []
    for body in bodies:
      self._entity.bodies_pos_sensors.append(
          self._entity.mjcf_model.sensor.add(
              'framepos', name=body.name + '_ego_body_pos',
              objtype='xbody', objname=body,
              reftype='xbody', refname=self._entity.root_body))
    def bodies_ego_pos(physics):
      return np.reshape(
          physics.bind(self._entity.bodies_pos_sensors).sensordata, -1)
    return observable.Generic(bodies_ego_pos)

  # @composer.observable
  # def absolute_root_pos(self):
  #   """Absolute position of the root body in the global frame."""
  #   return observable.Generic(lambda physics: physics.bind(self._entity.root_body).xpos)

  @composer.observable
  def absolute_root_mat(self):
    """3x3 rotation matrix of the root body in the global frame."""
    def root_matrix(physics):
      # Get the rotation matrix (3x3, flattened to length 9)
      return physics.bind(self._entity.root_body).xmat
    return observable.Generic(root_matrix)

  # @composer.observable
  # def absolute_root_zaxis(self):
  #   """Z-axis of the root body in the global frame."""
  #   def root_zaxis(physics):
  #     # Get the rotation matrix
  #     xmat = physics.bind(self._entity.root_body).xmat
  #     
  #     # Extract the z-axis (third column of the rotation matrix)
  #     # In a flattened 3x3 matrix, the third column is at indices 2, 5, 8
  #     z_axis = np.array([xmat[2], xmat[5], xmat[8]])
  #     
  #     return z_axis
  #   return observable.Generic(root_zaxis)

  @property
  def proprioception(self):
    return ([self.joints_pos, self.bodies_pos, self.absolute_root_mat] +
            self._collect_from_attachments('proprioception'))
