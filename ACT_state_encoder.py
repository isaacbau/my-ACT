# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import Tensor, nn


class ACTStateEncoder(nn.Module):
    """Linear projections for robot/environment state tokens."""

    def __init__(self, config):
        super().__init__()
        self.has_robot_state = config.robot_state_feature is not None
        self.has_env_state = config.env_state_feature is not None

        if self.has_robot_state:
            self.robot_state_proj = nn.Linear(config.robot_state_feature.shape[0], config.dim_model)
        else:
            self.robot_state_proj = None

        if self.has_env_state:
            self.env_state_proj = nn.Linear(config.env_state_feature.shape[0], config.dim_model)
        else:
            self.env_state_proj = None

    def project_robot_state(self, state: Tensor) -> Tensor:
        if self.robot_state_proj is None:
            raise ValueError("robot_state_proj is not initialized; robot_state_feature is missing.")

        return self.robot_state_proj(state)

    def project_env_state(self, state: Tensor) -> Tensor:
        if self.env_state_proj is None:
            raise ValueError("env_state_proj is not initialized; env_state_feature is missing.")

        return self.env_state_proj(state)
