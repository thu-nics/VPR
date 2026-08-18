# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_system.environments.env_manager import EnvironmentManagerBase, make_envs

__all__ = ["EnvironmentManagerBase", "make_envs"]


def __getattr__(name):
    """Load the environment manager only when its public symbols are requested.

    Parsers, oracles, and action validators remain importable in lightweight CPU
    environments that do not install the full Torch/Ray training stack.
    """
    if name in __all__:
        from agent_system.environments.env_manager import EnvironmentManagerBase, make_envs

        return {"EnvironmentManagerBase": EnvironmentManagerBase, "make_envs": make_envs}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
