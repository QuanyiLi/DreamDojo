# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if __name__ == "__main__":
    import os
    import torch
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    # Monkey-patch launch to save .pt after training
    import cosmos_oss.scripts.train as _train_module
    from cosmos_predict2._src.imaginaire.lazy_config import instantiate as _instantiate
    from cosmos_predict2._src.predict2.utils.model_loader import (
        create_model_from_consolidated_checkpoint_with_fsdp as _create_model,
    )
    from cosmos_oss.init import is_rank0 as _is_rank0
    from cosmos_predict2._src.imaginaire.utils.launch import log_reproducible_setup as _log_setup

    def _patched_launch(config, args):
        config.validate()
        config.freeze()
        trainer = config.trainer.type(config)
        _log_setup(config, args)

        if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
            model = _create_model(config)
        else:
            model = _instantiate(config.model)

        dataloader_train = _instantiate(config.dataloader_train)
        dataloader_val = _instantiate(config.dataloader_val)
        trainer.train(model, dataloader_train, dataloader_val)

        # Save consolidated .pt checkpoint after training
        # Use dcp_to_torch_save to properly consolidate sharded DCP checkpoint
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        latest_dcp = os.path.join(config.job.path_local, "checkpoints")
        latest_file = os.path.join(latest_dcp, "latest_checkpoint.txt")
        if os.path.exists(latest_file):
            with open(latest_file) as f:
                latest_iter = f.read().strip()
            dcp_model_dir = os.path.join(latest_dcp, latest_iter, "model")
            pt_path = os.path.join(config.job.path_local, "model_consolidated.pt")
            if _is_rank0():
                print(f"Consolidating DCP checkpoint {dcp_model_dir} -> {pt_path}...")
                dcp_to_torch_save(dcp_model_dir, pt_path)
                print(f"Saved consolidated checkpoint to {pt_path}")

    _train_module.launch = _patched_launch
    _train_module.main()
