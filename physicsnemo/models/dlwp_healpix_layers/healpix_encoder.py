# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Sequence

import torch as th
from hydra.utils import instantiate
from omegaconf import DictConfig


class UNetEncoder(th.nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes."""

    def __init__(
        self,
        conv_block: DictConfig,
        down_sampling_block: DictConfig,
        recurrent_block: DictConfig = None,
        input_block: DictConfig = None,
        input_channels: int = 3,
        n_channels: Sequence = (16, 32, 64),
        n_layers: Sequence = (2, 2, 1),
        conv_kernel_size: Sequence = (3, 3, 3),
        dilations: list = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        down_sampling_block: DictConfig
            dictionary of instantiable parameters for the downsample block
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        input_channels: int, optional
            Number of input channels
        n_channels: Sequence, optional
            The number of channels in each encoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        enable_healpixpad, bool, optional
            If the healpixpad library should be used (if installed)
        dropout, float, optional
            Probability to use in dropout (if 0, don't use dropout)
        """
        super().__init__()
        self.n_channels = n_channels

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        old_channels = input_channels
        self.encoder = []

        # Add input block if specified
        if input_block is not None:
            self.encoder.append(th.nn.Sequential(
                instantiate(
                    config=input_block,
                    in_channels=input_channels,
                    out_channels=n_channels[0],
                    kernel_size=2,
                    stride=2,
                    )
                )
            )
            old_channels = n_channels[0]

        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                kwargs = {
                    "config": down_sampling_block,
                    "enable_nhwc": enable_nhwc,
                    "enable_healpixpad": enable_healpixpad,
                }
                # {'_target_': 'physicsnemo.models.dlwp_healpix_layers.healpix_blocks.downsample_conv_block'}
                if "downsample_conv" in down_sampling_block._target_:
                    kwargs["in_channels"] = old_channels
                    kwargs["out_channels"] = old_channels
                modules.append(instantiate(**kwargs))

            modules.append(
                instantiate(
                    config=conv_block,
                    in_channels=old_channels,
                    latent_channels=curr_channel,
                    out_channels=curr_channel,
                    dilation=dilations[n],
                    kernel_size=conv_kernel_size[n],
                    n_layers=n_layers[n],
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                    dropout=dropout,
                )
            )
            old_channels = curr_channel

            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to enccode

        Returns
        -------
        Sequence: The encoded values
        """
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass





class UNetEncoder_depthweise(th.nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes.
    add input block to the encoder, move the channel change to the downsampling block
    """

    def __init__(
        self,
        conv_block: DictConfig,
        down_sampling_block: DictConfig,
        recurrent_block: DictConfig = None,
        input_block: DictConfig = None,
        input_channels: int = 3,
        n_channels: Sequence = (16, 32, 64),
        n_layers: Sequence = (2, 2, 1),
        conv_kernel_size: Sequence = (7, 7, 7),
        dilations: list = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        down_sampling_block: DictConfig
            dictionary of instantiable parameters for the downsample block
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        input_block: DictConfig, optional
            dictionary of instantiable parameters for the input block
        input_channels: int, optional
            Number of input channels
        n_channels: Sequence, optional
            The number of channels in each encoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        enable_healpixpad, bool, optional
            If the healpixpad library should be used (if installed)
        """
        super().__init__()
        self.n_channels = n_channels

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        self.encoder = []
        # ## Input block ##
        self.encoder.append(th.nn.Sequential(
            instantiate(
                config=input_block,
                in_channels=input_channels,
                out_channels=n_channels[0],
                kernel_size=3,
                stride=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
                )
            )
        )

        # Add downsampling and conv blocks
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                kwargs = {
                    "config": down_sampling_block,
                    "enable_nhwc": enable_nhwc,
                    "enable_healpixpad": enable_healpixpad,
                }
                # {'_target_': 'physicsnemo.models.dlwp_healpix_layers.healpix_blocks.downsample_conv_block'}
                if "downsample_conv" in down_sampling_block._target_:
                    kwargs["in_channels"] = old_channels
                    kwargs["out_channels"] = curr_channel
                modules.append(instantiate(**kwargs))

            modules.append(
                instantiate(
                    config=conv_block,
                    in_channels=curr_channel,
                    latent_channels=curr_channel,
                    out_channels=curr_channel,
                    dilation=dilations[n],
                    kernel_size=conv_kernel_size[n],
                    n_layers=n_layers[n],
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                )
            )
            old_channels = curr_channel

            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to enccode

        Returns
        -------
        Sequence: The encoded values
        """
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass



class UNetEncoder_Inception(th.nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes.
    add input block to the encoder, move the channel change to the downsampling block
    """

    def __init__(
        self,
        conv_block: DictConfig,
        down_sampling_block: DictConfig,
        recurrent_block: DictConfig = None,
        input_block: DictConfig = None,
        input_channels: int = 3,
        n_channels: Sequence = (16, 32, 64),
        n_layers: Sequence = (2, 2, 1),
        conv_kernel_size: Sequence = (3, 3, 3),
        dilations: list = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        down_sampling_block: DictConfig
            dictionary of instantiable parameters for the downsample block
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        input_block: DictConfig, optional
            dictionary of instantiable parameters for the input block
        input_channels: int, optional
            Number of input channels
        n_channels: Sequence, optional
            The number of channels in each encoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        enable_healpixpad, bool, optional
            If the healpixpad library should be used (if installed)
        """
        super().__init__()
        self.n_channels = n_channels

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        self.encoder = []
        # Add input block if specified
        self.encoder.append(th.nn.Sequential(
            instantiate(
                config=input_block,
                in_channels=input_channels,
                out_channels=n_channels[0],
                kernel_size=2,
                stride=2,
                )
            )
        )

        # Add downsampling and conv blocks
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                kwargs = {
                    "config": down_sampling_block,
                    "enable_nhwc": enable_nhwc,
                    "enable_healpixpad": enable_healpixpad,
                }
                # {'_target_': 'physicsnemo.models.dlwp_healpix_layers.healpix_blocks.downsample_conv_block'}
                if "downsample_conv" in down_sampling_block._target_:
                    kwargs["in_channels"] = old_channels
                    kwargs["out_channels"] = curr_channel
                modules.append(instantiate(**kwargs))

            modules.append(
                instantiate(
                    config=conv_block,
                    in_channels=curr_channel,
                    latent_channels=curr_channel,
                    out_channels=curr_channel,
                    dilation=dilations[n],
                    kernel_size=conv_kernel_size[n],
                    n_layers=n_layers[n],
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                )
            )
            old_channels = curr_channel

            self.encoder.append(th.nn.Sequential(*modules))

        self.encoder = th.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to enccode

        Returns
        -------
        Sequence: The encoded values
        """
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass