import re
from typing import List, Union, Dict
from omegaconf import OmegaConf
from omegaconf import DictConfig


class BlockArgs:
    def __init__(
        self, 
        reflection: int, 
        group: int, 
        kernel_size: int, 
        out_channel: Union[int, float],
        stride: int = None,
        num_layers: int = None, 
        conv_op: str = None,
        se_ratio: float = None, 
        skip: str = None,
    ) -> None:
        self.reflection = reflection
        self.group = group
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channel = out_channel
        self.num_layers = num_layers
        self.conv_op = conv_op
        self.se_ratio = se_ratio
        self.skip = skip

    @property
    def reflection(self):
        return self._reflection

    @reflection.setter
    def reflection(self, value):
        assert isinstance(value, int) and value in [-1, 0]
        self._reflection = value

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        assert isinstance(value, int) and value >= 0
        self._group = value

    @property
    def kernel_size(self):
        return self._kernel_size
    
    @kernel_size.setter
    def kernel_size(self, value):
        assert isinstance(value, int) and value in [0, 1, 3, 5, 7]
        self._kernel_size = value

    @property
    def stride(self):
        return self._stride
    
    @stride.setter
    def stride(self, value):
        assert isinstance(value, int) and value in [1, 2]
        self._stride = value

    @property
    def out_channel(self):
        return self._out_channel

    @out_channel.setter
    def out_channel(self, value):
        assert value > 0
        self._out_channel = value

    @property
    def num_layers(self):
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        if value is None:
            self._num_layers = None
        else:
            assert isinstance(value, int) and value > 0
            self._num_layers = value

    @property
    def se_ratio(self):
        return self._se_ratio

    @se_ratio.setter
    def se_ratio(self, value):
        if value is None:
            self._se_ratio = None
            return
        assert isinstance(value, float) and 0 <= value <= 1
        self._se_ratio = value

    @property
    def skip(self):
        return self._skip
    
    @skip.setter
    def skip(self, value):
        if value is None:
            self._skip = None
            return
        assert isinstance(value, str) and value in ["identity", "no", "conv"]
        self._skip = value

    @property
    def conv_op(self):
        return self._conv_op
    
    @conv_op.setter
    def conv_op(self, value):
        if value is None:
            self._conv_op = None
            return
        assert isinstance(value, str) and value in ["conv", "dconv", "mbconv"]
        self._conv_op = value

    @staticmethod
    def decode_block_string(block_string):
        options = {}
        ops = block_string.split('_')
        for op in ops:
            key, value = re.split(r'(?<=[a-zA-Z])(?=[^a-zA-Z])', op)
            options[key] = float(value) if '.' in value else int(value)
        
        return BlockArgs(
            reflection=options.get('r'),
            group=options.get('g'),
            kernel_size=options.get('k'),
            stride=options.get('s'),
            out_channel=options.get('o'),
            num_layers=options.get('n'),
            conv_op=options.get('c'),
            se_ratio=options.get('se'),
            skip=options.get('sk')
        )


class BlockArgsList:
    def __init__(self, blocks_args: List[BlockArgs]):
        self.blocks_args = blocks_args

    def __getitem__(self, index):
        return self.blocks_args[index]

    def __len__(self):
        return len(self.blocks_args)

    @classmethod
    def from_dict(
        cls, 
        blocks_args_dict: Union[Dict, DictConfig],
        stem_channels: int = None, 
        width_coefficient: float = None,
        depth_coefficient: float = None,
    ):
        # Convert DictConfig to dict if necessary
        if isinstance(blocks_args_dict, DictConfig):
            blocks_args_dict = OmegaConf.to_container(blocks_args_dict)
            blocks_args_dict = {int(k[1]): v for k, v in blocks_args_dict.items()}

        # Create BlockArgs objects
        blocks_args = [BlockArgs(**block_args_dict) for block_args_dict in blocks_args_dict.values()]

        # Create an instance of BlockArgsList
        instance = cls(blocks_args)

        # Perform checks and updates
        instance.check_valid_blocks_args()
        if stem_channels and width_coefficient:
            instance.update_channel_sizes(stem_channels, width_coefficient)
        if depth_coefficient:
            instance.update_layers_per_block(depth_coefficient)

        return instance

    def update_channel_sizes(self, initial_channel_size: int, width_coefficient: float):
        old_channels = initial_channel_size
        for i, block_args in enumerate(self.blocks_args):
            increase_factor = block_args.out_channel
            if i == 0:
                increase_factor *= width_coefficient

            out_channel = old_channels * increase_factor
            block_args.out_channel = out_channel
            old_channels = out_channel

    def update_layers_per_block(self, depth_coefficient: float):
        if not depth_coefficient:
            return

        for i, block_args in enumerate(self.blocks_args):
            if i == 0 or i == len(self.blocks_args) - 1:
                continue

            num_layers = int(round(depth_coefficient * block_args.num_layers))
            block_args.num_layers = num_layers

    def check_valid_blocks_args(self):
        if not self.blocks_args:
            raise ValueError("blocks_args is empty")

        previous_block = None
        for i, block in enumerate(self.blocks_args):
            # Checks that are specific to the list context
            if i != len(self.blocks_args) - 1:
                # The last block may not have a stride
                assert isinstance(block.stride, int) and block.stride > 0

            if 0 < i < len(self.blocks_args) - 1:
                # Middle blocks specific checks
                assert isinstance(block.num_layers, int) and block.num_layers > 0
                assert block.conv_op in ["conv", "dconv", "mbconv"]
                assert isinstance(block.se_ratio, float) and 0 <= block.se_ratio <= 1
                assert block.skip in ["identity", "no", "conv"], f"Invalid skip value: {block.skip}"

            if previous_block:
                # Checks involving the previous block
                assert previous_block.reflection >= block.reflection
                assert previous_block.group >= block.group

            previous_block = block



