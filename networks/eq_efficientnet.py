"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import math
from typing import List, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

sys.path.append('../scaling-laws-ecnn') # add parent directory
import torch
from torch import nn
from torch.nn import functional as F
from networks.eq_efficientnet_util import (
    BlockDecoder,
    eq_drop_connect,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    Eq_Conv2dSamePadding,
    Conv2dSamePadding
)
from networks.efficientnet import EfficientNet
from networks.eq_layers import EquivariantPool, EquivariantSqueezeExcitation, Restriction
from networks.util import calculate_output_image_size, iter_fix_param

from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    GroupTensor,
    FieldType,
    EquivariantModule,
    SequentialModule,
    R2Conv,
    GroupNorm,
    InducedNormGroupNorm,
    GroupStandardization,
    BatchNorm,
    InducedNormBatchNorm,
    Mish,
    ReLU,
    Swish,
    NormNonLinearity,
    InducedGatedNonLinearity,
    GroupPooling,
    NormPool,
    InducedNormPool,
    NormAvgPool,
    NormMaxPool,
    PointwiseAvgPool,
    PointwiseAdaptiveAvgPool,
    PointwiseMaxPool,
    DisentangleModule,
    RestrictionModule,
    MultipleModule,
)
from group_theory import Representation
from nn.modules import nonlinearities

import os
os.environ['HYDRA_FULL_ERROR'] = '1'

CHANNELS_CONSTANT = 1

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(EquivariantModule):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    """

    # TODO for fix params need to use both final 
    # _block_args.expand_ratio
    # _block_args.output_filters

    def __init__(self, in_type, block_args, global_params, image_size, fix_params, normal_block):
        super().__init__()
        self._block_args = block_args
        self.in_type = in_type
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        # inp = self._block_args.input_filters  # number of input channels
        inp = in_type
        # oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        oup = int((self._block_args.input_filters * self._block_args.expand_ratio))
        # oup = len(in_type) * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            kwargs = {'in_type': inp, 'out_channels': oup, 'image_size': image_size, 'kernel_size': 1, 'bias': False}
            self._expand_conv = iter_fix_param(Eq_Conv2dSamePadding, normal_block._expand_conv, fix_params, **kwargs)
            # self._expand_conv = Eq_Conv2dSamePadding(in_type=inp, out_channels=oup, 
            #                     image_size=image_size, kernel_size=1, bias=False)

            # self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            self._bn0 = BatchNorm(in_type=self._expand_conv.out_type, momentum=self._bn_mom, eps=self._bn_eps)
            self._swish0 = Swish(in_type=self._bn0.out_type)
            inp = self._swish0.out_type
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        kwargs = {'in_type': inp, 'out_channels': len(inp), 'image_size': image_size, 'groups': len(inp), 
                  'kernel_size': k, 'stride': s, 'bias': False}
        self._depthwise_conv = iter_fix_param(Eq_Conv2dSamePadding, normal_block._depthwise_conv, False, **kwargs)
        # self._depthwise_conv = Eq_Conv2dSamePadding(
        #     in_type=inp, out_channels=oup, image_size=image_size, groups=oup,  # groups makes it depthwise
        #     kernel_size=k, stride=s, bias=False
        # )
        self._bn1 = BatchNorm(in_type=self._depthwise_conv.out_type, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish1 = Swish(in_type=self._bn1.out_type)
        out_type = self._swish1.out_type
        # self._depthwise_conv = Conv2d(
        #     in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
        #     kernel_size=k, stride=s, bias=False)
        #self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            # we don't need same padding as it is a 1x1 conv
            # Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            input_channels_squeeze = len(in_type)
            num_squeezed_channels = max(1, int(input_channels_squeeze * self._block_args.se_ratio))
            kwargs = {'in_type': out_type, 'in_channels': input_channels_squeeze, 
                      'squeeze_channels': num_squeezed_channels, 'act_func': "Swish"}
            
            self.squeeze = iter_fix_param(EquivariantSqueezeExcitation, 
                                          nn.Sequential(*[normal_block._se_reduce, normal_block._se_expand]), 
                                          fix_params, channel_name='in_channels', **kwargs)
            # self.squeeze = EquivariantSqueezeExcitation(in_type=out_type, 
            #                 in_channels=input_channels_squeeze, squeeze_channels=num_squeezed_channels, 
            #                 act_func="Swish")
            out_type = self.squeeze.out_type

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        kwargs = {'in_type': out_type, 'out_channels': final_oup, 'image_size': image_size, 'kernel_size': 1, 'bias': False}
        self._project_conv = iter_fix_param(Eq_Conv2dSamePadding, normal_block._project_conv, fix_params, **kwargs)
        # self._project_conv = Eq_Conv2dSamePadding(in_type=out_type, out_channels=final_oup, image_size=image_size, kernel_size=1, bias=False)
        
        #self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn2 = BatchNorm(in_type=self._depthwise_conv.out_type, momentum=self._bn_mom, eps=self._bn_eps)

        self.out_type = self._bn2.out_type
        # self._swish = MemoryEfficientSwish()
        # self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish0(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish1(x)

        # Squeeze and Excitation
        if self.has_se:
            x = self.squeeze(x)

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and len(self.in_type) == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = eq_drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

class EquivariantEfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.
    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(
            self, blocks_args=None, global_params=None, image_size=None, 
            input_channels=3, num_classes=10, 
            group: str = "cyclic",
            rotation: int = 4,
            restrict: str = None,  # "invariant", "reflection", "halved"
            fix_params: bool = True,
            ):
        super().__init__()
        self.fix_params = fix_params
        self.restrict = restrict
        if self.fix_params:
            self.efficientnet = EfficientNet(
                blocks_args=blocks_args, global_params=global_params, image_size=image_size,
                input_channels=input_channels, num_classes=num_classes,
            )
        blocks_args = list(blocks_args)
        assert image_size is not None, 'Please provide image size'
        assert isinstance(blocks_args, list), f'blocks_args should be a list, is a {type(blocks_args)}'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        # BlockArgs
        blocks_args = BlockDecoder.decode(blocks_args)
        self._blocks_args = blocks_args

        self.group = group
        self.rotation = rotation
        self.input_channels = input_channels
        image_size = list(image_size)
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        # Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Get group spaces for specified rotations and flips
        if self.group == "cyclic":
            self.gspace = rot2dOnR2(self.rotation)
        elif self.group == "dihedral":
            self.gspace = flipRot2dOnR2(self.rotation)
        elif self.group == "orthogonal":
            self.gspace = flipRot2dOnR2(-1)
        else:
            raise ValueError(
                f'Group "{self.group}" is not know. Available groups: [cyclic, dihedral, orthogonal]'
            )

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.input_channels
        )

        # Stem
        out_channels = round_filters(32, self._global_params, rotation=1, fix_params=False)
        # self._conv_stem = Eq_Conv2dSamePadding()
        kwargs = {'in_type': self.input_field_type, 'out_channels': out_channels,
            'kernel_size': 3, 'stride': 2, 'image_size': image_size, 'bias': False}
        self._conv_stem = iter_fix_param(Eq_Conv2dSamePadding, self.efficientnet._conv_stem,
                                               fix_params=self.fix_params, **kwargs)

        # size params of conv_stem
        self._bn0 = BatchNorm(in_type=self._conv_stem.out_type, momentum=bn_mom, eps=bn_eps)
        self._swish0 = Swish(in_type=self._bn0.out_type)
        #self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self.field_type = self._swish0.out_type
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        # self._blocks = nn.ModuleList([])
        self._blocks = []
        counter = 0
        for i, block_args in enumerate(self._blocks_args):
            print(f"Building block: {i}")
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params, rotation=self.rotation, fix_params=self.fix_params),
                output_filters=round_filters(block_args.output_filters, self._global_params, rotation=self.rotation, fix_params=self.fix_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(self.field_type, block_args, 
                                            self._global_params, image_size=image_size, 
                                            fix_params=self.fix_params, normal_block=self.efficientnet._blocks[counter]))
            counter += 1
            self.field_type = self._blocks[-1].out_type
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(self.field_type, block_args, 
                                                self._global_params, image_size=image_size, 
                                                fix_params=self.fix_params, normal_block=self.efficientnet._blocks[counter]))
                counter += 1
                self.field_type = self._blocks[-1].out_type
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        self._blocks = SequentialModule(*self._blocks)

        # Restrict
        self.restriction = Restriction(self.field_type, self.group, self.rotation, self.restrict)
        self.field_type = self.restriction.out_type

        # Head
        input_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params, rotation=self.rotation, fix_params=self.fix_params)
        self._conv_head = Eq_Conv2dSamePadding(self.field_type, out_channels, 
                                               kernel_size=1, image_size=image_size, 
                                               bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._bn1 = BatchNorm(in_type=self._conv_head.out_type, momentum=bn_mom, eps=bn_eps)
        self._swish1 = Swish(in_type=self._bn1.out_type)


        # Final linear layer
        self.invariant_map = EquivariantPool(self._swish1.out_type, invariant_map=True)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.drop_out)
            self._fc = nn.Linear(out_channels, self.num_classes)

        # set activation to memory efficient swish by default
        # self._swish = Swish()
        # self._swish = MemoryEfficientSwish()

        if self.fix_params:
            # size of wrn total and size of equivariant part
            norm_para = sum([p.numel() for p in self.efficientnet.parameters() if p.requires_grad])
            del self.efficientnet
            equi_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
            current_ratio = equi_param / norm_para
            print(f"Equivariant_EfficientNet / EfficientNet parameter ratio: {current_ratio:.3f}")

    
    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = GroupTensor(inputs, self.input_field_type)
        x = self._swish0(self._bn0(self._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish1(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish0(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish1(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = GroupTensor(inputs, self.input_field_type)
        x = self.extract_features(x)
        # Pooling and final linear layer
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.
        Args:
            model_name (str): Name for efficientnet.
        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res


@hydra.main(config_path="../experiment/conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    inp = torch.rand(1, 1, 32, 32)
    image_size = [inp.shape[2], inp.shape[3]]
    n_inputs = inp.shape[1]
    n_outputs = 10
    # depth, num_classes, widen_factor=1, dropRate=0.0
    #net = EquivariantWideResNet()
    net = hydra.utils.instantiate(
            cfg.model,
            image_size=image_size,
            input_channels=n_inputs,
            num_classes=n_outputs,
        )
    tot_param = sum([p.numel() for p in net.parameters()  if p.requires_grad])
    print(f'Total number of parameters: {tot_param}')
    # print(net)
    return
    inp = inp.cuda()
    net.cuda()
    print(net(inp).size())

    #y = net(torch.randn(1,3,32,32))
    #print(y.size())


if __name__ == "__main__":
    main()
