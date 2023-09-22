def remove_keys_from_dict(
        replacement_dict: dict,
        adjust_list,
    ):
    keys_to_remove = []
    for k1, v1 in replacement_dict.items():
        keys_to_remove_inner = []
        for k2, v2 in v1.items():
            if k2 not in adjust_list:
                keys_to_remove_inner.append(k2)

        for k2 in keys_to_remove_inner:
            del replacement_dict[k1][k2]

        if len(replacement_dict[k1]) == 0:
            keys_to_remove.append(k1)

    for k in keys_to_remove:
        del replacement_dict[k]


d = {0: {'reflection': 0, 'group': 16, 'out_channel': 1.5, 'kernel_size': 5, 'stride': 2}, 1: {'reflection': 0, 'group': 16, 'num_layers': 1, 'conv_op': 'mbconv', 'kernel_size': 3, 'se_ratio': 0.75, 'out_channel': 2.5, 'skip': 'conv', 'stride': 1}, 2: {'reflection': 0, 'group': 8, 'num_layers': 1, 'conv_op': 'conv', 'kernel_size': 3, 'se_ratio': 0.5, 'out_channel': 3.75, 'skip': 'no', 'stride': 1}, 3: {'reflection': 0, 'group': 2, 'num_layers': 2, 'conv_op': 'mbconv', 'kernel_size': 5, 'se_ratio': 0.5, 'out_channel': 1.0, 'skip': 'identity', 'stride': 2}, 4: {'reflection': -1, 'group': 2, 'out_channel': 1.0, 'kernel_size': 4, 'stride': 2}}
keys =  ['group', 'out_channel', 'reflection']

remove_keys_from_dict(d, keys)