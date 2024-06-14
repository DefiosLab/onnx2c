import math


def same_upper(input_data, stride, kernel_size):
    kernel_height, kernel_width = kernel_size
    output_height = math.ceil(input_data.shape[0] / stride[0])
    output_width = math.ceil(input_data.shape[1] / stride[1])

    # パディング計算
    pad_height = max(
        (output_height - 1) * stride[0] + kernel_height - input_data.shape[0], 0
    )
    pad_width = max(
        (output_width - 1) * stride[1] + kernel_width - input_data.shape[1], 0
    )

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return [pad_top, pad_left, pad_bottom, pad_right]


def same_lower(input_data, stride, kernel_size):
    kernel_height, kernel_width = kernel_size
    output_height = math.ceil(input_data.shape[0] / stride[0])
    output_width = math.ceil(input_data.shape[1] / stride[1])

    # パディング計算
    pad_height = max(
        (output_height - 1) * stride[0] + kernel_height - input_data.shape[0], 0
    )
    pad_width = max(
        (output_width - 1) * stride[1] + kernel_width - input_data.shape[1], 0
    )

    pad_bottom = pad_height // 2
    pad_top = pad_height - pad_bottom
    pad_right = pad_width // 2
    pad_left = pad_width - pad_right
    return [pad_top, pad_left, pad_bottom, pad_right]
