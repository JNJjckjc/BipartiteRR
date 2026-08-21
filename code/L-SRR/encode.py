import math

EARTH_RADIUS = 6378137
MIN_LATITUDE = -85.05112878
MAX_LATITUDE = 85.05112878
MIN_LONGITUDE = -180
MAX_LONGITUDE = 180

def clip(n, min_value, max_value):
    return max(min(n, max_value), min_value)

def map_size(level_of_detail):
    return 256 << level_of_detail  # 256 * 2^level_of_detail


def lat_long_to_pixel_xy(latitude, longitude, level_of_detail):
    latitude = clip(latitude, MIN_LATITUDE, MAX_LATITUDE)
    longitude = clip(longitude, MIN_LONGITUDE, MAX_LONGITUDE)

    x = (longitude + 180) / 360
    sin_latitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)

    map_size_value = map_size(level_of_detail)

    # 转换为像素坐标
    pixel_x = int(clip(x * map_size_value + 0.5, 0, map_size_value - 1))
    pixel_y = int(clip(y * map_size_value + 0.5, 0, map_size_value - 1))

    return pixel_x, pixel_y


def pixel_xy_to_tile_xy(pixel_x, pixel_y):
    tile_x = pixel_x // 256
    tile_y = pixel_y // 256
    return tile_x, tile_y

def tile_xy_to_quad_key(tile_x, tile_y, level_of_detail):
    bit_string = []

    # 从最细致的层级开始，逐层转换为二进制编码
    for i in range(level_of_detail, 0, -1):
        mask = 1 << (i - 1)
        bits = 0
        if (tile_x & mask) != 0:
            bits += 1
        if (tile_y & mask) != 0:
            bits += 2

        bit_string.append(f"{bits:02b}")

    full_bit_string = ''.join(bit_string)

    return full_bit_string


def lat_lon_to_bit_string(latitude, longitude, level_of_detail):
    pixel_x, pixel_y = lat_long_to_pixel_xy(latitude, longitude, level_of_detail)

    tile_x, tile_y = pixel_xy_to_tile_xy(pixel_x, pixel_y)

    bit_string = tile_xy_to_quad_key(tile_x, tile_y, level_of_detail)

    return bit_string


def process_coordinates(input_file, output_file, level_of_detail):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            coordinates = line.strip()
            latitude, longitude = map(float, coordinates.split(','))
            bit_string = lat_lon_to_bit_string(latitude, longitude, level_of_detail)
            # 将编码结果写入到输出文件
            outfile.write(bit_string + '\n')


input_file = 'data_gowalla.txt'
output_file = 'data_gowalla_encode.txt'
level_of_detail = 23  # 使用23级详细度

process_coordinates(input_file, output_file, level_of_detail)
print(output_file)
