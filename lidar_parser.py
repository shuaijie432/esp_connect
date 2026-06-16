# """激光雷达数据点解析模块"""

# import struct
# import math


# class LidarPoint:
#     """激光雷达数据点（A1 格式，5字节）"""

#     def __init__(self):
#         self.quality = 0
#         self.start_bit = False
#         self.angle = 0.0
#         self.distance = 0.0

#     @staticmethod
#     def parse(buf: bytes, offset: int):
#         """从 byte[] 解析一个点（5字节）"""
#         if buf is None or len(buf) - offset < 5:
#             return None

#         b0 = buf[offset] & 0xFF
#         b1 = buf[offset + 1] & 0xFF
#         b2 = buf[offset + 2] & 0xFF
#         b3 = buf[offset + 3] & 0xFF
#         b4 = buf[offset + 4] & 0xFF

#         pt = LidarPoint()
#         pt.start_bit = (b0 & 0x01) != 0
#         s_bar = (b0 >> 1) & 0x01
#         pt.quality = (b0 >> 2) & 0x3F

#         # A1 校验：start_bit 和 s_bar 必须相反
#         if s_bar != (0 if pt.start_bit else 1):
#             return None

#         angle_q6 = (b2 << 7) | ((b1 >> 1) & 0x7F)
#         pt.angle = angle_q6 / 64.0

#         distance_q2 = b3 | (b4 << 8)
#         pt.distance = distance_q2 / 4.0

#         if pt.distance > 5000:
#             return None

#         return pt

#     def __str__(self):
#         return f"雷达{{{self.angle:.2f}°, {self.distance:.1f}mm, Q{self.quality}}}"


# class OdomData:
#     """里程计数据结构"""

#     def __init__(self):
#         self.x = 0.0
#         self.y = 0.0
#         self.theta = 0.0
#         self.vx = 0.0
#         self.vy = 0.0
#         self.wz = 0.0
#         self.valid = False

#     def __str__(self):

#         return (f"里程计{{x={self.x:.1f}, y={self.y:.1f}, "
#                 f"θ={math.degrees(self.theta):.2f}°, "
#                 f"v=({self.vx:.1f}, {self.vy:.1f}) mm/s, "
#                 f"w={self.wz:.4f} rad/s}}")


# def calc_checksum(data: bytes) -> int:
#     """累加校验和"""
#     return sum(b & 0xFF for b in data) & 0xFF


# def parse_frame(payload: bytes, min_quality: int = 0, max_distance: float = 12000):
#     """
#     解析 ESP32 发来的完整帧（新协议）

#     格式: [2字节总长度(小端)][雷达数据][里程计帧(27B)]
#     """
#     points = []
#     odom = OdomData()

#     # ========== 新协议：先检查最小长度 ==========
#     if len(payload) < 2 + 27:
#         return points, odom

#     # ========== 新协议：读取总长度 ==========
#     declared_len = payload[0] | (payload[1] << 8)

#     if declared_len != len(payload):
#         # 长度不匹配，但仍尝试解析
#         pass

#     # ========== 计算各段长度 ==========
#     radar_len = len(payload) - 2 - 27
#     if radar_len < 0:
#         return points, odom

#     radar_payload = payload[2:2 + radar_len]
#     odom_data = payload[2 + radar_len:2 + radar_len + 27]

#     # ========== 解析里程计 ==========
#     if len(odom_data) == 27:
#         if odom_data[0] == 0xAA and odom_data[1] == 0x55:
#             calc_sum = calc_checksum(odom_data[:26])
#             if calc_sum == odom_data[26]:
#                 x_bytes = odom_data[2:6]
#                 y_bytes = odom_data[6:10]
#                 theta_bytes = odom_data[10:14]
#                 vx_bytes = odom_data[14:18]
#                 vy_bytes = odom_data[18:22]
#                 wz_bytes = odom_data[22:26]

#                 # ESP32 发来的是米(m)，转换为毫米(mm)
#                 odom.x = struct.unpack('<f', x_bytes)[0] * 1000.0
#                 odom.y = struct.unpack('<f', y_bytes)[0] * 1000.0
#                 odom.theta = struct.unpack('<f', theta_bytes)[0]
#                 odom.vx = struct.unpack('<f', vx_bytes)[0] * 1000.0
#                 odom.vy = struct.unpack('<f', vy_bytes)[0] * 1000.0
#                 odom.wz = struct.unpack('<f', wz_bytes)[0]
#                 odom.valid = True

#     # ========== 解析雷达数据 ==========
#     i = 0
#     while i + 4 < len(radar_payload):
#         pt = LidarPoint.parse(radar_payload, i)
#         if pt is not None:
#             if 0 < pt.distance < max_distance and pt.quality >= min_quality:
#                 points.append(pt)
#             i += 5
#         else:
#             i += 1

#     return points, odom


# def bytes_to_hex(data: bytes) -> str:
#     return " ".join(f"{b:02X}" for b in data)


















"""激光雷达数据点解析模块 - 优化版"""

import struct
import math


"""激光雷达数据点解析模块 - 优化版"""

import struct
import math


class LidarPoint:
    """激光雷达数据点（A1 格式，5字节）"""

    __slots__ = ['quality', 'start_bit', 'angle', 'distance']

    def __init__(self):
        self.quality = 0
        self.start_bit = False
        self.angle = 0.0
        self.distance = 0.0

    @staticmethod
    def parse(buf: bytes, offset: int):
        """从 byte[] 解析一个点（5字节）"""
        if buf is None or len(buf) - offset < 5:
            return None

        b0 = buf[offset]
        b1 = buf[offset + 1]
        b2 = buf[offset + 2]
        b3 = buf[offset + 3]
        b4 = buf[offset + 4]

        # 快速校验：start_bit 和 s_bar 必须相反
        start_bit = (b0 & 0x01) != 0
        s_bar = (b0 >> 1) & 0x01
        if s_bar == start_bit:
            return None

        pt = LidarPoint()
        pt.start_bit = start_bit
        pt.quality = (b0 >> 2) & 0x3F

        angle_q6 = (b2 << 7) | ((b1 >> 1) & 0x7F)
        pt.angle = angle_q6 / 64.0

        distance_q2 = b3 | (b4 << 8)
        pt.distance = distance_q2 / 4.0

        # 过滤无效距离
        if pt.distance == 0 or pt.distance > 12000:
            return None

        return pt


class OdomData:
    """里程计数据结构"""

    __slots__ = ['x', 'y', 'theta', 'vx', 'vy', 'wz', 'valid']

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.valid = False


def calc_checksum(data: bytes) -> int:
    """累加校验和"""
    return sum(data) & 0xFF


def parse_frame(payload: bytes, min_quality: int = 0, max_distance: float = 12000):
    """
    解析 ESP32 发来的完整帧（优化版）
    格式: [2字节总长度(小端)][雷达数据][里程计帧(27B)]
    """
    points = []
    odom = OdomData()

    # 最小长度：2字节头 + 27字节里程计
    if len(payload) < 29:
        return points, odom

    # 读取声明长度
    declared_len = payload[0] | (payload[1] << 8)
    actual_len = len(payload)

    # 健壮性：如果声明长度不合理，使用实际长度
    if declared_len < 29 or declared_len > actual_len:
        usable_len = actual_len
    else:
        usable_len = declared_len

    # 里程计固定在最后 27 字节
    odom_offset = usable_len - 27
    radar_payload = payload[2:odom_offset]
    odom_data = payload[odom_offset:usable_len]

    # ========== 解析里程计 ==========
    if len(odom_data) == 27:
        if odom_data[0] == 0xAA and odom_data[1] == 0x55:
            calc_sum = calc_checksum(odom_data[:26])
            if calc_sum == odom_data[26]:
                # ESP32 发来的是米(m)，转换为毫米(mm)
                odom.x = struct.unpack_from('<f', odom_data, 2)[0] * 1000.0
                odom.y = struct.unpack_from('<f', odom_data, 6)[0] * 1000.0
                odom.theta = struct.unpack_from('<f', odom_data, 10)[0]
                odom.vx = struct.unpack_from('<f', odom_data, 14)[0] * 1000.0
                odom.vy = struct.unpack_from('<f', odom_data, 18)[0] * 1000.0
                odom.wz = struct.unpack_from('<f', odom_data, 22)[0]
                odom.valid = True

    # ========== 解析雷达数据（批量处理） ==========
    if len(radar_payload) < 5:
        return points, odom

    buf = radar_payload
    n = len(buf)
    i = 0
    append = points.append

    while i + 4 < n:
        b0 = buf[i]
        start_bit = b0 & 0x01
        s_bar = (b0 >> 1) & 0x01

        if s_bar != (0 if start_bit else 1):
            i += 1
            continue

        # 快速距离检查（提前过滤）
        distance_q2 = buf[i + 3] | (buf[i + 4] << 8)
        distance = distance_q2 / 4.0
        if distance == 0 or distance > max_distance:
            i += 5
            continue

        quality = (b0 >> 2) & 0x3F
        if quality < min_quality:
            i += 5
            continue

        angle_q6 = (buf[i + 2] << 7) | ((buf[i + 1] >> 1) & 0x7F)
        angle = angle_q6 / 64.0

        pt = LidarPoint()
        pt.start_bit = start_bit
        pt.quality = quality
        pt.angle = angle
        pt.distance = distance
        append(pt)
        i += 5

    return points, odom


def bytes_to_hex(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)