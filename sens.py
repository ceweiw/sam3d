import os
import struct
import numpy as np
from PIL import Image
import re  # 用于正则表达式匹配

class CalibrationData:
    def __init__(self):
        self.m_intrinsic = np.zeros(16)
        self.m_extrinsic = np.zeros(16)

    def load_from_file(self, file):
        def extract_matrix_from_line(line):
            # 匹配类似 m_calibrationColorIntrinsic = ... 的矩阵数据行
            matrix_values = list(map(float, line.split("=")[1].strip().split()))
            return np.array(matrix_values).reshape(4, 4)  # 将数据转为4x4矩阵

        # 解析并加载颜色和深度的标定矩阵
        for line in file:
            line = line.strip()
            if "m_calibrationColorIntrinsic" in line:
                self.m_intrinsic = extract_matrix_from_line(line)
            elif "m_calibrationColorExtrinsic" in line:
                self.m_extrinsic = extract_matrix_from_line(line)
            elif "m_calibrationDepthIntrinsic" in line:
                self.m_intrinsic = extract_matrix_from_line(line)
            elif "m_calibrationDepthExtrinsic" in line:
                self.m_extrinsic = extract_matrix_from_line(line)

    def save_to_file(self, out_file):
        out_file.write(struct.pack('16f', *self.m_intrinsic.flatten()))
        out_file.write(struct.pack('16f', *self.m_extrinsic.flatten()))


class SensorData:
    def __init__(self):
        self.m_version_number = 4
        self.m_sensor_name = "StructureSensor"
        self.m_color_width = 1600
        self.m_color_height = 900
        self.m_depth_width = 1600
        self.m_depth_height = 900
        self.m_depth_shift = 1000
        self.m_calibration_color = CalibrationData()
        self.m_calibration_depth = CalibrationData()
        self.m_frames = []
        self.m_frames_size=0
        self.m_color_compression_type = 'jpg'  # or 'JPEG'
        self.m_depth_compression_type = 'PNG'  # Compression type

    def load_from_file(self, source_folder):
        # 读取info.txt文件
        with open(os.path.join(source_folder, "info.txt"), 'r', encoding='utf-8') as in_meta:
            # 正则表达式匹配每一行的键值对
            for line in in_meta:
                line = line.strip()
                if "m_versionNumber" in line:
                    self.m_version_number = int(line.split("=")[1].strip())
                elif "m_sensorName" in line:
                    self.m_sensor_name = line.split("=")[1].strip()
                elif "m_colorWidth" in line:
                    self.m_color_width = int(line.split("=")[1].strip())
                    print("self.m_color_width",self.m_color_width)
                elif "m_colorHeight" in line:
                    self.m_color_height = int(line.split("=")[1].strip())
                elif "m_depthWidth" in line:
                    self.m_depth_width = int(line.split("=")[1].strip())
                elif "m_depthHeight" in line:
                    self.m_depth_height = int(line.split("=")[1].strip())
                elif "m_depthShift" in line:
                    self.m_depth_shift = float(line.split("=")[1].strip())
                elif "m_calibrationColorIntrinsic" in line:
                    self.m_calibration_color.load_from_file(in_meta)
                elif "m_calibrationDepthIntrinsic" in line:
                    self.m_calibration_depth.load_from_file(in_meta)

            # 读取帧数
            frame_line = re.search(r"m_frames\.size\s*=\s*(\d+)", line)
            if frame_line:
                self.m_frames_size = int(line.split('=')[1].strip())

    def load_from_images(self, source_folder, basename="frame-", color_ending="png"):
        # Now load images and poses
        for i in range(self.m_frames_size):  # Use frames size from info.txt
            color_file = os.path.join(source_folder, f"{basename}{i:06d}.color.{color_ending}")
            depth_file = os.path.join(source_folder, f"{basename}{i:06d}.depth.png")
            pose_file = os.path.join(source_folder, f"{basename}{i:06d}.pose.txt")

            if not os.path.exists(color_file) or not os.path.exists(depth_file) or not os.path.exists(pose_file):
                print("DONE")
                break

            # Read color image
            color_image = Image.open(color_file)
            color_data = np.array(color_image)

            # Read depth image
            depth_image = Image.open(depth_file)
            depth_data = np.array(depth_image)

            # Read pose file
            pose = np.loadtxt(pose_file)

            # Save the frame
            frame = RGBDFrame(color_data, depth_data, pose)
            self.m_frames.append(frame)

    def save_to_file(self, filename):
        with open(filename, 'wb') as out_file:
            self.write_header_to_file(out_file)
            self.write_rgb_frames_to_file(out_file)

    def write_header_to_file(self, out_file):
        out_file.write(struct.pack('I', self.m_version_number))
        sensor_name_len = len(self.m_sensor_name)
        out_file.write(struct.pack('Q', sensor_name_len))
        out_file.write(self.m_sensor_name.encode('utf-8'))

        self.m_calibration_color.save_to_file(out_file)
        self.m_calibration_depth.save_to_file(out_file)

        out_file.write(self.m_color_compression_type.encode('utf-8'))
        out_file.write(self.m_depth_compression_type.encode('utf-8'))
        out_file.write(struct.pack('I', self.m_color_width))
        out_file.write(struct.pack('I', self.m_color_height))
        out_file.write(struct.pack('I', self.m_depth_width))
        out_file.write(struct.pack('I', self.m_depth_height))
        out_file.write(struct.pack('f', self.m_depth_shift))

    def write_rgb_frames_to_file(self, out_file):
        out_file.write(struct.pack('Q', len(self.m_frames)))
        for frame in self.m_frames:
            frame.save_to_file(out_file)


class RGBDFrame:
    def __init__(self, color_data, depth_data, pose):
        self.color_data = color_data
        self.depth_data = depth_data
        self.pose = pose

    def save_to_file(self, out_file):
        # Save the color image data
        color_data_bytes = self.color_data.tobytes()
        out_file.write(struct.pack('Q', len(color_data_bytes)))
        out_file.write(color_data_bytes)

        # Save the depth image data
        depth_data_bytes = self.depth_data.tobytes()
        out_file.write(struct.pack('Q', len(depth_data_bytes)))
        out_file.write(depth_data_bytes)

        # Save pose data
        out_file.write(self.pose.tobytes())


if __name__ == "__main__":
    sd = SensorData()
    sd.load_from_file("E:/desktop/mydata")
    sd.load_from_images("E:/desktop/mydata", "frame-", "jpg")
    sd.save_to_file("E:/desktop/mydata/test.sens")
    print("Generated .sens data")
