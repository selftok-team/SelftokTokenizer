# -*- coding: utf-8 -*-

import os
import math
import torch
import numpy as np
from PIL import Image

try:
    from decord import cpu, gpu
    from decord import VideoReader
except:
    print("# Ascend didn't support decord, skipped")


def load_image(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def image_to_tensor(image_file, preprocess):
    image_data = preprocess(load_image(image_file))
    return image_data


def video_to_tensor_decord(video_file, preprocess, max_frames=8, start=None, end=None):
    vr = VideoReader(video_file, ctx=cpu(0))
    fps = vr.get_avg_fps()

    if start is None or start == 0:
        start = 0
    else:
        start = max(math.floor(start * fps), 0)

    if end is None or end == -1:
        end = len(vr)
    else:
        end = min(math.ceil(end * fps), len(vr))
    assert start < end, "vid name: {}, start: {} and end: {}".format(os.path.basename(video_file), start, end)

    idx_list = np.linspace(start, end - 1, max_frames, dtype="int").tolist()
    frames = vr.get_batch(idx_list).asnumpy()
    video_data = [preprocess(Image.fromarray(img, mode="RGB")) for img in frames]
    return torch.tensor(np.stack(video_data))


# 废弃代码，仅供参考
# import cv2
# import numpy as np
# import math
# import os
# import subprocess
# import torch
#
# from decord import VideoReader
# from decord import cpu, gpu
# from PIL import Image
# from multi_modality.data.transforms import set_visual_transforms
#
# cv2.setNumThreads(1)
#
#
# class RawVideoExtractorCV2():
#     def __init__(self, size=224, max_frames=12, centercrop=False, ffmpeg_dir="",
#                  frame_extractor_offline="decord", frame_extractor_online="decord"):
#         self.size = size
#         self.max_frames = max_frames  # 12
#         self.centercrop = centercrop
#         self.frame_extractor_offline = frame_extractor_offline
#         self.frame_extractor_online = frame_extractor_online
#         self.frame_extract_record = set()
#
#         self.ffmpeg_dir = ffmpeg_dir
#         self.ffmpeg = os.path.join(self.ffmpeg_dir, "ffmpeg")
#         self.ffprobe = os.path.join(self.ffmpeg_dir, "ffprobe")
#
#         self.transform_training = set_visual_transforms("train")
#         self.transform_testing = set_visual_transforms("test")
#
#     def get_vid_info(self, video_file, attribute, ffprobe):
#         cmd_str = f"{ffprobe} -v error -select_streams v:0 -show_entries stream={attribute} \
#                     -of default=noprint_wrappers=1:nokey=1 {video_file} -loglevel quiet"
#         out_bytes = subprocess.check_output(cmd_str, stderr=subprocess.STDOUT, shell=True)
#         out_text = out_bytes.decode('utf-8')
#
#         if attribute == "duration":
#             return float(out_text)
#         return int(out_text)
#
#     def video_to_tensor_opencv(self, video_file, start, end, preprocess, max_frames=8):
#         cap = cv2.VideoCapture(video_file)
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         total_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if start is None:
#             start_frame = 0
#         else:
#             start_frame = math.floor(start * fps)
#
#         if end is None:
#             end_frame = total_frame_cnt
#         else:
#             end_frame = math.ceil(end * fps)
#
#         frameCount = end_frame - start_frame
#         interval = math.ceil(frameCount / max_frames)
#         sample_frames = [idx * interval for idx in range(max_frames)]  # 0~11
#
#         ret = True
#         images = []
#
#         prev_frame_idx = None
#         valid_frames = {}
#         invalid_frames_record = set()
#         valid_frames_record = set()
#         for frame_num in sample_frames:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#             ret, frame = cap.read()
#             flag = False
#             temp_num = frame_num + 1
#             if not ret:
#                 while (temp_num < frameCount):
#                     if temp_num in invalid_frames_record:
#                         temp_num += 1
#                         continue
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, temp_num)
#                     ret, frame = cap.read()
#                     if ret:
#                         if temp_num in valid_frames_record:
#                             temp_num += 1
#                             continue
#                         valid_frames[frame_num] = frame
#                         valid_frames_record.add(temp_num)
#                         flag = True
#                         break
#                     else:
#                         invalid_frames_record.add(temp_num)
#                     temp_num += 1
#                 if not flag:
#                     if prev_frame_idx is None:
#                         try:
#                             frame = valid_frames[sample_frames[0]]
#                         except:
#                             print(
#                                 "find video {} start:{} end:{} fps:{} start_frame:{} end_frame:{} all frames is None exit...".format(
#                                     video_file, start, end, fps, start_frame, end_frame))
#                             exit()
#                         prev_frame_idx = sample_frames[1] if sample_frames[1] in valid_frames.keys() else None
#                     else:
#                         frame = valid_frames[prev_frame_idx]
#                         prev_frame_idx = prev_frame_idx + interval
#                         if prev_frame_idx > sample_frames[-1] or prev_frame_idx not in valid_frames.keys():
#                             prev_frame_idx = None
#             else:
#                 valid_frames[frame_num] = frame
#
#             try:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             except:
#                 print("find video {} all frames is None exit...".format(video_file))
#                 exit()
#
#             img = Image.fromarray(frame_rgb)
#             img = img.convert("RGB")
#             img = preprocess(img)  # torch.Size([3, 224, 224])
#             images.append(img)
#
#         cap.release()
#
#         if len(images) > 0:
#             video_data = torch.tensor(np.stack(images))
#         else:
#             video_data = torch.zeros(1)
#
#         return video_data  # [max_frames, 3, 244, 244]
#
#     def video_to_tensor_decord_uniform(self, video_file, start, end, preprocess, max_frames=8):
#         vr = VideoReader(video_file, ctx=cpu(0))  # ctx=gpu(0)
#         fps = vr.get_avg_fps()
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = 100000  # set a big num
#         start_idx = max(math.floor(start * fps), 0)
#         end_idx = min(math.ceil(end * fps), len(vr))
#
#         if start_idx >= end_idx:
#             print(f"vid name: {os.path.basename(video_file)}, start: {start} and end: {end} time out of vid")
#             return
#
#         gap = max((end_idx - start_idx) // max_frames, 1)
#         idx_list = list(range(start_idx, end_idx, gap))
#         if len(idx_list) != max_frames:
#             while len(idx_list) < max_frames:
#                 idx_list *= 2
#             idx_list = idx_list[:max_frames]
#
#         frames = vr.get_batch(idx_list).asnumpy()
#
#         img_list = []
#         for img in frames:
#             img = Image.fromarray(img, mode="RGB")
#             img = preprocess(img)
#             img_list.append(img)
#
#         video_data = torch.tensor(np.stack(img_list))
#         return video_data
#
#     def video_to_tensor_decord_keyframe(self, video_file, start, end, preprocess, max_frames=8):
#         vr = VideoReader(video_file, ctx=cpu(0), width=224, height=224)  # ctx=gpu(0)
#         fps = vr.get_avg_fps()
#         key_indices = vr.get_key_indices()
#         # print(f"idx_list ori: {key_indices}")
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = 100000  # set a big num
#         start_idx = max(math.floor(start * fps), 0)
#         end_idx = min(math.ceil(end * fps), len(vr) - 1)
#
#         if start_idx >= end_idx:
#             print(f"vid name: {os.path.basename(video_file)}, start: {start} and end: {end} time out of vid")
#             return
#
#         while len(key_indices) != 0 and key_indices[0] < start_idx:
#             key_indices.pop(0)
#         while len(key_indices) != 0 and key_indices[-1] > end_idx:
#             key_indices.pop(-1)
#
#         idx_list = key_indices
#         length = len(idx_list)
#         if length == 0:
#             idx_list = [start_idx, end_idx]
#         elif length > max_frames:
#             gap = max(length // max_frames, 1)
#             idx_list = [idx_list[i] for i in range(0, length, gap)]
#
#         if idx_list[0] < start_idx or idx_list[-1] > end_idx:
#             print(f"ERROR! start_idx: {start_idx}, end_idx: {end_idx}, idx_list: {idx_list}")
#
#         if len(idx_list) != max_frames:
#             while len(idx_list) < max_frames:
#                 idx_list *= 2
#             idx_list = idx_list[:max_frames]
#
#         # print(f"idx_list final: {idx_list}")
#         frames = vr.get_batch(idx_list).asnumpy()
#         img_list = []
#         for img in frames:
#             img = Image.fromarray(img, mode="RGB")
#             img = preprocess(img)
#             img_list.append(img)
#
#         video_data = torch.tensor(np.stack(img_list))
#         return video_data
#
#     def video_to_tensor_ffp_sup(self, video_file, start, end, preprocess, max_frames=8):
#         vid_name = os.path.splitext(os.path.basename(video_file))[0]
#         vid_time = self.get_vid_info(video_file, "duration", self.ffprobe)
#         vid_width = self.get_vid_info(video_file, "width", self.ffprobe)
#         vid_height = self.get_vid_info(video_file, "height", self.ffprobe)
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = vid_time
#         end = min(end, vid_time)
#
#         fps = f"{max_frames}/{end - start}"
#         cmd_str = f"{self.ffmpeg} -ss {start} -to {end} -i {video_file} -vf fps={fps} -vsync 0 -vcodec \
#                     rawvideo -pix_fmt rgb24 -f image2pipe -"
#         pipe = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout, _ = pipe.communicate()
#
#         bytes_per_img = vid_width * vid_height * 3
#         if len(stdout) % bytes_per_img != 0:
#             print("Error, get wrong video frames. vid name: {vid_name}, width: {vid_width}, height: {vid_height}")
#         img_num = len(stdout) // bytes_per_img
#         img_list = []
#         idx = 0
#         while idx < max_frames:
#             start_byte = (idx % img_num) * bytes_per_img
#             end_byte = ((idx % img_num) + 1) * bytes_per_img
#             img = Image.frombytes('RGB', (vid_width, vid_height), stdout[start_byte:end_byte])
#             img = preprocess(img)
#             img_list.append(img)
#             idx += 1
#
#         video_data = torch.tensor(np.stack(img_list))
#         return video_data
#
#     def video_to_tensor_ffp_py(self, video_file, start, end, preprocess, max_frames=8):
#         import ffmpeg as ffp
#         probe = ffp.probe(video_file)
#         vid_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
#         vid_width = int(vid_stream['width'])
#         vid_height = int(vid_stream['height'])
#         vid_time = float(vid_stream['duration'])
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = vid_time
#         end = min(end, vid_time)
#         fps = f"{max_frames}/{end - start}"
#
#         out, _ = (
#             ffp
#             .input(video_file, ss=start, to=end, loglevel='quiet')
#             .filter('fps', fps)
#             .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#             .run(capture_stdout=True)
#         )
#         video = (
#             np
#             .frombuffer(out, np.uint8)
#             .reshape([-1, vid_height, vid_width, 3])
#         )
#
#         if video.shape[0] != max_frames:
#             while video.shape[0] < max_frames:
#                 video = np.concatenate((video, video), axis=0)
#             video = video[:max_frames, :, :, :]
#
#         img_list = []
#         for i, img in enumerate(video):
#             img = Image.fromarray(img, mode="RGB")
#             img = preprocess(img)
#             img_list.append(img)
#
#         video_data = torch.tensor(np.stack(img_list))
#         return video_data
#
#     def frame_extract_opencv(self, video_file, start, end, width, height, frame_count=32, frame_saved_path=None):
#         if tuple([video_file, start, end]) in self.frame_extract_record:
#             return 0
#         vid = os.path.splitext(os.path.basename(video_file))[0]
#         os.makedirs(os.path.join(frame_saved_path, vid), exist_ok=True)
#
#         cap = cv2.VideoCapture(video_file)
#         total_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#         frames = []
#         frame_idxs_ori = [i for i in range(total_frame_cnt)]
#         frame_idxs_valid = []
#
#         if start is None:
#             start_frame = 0
#         else:
#             start_frame = math.floor(start * fps)
#
#         if end is None:
#             end_frame = total_frame_cnt
#         else:
#             end_frame = math.ceil(end * fps)
#
#         for frame_idx in range(start_frame, end_frame + 1, 1):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             if ret:
#                 frame_idxs_valid.append(frame_idx)
#
#         cnt = len(frame_idxs_valid)
#
#         if cnt == 0:
#             return
#
#         frame_splits = []
#         interval = float(cnt) / frame_count
#
#         prev = 0
#         for idx in range(frame_count):
#             if idx == 0:
#                 frame_splits.append(0)
#             else:
#                 frame_splits.append(int(prev))
#             prev = prev + interval
#
#         frames = [frame_idxs_valid[idx] for idx in frame_splits]
#
#         os.makedirs(os.path.join(frame_saved_path, vid, "start_{}_end_{}_opencv".format(start, end)), exist_ok=True)
#
#         for index, frame_idx in enumerate(frames):
#             # print("{}--->{}--->{}--->{}".format(os.path.basename(video_file), frame_idx, start_frame, end_frame))
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame_rgb)
#             img.save(os.path.join(frame_saved_path, vid, "start_{}_end_{}_opencv".format(start, end),
#                                   "frame_{}.jpg".format(index)), quality=30)
#
#         self.frame_extract_record.add(tuple([video_file, start, end]))
#         return 0
#
#     def frame_extract_ffmpeg(self, video_file, start, end, width, height, frame_count=32, frame_saved_path=None):
#         vid_name = os.path.splitext(os.path.basename(video_file))[0]
#         img_path = os.path.join(frame_saved_path, vid_name, f"start_{start}_end_{end}")
#         if os.path.exists(img_path):
#             img_num = len(os.listdir(img_path))
#             if img_num == frame_count:
#                 return
#             elif img_num == 0:
#                 print(f"img_path: {img_path}, img num is 0, remove dir!")
#                 os.removedirs(img_path)
#                 return
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = 100000  # set a big num
#         end = min(end, self.get_vid_info(video_file, "duration", self.ffprobe))
#
#         if start >= end:
#             print(f"vid name: {os.path.basename(video_file)}, start: {start} and end: {end} time out of vid")
#             return
#
#         os.makedirs(img_path, exist_ok=True)
#
#         fps = f"{frame_count}/{end - start}"
#         quality = 11  # the larger the number, the lower the image quality
#         cmd_str = f"{self.ffmpeg} -ss {start} -to {end} -i {video_file} -vf fps={fps},scale={width}:{height} \
#                     -vsync 0 -q {quality} {img_path}/frame_%d.jpg -loglevel quiet"
#         subprocess.run(cmd_str, encoding="utf-8", shell=True)
#
#     def frame_extract_all_ffmpeg(self, video_file, start, end, width, height, frame_saved_path=None):
#         vid_name = os.path.splitext(os.path.basename(video_file))[0]
#         img_path = os.path.join(frame_saved_path, vid_name, f"start_{start}_end_{end}")
#         if os.path.exists(img_path):
#             img_num = len(os.listdir(img_path))
#             if img_num == 0:
#                 print(f"img_path: {img_path}, img num is 0, remove dir!")
#                 os.removedirs(img_path)
#                 return
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = 100000  # set a big num
#         end = min(end, self.get_vid_info(video_file, "duration", self.ffprobe))
#
#         if start >= end:
#             print(f"vid name: {os.path.basename(video_file)}, start: {start} and end: {end} time out of vid")
#             return
#
#         os.makedirs(img_path, exist_ok=True)
#
#         quality = 11  # the larger the number, the lower the image quality
#         cmd_str = f"{self.ffmpeg} -ss {start} -to {end} -i {video_file} -vf scale={width}:{height} \
#                     -vsync 0 -q {quality} {img_path}/frame_%d.jpg -loglevel quiet"
#         subprocess.run(cmd_str, encoding="utf-8", shell=True)
#
#     def frame_extract_decord(self, video_file, start, end, width, height, frame_count=32, frame_saved_path=None):
#         vid_name = os.path.splitext(os.path.basename(video_file))[0]
#         img_path = os.path.join(frame_saved_path, vid_name, f"start_{start}_end_{end}")
#         if os.path.exists(img_path):
#             img_num = len(os.listdir(img_path))
#             if img_num == frame_count:
#                 return
#             elif img_num == 0:
#                 print(f"img_path: {img_path}, img num is 0, remove dir!")
#                 os.removedirs(img_path)
#                 return
#
#         vr = VideoReader(video_file, ctx=cpu(0), width=width, height=height)  # ctx=gpu(0)
#         fps = vr.get_avg_fps()
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = 100000  # set a big num
#         start_idx = max(math.floor(start * fps), 0)
#         end_idx = min(math.ceil(end * fps), len(vr))
#
#         if start_idx >= end_idx:
#             print(f"vid name: {os.path.basename(video_file)}, start: {start} and end: {end} time out of vid")
#             return
#
#         os.makedirs(img_path, exist_ok=True)
#
#         gap = max((end_idx - start_idx) // frame_count, 1)
#         idx_list = list(range(start_idx, end_idx, gap))
#         if len(idx_list) != frame_count:
#             while len(idx_list) < frame_count:
#                 idx_list *= 2
#             idx_list = idx_list[:frame_count]
#
#         frames = vr.get_batch(idx_list).asnumpy()
#
#         for i, img in enumerate(frames):
#             img = Image.fromarray(img, mode="RGB")
#             img.save(os.path.join(img_path, "frame_{}.jpg".format(i + 1)), quality=30)
#
#     def frame_extract_all_decord(self, video_file, start, end, width, height, frame_saved_path=None):
#         vid_name = os.path.splitext(os.path.basename(video_file))[0]
#         img_path = os.path.join(frame_saved_path, vid_name, f"start_{start}_end_{end}")
#         if os.path.exists(img_path):
#             img_num = len(os.listdir(img_path))
#             if img_num == 0:
#                 print(f"img_path: {img_path}, img num is 0, remove dir!")
#                 os.removedirs(img_path)
#                 return
#
#         vr = VideoReader(video_file, ctx=cpu(0), width=width, height=height)  # ctx=gpu(0)
#         fps = vr.get_avg_fps()
#
#         if start is None:
#             start = 0
#         if end is None:
#             end = 100000  # set a big num
#         start_idx = max(math.floor(start * fps), 0)
#         end_idx = min(math.ceil(end * fps), len(vr))
#
#         if start_idx >= end_idx:
#             print(f"vid name: {os.path.basename(video_file)}, start: {start} and end: {end} time out of vid")
#             return
#
#         os.makedirs(img_path, exist_ok=True)
#
#         idx_list = list(range(start_idx, end_idx, 1))
#         frames = vr.get_batch(idx_list).asnumpy()
#
#         for i, img in enumerate(frames):
#             img = Image.fromarray(img, mode="RGB")
#             img.save(os.path.join(img_path, "frame_{}.jpg".format(i + 1)), quality=30)
#
#     def get_video_frame(self, video_path, start, end, frame_num, width, height, frame_saved_path=None):
#         if self.frame_extractor_offline == "opencv":
#             try:
#                 self.frame_extract_opencv(video_path, start, end, width, height,
#                                           frame_count=frame_num, frame_saved_path=frame_saved_path)
#             except:
#                 print(f"Error! video_path:{os.path.basename(video_path)}, start:{start}, end:{end}")
#
#         elif self.frame_extractor_offline == "decord":
#             try:
#                 if frame_num == -1:
#                     self.frame_extract_all_decord(video_path, start, end, width, height,
#                                                   frame_saved_path=frame_saved_path)
#                 else:
#                     self.frame_extract_decord(video_path, start, end, width, height,
#                                               frame_count=frame_num, frame_saved_path=frame_saved_path)
#             except:
#                 print(f"Error! video_path:{os.path.basename(video_path)}, start:{start}, end:{end}")
#
#         elif self.frame_extractor_offline == "ffmpeg":
#             try:
#                 if frame_num == -1:
#                     self.frame_extract_all_ffmpeg(video_path, start, end, width, height,
#                                                   frame_saved_path=frame_saved_path)
#                 else:
#                     self.frame_extract_ffmpeg(video_path, start, end, width, height,
#                                               frame_count=frame_num, frame_saved_path=frame_saved_path)
#             except:
#                 print(f"Error! video_path:{os.path.basename(video_path)}, start:{start}, end:{end}")
#
#     def get_video_data(self, video_path, start, end):
#         # tic = time.time()
#         if self.frame_extractor_online == "opencv":
#             vid_tensor = self.video_to_tensor_opencv(video_path, start, end, self.transform_training,
#                                                      max_frames=self.max_frames)
#         elif self.frame_extractor_online == "decord":
#             vid_tensor = self.video_to_tensor_decord_uniform(video_path, start, end, self.transform_training,
#                                                              max_frames=self.max_frames)
#             # vid_tensor = self.video_to_tensor_decord_keyframe(video_path, start, end, self.transform_training,
#             #                                               max_frames=self.max_frames)
#         elif self.frame_extractor_online == "ffmpeg-subprocess":
#             vid_tensor = self.video_to_tensor_ffp_sup(video_path, start, end, self.transform_training,
#                                                       max_frames=self.max_frames)
#         elif self.frame_extractor_online == "ffmpeg-python":
#             vid_tensor = self.video_to_tensor_ffp_py(video_path, start, end, self.transform_training,
#                                                      max_frames=self.max_frames)
#         # toc = time.time()
#         # print("frame_extractor_online:{}, vid name:{}--> time:{:02.4f}".format(self.frame_extractor_online,
#         #     os.path.basename(video_path), toc-tic))
#
#         return vid_tensor
#
#     def get_test_video_data(self, video_path, start, end):
#         vid_tensor = self.video_to_tensor_decord_uniform(video_path, start, end, self.transform_testing,
#                                                          max_frames=self.max_frames)
#         # vid_tensor = self.video_to_tensor_decord_keyframe(video_path, start, end, self.transform_testing, max_frames=self.max_frames)
#         return vid_tensor
#
#     def process_raw_data(self, raw_video_data):
#         tensor_size = raw_video_data.size()
#         tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
#         return tensor
#
#     def process_frame_order(self, raw_video_data, frame_order=0):
#         # 0: ordinary order; 1: reverse order; 2: random order.
#         if frame_order == 0:
#             pass
#         elif frame_order == 1:
#             reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
#             raw_video_data = raw_video_data[reverse_order, ...]
#         elif frame_order == 2:
#             random_order = np.arange(raw_video_data.size(0))
#             np.random.shuffle(random_order)
#             raw_video_data = raw_video_data[random_order, ...]
#
#         return raw_video_data
#
#
# # An ordinary video frame extractor based CV2
# RawVideoExtractor = RawVideoExtractorCV2
