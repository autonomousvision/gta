import os
import sys
import glob
import subprocess
import datetime
import numpy as np

from skimage import io
from skimage.transform import resize as imresize

from multiprocessing import Pool
from pytube import YouTube
from time import sleep

class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = []
        self.list_list_timestamps = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def process(data, seq_id, videoname, output_root):
    seqname = data.list_seqnames[seq_id]
    save_path = output_root+"/"+seqname+"/data.npz"

    if os.path.exists(save_path):
        return False

    if not os.path.exists(output_root + seqname):
        os.makedirs(output_root + seqname)
    else:
        print("[INFO] Something Wrong, stop process")
        return True

    list_str_timestamps = []
    for timestamp in data.list_list_timestamps[seq_id]:
        timestamp = int(timestamp/1000)
        str_hour = str(int(timestamp/3600000)).zfill(2)
        str_min = str(int(int(timestamp%3600000)/60000)).zfill(2)
        str_sec = str(int(int(int(timestamp%3600000)%60000)/1000)).zfill(2)
        str_mill = str(int(int(int(timestamp%3600000)%60000)%1000)).zfill(3)
        _str_timestamp = str_hour+":"+str_min+":"+str_sec+"."+str_mill
        list_str_timestamps.append(_str_timestamp)

    # extract frames from a video
    for idx, str_timestamp in enumerate(list_str_timestamps):
        # command = 'ffmpeg -ss '+str_timestamp+' -i '+videoname+' -vframes 1 -f image2 '+output_root+seqname+'/'+str(data.list_list_timestamps[seq_id][idx])+'.png'

        command = 'ffmpeg -ss '+str_timestamp+' -i '+videoname+' -vframes 1 -q:v 1 -f image2 '+output_root+seqname+'/'+str(data.list_list_timestamps[seq_id][idx])+'.jpg'
        # print("current command is {}".format(command))
        os.system(command)

    png_list = glob.glob(output_root+"/"+seqname+"/*.jpg")
    images = {}

    for pngname in png_list:
        image = io.imread(pngname)
        basefile = pngname.split("/")[-1]
        images[basefile] = image
        os.remove(pngname)

    np.savez(output_root+"/"+seqname+"/data.npz", **images)

    return False

def wrap_process(list_args):
    return process(*list_args)

class DataDownloader:
    def __init__ (self, dataroot, mode='test'):
        print("[INFO] Loading data list ... ",end='')
        self.dataroot = dataroot
        self.list_seqnames = sorted(glob.glob(dataroot + '/*.txt'))
        self.output_root = './acid/' + mode + '/'
        self.mode =  mode

        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
            print(f"Directory {self.output_root} created.")

        self.isDone = False

        self.list_data = []
        if not self.isDone:
            for txt_file in self.list_seqnames:
                dir_name = txt_file.split('/')[-1]
                seq_name = dir_name.split('.')[0]

                # extract info from txt
                seq_file = open(txt_file, "r")
                lines = seq_file.readlines()
                youtube_url = ""
                list_timestamps= []
                for idx, line in enumerate(lines):
                    if idx == 0:
                        youtube_url = line.strip()
                    else:
                        timestamp = int(line.split(' ')[0])
                        list_timestamps.append(timestamp)
                seq_file.close()

                isRegistered = False
                for i in range(len(self.list_data)):
                    if youtube_url == self.list_data[i].url:
                        isRegistered = True
                        self.list_data[i].add(seq_name, list_timestamps)
                    else:
                        pass

                if not isRegistered:
                    self.list_data.append(Data(youtube_url, seq_name, list_timestamps))

            # self.list_data.reverse()
            print(" Done! ")
            print("[INFO] {} movies are used in {} mode".format(len(self.list_data), self.mode))


    def Run(self):
        print("[INFO] Start downloading {} movies".format(len(self.list_data)))

        for global_count, data in enumerate(self.list_data):
            print("[INFO] Downloading {} ".format(data.url))
            try:
                # sometimes this fails because of known issues of pytube and unknown factors
                yt = YouTube(data.url)
                stream = yt.streams.filter(res="360p").first()
                stream.download('./','current_'+mode)
            except Exception as e:
                print(e)
                failure_log = open('failed_videos_'+mode+'.txt', 'a')
                for seqname in data.list_seqnames:
                    failure_log.writelines(seqname + '\n')
                failure_log.close()
                continue

            videoname_candinate_list = glob.glob('./*')
            for videoname_candinate in videoname_candinate_list:
                if videoname_candinate.split('.')[-1] == '/current_'+mode:
                    videoname = videoname_candinate

            if len(data) == 1: # len(data) is len(data.list_seqnames)
                process(data, 0, videoname, self.output_root)
            else:
                with Pool(processes=4) as pool:
                    pool.map(wrap_process, [(data, seq_id, videoname, self.output_root) for seq_id in range(len(data))])

            # remove videos
            command = "rm " + videoname
            os.system(command)

            if self.isDone:
                return False

        return True

    def Show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data:
            print(" URL : {}".format(data.url))
            for idx in range(len(data)):
                print(" SEQ_{} : {}".format(idx, data.list_seqnames[idx]))
                print(" LEN_{} : {}".format(idx, len(data.list_list_timestamps[idx])))
                global_count = global_count + 1
            print("----------------------------------------")

        print("TOTAL : {} sequnces".format(global_count))

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: this.py [test or train]")
        quit()

    if sys.argv[1] == "test":
        mode = "test"
    elif sys.argv[1] == "train":
        mode = "train"
    elif sys.argv[1] == "validation":
        mode = "validation"
    else:
        print("invalid mode")
        quit()

    dataroot = "./ACID/" + mode
    downloader = DataDownloader(dataroot, mode)

    downloader.Show()
    isOK = downloader.Run()

    if isOK:
        print("Done!")
    else:
        print("Failed")


