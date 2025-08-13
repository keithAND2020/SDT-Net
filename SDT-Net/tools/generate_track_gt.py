import json
import os
import glob
from tqdm import tqdm

def main():
    
    data_root = '/ailab/user/zhuangguohang/data/test_v4_ztf'
    dir_list = os.listdir(data_root)
    for dir in tqdm(dir_list, desc='generating tracks .txt file'):
        tracks = []
        with open(os.path.join(data_root, dir, 'output.json'), 'r') as f:
            data = json.load(f)
        # frame_name = [os.path.splitext(os.path.basename(i))[0] for i in glob.glob(os.path.join(data_root, dir, '*.fit'))]
        

        for item in data:
            for key,value in item.items():
                frame_number = int(key.split('_')[0][-2:]) + 1
                id = int(key.split('_')[1]) + 1
                st, ed = value ## (st_y, st_x) (ed_y, ed_x)
                bb_left = int(min(st[1], ed[1]))
                bb_top = int(max(st[0], ed[0]))
                bb_width = int(abs(st[1]- ed[1]))
                bb_height = int(abs(st[0]-ed[0]))
                tracks.append([frame_number, id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1])

        with open(os.path.join(data_root, dir, 'tracks.txt'), 'w') as f:
            for line in tracks:
                f.write(','.join(map(str, line)) + "\n")


                  

    



if __name__ == '__main__':

    main()