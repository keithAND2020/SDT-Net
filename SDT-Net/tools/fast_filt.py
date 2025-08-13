import astropy.io.fits as fits
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as con
from matplotlib import pyplot as plt 
import os.path as osp
import pdb
import cv2
import io
from tqdm import tqdm
from petrel_client.client import Client
import json 

def load_alfa(file_name):
    with fits.open(file_name) as hdu_cata: data = hdu_cata[1].data
    coord = SkyCoord(data['RAJ2000'],data['DEJ2000'],unit=(u.hourangle,u.degree))
    alfa_cata  = {'AGC':data['AGC'],'RA':coord.ra.deg,'DEC':coord.dec.deg,'Vhel':data['Vhel']}
    alfa_coords = SkyCoord(alfa_cata['RA']*u.degree,alfa_cata['DEC']*u.degree, frame='icrs')
    return alfa_cata, alfa_coords

def load_cube_ceph(client, file_name):
    ceph_buf = client.get(file_name, enable_cache=True)
    hdu_cube=fits.open(io.BytesIO(ceph_buf))
    cube_data = hdu_cube[0].data
    np.nan_to_num(cube_data,0) 
    cube_header = hdu_cube[0].header
    return cube_header, cube_data

alfa_cata, alfa_coords = load_alfa('./alfalfa.fit')
ceph_base_dir = 's3://Dataset/astronomy/fast/'
client = Client('~/petreloss_new.conf', enable_mc=True)

keep_datas = {}
all_num, useful_num = 0, 0

for fast_1lv_dir in client.list('s3://Dataset/astronomy/fast/'):
    for fast_2lv_dir in tqdm(client.list(osp.join(ceph_base_dir, fast_1lv_dir))):
        cube_header, cube_data = load_cube_ceph(client, osp.join(ceph_base_dir, fast_1lv_dir,fast_2lv_dir))
        Wcs = WCS(cube_header,naxis=2)
        ra_pixel, dec_pixel = Wcs.world_to_pixel(alfa_coords)
        freq_pixel = (1420.4/(alfa_cata['Vhel']*1000/con.c.value+1)-cube_header['CRVAL3'])/cube_header['CDELT3']

        keep_index = (ra_pixel>=0) * (ra_pixel<=cube_header['NAXIS1']) *\
                     (dec_pixel>=0) * (dec_pixel<=cube_header['NAXIS2']) *\
                     (freq_pixel>=0) * (freq_pixel<=cube_header['NAXIS3'])
        if keep_index.max():
            if fast_1lv_dir not in keep_datas.keys():
                keep_datas[fast_1lv_dir] = [fast_2lv_dir]
            else:
                keep_datas[fast_1lv_dir].append(fast_2lv_dir)

            useful_num += 1
        all_num += 1
print([useful_num, all_num])
with open("fast_filted.json", "w") as outfile:
    json.dump(keep_datas, outfile)
exit()