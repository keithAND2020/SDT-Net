import pdb
import argparse

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from libs.builders import build_models
from libs.dataset import Fast_demo
from libs.utils import init_all,coord_data2world
from torch.utils.data import DataLoader
import torch
from astropy.io.fits.header import Header as astropy_header
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import cv2
from selenium import webdriver

#parser
parser = argparse.ArgumentParser(description='3D dark matter reconstruction')
parser.add_argument('config', type=str)
parser.add_argument('-q', '--query_web', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('-v', '--visualize', action='store_true')
parser.add_argument('--log_dir', type=str)
parser.add_argument('--resume', type=str, default=None)

parser.add_argument('--launcher', choices=['none', 'slurm', 'pytorch'], default='none')
parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
args = parser.parse_args()

if args.query_web:
    from selenium import webdriver
    from selenium.webdriver.edge.service import Service
    from selenium.webdriver import Edge,Chrome,ChromeOptions
    from selenium.webdriver.chrome.options import Options

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver=Chrome(executable_path='/mnt/workspace/luyan/Fast_detection/tools/chromedriver',options=chrome_options)
    url='http://simbad.cds.unistra.fr/simbad/sim-fcoo'
    driver.get(url)
    driver.maximize_window()
    def query_web(driver, query_info):
        #query_info = '14 30 37.42174998160 +44 22 58.0036271396'
        query_bar=driver.find_element_by_name('Coord')
        query_bar.clear()
        query_bar.send_keys(query_info)

        imgs = []
        vis_shape = None  # also equal to has a acceptable vis or not
        for i in range(1,4):
            try:
                radius_bar = driver.find_element_by_name('Radius')
                radius_bar.clear()
                radius_bar.send_keys(i)
                submit_button=driver.find_element_by_name('submit')
                submit_button.click()
                full_screen_button=driver.find_element_by_xpath('//*[@title="Full screen"]')
                full_screen_button.click()

                img = driver.find_elements_by_class_name("aladin-imageCanvas")[0]
                data = img.screenshot_as_png
                img = Image.open(BytesIO(data))
                img_array = np.asarray(img)
                img_array = img_array[:,:,[2,1,0]]
                vis_shape = img_array.shape
                imgs.append(img_array)

                full_screen_button=driver.find_element_by_xpath('//*[@title="Restore original size"]')
                full_screen_button.click()
            except:
                imgs.append(None)
        if vis_shape is not None:
            new_imgs = []
            for img in imgs:
                if img is None: img = np.ones(vis_shape)*255
                new_imgs.append(img)
            img_array = np.concatenate(new_imgs,1)
        else:
            img_array = None

        try: #find basic infomation
            basic_info=driver.find_element_by_id('basic_data')
            _, target_kind = basic_info.text.split('\n')[0].split(' -- ')
        except:
            try:
                datatable = driver.find_elements_by_class_name("datatable")
                dist_asec = driver.find_elements_by_class_name("computed")[1].text
                target_kind = 'Otype-'+datatable[1].text.split(dist_asec)[1].split(' ')[1]
            except:
                target_kind = 'Not known'
        
        return img_array,target_kind


configs, logger = init_all(args)

# build model
net = build_models(**configs['model']).to('cuda')
net.eval()
pdb.set_trace()
all_files = os.listdir('/mnt/workspace/luyan/Datasets/astronomy/bootes/')
all_files = [_ for _ in all_files if 'Dec_3' not in _]

def int_round(x):
    x = int(np.array(float(x)).round())
    return x

save_shape = None

for current_file in all_files:
    print(current_file)
    demo_vis_dir = 'demo_vis'
    demo_tmp_dir = './data'
    demo_data_dir = '/mnt/workspace/luyan/Datasets/astronomy/bootes/'+current_file
    demo_tmp_dir += demo_data_dir[len(os.path.dirname(os.path.dirname(demo_data_dir))):]
    save_csv = os.path.basename(demo_data_dir)+'.csv'
    # demo dataset
    dataset = Fast_demo(demo_data_dir,demo_tmp_dir)
    dataloader = DataLoader(dataset=dataset,
                   batch_size=1,
                   num_workers=4,
                   shuffle=False,
                   pin_memory=True)

    csv_header = ['img_folder', 'score', 'ra', 'dec', 'vel']
    csv_data = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            pz_input_cube = data['input']
            _, _, freq_size, dec_size, ra_size = pz_input_cube.size()
            out = net(pz_input_cube.cuda())

            # get coord
            keepindex = out['scores']>0.2
            pred_score = out['scores'][keepindex].cpu()
            pred_loc = out['locations'][keepindex].cpu()


            if int_round(keepindex.sum())>0:
                cube_header = astropy_header.fromstring(data['cube_header'][0])
                pz_input_cube = pz_input_cube.numpy()[0]
                input_cube = pz_input_cube.mean(0)
                ra_dec_pred,vel_pred = coord_data2world(pred_loc, cube_header, input_cube.shape, data['cut_info'][0]) 
                # vis
                freq_arcmin_per_pixel = (cube_header['NAXIS3']-1)/(input_cube.shape[0]-1) * cube_header['CDELT3']
                freq_range_pixel = int_round(3/freq_arcmin_per_pixel)
                for ra_dec, vel, score in zip(ra_dec_pred, vel_pred, pred_score):
                    icrs_coord = SkyCoord(ra=ra_dec.ra.to_value()*u.deg, dec=ra_dec.dec.to_value()*u.deg, radial_velocity=vel.item()*u.km/u.s, frame='icrs')
                    #pred_sky_coord = '%d %d %f +%d %d %f'%(icrs_coord.ra.hms.h,icrs_coord.ra.hms.m,icrs_coord.ra.hms.s,icrs_coord.dec.dms.d,icrs_coord.dec.dms.m,icrs_coord.dec.dms.s)
                    csv_data.append([os.path.join(demo_vis_dir,data['filename'][0]), 
                                     score.item(), 
                                     '%d %d %f'%(icrs_coord.ra.hms.h,icrs_coord.ra.hms.m,icrs_coord.ra.hms.s),
                                     '+%d %d %f'%(icrs_coord.dec.dms.d,icrs_coord.dec.dms.m,icrs_coord.dec.dms.s),
                                     vel.item()])

                os.makedirs(os.path.join(demo_vis_dir,data['filename'][0]),exist_ok=True)
                for score, (freq, dec, ra), ra_dec_real, vel_real in zip(pred_score,pred_loc,ra_dec_pred,vel_pred):
                    icrs_coord = SkyCoord(ra=ra_dec_real.ra.to_value()*u.deg, dec=ra_dec_real.dec.to_value()*u.deg, radial_velocity=vel_real.item()*u.km/u.s, frame='icrs')
                    img_dir = '%f_%d %d %f_+%d %d %f_%f.jpg'%(score.item(),icrs_coord.ra.hms.h,icrs_coord.ra.hms.m,icrs_coord.ra.hms.s,icrs_coord.dec.dms.d,icrs_coord.dec.dms.m,icrs_coord.dec.dms.s,vel_real.item())
                    # accumulate the useless dimension
                    ori_spatial_vis = input_cube[max(freq-13,0):min(freq+13,freq_size-1)].sum(0)
                    ori_decfreq_pz_vis = pz_input_cube[:,:,:,max(ra-1,0):min(ra+1,ra_size-1)].sum(3)
                    ori_rafreq_vis = input_cube[:,max(dec-1,0):min(dec+1,dec_size-1):].sum(1)
                    # padding for cropping
                    spatial_vis_padding = np.zeros([ori_spatial_vis.shape[0]+20, ori_spatial_vis.shape[1]+20])
                    spatial_vis_padding[10:-10,10:-10] = ori_spatial_vis
                    decfreq_pz_vis_padding = np.zeros([2,ori_decfreq_pz_vis.shape[1]+2*freq_range_pixel, ori_decfreq_pz_vis.shape[2]+20])
                    decfreq_pz_vis_padding[:,freq_range_pixel:-freq_range_pixel,10:-10] = ori_decfreq_pz_vis
                    rafreq_vis_padding = np.zeros([ori_rafreq_vis.shape[0]+2*freq_range_pixel, ori_rafreq_vis.shape[1]+20])
                    rafreq_vis_padding[freq_range_pixel:-freq_range_pixel,10:-10] = ori_rafreq_vis
                    # zoom up cropping
                    spatial_vis = spatial_vis_padding[dec:dec+20,ra:ra+20]
                    decfreq_pz_vis = decfreq_pz_vis_padding[:,freq:freq+2*freq_range_pixel,dec:dec+20]
                    rafreq_vis = rafreq_vis_padding[freq:freq+2*freq_range_pixel,ra:ra+20]
                    
                    ori_decfreq_vis = ori_decfreq_pz_vis.mean(0)
                    decfreq_vis = decfreq_pz_vis.mean(0)

                    plt.figure(figsize=(10,6))
                    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

                    # up left (ra dec)
                    ori_spatial_vis = np.clip(ori_spatial_vis,a_min=-20,a_max=30)
                    ax1.imshow(ori_spatial_vis,cmap='rainbow',vmin=ori_spatial_vis.min(),vmax=ori_spatial_vis.max()+1e-12)
                    ax1.scatter(ra,dec,color='black',s=10,marker='x',linewidths=1.5)
                    ax1.set_title('ra dec')
                    #ax1.get_xaxis().set_visible(False)
                    #ax1.get_yaxis().set_visible(False)
                    ax1.set(xticks=np.arange(0, 256, 51), xticklabels=np.linspace(1, cube_header['NAXIS1'], 6).round().astype(int),
                            yticks=[0,31], yticklabels=[1, cube_header['NAXIS2']])
                    

                    # up right (zoom up ra dec)
                    spatial_vis = np.clip(spatial_vis,a_min=-20,a_max=30)
                    spatial_vis_img = ax2.imshow(spatial_vis,cmap='rainbow',vmin=ori_spatial_vis.min(),vmax=ori_spatial_vis.max()+1e-12)
                    ax2.scatter(10,10,color='black',s=10,marker='x',linewidths=1)
                    ax2.set(xticks=[0,9.5,19], xticklabels=[int_round((ra-9.5)*(cube_header['NAXIS1']/256)),int_round(ra*(cube_header['NAXIS1']/256)),int_round((ra+9.5)*(cube_header['NAXIS1']/256))],
                            yticks=[0,9.5,19], yticklabels=[int_round((dec-9.5)*(cube_header['NAXIS2']/32)),int_round(dec*(cube_header['NAXIS2']/32)),int_round((dec+9.5)*(cube_header['NAXIS2']/32))])                   
                    ax2.set_title('zoom up ra dec')
                    divider2 = make_axes_locatable(ax2)
                    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(spatial_vis_img, cax=cax2, orientation='vertical')

                    # mid left (freq ra)
                    ori_rafreq_vis = np.clip(ori_rafreq_vis,a_min=-20,a_max=30)
                    ax3.imshow(ori_rafreq_vis.T,cmap='rainbow',vmin=ori_rafreq_vis.min(),vmax=ori_rafreq_vis.max()+1e-12)
                    ax3.scatter(freq,ra,color='black',s=10,marker='x',linewidths=1.5)
                    ax3.set_title('freq ra')
                    ax3.set(xticks=np.arange(0, 512, 102), xticklabels=np.linspace(cube_header['CRVAL3'], cube_header['CRVAL3']+cube_header['NAXIS3']*cube_header['CDELT3'], 6).round().astype(int),
                            yticks=np.arange(0, 255, 127), yticklabels=np.linspace(1, cube_header['NAXIS1'], 3).round().astype(int))

                    # mid right (zoom up freq ra)
                    rafreq_vis = np.clip(rafreq_vis,a_min=-20,a_max=30)
                    rafreq_vis_img = ax4.imshow(rafreq_vis.T,cmap='rainbow',vmin=ori_rafreq_vis.min(),vmax=ori_rafreq_vis.max()+1e-12)
                    ax4.scatter(freq_range_pixel,10,color='black',s=10,marker='x',linewidths=1)
                    ax4.set(xticks=[0,freq_range_pixel,2*freq_range_pixel-1], xticklabels=[int_round(cube_header['CRVAL3']+(freq-freq_range_pixel)*freq_arcmin_per_pixel),int_round(cube_header['CRVAL3']+freq*freq_arcmin_per_pixel),int_round(cube_header['CRVAL3']+(freq+freq_range_pixel)*freq_arcmin_per_pixel)],
                            yticks=[0,9.5,19], yticklabels=[int_round((ra-9.5)*(cube_header['NAXIS1']/256)),int_round(ra*(cube_header['NAXIS1']/256)),int_round((ra+9.5)*(cube_header['NAXIS1']/256))])
                    ax4.set_title('zoom up freq ra')
                    divider4 = make_axes_locatable(ax4)
                    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(rafreq_vis_img, cax=cax4, orientation='vertical')

                    # down left (freq dec)
                    ori_decfreq_vis = np.clip(ori_decfreq_vis,a_min=-20,a_max=30)
                    ax5.imshow(ori_decfreq_vis.T,cmap='rainbow',vmin=ori_decfreq_vis.min(),vmax=ori_decfreq_vis.max()+1e-12)
                    ax5.scatter(freq,dec,color='black',s=10,marker='x',linewidths=1.5)
                    ax5.set_title('freq dec')
                    ax5.set(xticks=np.arange(0, 512, 102), xticklabels=np.linspace(cube_header['CRVAL3'], cube_header['CRVAL3']+cube_header['NAXIS3']*cube_header['CDELT3'], 6).round().astype(int),
                            yticks=[0,31], yticklabels=[1, cube_header['NAXIS2']])
                    fig.tight_layout()

                    # down right (zoom up freq dec)
                    decfreq_vis = np.clip(decfreq_vis,a_min=-20,a_max=30)
                    decfreq_vis_img = ax6.imshow(decfreq_vis.T,cmap='rainbow',vmin=ori_decfreq_vis.min(),vmax=ori_decfreq_vis.max()+1e-12)
                    ax6.scatter(freq_range_pixel,10,color='black',s=10,marker='x',linewidths=1)
                    ax6.set(xticks=[0,freq_range_pixel,2*freq_range_pixel-1], xticklabels=[int_round(cube_header['CRVAL3']+(freq-freq_range_pixel)*freq_arcmin_per_pixel),int_round(cube_header['CRVAL3']+freq*freq_arcmin_per_pixel),int_round(cube_header['CRVAL3']+(freq+freq_range_pixel)*freq_arcmin_per_pixel)],
                            yticks=[0,9.5,19], yticklabels=[int_round((dec-9.5)*(cube_header['NAXIS2']/32)),int_round(dec*(cube_header['NAXIS2']/32)),int_round((dec+9.5)*(cube_header['NAXIS2']/32))])
                    ax6.set_title('zoom up freq dec')
                    divider6 = make_axes_locatable(ax6)
                    cax6 = divider6.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(decfreq_vis_img, cax=cax6, orientation='vertical')

                    # plt to array
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png',bbox_inches='tight',dpi=150)
                    img_array = np.asarray(Image.open(buffer))[:,:,[2,1,0]]
                    plt.close('all')

                    # freq wave
                    freq_wave = np.concatenate([decfreq_pz_vis[:,:,9:11].sum(-1), np.expand_dims(decfreq_vis[:,9:11].sum(-1),0)])
                    x_range = np.linspace(float(freq-freq_range_pixel)*freq_arcmin_per_pixel,float(freq+freq_range_pixel)*freq_arcmin_per_pixel,freq_range_pixel*2) + cube_header['CRVAL3']
                    plt.plot(x_range,freq_wave[0],'s-', label='pz1')
                    plt.plot(x_range,freq_wave[1],'s-', label='pz2')
                    #plt.plot(x_range,freq_wave[2],'s-', label='all')
                    plt.legend()
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png',bbox_inches='tight',dpi=150)
                    wave_array = np.asarray(Image.open(buffer))[:,:,[2,1,0]]
                    plt.close('all')
    
                    if save_shape is None:
                        save_shape = (img_array.shape[1],img_array.shape[0])
                        save_shape_wav = (int_round(wave_array.shape[1]*img_array.shape[0]/wave_array.shape[0]),img_array.shape[0])

                    # query web
                    if args.query_web:
                        try:
                            web_img,web_type = query_web(driver, '%d %d %f +%d %d %f'%(icrs_coord.ra.hms.h,icrs_coord.ra.hms.m,icrs_coord.ra.hms.s,icrs_coord.dec.dms.d,icrs_coord.dec.dms.m,icrs_coord.dec.dms.s))
                        except:
                            driver.get(url)
                            driver.maximize_window()
                        if web_img is not None:
                            web_img = web_img[:,:,:3]
                            web_img = cv2.resize(web_img,(int(web_img.shape[1]*save_shape[1]/web_img.shape[0]),save_shape[1]))
                        else:
                            web_img = 255*np.ones([save_shape[1],3*save_shape[1],3])
                        cv2.imwrite(os.path.join(demo_vis_dir,data['filename'][0],img_dir[:-4]+'_%s'%web_type+img_dir[-4:]),np.concatenate([cv2.resize(img_array,save_shape),cv2.resize(wave_array,save_shape_wav),web_img],1))
                    else:
                        cv2.imwrite(os.path.join(demo_vis_dir,data['filename'][0],img_dir),np.concatenate([cv2.resize(img_array,save_shape),cv2.resize(wave_array,save_shape_wav)],1))

    with open(save_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
if args.query_web:
    driver.close()
exit()