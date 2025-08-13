import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from skimage.transform import resize
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import pdb
import json
import os
import time
import glob
from tqdm import tqdm
import cv2
import argparse


def gauss1(x, params): 
    #gauss1(x, params) params=[mean,sigma,peak]
    mean, sigma, peak = params
    return peak * np.exp(-(x - mean)**2 / (2 * sigma**2))

def add_array_to_data(array, data0, x_start, y_start, angle):
    data=data0.copy() 
    #Rotate a rectangular array by angle degrees and add it to the large array data at center coordinates x_center, y_center
    width, length = array.shape

    x_end = x_start
    y_end = y_start
    # Calculate rotation angle
    theta = np.radians(angle)
    
    # Corresponding rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    
    for i in range(width):
        for j in range(length):
            # Calculate rotated coordinates
            rotated_x, rotated_y = np.dot(rotation_matrix, np.array([i, j]))
            rotated_x = int(rotated_x)
            rotated_y = int(rotated_y)
            
            # Calculate corresponding coordinates in data
            data_x = x_start + rotated_x
            data_y = y_start + rotated_y
            
            # If coordinates are within data range, add array value to data
            if 0 <= data_x < data.shape[0] and 0 <= data_y < data.shape[1]:
                data[data_x, data_y] += array[i, j]
                x_end = data_x
                y_end = data_y

    
    return data,x_start,x_end,y_start,y_end

def calculate_next_point(x_center, y_center, angle, v):
    # Convert angle to radians
    radian = math.radians(angle)
    x_next = x_center + v * math.cos(radian)
    y_next = y_center + v * math.sin(radian)
    return (int(x_next), int(y_next)) 

def create_array(length, width, params):
    # Create array corresponding to line segment
    array = np.zeros((length, width))
    mean, sigma, peak = params
    peak_lb = peak*0.5
    peak_ub = peak*1.1
    for i in range(length):
        for j in range(width):
            peak0 = np.random.uniform(0.95,1.05)*peak
            if peak0 > peak_ub:
                peak=peak_ub
            elif peak0 < peak_lb:
                peak=peak_lb
            else:
                peak = peak0
            array[i, j] = gauss1(j+np.random.uniform(-0.5,0.5), [mean, sigma, peak])
    return array

def add_noise(array, noise_level):
    #Add random noise to array
    noise = np.random.normal(0, noise_level, array.shape)
    noisy_array = array + noise
    return noisy_array


def generate_space_debris_simulation(args):
    """
    Generate space debris simulation images
    
    Args:
        args: argparse object containing all parameters
    """
    # Read background image
    hdul = fits.open(args.space_path)
    try:
        data0 = hdul[0].data
    except:
        data0 = hdul[1].data
    np.nan_to_num(data0, 0) 
    data0 = resize(data0, (3072, 3072))
    
    # Randomly select image transformation
    ind = np.random.choice(range(0, 8))
    if ind == 0:
        data0 = data0
    elif ind == 1:
        data0 = data0[::-1, :]
    elif ind == 2:
        data0 = data0[::-1, ::-1]
    elif ind == 3:
        data0 = data0[:, ::-1]
    elif ind == 4:
        data0 = data0[::-1, :].T
    elif ind == 5:
        data0 = data0[::-1, ::-1].T
    elif ind == 6:
        data0 = data0[:, ::-1].T
    elif ind == 7:
        data0 = data0.T

    # Image normalization
    z = ZScaleInterval()
    z1, z2 = z.get_limits(data0)
    data0 = np.clip(255*(data0-z1)/(z2-z1), 0, 255)

    # Generate debris parameters
    debris_nums = np.random.randint(1, args.debris_range) 
    center_list = []
    label_data_list = []
    
    for n in range(debris_nums):
        # Use parameter settings or random generation
        if args.x_init is not None and args.y_init is not None:
            x_init = args.x_init
            y_init = args.y_init
        else:
            x_init = np.random.choice(range(300, 2800)) 
            y_init = np.random.choice(range(300, 2800))
            
        if args.angle is not None:
            angle = args.angle
        else:
            angle = np.random.randint(0, 360)
            
        if args.length is not None:
            length = args.length
        else:
            length = np.random.randint(100, 800)
            
        if args.sigma is not None:
            sigma = args.sigma
        else:
            sigma = np.random.uniform(1, 3)
            
        if args.width is not None:
            width = args.width
        else:
            width = np.random.randint(15, 30)
            
        if args.peak is not None:
            peak = args.peak
        else:
            peak = np.random.randint(40, 160)
            
        if args.noise is not None:
            noise = args.noise
        else:
            noise = np.random.randint(20, 100)
            
        if args.velocity is not None:
            v = args.velocity
        else:
            v = np.random.randint(100, 300)
            
        params = [width/2, width/8, peak]
        center_list.append([x_init, y_init, angle, length, width, params, noise, v])
    
    # Generate continuous images
    error = []
    os.makedirs(args.output_path, exist_ok=True)

    for i in range(args.n_images):
        tmp_list = []
        data = add_noise(data0, args.image_noise)
        
        for id, data_ in enumerate(center_list):
            x_init, y_init, angle, length, width, params, noise, v = data_
            x_next, y_next = calculate_next_point(x_init, y_init, angle, v*i)
            
            if x_next >= data.shape[0] or x_next < 0 or y_next >= data.shape[1] or y_next < 0:
                continue
                
            array = create_array(length, width, params)
            result, x_start, x_end, y_start, y_end = add_array_to_data(array, data, x_next, y_next, angle)
            
            if x_start == y_start and x_end == y_end:
                continue
                
            if x_end//4 >= 768 or x_end < 0 or y_end//4 >= 768 or y_end < 0:
                error.append(id)
                print(f'{x_end} {y_end}')
                print(f'{x_end//4} {y_end//4}')
                print('error')

            label = np.array([[x_start, y_start], [x_end, y_end]])
            label_list = label.tolist()
            label_name = "result{:02d}_{:02d}".format(i, id)
            tmp_list.append({
                label_name: label_list,
            })
            data = result

        if len(tmp_list) == 0:
            break
            
        label_data_list.extend(tmp_list)
        result = np.clip(result, 0, 255)
        
        # Save image
        output = os.path.join(args.output_path, "result{:02d}.png".format(i))
        cv2.imwrite(output, result)
        
        # Save labels
        if args.save_labels:
            label_file = os.path.join(args.output_path, "labels{:02d}.json".format(i))
            with open(label_file, 'w') as f:
                json.dump(tmp_list, f, indent=2)
    
    print(f"Generated {len(label_data_list)} debris labels")
    if error:
        print(f"{len(error)} debris exceeded boundaries")


def main():
    parser = argparse.ArgumentParser(description='Space Debris Simulation Image Generator')
    
    # Input/Output parameters
    parser.add_argument('--space_path', type=str, 
                       default='/ailab/group/pjlab-ai4s/ai4astro/move_4.24/Datasets/astronomy/transient_det/r_data/ztf_20181230098148_000545_zr_c16_o_q4_sciimg.fits',
                       help='Background image path')
    parser.add_argument('--output_path', type=str, 
                       default='/ailab/user/zhuangguohang/ai4stronomy/zhuangguohang/ai4astronomy/sd_docs/cp/sumik',
                       help='Output path')
    
    # Debris count parameters
    parser.add_argument('--debris_range', type=int, default=2,
                       help='Debris count range (1 to debris_range)')
    parser.add_argument('--n_images', type=int, default=2,
                       help='Number of continuous images to generate')
    
    # Debris position parameters
    parser.add_argument('--x_init', type=int, default=None,
                       help='Initial x coordinate of debris (300-2800)')
    parser.add_argument('--y_init', type=int, default=None,
                       help='Initial y coordinate of debris (300-2800)')
    
    # Debris geometry parameters
    parser.add_argument('--angle', type=int, default=None,
                       help='Debris angle (0-360 degrees)')
    parser.add_argument('--length', type=int, default=None,
                       help='Debris length (100-800)')
    parser.add_argument('--width', type=int, default=None,
                       help='Debris width (15-30)')
    parser.add_argument('--sigma', type=float, default=None,
                       help='Gaussian distribution standard deviation (1-3)')
    
    # Debris intensity parameters
    parser.add_argument('--peak', type=int, default=None,
                       help='Peak intensity (40-160)')
    parser.add_argument('--noise', type=int, default=None,
                       help='Debris noise level (20-100)')
    parser.add_argument('--velocity', type=int, default=None,
                       help='Debris velocity (100-300)')
    
    # Image noise parameters
    parser.add_argument('--image_noise', type=float, default=None,
                       help='Image background noise (9-10)')
    
    # Other parameters
    parser.add_argument('--save_labels', action='store_true',
                       help='Whether to save label files')
    
    args = parser.parse_args()
    
    # Set default values
    if args.image_noise is None:
        args.image_noise = np.random.uniform(9, 10)
    
    # Generate simulation images
    generate_space_debris_simulation(args)


if __name__ == "__main__":
    main()
