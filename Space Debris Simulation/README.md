# Space Debris Simulation Image Generator

A tool for generating realistic space debris simulation images that can simulate the motion trajectories of space debris in astronomical images.

## Features

- 🛰️ Generate realistic space debris simulation images
- 📍 Control debris position, angle, size and other parameters
- 🎯 Support continuous multi-frame image generation to simulate debris motion
- 📊 Automatically generate label files for subsequent analysis
- 🔧 Flexible parameter configuration, support random generation or specified parameters

## Installation

### Manual Installation

```bash
pip install -r requirements.txt
```

### Dependencies

```bash
pip install numpy matplotlib scikit-image astropy opencv-python tqdm
```

## Quick Start

### 1. Basic Usage

Generate simulation images with default parameters:

```bash
python space_debris_su.py --output_path ./output
```

### 2. Specify Debris Parameters

Generate debris with specified position and angle:

```bash
python space_debris_su.py \
    --x_init 1500 \
    --y_init 1500 \
    --angle 45 \
    --output_path ./output_positioned
```

### 3. Generate Multiple Debris

Generate multiple debris and continuous images:

```bash
python space_debris_su.py \
    --debris_range 5 \
    --n_images 5 \
    --save_labels \
    --output_path ./output_multiple
```

## Parameter Description

### Input/Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--space_path` | str | Default FITS file path | Background image path |
| `--output_path` | str | Default output path | Output image save path |

### Debris Count Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--debris_range` | int | 2 | Debris count range (1 to debris_range) |
| `--n_images` | int | 2 | Number of continuous images to generate |

### Debris Position Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--x_init` | int | None | Initial x coordinate of debris (300-2800) |
| `--y_init` | int | None | Initial y coordinate of debris (300-2800) |

### Debris Geometry Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--angle` | int | None | Debris angle (0-360 degrees) |
| `--length` | int | None | Debris length (100-800) |
| `--width` | int | None | Debris width (15-30) |
| `--sigma` | float | None | Gaussian distribution standard deviation (1-3) |

### Debris Intensity Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--peak` | int | None | Peak intensity (40-160) |
| `--noise` | int | None | Debris noise level (20-100) |
| `--velocity` | int | None | Debris velocity (100-300) |

### Image Noise Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--image_noise` | float | None | Image background noise (9-10) |

### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_labels` | flag | False | Whether to save label files |

## Usage Examples

### Example 1: Use Default Parameters

```bash
python space_debris_su.py --output_path ./output_default
```

This will generate simulation images using all random parameters.

### Example 2: Specify Debris Position and Angle

```bash
python space_debris_su.py \
    --x_init 1500 \
    --y_init 1500 \
    --angle 45 \
    --output_path ./output_positioned
```

This will generate a debris located at (1500,1500) with an angle of 45 degrees.

### Example 3: Specify Debris Geometry Parameters

```bash
python space_debris_su.py \
    --length 500 \
    --width 20 \
    --sigma 2.0 \
    --output_path ./output_geometry
```

This will generate debris with length 500, width 20, and sigma 2.0.

### Example 4: Specify Debris Intensity Parameters

```bash
python space_debris_su.py \
    --peak 100 \
    --noise 50 \
    --velocity 200 \
    --output_path ./output_intensity
```

This will generate debris with peak intensity 100, noise 50, and velocity 200.

### Example 5: Generate Multiple Debris and Images

```bash
python space_debris_su.py \
    --debris_range 5 \
    --n_images 5 \
    --save_labels \
    --output_path ./output_multiple
```

This will generate 1-5 debris, 5 continuous images, and save label files.

### Example 6: Complete Parameter Setting

```bash
python space_debris_su.py \
    --x_init 1000 \
    --y_init 1000 \
    --angle 30 \
    --length 400 \
    --width 25 \
    --sigma 1.5 \
    --peak 120 \
    --noise 60 \
    --velocity 250 \
    --debris_range 3 \
    --n_images 3 \
    --image_noise 9.5 \
    --save_labels \
    --output_path ./output_complete
```

This will generate simulation images using all specified parameters.

## Output Files

### Image Files

- `result00.png`, `result01.png`, ...: Generated simulation image files

### Label Files (when using `--save_labels`)

- `labels00.json`, `labels01.json`, ...: JSON files containing debris position information

Label file format example:
```json
[
  {
    "result00_0": [[x_start, y_start], [x_end, y_end]]
  },
  {
    "result00_1": [[x_start, y_start], [x_end, y_end]]
  }
]
```

## Example Usage Script

We also provide an example script `example_usage.py` that demonstrates various usage scenarios:

```bash
# View all examples
python example_usage.py

# Run specific example
python example_usage.py 1  # Run example 1
python example_usage.py 2  # Run example 2
```

## Parameter Range Description

### Coordinate Range
- x_init, y_init: 300-2800 (avoid boundary effects)

### Geometry Parameters
- angle: 0-360 degrees
- length: 100-800 pixels
- width: 15-30 pixels
- sigma: 1-3 (Gaussian distribution standard deviation)

### Intensity Parameters
- peak: 40-160 (peak intensity)
- noise: 20-100 (noise level)
- velocity: 100-300 (motion velocity)

### Other Parameters
- debris_range: 1-any positive integer (debris count range)
- n_images: 1-any positive integer (continuous image count)
- image_noise: 9-10 (background noise)

## Important Notes

1. **Coordinate Range**: Initial debris coordinates should be within 300-2800 range to avoid boundary effects
2. **Image Size**: Output image size is 3072x3072 pixels
3. **File Format**: Input supports FITS format, output is PNG format
4. **Memory Usage**: Pay attention to memory usage when generating large numbers of images
5. **Randomness**: Unspecified parameters will use random values

## Troubleshooting

### Common Issues

1. **File Path Error**: Ensure input file path is correct and file exists
2. **Parameter Range Error**: Ensure parameter values are within specified ranges
3. **Insufficient Memory**: Reduce `n_images` or `debris_range` parameters
4. **Output Directory Permissions**: Ensure write permissions for output directory

### View Help

```bash
python space_debris_su.py --help
```

## Technical Details

### Algorithm Principle

1. **Background Image Processing**: Read FITS file, perform size adjustment and normalization
2. **Debris Generation**: Use Gaussian function to generate debris shape
3. **Motion Simulation**: Calculate debris motion trajectory based on velocity and angle
4. **Noise Addition**: Add random noise to images and debris
5. **Label Generation**: Record debris position information in each frame

### Core Functions

- `gauss1()`: Generate Gaussian distribution
- `create_array()`: Create debris array
- `add_array_to_data()`: Add debris to background image
- `calculate_next_point()`: Calculate debris position in next frame
- `add_noise()`: Add noise

## License

This project follows the MIT License.

## Contributing

Welcome to submit Issues and Pull Requests to improve this tool.
