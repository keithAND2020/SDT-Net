#!/usr/bin/env python3
"""
Space Debris Simulation Image Generator Usage Examples

This file demonstrates how to use the refactored space debris simulation code 
to control various parameters through command line arguments.
"""

import subprocess
import sys

def run_example():
    """Run various examples"""
    
    print("=== Space Debris Simulation Image Generator Usage Examples ===\n")
    
    # Example 1: Use default parameters
    print("1. Generate simulation images with default parameters:")
    cmd1 = [
        "python", "space_debris_su.py",
        "--output_path", "./output_default"
    ]
    print(f"Command: {' '.join(cmd1)}")
    print("This will generate simulation images using all random parameters\n")
    
    # Example 2: Specify debris position and angle
    print("2. Specify debris position and angle:")
    cmd2 = [
        "python", "space_debris_su.py",
        "--x_init", "1500",
        "--y_init", "1500", 
        "--angle", "45",
        "--output_path", "./output_positioned"
    ]
    print(f"Command: {' '.join(cmd2)}")
    print("This will generate a debris located at (1500,1500) with an angle of 45 degrees\n")
    
    # Example 3: Specify debris geometry parameters
    print("3. Specify debris geometry parameters:")
    cmd3 = [
        "python", "space_debris_su.py",
        "--length", "500",
        "--width", "20",
        "--sigma", "2.0",
        "--output_path", "./output_geometry"
    ]
    print(f"Command: {' '.join(cmd3)}")
    print("This will generate debris with length 500, width 20, and sigma 2.0\n")
    
    # Example 4: Specify debris intensity parameters
    print("4. Specify debris intensity parameters:")
    cmd4 = [
        "python", "space_debris_su.py",
        "--peak", "100",
        "--noise", "50",
        "--velocity", "200",
        "--output_path", "./output_intensity"
    ]
    print(f"Command: {' '.join(cmd4)}")
    print("This will generate debris with peak intensity 100, noise 50, and velocity 200\n")
    
    # Example 5: Generate multiple debris and images
    print("5. Generate multiple debris and images:")
    cmd5 = [
        "python", "space_debris_su.py",
        "--debris_range", "5",
        "--n_images", "5",
        "--save_labels",
        "--output_path", "./output_multiple"
    ]
    print(f"Command: {' '.join(cmd5)}")
    print("This will generate 1-5 debris, 5 continuous images, and save label files\n")
    
    # Example 6: Complete parameter setting
    print("6. Complete parameter setting:")
    cmd6 = [
        "python", "space_debris_su.py",
        "--x_init", "1000",
        "--y_init", "1000",
        "--angle", "30",
        "--length", "400",
        "--width", "25",
        "--sigma", "1.5",
        "--peak", "120",
        "--noise", "60",
        "--velocity", "250",
        "--debris_range", "3",
        "--n_images", "3",
        "--image_noise", "9.5",
        "--save_labels",
        "--output_path", "./output_complete"
    ]
    print(f"Command: {' '.join(cmd6)}")
    print("This will generate simulation images using all specified parameters\n")
    
    # Example 7: View help information
    print("7. View all available parameters:")
    cmd7 = ["python", "space_debris_su.py", "--help"]
    print(f"Command: {' '.join(cmd7)}")
    print("This will display all available command line parameters\n")

def run_specific_example(example_num):
    """Run specific example"""
    examples = {
        1: [
            "python", "space_debris_su.py",
            "--output_path", "./output_default"
        ],
        2: [
            "python", "space_debris_su.py",
            "--x_init", "1500",
            "--y_init", "1500", 
            "--angle", "45",
            "--output_path", "./output_positioned"
        ],
        3: [
            "python", "space_debris_su.py",
            "--length", "500",
            "--width", "20",
            "--sigma", "2.0",
            "--output_path", "./output_geometry"
        ],
        4: [
            "python", "space_debris_su.py",
            "--peak", "100",
            "--noise", "50",
            "--velocity", "200",
            "--output_path", "./output_intensity"
        ],
        5: [
            "python", "space_debris_su.py",
            "--debris_range", "5",
            "--n_images", "5",
            "--save_labels",
            "--output_path", "./output_multiple"
        ],
        6: [
            "python", "space_debris_su.py",
            "--x_init", "1000",
            "--y_init", "1000",
            "--angle", "30",
            "--length", "400",
            "--width", "25",
            "--sigma", "1.5",
            "--peak", "120",
            "--noise", "60",
            "--velocity", "250",
            "--debris_range", "3",
            "--n_images", "3",
            "--image_noise", "9.5",
            "--save_labels",
            "--output_path", "./output_complete"
        ]
    }
    
    if example_num in examples:
        print(f"Running example {example_num}...")
        subprocess.run(examples[example_num])
    else:
        print(f"Example {example_num} does not exist")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            run_specific_example(example_num)
        except ValueError:
            print("Please provide a valid example number (1-6)")
    else:
        run_example()
