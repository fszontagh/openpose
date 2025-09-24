# openpose: A Simple Pose Estimation CLI Tool

openpose is a command-line interface (CLI) tool for pose estimation, built with OpenCV. It processes input images or videos to detect poses including hand, body, face, or all.

## Installation

### Prerequisites
- OpenCV (compiled with C++17 standard support)
- CMake
- A C++ compiler supporting C++17 (e.g., g++ or clang++)

### Build Steps
1. Navigate to the project root directory.
2. Create and enter the build directory:
   ```
   mkdir build && cd build
   ```
3. Configure the build:
   ```
   cmake ..
   ```
4. Build the project:
   ```
   make
   ```

The executable will be available at `./build/openpose`.

## Usage

Run the tool with the following command structure:

```
Usage: ./build/openpose --input <file> --output <file> [OPTIONS]
Options:
  -t, --threshold <value>    Detection threshold (0.0-1.0, default: 0.1)
  -b, --blend <factor>       Original image opacity (0.0-1.0, default: 0.5)
  -m, --models <directory>   Models base directory (default: ./models)
  --mode <mode>              Mode: hand, body, both, face, all (default: all)
  -v, --verbose              Enable verbose output.
  --no-validate-connectivity Deactivates filtering of noisy points (on by default)
  --no-interpolate           Deactivates estimation of missing joints (on by default)
```

### Required Arguments
- `--input <file>`: Path to the input image or video file to process.
- `--output <file>`: Path to the output file where the results (annotated image or video) will be saved.

### Optional Options
- `-t, --threshold <value>`: Sets the detection confidence threshold (range: 0.0 to 1.0; default: 0.1). Higher values make detection stricter.
- `-b, --blend <factor>`: Controls the opacity of the original image in the output (range: 0.0 to 1.0; default: 0.5). 1.0 shows full original, 0.0 shows only annotations.
- `-m, --models <directory>`: Specifies the base directory for model files (default: `./models`).
- `--mode <mode>`: Selects the pose detection mode: `hand`, `body`, `both`, `face`, or `all` (default: `all`).
- `-v, --verbose`: Enables detailed output during processing.
- `--no-validate-connectivity`: Disables filtering of noisy keypoints (enabled by default for better accuracy).
- `--no-interpolate`: Disables estimation of missing joints (enabled by default for smoother results).

Note: `--help` is not implemented; run without arguments to see the usage message.

## Dependencies
- OpenCV (for computer vision and image processing)
- C++17 standard (for modern language features)

## Models
The tool requires pre-trained models for pose estimation. Download them from:
https://huggingface.co/fszontagh/openpose-pose-estimation

Extract the models and place them in the `./models` directory (relative to the project root). Ensure the directory structure matches the expected model paths used in the code.