# openpose: A Simple Pose Estimation CLI Tool

openpose is a command-line interface (CLI) tool for pose estimation, built with OpenCV. It processes input images to detect poses including body, hand, face, or all.

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
  -i, --input <file>         Required. Path to the input image.
  -o, --output <file>        Optional. Path to save the output image (default: unified_output.jpg).
  -t, --threshold <value>    Pose keypoint detection threshold (0.0-1.0, default: 0.1).
  --person-threshold <value> Person detection confidence threshold for multi-person (default: 0.5).
  --nms-threshold <value>    Non-Maximum Suppression threshold for multi-person (default: 0.4).
  -b, --blend <factor>       Original image opacity (0.0-1.0, default: 0.5).
  -m, --models <directory>   Models base directory (default: ./models).
  -M, --mode <modes>         Modes: comma-separated list of 'body', 'hand', 'face', 'all' (default: body).
  -v, --verbose              Enable verbose output.
  --single-person            Disable multi-person detection (multi-person is on by default).
  --no-validate-connectivity Deactivates filtering of noisy points (on by default).
  --no-interpolate           Deactivates estimation of missing joints (on by default).
  --draw-foot                Draw the foot keypoints (off by default).
  --no-rainbow               Disable rainbow coloring for skeletons.
  -h, --help                 Display this help message.
```

### Required Arguments
- `-i, --input <file>`: Path to the input image to process.

### Optional Arguments
- `-o, --output <file>`: Path to save the output image (default: unified_output.jpg).
- `-t, --threshold <value>`: Pose keypoint detection threshold (0.0-1.0, default: 0.1). Higher values make detection more strict.
- `--person-threshold <value>`: Person detection confidence threshold for multi-person scenarios (default: 0.5).
- `--nms-threshold <value>`: Non-Maximum Suppression threshold to reduce overlapping detections in multi-person (default: 0.4).
- `-b, --blend <factor>`: Opacity of the original image in the output (0.0-1.0, default: 0.5). 1.0 = full original, 0.0 = annotations only.
- `-m, --models <directory>`: Base directory for models (default: ./models).
- `-M, --mode <modes>`: Comma-separated list of detection modes: body, hand, face, all (default: body).
- `-v, --verbose`: Enable verbose logging.
- `--single-person`: Use single-person mode (disables multi-person detection; multi-person is default).
- `--no-validate-connectivity`: Disable filtering of noisy keypoints (enabled by default).
- `--no-interpolate`: Disable estimation of missing joints (enabled by default).
- `--draw-foot`: Enable drawing of foot keypoints (disabled by default).
- `--no-rainbow`: Disable rainbow colors for skeleton visualization (enabled by default).
- `-h, --help`: Show the help message and exit.

## Dependencies
- OpenCV (for computer vision and image processing)
- C++17 standard (for modern language features)

## Models
The tool requires pre-trained models for pose estimation. Download them from:
https://huggingface.co/fszontagh/openpose-pose-estimation

Extract the models and place them in the `./models` directory (relative to the project root). Ensure the directory structure matches the expected model paths used in the code.