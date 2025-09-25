#include "utils.hpp"
#include <getopt.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

void print_usage(const char *prog_name) {
  cerr << "Usage: " << prog_name
       << " --input <file> --output <file> [OPTIONS]\n"
       << "Options:\n"
       << "  -t, --threshold <value>    Detection threshold (0.0-1.0, default: "
          "0.1)\n"
       << "  -b, --blend <factor>       Original image opacity (0.0-1.0, "
          "default: 0.5)\n"
       << "  -m, --models <directory>   Models base directory (default: "
          "./models)\n"
       << "  --mode <mode>              Mode: hand, body, both, face, all "
          "(default: all)\n"
       << "  -v, --verbose              Enable verbose output.\n"
       << "  --no-validate-connectivity Deactivates filtering of noisy points "
          "(on by default)\n"
       << "  --no-interpolate           Deactivates estimation of missing "
          "joints (on by default)\n";
}

int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);
  if (args.image_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  if (args.verbose)
    cout << "Verbose: Reading image: " << args.image_path << endl;
  Mat img = imread(args.image_path);
  if (img.empty()) {
    cerr << "Error: Could not read the input image: " << args.image_path
         << "\n";
    return 2;
  }
  if (args.verbose)
    cout << "Verbose: Image loaded successfully. Size: " << img.cols << "x"
         << img.rows << endl;

  Net body_net, hand_net, face_net;
  vector<float> bodyKeypoints, handKeypoints, faceKeypoints;

  bool process_body =
      (args.mode == Mode::BODY || args.mode == Mode::BODY_AND_HAND ||
       args.mode == Mode::ALL);
  bool process_hand =
      (args.mode == Mode::HAND || args.mode == Mode::BODY_AND_HAND ||
       args.mode == Mode::ALL);
  bool process_face = (args.mode == Mode::FACE || args.mode == Mode::ALL);

  bool run_body_model_first =
      process_body || process_hand || (process_face && args.mode != Mode::FACE);

  if (run_body_model_first) {
    if (args.verbose)
      cout << "Verbose: Loading BODY model..." << endl;
    auto body_paths = resolveModelPaths(ModelType::BODY, args.models_dir);
    if (args.verbose)
      cout << "Verbose: Body Proto: " << body_paths["prototxt"]
           << ", Body Model: " << body_paths["caffemodel"] << endl;
    loadModels(body_net, body_paths);
    if (args.verbose)
      cout << "Verbose: BODY model loaded. Running estimation..." << endl;
    bodyKeypoints = runOpenPose(img, body_net, ModelType::BODY, args.threshold);
    if (args.verbose)
      cout << "Verbose: BODY estimation completed." << endl;
  }

  if (process_hand) {
    if (bodyKeypoints.empty()) {
      cerr << "Warning: Hand detection requires body detection to locate "
              "wrists. Skipping hand detection."
           << endl;
    } else {
      if (args.verbose)
        cout << "Verbose: Loading HAND model..." << endl;
      auto hand_paths = resolveModelPaths(ModelType::HAND, args.models_dir);
      loadModels(hand_net, hand_paths);
      if (args.verbose)
        cout << "Verbose: HAND model loaded. Detecting hands..." << endl;

      Point right_wrist(-1, -1), left_wrist(-1, -1);
      if (bodyKeypoints.size() >= 25 * 3) {
        if (bodyKeypoints[4 * 3 + 2] > 0)
          right_wrist = Point(bodyKeypoints[4 * 3], bodyKeypoints[4 * 3 + 1]);
        if (bodyKeypoints[7 * 3 + 2] > 0)
          left_wrist = Point(bodyKeypoints[7 * 3], bodyKeypoints[7 * 3 + 1]);
      }

      Rect right_rect = makeHandRect(right_wrist, 368, img),
           left_rect = makeHandRect(left_wrist, 368, img);
      vector<float> leftHand, rightHand;

      if (left_rect.area() > 0) {
        leftHand = runOpenPose(img(left_rect), hand_net, ModelType::HAND,
                               args.threshold);
        for (size_t i = 0; i < leftHand.size(); i += 3)
          if (leftHand[i + 2] > 0) {
            leftHand[i] += left_rect.x;
            leftHand[i + 1] += left_rect.y;
          }
      } else {
        leftHand = vector<float>(21 * 3, -1.0f);
      }

      if (right_rect.area() > 0) {
        rightHand = runOpenPose(img(right_rect), hand_net, ModelType::HAND,
                                args.threshold);
        for (size_t i = 0; i < rightHand.size(); i += 3)
          if (rightHand[i + 2] > 0) {
            rightHand[i] += right_rect.x;
            rightHand[i + 1] += right_rect.y;
          }
      } else {
        rightHand = vector<float>(21 * 3, -1.0f);
      }

      handKeypoints.insert(handKeypoints.end(), leftHand.begin(),
                           leftHand.end());
      handKeypoints.insert(handKeypoints.end(), rightHand.begin(),
                           rightHand.end());
      if (args.verbose)
        cout << "Verbose: HAND estimation completed." << endl;
    }
  }

  if (process_face) {
    if (args.verbose)
      cout << "Verbose: Loading FACE model..." << endl;
    auto face_paths = resolveModelPaths(ModelType::FACE, args.models_dir);
    loadModels(face_net, face_paths);
    if (args.verbose)
      cout << "Verbose: FACE model loaded. Detecting face..." << endl;

    auto haar_paths = resolveModelPaths(ModelType::FACE_HAAR, args.models_dir);
    string haar_path = haar_paths["haarcascade"];

    Rect face_rect = getFaceRect(bodyKeypoints, img, haar_path, args.verbose);

    if (face_rect.area() > 0) {
      faceKeypoints = runOpenPose(img(face_rect), face_net, ModelType::FACE,
                                  args.threshold);
      for (size_t i = 0; i < faceKeypoints.size(); i += 3) {
        if (faceKeypoints[i + 2] > 0) {
          faceKeypoints[i] += face_rect.x;
          faceKeypoints[i + 1] += face_rect.y;
        }
      }
    } else {
      if (args.verbose)
        cout << "Verbose: Could not locate face for detailed analysis." << endl;
      faceKeypoints = vector<float>(70 * 3, -1.0f);
    }
    if (args.verbose)
      cout << "Verbose: FACE estimation completed." << endl;
  }

  if (args.verbose)
    cout << "Verbose: Post-processing keypoints..." << endl;
  if (process_body) {
    bodyKeypoints = postProcessKeypoints(
        bodyKeypoints, ModelType::BODY, args.enable_validation,
        args.enable_interpolation, args.verbose);
  }
  if (process_hand) {
    Point right_wrist(-1, -1), left_wrist(-1, -1);
    if (!bodyKeypoints.empty()) {
      if (bodyKeypoints.size() > 4 * 3 + 2 && bodyKeypoints[4 * 3 + 2] > 0)
        right_wrist = Point(bodyKeypoints[4 * 3], bodyKeypoints[4 * 3 + 1]);
      if (bodyKeypoints.size() > 7 * 3 + 2 && bodyKeypoints[7 * 3 + 2] > 0)
        left_wrist = Point(bodyKeypoints[7 * 3], bodyKeypoints[7 * 3 + 1]);
    }
    handKeypoints =
        postProcessKeypoints(handKeypoints, ModelType::HAND,
                             args.enable_validation, args.enable_interpolation,
                             args.verbose, right_wrist, left_wrist, 150.0f);
  }
  if (process_face) {
    faceKeypoints = postProcessKeypoints(faceKeypoints, ModelType::FACE, false,
                                         false, args.verbose);
  }

  if (args.verbose)
    cout << "Verbose: Drawing keypoints on the image..." << endl;
  // --- DRAWING ---
  // FIX: Create a black overlay instead of cloning the image.
  Mat overlay = Mat::zeros(img.size(), img.type());

  if (process_body) {
    drawKeypoints(overlay, bodyKeypoints, ModelType::BODY, Scalar(255, 0, 0));
  }
  if (process_hand) {
    drawKeypoints(overlay, handKeypoints, ModelType::HAND, Scalar(0, 255, 0));
  }
  if (process_face) {
    drawKeypoints(overlay, faceKeypoints, ModelType::FACE, Scalar(0, 255, 255));
  }

  // Draw connecting lines for hands
  if (process_hand && !bodyKeypoints.empty() && !handKeypoints.empty()) {
    if (bodyKeypoints.size() > 4 * 3 + 2 && handKeypoints.size() > 21 * 3 + 2 &&
        bodyKeypoints[4 * 3 + 2] > 0 && handKeypoints[21 * 3 + 2] > 0)
      line(overlay, Point(bodyKeypoints[4 * 3], bodyKeypoints[4 * 3 + 1]),
           Point(handKeypoints[21 * 3], handKeypoints[21 * 3 + 1]),
           Scalar(0, 255, 0), max(1, (img.cols + img.rows) / 400));
    if (bodyKeypoints.size() > 7 * 3 + 2 && handKeypoints.size() > 0 * 3 + 2 &&
        bodyKeypoints[7 * 3 + 2] > 0 && handKeypoints[0 * 3 + 2] > 0)
      line(overlay, Point(bodyKeypoints[7 * 3], bodyKeypoints[7 * 3 + 1]),
           Point(handKeypoints[0 * 3], handKeypoints[0 * 3 + 1]),
           Scalar(0, 255, 0), max(1, (img.cols + img.rows) / 400));
  }

  // --- FINAL OUTPUT ---
  // FIX: Blend the background image (img) with the keypoint overlay.
  // The blend_factor now correctly controls the opacity of the original image.
  Mat final_output;
  addWeighted(img, args.blend_factor, overlay, 1.0, 0, final_output);

  string output_path =
      args.output_file.empty() ? "unified_output.jpg" : args.output_file;
  imwrite(output_path, final_output);
  cout << "Done. Output saved to: " << output_path << "\n";
  return 0;
}
