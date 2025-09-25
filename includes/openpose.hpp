#ifndef OPENPOSE_HPP
#define OPENPOSE_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <map>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// --- Public Enums and Constants ---

enum class ModelType { BODY, HAND, FACE, FACE_HAAR, PERSON, HAND_HAAR };

const vector<Scalar> PERSON_COLORS = {Scalar(255, 0, 0),   Scalar(0, 255, 0),
                                      Scalar(0, 0, 255),   Scalar(255, 255, 0),
                                      Scalar(255, 0, 255), Scalar(0, 255, 255)};

const vector<pair<int, int>> POSE_PAIRS_BODY_25 = {
    {1, 8},   {1, 2},   {1, 5},   {2, 3},   {3, 4},   {5, 6},   {6, 7},
    {8, 9},   {9, 10},  {10, 11}, {8, 12},  {12, 13}, {13, 14}, {1, 0},
    {0, 15},  {15, 17}, {0, 16},  {16, 18}, {14, 19}, {19, 20}, {14, 21},
    {11, 22}, {22, 23}, {11, 24}};
const vector<pair<int, int>> HAND_LEFT_PAIRS = {
    {0, 1},   {1, 2},   {2, 3},  {3, 4},   {0, 5},   {5, 6},  {6, 7},
    {7, 8},   {0, 9},   {9, 10}, {10, 11}, {11, 12}, {0, 13}, {13, 14},
    {14, 15}, {15, 16}, {0, 17}, {17, 18}, {18, 19}, {19, 20}};
const vector<pair<int, int>> HAND_RIGHT_PAIRS = {
    {21, 22}, {22, 23}, {23, 24}, {24, 25}, {21, 26}, {26, 27}, {27, 28},
    {28, 29}, {21, 30}, {30, 31}, {31, 32}, {32, 33}, {21, 34}, {34, 35},
    {35, 36}, {36, 37}, {21, 38}, {38, 39}, {39, 40}, {40, 41}};
const vector<pair<int, int>> POSE_PAIRS_FACE = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 6},   {6, 7},
    {7, 8},   {8, 9},   {9, 10},  {10, 11}, {11, 12}, {12, 13}, {13, 14},
    {14, 15}, {15, 16}, {17, 18}, {18, 19}, {19, 20}, {20, 21}, {22, 23},
    {23, 24}, {24, 25}, {25, 26}, {27, 28}, {28, 29}, {29, 30}, {31, 32},
    {32, 33}, {33, 34}, {34, 35}, {36, 37}, {37, 38}, {38, 39}, {39, 40},
    {40, 41}, {41, 36}, {42, 43}, {43, 44}, {44, 45}, {45, 46}, {46, 47},
    {47, 42}, {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54},
    {54, 55}, {55, 56}, {56, 57}, {57, 58}, {58, 59}, {59, 48}, {60, 61},
    {61, 62}, {62, 63}, {63, 64}, {64, 65}, {65, 66}, {66, 67}, {67, 60}};

// --- OpenPose Main Class ---

class OpenPose {
public:
  // The constructor receives the command line arguments
  OpenPose(int argc, char **argv) {
    parseArgs(argc, argv);
    loadAllModels();
  }

  // The main run method that controls the entire process
  void run() {
    if (args.image_path.empty()) {
        printUsage();
        return;
    }

    if (args.verbose) cout << "Verbose: Reading image: " << args.image_path << endl;
    Mat img = imread(args.image_path);
    if (img.empty()) {
        cerr << "Error: Could not read the input image: " << args.image_path << "\n";
        return;
    }
    if (args.verbose) cout << "Verbose: Image loaded successfully. Size: " << img.cols << "x" << img.rows << endl;

    Mat overlay = Mat::zeros(img.size(), img.type());

    if (args.multi_person) {
        runMultiPersonPipeline(img, overlay);
    } else {
        runSinglePersonPipeline(img, overlay);
    }

    Mat final_output;
    addWeighted(img, args.blend_factor, overlay, 1.0, 0, final_output);

    string output_path = args.output_file.empty() ? "unified_output.jpg" : args.output_file;
    imwrite(output_path, final_output);
    cout << "Done. Output saved to: " << output_path << "\n";
  }

private:
  // Internal configuration structure
  struct Args {
    string image_path;
    string output_file;
    float blend_factor = 0.5f;
    float threshold = 0.1f;
    float person_threshold = 0.5f;
    float nms_threshold = 0.4f;
    string models_dir = "./models";
    bool enable_validation = true;
    bool enable_interpolation = true;
    bool process_body = true; // Default mode is now just body
    bool process_hand = false;
    bool process_face = false;
    bool verbose = false;
    bool multi_person = false;
  };
  Args args;

  // Model containers
  Net body_net, hand_net, face_net, person_net;

  // --- Private Methods ---

  void printUsage() {
      cerr << "Usage: openpose --input <file> --output <file> [OPTIONS]\n"
           << "Options:\n"
           << "  -t, --threshold <value>    Pose keypoint detection threshold (0.0-1.0, default: 0.1)\n"
           << "  --person-threshold <value> Person detection confidence threshold for multi-person (default: 0.5)\n"
           << "  --nms-threshold <value>    Non-Maximum Suppression threshold for multi-person (default: 0.4)\n"
           << "  -b, --blend <factor>       Original image opacity (0.0-1.0, default: 0.5)\n"
           << "  -m, --models <directory>   Models base directory (default: ./models)\n"
           << "  -M, --mode <modes>         Modes: comma-separated list of 'body', 'hand', 'face', 'all'. (default: body)\n"
           << "  -v, --verbose              Enable verbose output.\n"
           << "  --multi-person             Enable multi-person detection (requires person model).\n"
           << "  --no-validate-connectivity Deactivates filtering of noisy points (on by default)\n"
           << "  --no-interpolate           Deactivates estimation of missing joints (on by default)\n";
  }

  void parseFloat(const char *s, float &out) {
      try {
          out = stof(s);
      } catch (...) {
          // Keep default value on parse error
      }
  }

  void parseArgs(int argc, char **argv) {
      int opt;
      int option_index = 0;
      static struct option long_options[] = {
          {"input", required_argument, 0, 'i'},
          {"output", required_argument, 0, 'o'},
          {"blend", required_argument, 0, 'b'},
          {"threshold", required_argument, 0, 't'},
          {"person-threshold", required_argument, 0, 5},
          {"nms-threshold", required_argument, 0, 4},
          {"models", required_argument, 0, 'm'},
          {"mode", required_argument, 0, 'M'},
          {"verbose", no_argument, 0, 'v'},
          {"multi-person", no_argument, 0, 3},
          {"no-validate-connectivity", no_argument, 0, 1},
          {"no-interpolate", no_argument, 0, 2},
          {0, 0, 0, 0}};

      bool mode_was_set = false;
      while ((opt = getopt_long(argc, argv, "i:o:b:t:m:M:v", long_options, &option_index)) != -1) {
          switch (opt) {
          case 'i': args.image_path = optarg; break;
          case 'o': args.output_file = optarg; break;
          case 'b': parseFloat(optarg, args.blend_factor); break;
          case 't': parseFloat(optarg, args.threshold); break;
          case 'm': args.models_dir = optarg; break;
          case 'v': args.verbose = true; break;
          case 'M':
              mode_was_set = true;
              args.process_body = false;
              args.process_hand = false;
              args.process_face = false;
              {
                  string modes_str(optarg);
                  stringstream ss(modes_str);
                  string mode;
                  while(getline(ss, mode, ',')) {
                      mode.erase(0, mode.find_first_not_of(" \t\n\r"));
                      mode.erase(mode.find_last_not_of(" \t\n\r") + 1);
                      if (mode == "body") args.process_body = true;
                      else if (mode == "hand" || mode == "hands") args.process_hand = true;
                      else if (mode == "face") args.process_face = true;
                      else if (mode == "all") {
                          args.process_body = true;
                          args.process_hand = true;
                          args.process_face = true;
                      } else {
                          cerr << "Invalid mode specified: " << mode << endl;
                          printUsage();
                          exit(1);
                      }
                  }
              }
              break;
          case 1: args.enable_validation = false; break;
          case 2: args.enable_interpolation = false; break;
          case 3: args.multi_person = true; break;
          case 4: parseFloat(optarg, args.nms_threshold); break;
          case 5: parseFloat(optarg, args.person_threshold); break;
          case '?': exit(1);
          default: break;
          }
      }
      if (mode_was_set && !args.process_body && !args.process_hand && !args.process_face) {
          args.process_body = true;
      }
  }

  map<string, string> resolveModelPaths(ModelType type) {
    map<string, string> paths;
    string models_dir_clean = args.models_dir;
    if (!models_dir_clean.empty() && models_dir_clean.back() == '/')
        models_dir_clean.pop_back();
    string base;
    if (type == ModelType::BODY) {
        base = models_dir_clean + "/pose";
        string body25_proto = base + "/body_25/pose_deploy.prototxt";
        string body25_model = base + "/body_25/pose_iter_584000.caffemodel";
        paths["prototxt"] = body25_proto;
        paths["caffemodel"] = body25_model;
    } else if (type == ModelType::HAND) {
        base = models_dir_clean + "/hand";
        paths["prototxt"] = base + "/pose_deploy.prototxt";
        paths["caffemodel"] = base + "/pose_iter_102000.caffemodel";
    } else if (type == ModelType::FACE) {
        base = models_dir_clean + "/face";
        paths["prototxt"] = base + "/pose_deploy.prototxt";
        paths["caffemodel"] = base + "/pose_iter_116000.caffemodel";
    } else if (type == ModelType::FACE_HAAR) {
        base = models_dir_clean + "/face";
        paths["haarcascade"] = base + "/haarcascade_frontalface_alt.xml";
    } else if (type == ModelType::PERSON) {
        base = models_dir_clean + "/person";
        paths["prototxt"] = base + "/MobileNetSSD_deploy.prototxt";
        paths["caffemodel"] = base + "/MobileNetSSD_deploy.caffemodel";
    } else if (type == ModelType::HAND_HAAR) {
        base = models_dir_clean + "/hand";
        paths["haarcascade"] = base + "/haarcascade_hand.xml";
    }
    return paths;
  }

  bool loadModel(Net &net, const map<string, string> &paths) {
      const string& proto = paths.at("prototxt");
      const string& model = paths.at("caffemodel");

      if (args.verbose) {
          cout << "Verbose: Loading model files:" << endl;
          cout << "  Prototxt: " << proto << endl;
          cout << "  Caffemodel: " << model << endl;
      }
      try {
          if (access(proto.c_str(), F_OK) != 0 || access(model.c_str(), F_OK) != 0) {
              throw runtime_error("Model file not found.");
          }
          net = readNetFromCaffe(proto, model);
          if (net.empty()) {
              throw runtime_error("ReadNetFromCaffe returned an empty network.");
          }
          net.setPreferableBackend(DNN_BACKEND_OPENCV);
          net.setPreferableTarget(DNN_TARGET_CPU);
      } catch (const std::exception& e) {
          cerr << "Error: Exception while loading model: " << e.what() << endl;
          cerr << "  Prototxt: " << proto << endl;
          cerr << "  Caffemodel: " << model << endl;
          return false;
      }
      return true;
  }

  void loadAllModels() {
      // Body model is needed for body AND for hand detection (to find wrists).
      if ((args.process_body || args.process_hand) && body_net.empty()) {
          if (args.verbose) cout << "Verbose: Loading BODY model..." << endl;
          auto paths = resolveModelPaths(ModelType::BODY);
          if (!loadModel(body_net, paths)) throw runtime_error("Failed to load BODY model.");
      }
      if (args.process_hand && hand_net.empty()) {
          if (args.verbose) cout << "Verbose: Loading HAND model..." << endl;
          auto paths = resolveModelPaths(ModelType::HAND);
          if (!loadModel(hand_net, paths)) throw runtime_error("Failed to load HAND model.");
      }
      if (args.process_face && face_net.empty()) {
          if (args.verbose) cout << "Verbose: Loading FACE model..." << endl;
          auto paths = resolveModelPaths(ModelType::FACE);
          if (!loadModel(face_net, paths)) throw runtime_error("Failed to load FACE model.");
      }
      if (args.multi_person && person_net.empty()) {
          if (args.verbose) cout << "Verbose: Loading PERSON detector model..." << endl;
          auto paths = resolveModelPaths(ModelType::PERSON);
          if(!loadModel(person_net, paths)) throw runtime_error("Failed to load PERSON model.");
      }
  }

  void processPersons(Mat& img, Mat& overlay, const vector<Rect>& person_rects);
  void runMultiPersonPipeline(Mat& img, Mat& overlay);
  void runSinglePersonPipeline(Mat& img, Mat& overlay);
  vector<Rect> detectPersons(Net &net, const Mat &frame, float conf_threshold, float nms_threshold, bool verbose);
  vector<float> runOpenPose(const cv::Mat &image, Net &net, ModelType type, float threshold = 0.05f);
  void drawKeypoints(cv::Mat &image, const vector<float> &keypoints, ModelType type, Scalar color);
  Rect makeHandRect(Point wrist, int box_size, const Mat &img);
  vector<float> postProcessKeypoints(const vector<float> &keypoints, ModelType type, bool enable_validation, bool enable_interpolation);
  vector<Point> validate_connectivity(const vector<Point> &points, const vector<pair<int, int>> &pairs, float max_dist);
  vector<Point> interpolate_chain(const vector<Point> &points, const vector<int> &chain);
  vector<Point> interpolate_missing_joints(const vector<Point> &points, ModelType type);
};

#endif // OPENPOSE_HPP

