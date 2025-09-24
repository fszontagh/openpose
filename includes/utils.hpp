#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm> // Needed for std::max_element
#include <cmath>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <map>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp> // Needed for the Haar Cascade face detector
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

enum class ModelType { BODY, HAND, FACE, FACE_HAAR };

enum class Mode { HAND, BODY, BODY_AND_HAND, FACE, ALL };

// OpenPose 25-point BODY_25 model pairs
const vector<pair<int, int>> POSE_PAIRS_BODY_25 = {
    {1, 8},   {1, 2},   {1, 5},   {2, 3},   {3, 4},   {5, 6},
    {6, 7},   {8, 9},   {9, 10},  {10, 11}, {8, 12},  {12, 13},
    {13, 14}, {1, 0},   {0, 15},  {15, 17}, {0, 16},  {16, 18},
    {14, 19}, {19, 20}, {14, 21}, {11, 22}, {22, 23}, {11, 24}};
// Hand pairs for left hand (0-20)
const vector<pair<int, int>> HAND_LEFT_PAIRS = {
    {0, 1},   {1, 2},   {2, 3},  {3, 4},   {0, 5},   {5, 6},  {6, 7},
    {7, 8},   {0, 9},   {9, 10}, {10, 11}, {11, 12}, {0, 13}, {13, 14},
    {14, 15}, {15, 16}, {0, 17}, {17, 18}, {18, 19}, {19, 20}};
// Hand pairs for right hand (21-41)
const vector<pair<int, int>> HAND_RIGHT_PAIRS = {
    {21, 22}, {22, 23}, {23, 24}, {24, 25}, {21, 26}, {26, 27}, {27, 28},
    {28, 29}, {21, 30}, {30, 31}, {31, 32}, {32, 33}, {21, 34}, {34, 35},
    {35, 36}, {36, 37}, {21, 38}, {38, 39}, {39, 40}, {40, 41}};
// Face 70-point model pairs
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

struct Args {
  string image_path;
  string output_file;
  float blend_factor = 0.5f;
  float threshold = 0.1f;
  string models_dir = "./models";
  bool enable_validation = true;
  bool enable_interpolation = true;
  Mode mode = Mode::ALL;
  bool verbose = false; // Verbose mode flag
};

static bool parseFloat(const char *s, float &out) {
  try {
    out = stof(s);
    return true;
  } catch (...) {
    return false;
  }
}

inline Args parseArgs(int argc, char **argv) {
  Args args;
  int opt;
  int option_index = 0;
  static struct option long_options[] = {
      {"input", required_argument, 0, 'i'},
      {"output", required_argument, 0, 'o'},
      {"blend", required_argument, 0, 'b'},
      {"threshold", required_argument, 0, 't'},
      {"models", required_argument, 0, 'm'},
      {"mode", required_argument, 0, 'M'},
      {"verbose", no_argument, 0, 'v'}, // verbose flag
      {"no-validate-connectivity", no_argument, 0, 1},
      {"no-interpolate", no_argument, 0, 2},
      {0, 0, 0, 0}};
  while ((opt = getopt_long(argc, argv, "i:o:b:t:m:M:v", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'i':
      args.image_path = optarg;
      break;
    case 'o':
      args.output_file = optarg;
      break;
    case 'b':
      parseFloat(optarg, args.blend_factor);
      break;
    case 't':
      parseFloat(optarg, args.threshold);
      break;
    case 'm':
      args.models_dir = optarg;
      break;
    case 'v':
      args.verbose = true;
      break; // verbose flag handler
    case 'M':
      if (!strcmp(optarg, "hand"))
        args.mode = Mode::HAND;
      else if (!strcmp(optarg, "body"))
        args.mode = Mode::BODY;
      else if (!strcmp(optarg, "both"))
        args.mode = Mode::BODY_AND_HAND;
      else if (!strcmp(optarg, "face"))
        args.mode = Mode::FACE;
      else if (!strcmp(optarg, "all"))
        args.mode = Mode::ALL;
      else {
        cerr << "Invalid mode: " << optarg << endl;
        exit(1);
      }
      break;
    case 1:
      args.enable_validation = false;
      break;
    case 2:
      args.enable_interpolation = false;
      break;
    default:
      break;
    }
  }
  return args;
}

inline map<string, string>
resolveModelPaths(ModelType type, const string &models_dir = "./models") {
  map<string, string> paths;
  string models_dir_clean = models_dir;
  if (!models_dir_clean.empty() && models_dir_clean.back() == '/')
    models_dir_clean.pop_back();
  string base;
  if (type == ModelType::BODY) {
    base = models_dir_clean + "/pose";
    string body25_proto = base + "/body_25/pose_deploy.prototxt";
    string body25_model = base + "/body_25/pose_iter_584000.caffemodel";
    if (access(body25_proto.c_str(), F_OK) == 0 &&
        access(body25_model.c_str(), F_OK) == 0) {
      paths["prototxt"] = body25_proto;
      paths["caffemodel"] = body25_model;
    } else {
      throw runtime_error("Body model not found: " + body25_proto);
    }
  } else if (type == ModelType::HAND) {
    base = models_dir_clean + "/hand";
    paths["prototxt"] = base + "/pose_deploy.prototxt";
    paths["caffemodel"] = base + "/pose_iter_102000.caffemodel";
    if (access(paths["prototxt"].c_str(), F_OK) != 0 ||
        access(paths["caffemodel"].c_str(), F_OK) != 0) {
      throw runtime_error("Hand model not found: " + paths["prototxt"]);
    }
  } else if (type == ModelType::FACE) {
    base = models_dir_clean + "/face";
    paths["prototxt"] = base + "/pose_deploy.prototxt";
    paths["caffemodel"] = base + "/pose_iter_116000.caffemodel";
    if (access(paths["prototxt"].c_str(), F_OK) != 0 ||
        access(paths["caffemodel"].c_str(), F_OK) != 0) {
      throw runtime_error("Face model not found: " + paths["prototxt"]);
    }
  } else if (type == ModelType::FACE_HAAR) {
    base = models_dir_clean + "/face";
    paths["haarcascade"] = base + "/haarcascade_frontalface_alt.xml";
    if (access(paths["haarcascade"].c_str(), F_OK) != 0) {
      throw runtime_error("Haar Cascade model not found: " +
                          paths["haarcascade"]);
    }
  }
  return paths;
}

inline void loadModels(Net &net, const map<string, string> &paths) {
  net = readNetFromCaffe(paths.at("prototxt"), paths.at("caffemodel"));
  if (net.empty()) {
    throw runtime_error("Failed to load model from " + paths.at("prototxt") +
                        " and " + paths.at("caffemodel"));
  }
  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);
}

inline vector<float> runOpenPose(const cv::Mat &image, Net &net, ModelType type,
                                 float threshold = 0.05f) {
  const int inHeight = 368;
  if (image.empty() || image.rows <= 0 || image.cols <= 0) {
    int nPoints;
    if (type == ModelType::BODY)
      nPoints = 25;
    else if (type == ModelType::HAND)
      nPoints = 21;
    else
      nPoints = 70; // FACE
    return vector<float>(nPoints * 3, -1.0f);
  }
  int inWidth = (int)round(((float)inHeight / image.rows) * image.cols);
  inWidth = (int)round((float)inWidth / 16.0) * 16;
  if (inWidth < 1)
    inWidth = 1;
  Mat inpBlob = blobFromImage(image, 1.0 / 255.0, Size(inWidth, inHeight),
                              Scalar(0, 0, 0), false, false);
  net.setInput(inpBlob);
  Mat output = net.forward();
  int mapH = output.size[2];
  int mapW = output.size[3];
  int nPoints;
  if (type == ModelType::BODY)
    nPoints = 25;
  else if (type == ModelType::HAND)
    nPoints = 21;
  else
    nPoints = 70; // FACE
  if (mapH <= 0 || mapW <= 0) {
    return vector<float>(nPoints * 3, -1.0f);
  }
  vector<tuple<Point, float>> points(nPoints, make_tuple(Point(-1, -1), -1.0f));
  for (int n = 0; n < nPoints; ++n) {
    Mat prob(mapH, mapW, CV_32F, output.ptr(0, n));
    Point maxLoc;
    double probVal;
    minMaxLoc(prob, 0, &probVal, 0, &maxLoc);
    if (probVal > threshold) {
      float x = (image.cols * (float)maxLoc.x) / mapW;
      float y = (image.rows * (float)maxLoc.y) / mapH;
      if (std::isfinite(x) && std::isfinite(y)) {
        points[n] = make_tuple(Point((int)x, (int)y), (float)probVal);
      }
    }
  }
  vector<float> keypoints;
  for (auto &t : points) {
    auto [p, conf] = t;
    keypoints.push_back(p.x);
    keypoints.push_back(p.y);
    keypoints.push_back(conf);
  }
  return keypoints;
}

inline void drawKeypoints(cv::Mat &image, const vector<float> &keypoints,
                          ModelType type, Scalar color = Scalar(255, 0, 0)) {
  vector<Point> points;
  for (size_t i = 0; i < keypoints.size(); i += 3) {
    if (keypoints[i + 2] > 0)
      points.push_back(Point(keypoints[i], keypoints[i + 1]));
    else
      points.push_back(Point(-1, -1));
  }
  const vector<pair<int, int>> *pairs;
  if (type == ModelType::BODY)
    pairs = &POSE_PAIRS_BODY_25;
  else if (type == ModelType::FACE)
    pairs = &POSE_PAIRS_FACE;
  else {
    static vector<pair<int, int>> HAND_PAIRS;
    HAND_PAIRS.clear();
    HAND_PAIRS.insert(HAND_PAIRS.end(), HAND_LEFT_PAIRS.begin(),
                      HAND_LEFT_PAIRS.end());
    HAND_PAIRS.insert(HAND_PAIRS.end(), HAND_RIGHT_PAIRS.begin(),
                      HAND_RIGHT_PAIRS.end());
    pairs = &HAND_PAIRS;
  }
  int lineThickness = max(1, (image.cols + image.rows) / 400);
  for (const auto &p : *pairs) {
    if (p.first >= points.size() || p.second >= points.size())
      continue;
    Point pA = points[p.first];
    Point pB = points[p.second];
    if (pA.x > 0 && pB.x > 0)
      line(image, pA, pB, color, lineThickness, LINE_AA);
  }
  for (const auto &point : points) {
    if (point.x > 0)
      circle(image, point, lineThickness + 2, color, FILLED, LINE_AA);
  }
}

inline Rect makeHandRect(Point wrist, int box_size, const Mat &img) {
  if (wrist.x < 0 || wrist.y < 0)
    return Rect();
  int x = wrist.x - box_size / 2;
  int y = wrist.y - box_size / 2;
  int x_clamped = max(0, x);
  int y_clamped = max(0, y);
  int w = min(box_size, img.cols - x_clamped);
  int h = min(box_size, img.rows - y_clamped);
  return Rect(x_clamped, y_clamped, w, h);
}

inline Rect getFaceRect(const vector<float> &bodyKeypoints, const Mat &img,
                        const string &cascade_path, bool verbose) {
  // 1. Try to find face ROI from body keypoints
  vector<Point> head_points;
  vector<int> head_indices = {0, 15, 16, 17,
                              18}; // Nose, REye, LEye, REar, LEar
  int valid_head_points = 0;
  for (int idx : head_indices) {
    if (bodyKeypoints.size() > idx * 3 + 2 &&
        bodyKeypoints[idx * 3 + 2] > 0.2) { // Stricter confidence
      head_points.push_back(
          Point(bodyKeypoints[idx * 3], bodyKeypoints[idx * 3 + 1]));
      valid_head_points++;
    }
  }

  // Only trust body keypoints if we have enough of them AND they form a
  // reasonably sized box
  if (valid_head_points >= 3) {
    Rect bbox = boundingRect(head_points);
    if (bbox.width > 50 && bbox.height > 50) { // New check for minimum size
      if (verbose)
        cout << "Verbose: Found plausible face ROI using body keypoints."
             << endl;
      bbox.x -= bbox.width * 0.2;
      bbox.y -= bbox.height * 0.3;
      bbox.width *= 1.4;
      bbox.height *= 1.5;
      return bbox & Rect(0, 0, img.cols, img.rows);
    }
  }

  // 2. Fallback to Haar Cascade if body keypoints are insufficient or
  // unreliable
  if (verbose)
    cout << "Verbose: Body keypoints unreliable for face ROI, falling back to "
            "Haar Cascade."
         << endl;
  CascadeClassifier face_cascade;
  if (!face_cascade.load(cascade_path)) {
    cerr << "Error: Could not load Haar Cascade model from: " << cascade_path
         << endl;
    return Rect();
  }
  vector<Rect> faces;
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);
  equalizeHist(gray, gray);
  face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE,
                                Size(50, 50));

  if (!faces.empty()) {
    if (verbose)
      cout << "Verbose: Found " << faces.size()
           << " face(s) using Haar Cascade. Using the largest one." << endl;
    Rect biggest_face = *std::max_element(
        faces.begin(), faces.end(),
        [](const Rect &a, const Rect &b) { return a.area() < b.area(); });
    biggest_face.x -= biggest_face.width * 0.2;
    biggest_face.y -= biggest_face.height * 0.2;
    biggest_face.width *= 1.4;
    biggest_face.height *= 1.4;
    return biggest_face & Rect(0, 0, img.cols, img.rows);
  }

  return Rect(); // Return empty rect if no face was found
}

// Post-processing functions moved before their first use to resolve build
// errors.
inline vector<Point> validate_connectivity(const vector<Point> &points,
                                           const vector<pair<int, int>> &pairs,
                                           float max_dist = 80.0f) {
  vector<Point> cleaned_points = points;
  for (const auto &pair : pairs) {
    int i = pair.first;
    int j = pair.second;
    if (i >= cleaned_points.size() || j >= cleaned_points.size())
      continue;
    if (cleaned_points[i].x > 0 && cleaned_points[j].x > 0) {
      if (norm(cleaned_points[i] - cleaned_points[j]) > max_dist) {
        cleaned_points[j] = Point(-1, -1);
      }
    }
  }
  return cleaned_points;
}

inline vector<Point> interpolate_chain(const vector<Point> &points,
                                       const vector<int> &chain) {
  vector<Point> interp = points;
  for (size_t k = 0; k < chain.size() - 2; ++k) {
    int start = chain[k];
    int middle = chain[k + 1];
    int end = chain[k + 2];
    if (start >= interp.size() || middle >= interp.size() ||
        end >= interp.size())
      continue;
    if (interp[start].x > 0 && interp[end].x > 0 && interp[middle].x < 0) {
      interp[middle].x = (interp[start].x + interp[end].x) / 2;
      interp[middle].y = (interp[start].y + interp[end].y) / 2;
    }
  }
  return interp;
}

inline vector<Point> interpolate_missing_joints(const vector<Point> &points,
                                                ModelType type) {
  if (type == ModelType::BODY) {
    vector<Point> interpolated_points = points;
    const vector<tuple<int, int, int>> limb_triplets = {
        {2, 3, 4}, {5, 6, 7}, {9, 10, 11}, {12, 13, 14}};
    for (const auto &triplet : limb_triplets) {
      int start_idx = get<0>(triplet), middle_idx = get<1>(triplet),
          end_idx = get<2>(triplet);
      if (start_idx >= interpolated_points.size() ||
          middle_idx >= interpolated_points.size() ||
          end_idx >= interpolated_points.size())
        continue;
      Point p_start = interpolated_points[start_idx],
            p_middle = interpolated_points[middle_idx],
            p_end = interpolated_points[end_idx];
      if (p_start.x > 0 && p_end.x > 0 && p_middle.x < 0) {
        interpolated_points[middle_idx].x = (p_start.x + p_end.x) / 2;
        interpolated_points[middle_idx].y = (p_start.y + p_end.y) / 2;
      }
    }
    return interpolated_points;
  }
  return points;
}

inline vector<float> postProcessKeypoints(
    const vector<float> &keypoints, ModelType type, bool enable_validation,
    bool enable_interpolation, bool verbose, Point right_wrist = Point(-1, -1),
    Point left_wrist = Point(-1, -1), float wrist_threshold = 150.0f) {
  vector<Point> points;
  for (size_t i = 0; i < keypoints.size(); i += 3) {
    if (keypoints[i + 2] > 0)
      points.push_back(Point(keypoints[i], keypoints[i + 1]));
    else
      points.push_back(Point(-1, -1));
  }

  vector<Point> final_points;
  if (type == ModelType::BODY) {
    final_points = points;
    if (enable_interpolation)
      final_points = interpolate_missing_joints(final_points, type);
    if (enable_validation) {
      float dynamic_max_dist = 150.0f;
      if (final_points.size() > 8 && final_points[1].x > 0 &&
          final_points[8].x > 0) {
        float torso_height = norm(final_points[1] - final_points[8]);
        dynamic_max_dist = torso_height * 1.5f;
        if (verbose)
          cout << "Verbose: Dynamic validation distance for BODY set to: "
               << dynamic_max_dist << endl;
      }
      final_points = validate_connectivity(final_points, POSE_PAIRS_BODY_25,
                                           dynamic_max_dist);
    }
  } else if (type == ModelType::HAND) {
    vector<Point> final_left(points.begin(), points.begin() + 21);
    if (enable_interpolation) {
      vector<vector<int>> chains = {{0, 1, 2, 3, 4},
                                    {0, 5, 6, 7, 8},
                                    {0, 9, 10, 11, 12},
                                    {0, 13, 14, 15, 16},
                                    {0, 17, 18, 19, 20}};
      for (auto &c : chains)
        final_left = interpolate_chain(final_left, c);
    }
    if (enable_validation)
      final_left = validate_connectivity(final_left, HAND_LEFT_PAIRS, 100.0f);
    vector<Point> final_right(points.begin() + 21, points.end());
    if (enable_interpolation) {
      vector<vector<int>> chains_g = {{21, 22, 23, 24, 25},
                                      {21, 26, 27, 28, 29},
                                      {21, 30, 31, 32, 33},
                                      {21, 34, 35, 36, 37},
                                      {21, 38, 39, 40, 41}};
      vector<vector<int>> chains_l;
      for (const auto &gc : chains_g) {
        vector<int> lc;
        for (int i : gc)
          lc.push_back(i - 21);
        chains_l.push_back(lc);
      }
      for (auto &c : chains_l)
        final_right = interpolate_chain(final_right, c);
    }
    if (enable_validation) {
      vector<pair<int, int>> pairs_l;
      for (const auto &p : HAND_RIGHT_PAIRS)
        pairs_l.push_back({p.first - 21, p.second - 21});
      final_right = validate_connectivity(final_right, pairs_l, 100.0f);
    }
    if (left_wrist.x >= 0)
      for (size_t i = 0; i < final_left.size(); ++i)
        if (final_left[i].x >= 0 &&
            norm(final_left[i] - left_wrist) > wrist_threshold)
          final_left[i] = Point(-1, -1);
    if (right_wrist.x >= 0)
      for (size_t i = 0; i < final_right.size(); ++i)
        if (final_right[i].x >= 0 &&
            norm(final_right[i] - right_wrist) > wrist_threshold)
          final_right[i] = Point(-1, -1);
    final_points = final_left;
    final_points.insert(final_points.end(), final_right.begin(),
                        final_right.end());
  } else if (type == ModelType::FACE) {
    final_points = points;
  }

  vector<float> processed_keypoints;
  for (auto &p : final_points) {
    processed_keypoints.push_back(p.x);
    processed_keypoints.push_back(p.y);
    processed_keypoints.push_back((p.x >= 0 && p.y >= 0) ? 1.0f : -1.0f);
  }
  return processed_keypoints;
}

#endif // UTILS_HPP
