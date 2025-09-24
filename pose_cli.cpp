// pose_cli.cpp
// Universal CLI that automatically detects COCO and BODY_25 models.
// Extended with validation and interpolation logic.

#include <cmath> // round()
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

// OpenPose 18-point COCO model pairs
const vector<pair<int, int>> POSE_PAIRS_COCO = {
    {1, 2}, {1, 5},  {2, 3},   {3, 4},  {5, 6},   {6, 7},
    {1, 8}, {8, 9},  {9, 10},  {1, 11}, {11, 12}, {12, 13},
    {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};

// OpenPose 25-point BODY_25 model pairs (VISUALIZATION-FRIENDLY)
const vector<pair<int, int>> POSE_PAIRS_BODY_25 = {
    {1, 8}, {1, 2},   {1, 5},   {2, 3},   {3, 4},   {5, 6},  {6, 7},
    {8, 9}, {9, 10},  {10, 11}, {8, 12},  {12, 13}, {13, 14}, {1, 0},
    {0, 15},{15, 17}, {0, 16},  {16, 18}, {14, 19}, {19, 20},
    {14, 21}, {11, 22}, {22, 23}, {11, 24}
};

// --- NEW, POST-PROCESSING FUNCTIONS ---

// Creates an adjacency list based on the connections of the points.
vector<vector<int>> build_adjacency_list(const vector<Point> &points, const vector<pair<int, int>> &pairs) {
    vector<vector<int>> adj(points.size());
    for (const auto &pair : pairs) {
        if (points[pair.first].x > 0 && points[pair.second].x > 0) {
            adj[pair.first].push_back(pair.second);
            adj[pair.second].push_back(pair.first);
        }
    }
    return adj;
}

// Deletes points that have too many lines connected to them.
vector<Point> validate_connectivity(const vector<Point> &points, const vector<vector<int>> &adj, int max_connections = 5) {
    vector<Point> cleaned_points = points;
    for (size_t i = 0; i < adj.size(); ++i) {
        if (adj[i].size() > (size_t)max_connections) {
            cout << "VALIDATION: Point deleted (" << i << "), too many connections: " << adj[i].size() << endl;
            cleaned_points[i] = Point(-1, -1);
        }
    }
    return cleaned_points;
}

// Estimates the position of missing joints in the limb chains.
vector<Point> interpolate_missing_joints(const vector<Point> &points) {
    vector<Point> interpolated_points = points;
    // Limb chains (start, middle, end) for the BODY_25 model
    const vector<tuple<int, int, int>> limb_triplets = {
        {2, 3, 4},   // Right arm: RShoulder, RElbow, RWrist
        {5, 6, 7},   // Left arm: LShoulder, LElbow, LWrist
        {9, 10, 11}, // Right lower leg: RKnee, RAnkle, RHeel
        {12, 13, 14} // Left lower leg: LKnee, LAnkle, LHeel
    };

    for (const auto &triplet : limb_triplets) {
        int start_idx = get<0>(triplet);
        int middle_idx = get<1>(triplet);
        int end_idx = get<2>(triplet);

        Point p_start = interpolated_points[start_idx];
        Point p_middle = interpolated_points[middle_idx];
        Point p_end = interpolated_points[end_idx];

        if (p_start.x > 0 && p_end.x > 0 && p_middle.x < 0) {
            Point new_middle_point;
            new_middle_point.x = (p_start.x + p_end.x) / 2;
            new_middle_point.y = (p_start.y + p_end.y) / 2;
            interpolated_points[middle_idx] = new_middle_point;
            cout << "INTERPOLATION: Point created (" << middle_idx << ") between points " << start_idx << " and " << end_idx << "." << endl;
        }
    }
    return interpolated_points;
}

// --- MAIN PROGRAM ---

struct option long_options[] = {
    {"input", required_argument, 0, 'i'},
    {"output", required_argument, 0, 'o'},
    {"blend", required_argument, 0, 'b'},
    {"threshold", required_argument, 0, 't'},
    {"models", required_argument, 0, 'm'},
    {"coco-prototxt", required_argument, 0, 1001},
    {"coco-caffemodel", required_argument, 0, 1002},
    {"body25-prototxt", required_argument, 0, 1003},
    {"body25-caffemodel", required_argument, 0, 1004},
    {"validate-connectivity", no_argument, 0, 2001}, // NEW SWITCH
    {"interpolate", no_argument, 0, 2002},           // NEW SWITCH
    {0, 0, 0, 0}};

static bool parseFloat(const char *s, float &out) {
  try { out = stof(s); return true; } catch (...) { return false; }
}

void print_usage(const char* prog_name) {
    cerr << "Usage: " << prog_name << " --input <file> --output <file> [OPTIONS]\n"
         << "Options:\n"
         << "  -t, --threshold <value>    Detection threshold (0.0-1.0, default: 0.05)\n"
         << "  -b, --blend <factor>       Original image opacity (0.0-1.0, default: 0.5)\n"
         << "  -m, --models <directory>   Models base directory (default: .)\n"
         << "  --coco-prototxt <file>     COCO model prototxt file\n"
         << "  --coco-caffemodel <file>   COCO model caffemodel file\n"
         << "  --body25-prototxt <file>   BODY_25 model prototxt file\n"
         << "  --body25-caffemodel <file> BODY_25 model caffemodel file\n"
         << "  --validate-connectivity    Activates filtering of noisy points\n"
         << "  --interpolate              Activates estimation of missing joints\n";
}

int main(int argc, char **argv) {
    string input_path, output_path, coco_proto, coco_model, body25_proto, body25_model;
    float blend_factor = 0.5f, threshold = 0.05f;
    string base_dir = ".";
    bool enable_validation = false;
    bool enable_interpolation = false;
    int opt;

    while ((opt = getopt_long(argc, argv, "i:o:b:t:m:", long_options, NULL)) != -1) {
        switch (opt) {
        case 'i': input_path = optarg; break;
        case 'o': output_path = optarg; break;
        case 'b': parseFloat(optarg, blend_factor); break;
        case 't': parseFloat(optarg, threshold); break;
        case 'm': base_dir = optarg; break;
        case 1001: coco_proto = optarg; break;
        case 1002: coco_model = optarg; break;
        case 1003: body25_proto = optarg; break;
        case 1004: body25_model = optarg; break;
        case 2001: enable_validation = true; break;
        case 2002: enable_interpolation = true; break;
        default: print_usage(argv[0]); return 1;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Determining the path of model files
    string proto_path, weights_path;
    // ... (the model selection logic remains unchanged) ...
    if (!coco_proto.empty()) { proto_path = coco_proto; weights_path = coco_model; }
    else if (!body25_proto.empty()) { proto_path = body25_proto; weights_path = body25_model; }
    else {
        string body25_proto_default = base_dir + "/body_25/pose_deploy.prototxt";
        string body25_model_default = base_dir + "/body_25/pose_iter_584000.caffemodel";
        if (ifstream(body25_proto_default) && ifstream(body25_model_default)) {
            proto_path = body25_proto_default; weights_path = body25_model_default;
        } else {
            string coco_proto_default = base_dir + "/coco/pose_deploy_linevec.prototxt";
            string coco_model_default = base_dir + "/coco/pose_iter_440000.caffemodel";
            if (ifstream(coco_proto_default) && ifstream(coco_model_default)) {
                proto_path = coco_proto_default; weights_path = coco_model_default;
            } else {
                cerr << "Error: Model files not found.\n"; return 1;
            }
        }
    }

    Mat frame = imread(input_path);
    if (frame.empty()) { cerr << "Error: Could not read the input: " << input_path << "\n"; return 2; }

    Net net = readNetFromCaffe(proto_path, weights_path);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    const int inHeight = 368;
    int inWidth = (int)round(((float)inHeight / frame.rows) * frame.cols);
    inWidth = (int)round((float)inWidth / 16.0) * 16;
    Mat inpBlob = blobFromImage(frame, 1.0 / 255.0, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

    net.setInput(inpBlob);
    Mat output = net.forward();

    int nParts = output.size[1];
    int mapH = output.size[2];
    int mapW = output.size[3];

    int nPoints = 0;
    const vector<pair<int, int>> *posePairs = nullptr;
    bool is_body25 = false;

    if (nParts == 57) {
        cout << "COCO (18-point) model detected." << endl;
        nPoints = 18;
        posePairs = &POSE_PAIRS_COCO;
    } else if (nParts == 78) {
        cout << "BODY_25 (25-point) model detected." << endl;
        nPoints = 25;
        posePairs = &POSE_PAIRS_BODY_25;
        is_body25 = true;
    } else {
        cerr << "Error: Unknown model type! Output channels: " << nParts << endl;
        return 6;
    }

    vector<Point> raw_points(nPoints, Point(-1, -1));
    for (int n = 0; n < nPoints; ++n) {
        Mat prob(mapH, mapW, CV_32F, output.ptr(0, n));
        Point maxLoc;
        double probVal;
        minMaxLoc(prob, 0, &probVal, 0, &maxLoc);
        if (probVal > threshold) {
            float x = (frame.cols * (float)maxLoc.x) / mapW;
            float y = (frame.rows * (float)maxLoc.y) / mapH;
            raw_points[n] = Point((int)x, (int)y);
        }
    }

    // --- POST-PROCESSING ---
    vector<Point> final_points = raw_points;
    if (enable_validation) {
        vector<vector<int>> adj = build_adjacency_list(final_points, *posePairs);
        final_points = validate_connectivity(final_points, adj);
    }
    if (enable_interpolation) {
        if (!is_body25) {
            cout << "Warning: Interpolation is only supported with the BODY_25 model." << endl;
        } else {
            final_points = interpolate_missing_joints(final_points);
        }
    }

    // --- DRAWING ---
    Mat pose_on_black = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
    int lineThickness = max(1, (frame.cols + frame.rows) / 400);

    // Drawing lines (white: original, yellow: interpolated)
    for (const auto &p : *posePairs) {
        Point pA = final_points[p.first];
        Point pB = final_points[p.second];

        if (pA.x > 0 && pB.x > 0) {
            Scalar line_color(0, 255, 255); // Alapértelmezett: Sárga (interpolált)
            if (raw_points[p.first].x > 0 && raw_points[p.second].x > 0) {
                 line_color = Scalar(0, 255, 0); // Ha eredetileg is megvolt: Zöld
            }
            line(pose_on_black, pA, pB, line_color, lineThickness, LINE_AA);
        }
    }
    // Drawing points (red)
    for (const auto &point : final_points) {
        if (point.x > 0) {
            circle(pose_on_black, point, lineThickness + 2, Scalar(0, 0, 255), FILLED, LINE_AA);
        }
    }

    Mat final_image;
    addWeighted(frame, blend_factor, pose_on_black, 1.0 - blend_factor, 0, final_image);

    imwrite(output_path, final_image);
    cout << "Done: " << output_path << "\n";
    return 0;
}
