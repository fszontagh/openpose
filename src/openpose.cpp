#include "openpose.hpp"


void OpenPose::runMultiPersonPipeline(Mat& img, Mat& overlay){
    if (args.verbose) cout << "Verbose: Multi-person mode enabled." << endl;
    vector<Rect> person_rects = detectPersons(person_net, img, args.person_threshold, args.nms_threshold, args.verbose);
    if (args.verbose) cout << "Verbose: Found " << person_rects.size() << " person(s) after NMS." << endl;

    if (person_rects.empty()) {
        cout << "No persons detected in the image." << endl;
        return;
    }

    bool process_body = (args.mode == Mode::BODY || args.mode == Mode::BODY_AND_HAND || args.mode == Mode::ALL);
    bool process_hand = (args.mode == Mode::HAND || args.mode == Mode::BODY_AND_HAND || args.mode == Mode::ALL);
    bool process_face = (args.mode == Mode::FACE || args.mode == Mode::ALL);

    for (size_t i = 0; i < person_rects.size(); ++i) {
        const auto& person_rect = person_rects[i];
        Scalar color = PERSON_COLORS[i % PERSON_COLORS.size()];
        if (args.verbose) cout << "Verbose: Processing person " << (i + 1) << " in ROI: " << person_rect << endl;

        Mat person_crop = img(person_rect);
        vector<float> bodyKeypoints, handKeypoints, faceKeypoints;

        if (process_body) {
            bodyKeypoints = runOpenPose(person_crop, body_net, ModelType::BODY, args.threshold);
            for(size_t j=0; j<bodyKeypoints.size(); j+=3) if(bodyKeypoints[j+2]>0){ bodyKeypoints[j]+=person_rect.x; bodyKeypoints[j+1]+=person_rect.y; }
        }
        if (process_hand) {
            // Hand detection now relies on body keypoints from the same person
            Point r_wrist(-1,-1), l_wrist(-1,-1);
            if(!bodyKeypoints.empty() && bodyKeypoints.size()>=25*3){ if(bodyKeypoints[4*3+2]>0)r_wrist=Point(bodyKeypoints[4*3],bodyKeypoints[4*3+1]); if(bodyKeypoints[7*3+2]>0)l_wrist=Point(bodyKeypoints[7*3],bodyKeypoints[7*3+1]); }
            Rect r_h_rect=makeHandRect(r_wrist,368,img), l_h_rect=makeHandRect(l_wrist,368,img);
            vector<float> l_hand, r_hand;
            if(l_h_rect.area()>0){ l_hand=runOpenPose(img(l_h_rect),hand_net,ModelType::HAND,args.threshold); for(size_t j=0;j<l_hand.size();j+=3)if(l_hand[j+2]>0){l_hand[j]+=l_h_rect.x;l_hand[j+1]+=l_h_rect.y;} } else { l_hand=vector<float>(21*3,-1.f); }
            if(r_h_rect.area()>0){ r_hand=runOpenPose(img(r_h_rect),hand_net,ModelType::HAND,args.threshold); for(size_t j=0;j<r_hand.size();j+=3)if(r_hand[j+2]>0){r_hand[j]+=r_h_rect.x;r_hand[j+1]+=r_h_rect.y;} } else { r_hand=vector<float>(21*3,-1.f); }
            handKeypoints.insert(handKeypoints.end(),l_hand.begin(),l_hand.end()); handKeypoints.insert(handKeypoints.end(),r_hand.begin(),r_hand.end());
        }
        if (process_face) {
            auto haar_paths = resolveModelPaths(ModelType::FACE_HAAR);
            CascadeClassifier face_cascade;
            if(face_cascade.load(haar_paths["haarcascade"])) {
                vector<Rect> faces;
                Mat gray;
                cvtColor(person_crop, gray, COLOR_BGR2GRAY);
                equalizeHist(gray, gray);
                face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));

                if(!faces.empty()){
                    Rect face_rect_local = faces[0];
                    Rect face_rect_global = face_rect_local + person_rect.tl();

                    faceKeypoints = runOpenPose(img(face_rect_global), face_net, ModelType::FACE, args.threshold);
                    for (size_t j = 0; j < faceKeypoints.size(); j += 3)
                        if (faceKeypoints[j + 2] > 0) {
                        faceKeypoints[j] += face_rect_global.x;
                        faceKeypoints[j + 1] += face_rect_global.y;
                        }
                } else { faceKeypoints = vector<float>(70 * 3, -1.f); }
            } else { cerr << "Error: Could not load Haar Cascade model for face." << endl; faceKeypoints = vector<float>(70*3, -1.f); }
        }

        // Post-processing and drawing for this person
        if (process_body) { bodyKeypoints=postProcessKeypoints(bodyKeypoints,ModelType::BODY,args.enable_validation,args.enable_interpolation); drawKeypoints(overlay,bodyKeypoints,ModelType::BODY,color); }
        if (process_hand) { handKeypoints=postProcessKeypoints(handKeypoints,ModelType::HAND,args.enable_validation,args.enable_interpolation); drawKeypoints(overlay,handKeypoints,ModelType::HAND,color); }
        if (process_face) { faceKeypoints=postProcessKeypoints(faceKeypoints,ModelType::FACE,false,false); drawKeypoints(overlay,faceKeypoints,ModelType::FACE,color); }
    }
}

void OpenPose::runSinglePersonPipeline(Mat& img, Mat& overlay){
    if (args.verbose) cout << "Verbose: Single-person mode enabled." << endl;
    vector<float> bodyKeypoints, handKeypoints, faceKeypoints;

    bool process_body = (args.mode == Mode::BODY || args.mode == Mode::BODY_AND_HAND || args.mode == Mode::ALL);
    bool process_hand = (args.mode == Mode::HAND || args.mode == Mode::BODY_AND_HAND || args.mode == Mode::ALL);
    bool process_face = (args.mode == Mode::FACE || args.mode == Mode::ALL);

    if (process_body) {
        bodyKeypoints = runOpenPose(img, body_net, ModelType::BODY, args.threshold);
    }
    if (process_hand) {
        auto hand_cascade_paths = resolveModelPaths(ModelType::HAND_HAAR);
        CascadeClassifier hand_cascade;
        if(hand_cascade.load(hand_cascade_paths["haarcascade"])){
            vector<Rect> hands = detectHands(hand_cascade, img);
            vector<float> all_hands_kps;
            for(const auto& hand_rect : hands){
                vector<float> single_hand_kps = runOpenPose(img(hand_rect), hand_net, ModelType::HAND, args.threshold);
                for(size_t i = 0; i < single_hand_kps.size(); i += 3){
                    if(single_hand_kps[i+2] > 0){
                        single_hand_kps[i] += hand_rect.x;
                        single_hand_kps[i+1] += hand_rect.y;
                    }
                }
                all_hands_kps.insert(all_hands_kps.end(), single_hand_kps.begin(), single_hand_kps.end());
            }
            handKeypoints = all_hands_kps;
        } else {
            cerr << "Error: Could not load Haar Cascade for hands." << endl;
        }
    }
    if (process_face) {
        auto haar_paths = resolveModelPaths(ModelType::FACE_HAAR);
        CascadeClassifier face_cascade;
        if(face_cascade.load(haar_paths["haarcascade"])) {
            vector<Rect> faces;
            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);
            face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));

            if(!faces.empty()){
                Rect face_rect = faces[0];
                faceKeypoints = runOpenPose(img(face_rect), face_net, ModelType::FACE, args.threshold);
                for (size_t i = 0; i < faceKeypoints.size(); i += 3)
                    if (faceKeypoints[i + 2] > 0) {
                    faceKeypoints[i] += face_rect.x;
                    faceKeypoints[i + 1] += face_rect.y;
                    }
            } else { faceKeypoints = vector<float>(70 * 3, -1.f); }
        } else { cerr << "Error: Could not load Haar Cascade model." << endl; faceKeypoints = vector<float>(70*3, -1.f); }
    }
    if (process_body) {
        bodyKeypoints = postProcessKeypoints(bodyKeypoints, ModelType::BODY, args.enable_validation, args.enable_interpolation);
        drawKeypoints(overlay, bodyKeypoints, ModelType::BODY, Scalar(255, 0, 0));
    }
    if (process_hand) {
        handKeypoints = postProcessKeypoints(handKeypoints, ModelType::HAND, args.enable_validation, args.enable_interpolation);
        drawKeypoints(overlay, handKeypoints, ModelType::HAND, Scalar(0, 255, 0));
    }
    if (process_face) {
        faceKeypoints = postProcessKeypoints(faceKeypoints, ModelType::FACE, false, false);
        drawKeypoints(overlay, faceKeypoints, ModelType::FACE, Scalar(0, 255, 255));
    }
}

// Implementations of all other helper methods
vector<Rect> OpenPose::detectPersons(Net &net, const Mat &frame, float conf_threshold, float nms_threshold, bool verbose) {
    int frame_height = frame.rows;
    int frame_width = frame.cols;
    Mat blob = blobFromImage(frame, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(blob);
    Mat output = net.forward();
    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    vector<Rect> boxes;
    vector<float> confidences;

    for (int i = 0; i < detectionMat.rows; i++) {
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);

        if (class_id == 15 && confidence > conf_threshold) {
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame_width);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame_height);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame_width - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * frame_height - box_y);
            boxes.push_back(Rect(box_x, box_y, box_width, box_height));
            confidences.push_back(confidence);
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    vector<Rect> persons;
    for (int idx : indices) {
        persons.push_back(boxes[idx] & Rect(0, 0, frame_width, frame_height));
    }
    return persons;
}

vector<float> OpenPose::runOpenPose(const cv::Mat &image, Net &net, ModelType type, float threshold) {
  const int inHeight = 368;
  if (image.empty() || image.rows <= 0 || image.cols <= 0) {
    int nPoints = (type == ModelType::BODY) ? 25 : (type == ModelType::HAND ? 21 : 70);
    return vector<float>(nPoints * 3, -1.0f);
  }
  int inWidth = (int)round(((float)inHeight / image.rows) * image.cols);
  inWidth = (int)round((float)inWidth / 16.0) * 16;
  if (inWidth < 1) inWidth = 1;
  Mat inpBlob = blobFromImage(image, 1.0 / 255.0, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
  net.setInput(inpBlob);
  Mat output = net.forward();
  int mapH = output.size[2], mapW = output.size[3];
  int nPoints = (type == ModelType::BODY) ? 25 : (type == ModelType::HAND ? 21 : 70);
  if (mapH <= 0 || mapW <= 0) { return vector<float>(nPoints * 3, -1.0f); }

  vector<tuple<Point, float>> points(nPoints, make_tuple(Point(-1, -1), -1.0f));
  for (int n = 0; n < nPoints; ++n) {
    Mat prob(mapH, mapW, CV_32F, output.ptr(0, n));
    Point maxLoc; double probVal;
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
  for (auto &t : points) { auto [p, conf] = t; keypoints.push_back(p.x); keypoints.push_back(p.y); keypoints.push_back(conf); }
  return keypoints;
}

void OpenPose::drawKeypoints(cv::Mat &image, const vector<float> &keypoints, ModelType type, Scalar color) {
  vector<Point> points;
  for (size_t i = 0; i < keypoints.size(); i += 3) {
    if (keypoints[i + 2] > 0) points.push_back(Point(keypoints[i], keypoints[i + 1]));
    else points.push_back(Point(-1, -1));
  }
  const vector<pair<int, int>> *pairs;
  if (type == ModelType::BODY) pairs = &POSE_PAIRS_BODY_25;
  else if (type == ModelType::FACE) pairs = &POSE_PAIRS_FACE;
  else {
    static vector<pair<int, int>> HAND_PAIRS; // Combined for simplicity
    HAND_PAIRS.clear();
    HAND_PAIRS.insert(HAND_PAIRS.end(), HAND_LEFT_PAIRS.begin(), HAND_LEFT_PAIRS.end());
    // Right hand pairs need offset if keypoints are concatenated
    if (keypoints.size() > 21*3) {
        vector<pair<int,int>> right_offset;
        for(const auto& p : HAND_RIGHT_PAIRS) right_offset.push_back({p.first-21, p.second-21}); // Example offset logic
         // This logic is complex and better handled by splitting hand keypoints before drawing
    }
    pairs = &HAND_PAIRS;
  }
  int lineThickness = max(1, (image.cols + image.rows) / 400);
  for (const auto &p : *pairs) {
    if (p.first >= points.size() || p.second >= points.size()) continue;
    Point pA = points[p.first]; Point pB = points[p.second];
    if (pA.x > 0 && pB.x > 0) line(image, pA, pB, color, lineThickness, LINE_AA);
  }
  for (const auto &point : points) { if (point.x > 0) circle(image, point, lineThickness + 2, color, FILLED, LINE_AA); }
}


Rect OpenPose::makeHandRect(Point wrist, int box_size, const Mat &img) {
  if (wrist.x < 0 || wrist.y < 0) return Rect();
  int x = wrist.x - box_size / 2;
  int y = wrist.y - box_size / 2;
  int x_clamped = max(0, x);
  int y_clamped = max(0, y);
  int w = min(box_size, img.cols - x_clamped);
  int h = min(box_size, img.rows - y_clamped);
  return Rect(x_clamped, y_clamped, w, h);
}

vector<Rect> OpenPose::detectHands(CascadeClassifier& cascade, const Mat& img) {
    vector<Rect> hands;
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    cascade.detectMultiScale(gray, hands, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(80, 80));
    return hands;
}


vector<float> OpenPose::postProcessKeypoints(const vector<float> &keypoints, ModelType type, bool enable_validation, bool enable_interpolation) {
    vector<Point> points;
    for (size_t i = 0; i < keypoints.size(); i += 3) {
        if (keypoints[i + 2] > 0) points.push_back(Point(keypoints[i], keypoints[i + 1]));
        else points.push_back(Point(-1, -1));
    }

    vector<Point> final_points = points;
    if (type == ModelType::BODY && enable_interpolation) final_points = interpolate_missing_joints(final_points, type);
    if (type == ModelType::BODY && enable_validation) {
        float dynamic_max_dist = 150.0f;
        if (final_points.size() > 8 && final_points[1].x > 0 && final_points[8].x > 0) {
            dynamic_max_dist = norm(final_points[1] - final_points[8]) * 1.5f;
        }
        final_points = validate_connectivity(final_points, POSE_PAIRS_BODY_25, dynamic_max_dist);
    }
    // Simplified post-processing for hands and face for now

    vector<float> processed_keypoints;
    for (auto &p : final_points) {
        processed_keypoints.push_back(p.x);
        processed_keypoints.push_back(p.y);
        processed_keypoints.push_back((p.x >= 0 && p.y >= 0) ? 1.0f : -1.0f);
    }
    return processed_keypoints;
}

vector<Point> OpenPose::validate_connectivity(const vector<Point> &points, const vector<pair<int, int>> &pairs, float max_dist) {
    vector<Point> cleaned_points = points;
    for (const auto &pair : pairs) {
        if (pair.first < cleaned_points.size() && pair.second < cleaned_points.size() &&
            cleaned_points[pair.first].x > 0 && cleaned_points[pair.second].x > 0) {
            if (norm(cleaned_points[pair.first] - cleaned_points[pair.second]) > max_dist) {
                cleaned_points[pair.second] = Point(-1, -1);
            }
        }
    }
    return cleaned_points;
}

vector<Point> OpenPose::interpolate_chain(const vector<Point> &points, const vector<int> &chain) {
    vector<Point> interp = points;
    for (size_t k = 0; k < chain.size() - 2; ++k) {
        int start = chain[k], middle = chain[k+1], end = chain[k+2];
        if (start < interp.size() && middle < interp.size() && end < interp.size() &&
            interp[start].x > 0 && interp[end].x > 0 && interp[middle].x < 0) {
            interp[middle] = (interp[start] + interp[end]) * 0.5;
        }
    }
    return interp;
}

vector<Point> OpenPose::interpolate_missing_joints(const vector<Point> &points, ModelType type) {
    if (type != ModelType::BODY) return points;
    vector<Point> interpolated = points;
    const vector<tuple<int, int, int>> limb_triplets = {{2,3,4}, {5,6,7}, {9,10,11}, {12,13,14}};
    for (const auto& t : limb_triplets) {
        int s = get<0>(t), m = get<1>(t), e = get<2>(t);
        if (s < interpolated.size() && m < interpolated.size() && e < interpolated.size() &&
            interpolated[s].x > 0 && interpolated[e].x > 0 && interpolated[m].x < 0) {
            interpolated[m] = (interpolated[s] + interpolated[e]) * 0.5;
        }
    }
    return interpolated;
}
