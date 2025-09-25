#include "parameter_handler.hpp"
#include <algorithm>
#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <sstream>

void ParameterHandler::displayHelp() {
  std::cerr
      << "Usage: openpose --input <file> --output <file> [OPTIONS]\n"
      << "Options:\n"
      << "  -i, --input <file>         Required. Path to the input image.\n"
      << "  -o, --output <file>        Optional. Path to save the output image "
         "(default: unified_output.jpg).\n"
      << "  -t, --threshold <value>    Pose keypoint detection threshold "
         "(0.0-1.0, default: 0.1).\n"
      << "  --person-threshold <value> Person detection confidence threshold "
         "for multi-person (default: 0.5).\n"
      << "  --nms-threshold <value>    Non-Maximum Suppression threshold for "
         "multi-person (default: 0.4).\n"
      << "  -b, --blend <factor>       Original image opacity (0.0-1.0, "
         "default: 0.5).\n"
      << "  -m, --models <directory>   Models base directory (default: "
         "./models).\n"
      << "  -M, --mode <modes>         Modes: comma-separated list of 'body', "
         "'hand', 'face', 'all' (default: body).\n"
      << "  -v, --verbose              Enable verbose output.\n"
      << "  --multi-person             Enable multi-person detection (requires "
         "person model).\n"
      << "  --no-validate-connectivity Deactivates filtering of noisy points "
         "(on by default).\n"

      << "  --no-interpolate           Deactivates estimation of missing "
         "joints (on by default).\n"
      << "  -h, --help                 Display this help message.\n";
}

float ParameterHandler::parseFloat(const std::string &s,
                                   const std::string &arg_name) {
  try {
    float val = std::stof(s);
    return val;
  } catch (const std::invalid_argument &) {
    throw ParameterException("Invalid numeric value for " + arg_name + ": " +
                             s);
  } catch (const std::out_of_range &) {
    throw ParameterException("Value for " + arg_name +
                             " is out of range: " + s);
  }
}

ProgramOptions ParameterHandler::parse(int argc, char **argv) {
  ProgramOptions opts;
  int opt;
  int option_index = 0;

  static struct option long_options[] = {
      {"input", required_argument, 0, 'i'},
      {"output", required_argument, 0, 'o'},
      {"blend", required_argument, 0, 'b'},
      {"threshold", required_argument, 0, 't'},
      {"person-threshold", required_argument, 0, 1001},
      {"nms-threshold", required_argument, 0, 1002},
      {"models", required_argument, 0, 'm'},
      {"mode", required_argument, 0, 'M'},
      {"verbose", no_argument, 0, 'v'},
      {"multi-person", no_argument, 0, 1003},
      {"no-validate-connectivity", no_argument, 0, 1004},
      {"no-interpolate", no_argument, 0, 1005},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  bool mode_was_set = false;
  while ((opt = getopt_long(argc, argv, "i:o:b:t:m:M:vh", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'i':
      opts.image_path = optarg;
      break;
    case 'o':
      opts.output_file = optarg;
      break;
    case 'b':
      opts.blend_factor = parseFloat(optarg, "--blend");
      break;
    case 't':
      opts.threshold = parseFloat(optarg, "--threshold");
      break;
    case 'm':
      opts.models_dir = optarg;
      break;
    case 'v':
      opts.verbose = true;
      break;
    case 'M': {
      mode_was_set = true;
      opts.process_body = false; // Reset defaults
      opts.process_hand = false;
      opts.process_face = false;
      std::string modes_str(optarg);
      std::stringstream ss(modes_str);
      std::string mode;
      while (getline(ss, mode, ',')) {
        mode.erase(0, mode.find_first_not_of(" \t\n\r"));
        mode.erase(mode.find_last_not_of(" \t\n\r") + 1);
        if (mode == "body")
          opts.process_body = true;
        else if (mode == "hand" || mode == "hands")
          opts.process_hand = true;
        else if (mode == "face")
          opts.process_face = true;
        else if (mode == "all") {
          opts.process_body = true;
          opts.process_hand = true;
          opts.process_face = true;
        } else {
          throw ParameterException("Invalid mode specified: " + mode);
        }
      }
      break;
    }
    case 'h':
      displayHelp();
      exit(0);
    case 1001:
      opts.person_threshold = parseFloat(optarg, "--person-threshold");
      break;
    case 1002:
      opts.nms_threshold = parseFloat(optarg, "--nms-threshold");
      break;
    case 1003:
      opts.multi_person = true;
      break;
    case 1004:
      opts.enable_validation = false;
      break;
    case 1005:
      opts.enable_interpolation = false;
      break;
    case '?':
      // getopt_long already printed an error message.
      throw ParameterException("Unknown or malformed option.");
    default:
      break;
    }
  }

  if (opts.image_path.empty()) {
    throw ParameterException("--input is a required argument.");
  }
  if (!std::filesystem::exists(opts.image_path)) {
    throw ParameterException("--input file does not exists");
  }

  // If --mode was not used, default to body. If it was used but no valid modes
  // were found, also default to body.
  if (!mode_was_set ||
      (!opts.process_body && !opts.process_hand && !opts.process_face)) {
    opts.process_body = true;
  }

  return opts;
}