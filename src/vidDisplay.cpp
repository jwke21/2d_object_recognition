/**
 * Jake Van Meter
 * Fall 2023
 * CS 5330
 */
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <unordered_map>
#include <utility>

#include "csv.hpp"
#include "util.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  // 0 is for internal webcam
  // 4 is for external webcam
  cv::VideoCapture* capdev = new cv::VideoCapture(0);

  if (!capdev->isOpened()) {
    cout << "Unable to open the video device" << endl;
    return -1;
  }

  const string filename = "features.csv";

  // get size of input
  const cv::Size refS = cv::Size(capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                                 capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

  // create window for raw image
  cv::namedWindow("Raw Webcam", cv::WindowFlags::WINDOW_AUTOSIZE);

  // create window for thresholded image
  cv::namedWindow("Threshold", cv::WindowFlags::WINDOW_AUTOSIZE);

  // create window for cleaned binary image
  cv::namedWindow("Clean Binary", cv::WindowFlags::WINDOW_AUTOSIZE);

  // create window for segmented image
  cv::namedWindow("Segmented Image", cv::WindowFlags::WINDOW_AUTOSIZE);

  // colors that will be used to display the regions in the segmented image
  const int max_regions = 50;
  vector<cv::Vec3b> colors(max_regions);
  for (int i = 0; i < max_regions; i++) {
    colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
  }

  // source frame from webcam or read from file
  cv::Mat src(refS, CV_8UC3);

  // features of training data
  vector<vector<string>> db_rows_str;
  csv::read_all_rows(filename, db_rows_str);
  // db will map image label to feature object
  vector<pair<string, Features>> db(db_rows_str.size());
  // unordered_map<string, Features> db;
  for (int i = 0; i < db_rows_str.size(); i++) {
    db[i] = make_pair(db_rows_str[i][0], Features());
    db[i].second.minBBRatio = stod(db_rows_str[i][1]);
    // db[i].second.minBBArea = stod(db_rows_str[i][2]);
    db[i].second.maxBBPercentFilled = stod(db_rows_str[i][2]);
    db[i].second.mu22a = stod(db_rows_str[i][3]);
    for (int j = 0; j < 7; j++) {
      db[i].second.hu[j] = stod(db_rows_str[i][4 + j]);
    }
  }

  // calculate the mean of each feature
  Features means;
  for (auto r : db) {
    means.minBBRatio += r.second.minBBRatio;
    // means.minBBArea += r.second.minBBArea;
    means.maxBBPercentFilled += r.second.maxBBPercentFilled;
    means.mu22a += r.second.mu22a;
    for (int i = 0; i < 7; i++) {
      means.hu[i] += r.second.hu[i];
    }
  }
  means.minBBRatio /= db.size();
  // means.minBBArea /= db.size();
  means.maxBBPercentFilled /= db.size();
  means.mu22a /= db.size();
  for (int i = 0; i < 7; i++) {
    means.hu[i] /= db.size();
  }

  // calculate the standard deviation of each feature
  Features stddev;
  for (auto r : db) {
    stddev.minBBRatio += pow(r.second.minBBRatio - means.minBBRatio, 2);
    // stddev.minBBArea += pow(r.second.minBBArea - means.minBBArea, 2);
    stddev.maxBBPercentFilled +=
        pow(r.second.maxBBPercentFilled - means.maxBBPercentFilled, 2);
    stddev.mu22a += pow(r.second.mu22a - means.mu22a, 2);
    for (int i = 0; i < 7; i++) {
      stddev.hu[i] += pow(r.second.hu[i] - means.hu[i], 2);
    }
  }
  // std = sqrt(sum((x - mean)^2) / (n - 1))
  stddev.minBBRatio = sqrt(stddev.minBBRatio / (db.size() - 1));
  // stddev.minBBArea = sqrt(stddev.minBBArea / (db.size() - 1));
  stddev.maxBBPercentFilled = sqrt(stddev.maxBBPercentFilled / (db.size() - 1));
  stddev.mu22a = sqrt(stddev.mu22a / (db.size() - 1));
  for (int i = 0; i < 7; i++) {
    stddev.hu[i] = sqrt(stddev.hu[i] / (db.size() - 1));
  }

  // booleans determining state of program

  bool aolcm = false;  // axis of least central moment
  bool term = false;
  bool minBB = false;
  bool maxBB = false;
  bool classify = false;

  // predicted label made of region in classify mode
  string pred;

  // enum for different types of classifiers
  enum Classifier { NEAREST_NEIGHBOR, KNN };
  int classifier;

  // performance object to track confusion matrix
  Performance perf = Performance();

  while (!term) {
    *capdev >> src;  // read frame from webcam
    // src = cv::imread("imgs/img4P3.png");
    // src = cv::imread("../imgs/img3P3.png");

    // blur the image
    // blur5x5(src, src);
    cv::GaussianBlur(src, src, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);

    cv::imshow("Raw Webcam", src);

    /* Task 1: Threshold the image */

    cv::Mat thresh;
    // https://docs.opencv.org/4.x/db/d8e/tutorial_threshold.html
    // cv::threshold(src, thresh, 100, 255, cv::THRESH_BINARY_INV);
    bin_threshold(src, thresh, 100);

    // display thresholded image
    cv::imshow("Threshold", thresh);

    /* Task 2: Clean up the binary image (using grassfire) */

    cv::Mat grass;
    grassfire(thresh, grass);

    cv::imshow("Clean Binary", grass);

    /* Task 3: Segment the image into regions */

    cv::Mat seg;
    cv::Mat stats;
    cv::Mat centroids;
    // https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
    int n = cv::connectedComponentsWithStats(
        thresh, seg, stats, centroids, 8, CV_32S,
        cv::ConnectedComponentsAlgorithmsTypes::CCL_WU);
    // region_growing_segmentation(grass, regionmap);

    vector<Region> regions;
    segment(seg, seg, stats, centroids, colors, regions);

    cv::imshow("Segmented Image", seg);

    /* Task 4: Calculate the features for kept regions */

    for (auto r : regions) {
      r.computeFeatures(seg);

      // draw the bounding boxes if user wants to

      if (minBB) {
        r.drawMinBoundingBox(seg);
        r.drawMinBoundingBox(src);
      }
      if (maxBB) {
        r.drawMaxBoundingBox(seg);
        r.drawMaxBoundingBox(src);
      }
      if (aolcm) {
        r.drawAxisOfLeastCentralMoment(seg);
        r.drawAxisOfLeastCentralMoment(src);
      }
    }

    /* Task 6: classify */

    if (classify) {
      // classify the regions
      for (auto r : regions) {
        // compute the distance to each training image
        vector<pair<double, string>> distances;
        for (auto row : db) {
          distances.push_back(make_pair(
              compute_distance(row.second, *r.feats, stddev), row.first));
        }

        // find the minimum distance

        // use nearest neighbor classifier
        if (classifier == NEAREST_NEIGHBOR) {
          pred = nearest_neighbor(distances);
        }
        if (classifier == KNN) {
          // use knn classifier
          pred = knn(distances, 3);
        }

        // draw the label on the image
        r.drawLabel(seg, pred);
        r.drawLabel(src, pred);
      }
    }

    cv::imshow("Segmented Image", seg);
    cv::imshow("Raw Webcam", src);

    int key = cv::waitKey(20);
    switch (key) {
      case 'A':
      case 'a': {
        if (aolcm) {
          cout << "Axis of least central moment mode disabled" << endl;
        } else {
          cout << "Axis of least central moment mode enabled" << endl;
        }
        aolcm = !aolcm;
        break;
      }
      case 'B': {
        if (maxBB) {
          cout << "Max bounding box mode disabled" << endl;
        } else {
          cout << "Max bounding box mode enabled" << endl;
        }
        maxBB = !maxBB;
        break;
      }
      case 'b': {
        if (minBB) {
          cout << "Min bounding box mode disabled" << endl;
        } else {
          cout << "Min bounding box mode enabled" << endl;
        }
        minBB = !minBB;
        break;
      }
      case 'C':
      case 'c': {
        if (classify) {
          cout << "Classify mode disabled" << endl;
        } else {
          cout << "Classify mode enabled" << endl;

          cout << "Which classifier would you like to use?\n\n";
          cout << "\t[1] Nearest Neighbor\n";
          cout << "\t[2] K-Nearest Neighbor (K=3)\n" << endl;

          while (!(cin >> classifier) || classifier < 1 || classifier > 2) {
            cout << "Invalid choice. Please enter 1 or 2: ";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
          }

          classifier--;
        }

        classify = !classify;
        break;
      }
      case 'E':
      case 'e': {
        if (regions.size() > 1) {
          cout << "Can only evaluate performance for one region at a time"
               << endl;
          break;
        }
        if (!classify) {
          cout << "Must be in classify mode to evaluate performance" << endl;
          break;
        }

        string gt;

        cout << "Enter the ground truth label for the image: ";
        cin >> gt;

        if (gt == pred) {
          perf.true_pos++;
          perf.true_neg += 9;
        } else if (gt != pred) {
          perf.false_pos++;
          perf.false_neg++;
          perf.true_neg += 8;
        }

        cout << "Would you like to save the performance metrics to "
                "performance.csv? (y/n): ";

        char save_choice;
        while (!(cin >> save_choice) ||
               (save_choice != 'y' && save_choice != 'Y' &&
                save_choice != 'n' && save_choice != 'N')) {
          cout << "Invalid choice. Please enter y or n: ";
          cin.clear();
          cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }

        if (save_choice == 'y' || save_choice == 'Y') {
          vector<vector<string>> rows(4);
          string f = "performance.csv";

          rows[0].push_back("true_pos");
          rows[0].push_back(to_string(perf.true_pos));
          rows[1].push_back("false_pos");
          rows[1].push_back(to_string(perf.false_pos));
          rows[2].push_back("true_neg");
          rows[2].push_back(to_string(perf.true_neg));
          rows[3].push_back("false_neg");
          rows[3].push_back(to_string(perf.false_neg));

          csv::write_rows(f, rows, true);

          cout << "Performance metrics saved to " << f << endl;

          perf.true_pos = 0;
          perf.false_pos = 0;
          perf.true_neg = 0;
          perf.false_neg = 0;
        }

        break;
      }
      case 'N':
      case 'n': {
        if (regions.size() != 1) {
          cout << "Can only save features for one region at a time" << endl;
          break;
        }
        // save the features
        vector<string> row;
        string label;
        cout << "Enter the label for the image: ";
        cin >> label;

        Features* f = regions[0].feats;

        row.push_back(label);
        row.push_back(to_string(f->minBBRatio));
        // row.push_back(to_string(f->minBBArea));
        row.push_back(to_string(f->maxBBPercentFilled));
        row.push_back(to_string(f->mu22a));
        for (int i = 0; i < 7; i++) {
          row.push_back(to_string(f->hu[i]));
        }

        csv::write_row(filename, row, false);

        cout << "Saved features for region " << label << " to " << filename
             << endl;

        break;
      }
      case 'Q':
      case 'q': {
        term = true;
        break;
      }
      case 'S':
      case 's': {
        cout << "Saving image..." << endl;
        cout << "Enter the label for the image: ";

        string label;
        cin >> label;
        cv::imwrite("imgs/" + label + "_raw.png", src);
        cv::imwrite("imgs/" + label + "_thresholded.png", thresh);
        cv::imwrite("imgs/" + label + "_clean_binary.png", grass);
        cv::imwrite("imgs/" + label + "_segmented.png", seg);

        cout << "Saved image to imgs/" << label << endl;
        break;
      }
      default:
        for (auto r : regions) {
          delete r.feats;
        }
        break;
    }
  }

  delete capdev;

  return 0;
}
