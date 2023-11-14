/**
 * Jake Van Meter
 * Fall 2023
 * CS 5330
 */

#include "util.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stack>
#include <utility>
#include <vector>

#include "disjoint_set.hpp"

using namespace std;

Region::Region(int label, const cv::Mat& stats, const cv::Mat& centroids,
               cv::Vec3b color, const cv::Mat& img) {
  this->label = label;
  this->left = stats.at<int>(label, cv::CC_STAT_LEFT);
  this->top = stats.at<int>(label, cv::CC_STAT_TOP);
  this->width = stats.at<int>(label, cv::CC_STAT_WIDTH);
  this->height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
  this->area = stats.at<int>(label, cv::CC_STAT_AREA);

  this->cx = centroids.at<double>(label, 0);
  this->cy = centroids.at<double>(label, 1);

  this->color = color;

  feats = new Features();

  getPointsFromImage(img);
}

void Region::getPointsFromImage(const cv::Mat& img) {
  for (int i = top; i < top + height; i++) {
    for (int j = left; j < left + width; j++) {
      // check that pixel is part of foreground
      if (img.at<int>(i, j) != 0) {
        points.push_back(cv::Point(j, i));
      }
    }
  }
}

void Region::colorRegion(const cv::Mat& region_map, cv::Mat& dst) {
  for (int i = top; i < top + height; i++) {
    for (int j = left; j < left + width; j++) {
      // check that pixel is part of foreground
      if (region_map.at<int>(i, j) != 0) {
        dst.at<cv::Vec3b>(i, j) = color;
      }
    }
  }
}

void Region::colorRegion(cv::Mat& dst) {
  int channels = dst.channels();
  for (auto p : points) {
    if (channels == 1) {
      dst.at<uchar>(p.y, p.x) = 255;
    } else {
      dst.at<cv::Vec3b>(p.y, p.x) = color;
      ;
    }
  }
}

void Region::drawMaxBoundingBox(cv::Mat& img) {
  for (int i = top; i < top + height; i++) {
    for (int j = left; j < left + width; j++) {
      // top and bottom of box
      if (i == top || i == top + height - 1) {
        img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
        continue;
      }
      // left and right of box
      img.at<cv::Vec3b>(i, left) = cv::Vec3b(0, 0, 255);
      img.at<cv::Vec3b>(i, left + width - 1) = cv::Vec3b(0, 0, 255);
    }
  }
}

void Region::drawMinBoundingBox(cv::Mat& img) {
  cv::Point2f vertices[4];
  cv::RotatedRect minbb = cv::minAreaRect(points);

  minbb.points(vertices);

  // https://docs.opencv.org/4.x/df/dee/samples_2cpp_2minarea_8cpp-example.html#a13
  for (int i = 0; i < 4; i++) {
    cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 1,
             cv::LINE_AA);
  }
}

void Region::drawAxisOfLeastCentralMoment(cv::Mat& img) {
  // get min bounding box
  cv::Point2f vertices[4];
  cv::RotatedRect minbb = cv::minAreaRect(points);

  minbb.points(vertices);

  // calculate the central axis angle
  double alpha = 0.5 * atan2(2 * mu.mu11, mu.mu20 - mu.mu02);

  // calculate the second moment about the central axis
  double beta = alpha * M_1_PI / 2;

  // calculate the length of the axis
  double len = sqrt(feats->mu22a);

  // calculate the endpoints of the axis
  int h = max(minbb.size.height, minbb.size.width) / 2;
  cv::Point2f p1 = cv::Point2f(cx + h * cos(alpha), cy + h * sin(alpha));
  cv::Point2f p2 = cv::Point2f(cx - h * cos(alpha), cy - h * sin(alpha));

  // draw the axis
  cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
}

void Region::drawCentroid(cv::Mat& img) {
  // draw a circle at the centroid
  cv::circle(img, cv::Point(cx, cy), 5, cv::Scalar(0, 0, 255), cv::FILLED);
}

void Region::drawLabel(cv::Mat& img, const string label) {
  // draw the label above the region
  cv::putText(img, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
}

void Region::computeFeatures(const cv::Mat& img) {
  // find the oriented bounding box
  // https://docs.opencv.org/4.x/df/dee/samples_2cpp_2minarea_8cpp-example.html#a13

  cv::Point2f vertices[4];
  cv::RotatedRect minbb = cv::minAreaRect(points);

  // calculate features related to bounding boxes
  feats->minBBRatio = max(minbb.size.height, minbb.size.width) /
                      min(minbb.size.height, minbb.size.width);
  // feats->minBBArea = minbb.size.height * minbb.size.width;
  for (int i = 0; i < 4; i++) {
    double total = height * width;
    feats->maxBBPercentFilled = total / (double)points.size();
  }

  // find the contours
  // https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0

  cv::Mat tmp = cv::Mat::zeros(img.size(), CV_8UC1);
  // cv::cvtColor(img, tmp, cv::COLOR_BGR2GRAY);
  colorRegion(tmp);

  vector<vector<cv::Point>> contours;
  cv::findContours(tmp, contours, cv::RETR_CCOMP, cv::RETR_FLOODFILL);

  // calculate the moments
  // https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga556a180f43cab22649c23ada36a8a139
  // https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html
  // https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html

  mu = cv::moments(contours[0]);

  // calculate the central axis angle
  double alpha = 0.5 * atan2(2 * mu.mu11, mu.mu20 - mu.mu02);
  // calculate the second moment about the central axis
  double beta = alpha * M_1_PI / 2;
  feats->mu22a = mu.mu20 * pow(cos(beta), 2) + mu.mu02 * pow(sin(beta), 2) -
                 mu.mu11 * sin(2 * beta);

  // calculate the Hu moments (invariants)
  // https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944
  cv::HuMoments(mu, feats->hu);

  // cout << "central axis angle: " << alpha << endl;
  // cout << minbb.angle << endl;
}

void bin_threshold(const cv::Mat& src, cv::Mat& dst, int threshold) {
  if (src.empty()) {
    cout << "[bin_threshold] Error: src is empty" << endl;
    return;
  }

  // get grayscale of source
  cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);

  int R = dst.rows, C = dst.cols;
  // threshold the dest image
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      dst.at<uchar>(i, j) = dst.at<uchar>(i, j) > threshold ? 0 : 255;
    }
  }
}

void grassfire(const cv::Mat& src, cv::Mat& dst) {
  if (src.empty()) {
    cout << "[grassfire] Error: src is empty" << endl;
    return;
  }

  if (src.channels() != 1) {
    cout << "[grassfire] Error: src is not grayscale" << endl;
    return;
  }

  if (src.size() != dst.size()) {
    dst = cv::Mat::zeros(src.size(), CV_8UC1);
  }

  int R = src.rows, C = src.cols;

  // forward pass
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      // check if foreground
      if (src.at<uchar>(i, j) > 0) {
        int up = i == 0 ? 0 : dst.at<uchar>(i - 1, j) + 1;
        int left = j == 0 ? 0 : dst.at<uchar>(i, j - 1) + 1;
        dst.at<uchar>(i, j) = min(up, left);
      }
    }
  }

  // backward pass
  int maxval = 0;
  for (int i = R - 1; i >= 0; i--) {
    for (int j = C - 1; j >= 0; j--) {
      // check if foreground
      if (src.at<uchar>(i, j) > 0) {
        int cur = dst.at<uchar>(i, j);
        int down = i == R - 1 ? 0 : dst.at<uchar>(i + 1, j) + 1;
        int right = j == C - 1 ? 0 : dst.at<uchar>(i, j + 1) + 1;

        cur = min(cur, down);
        cur = min(cur, right);

        dst.at<uchar>(i, j) = cur;
        maxval = max(maxval, cur);
      }
    }
  }

  dst = 255 * dst / maxval;
}

void region_growing_segmentation(const cv::Mat& src, cv::Mat& dst) {
  int R = src.rows, C = src.cols;

  dst = cv::Mat::zeros(src.size(), CV_8UC1);

  stack<pair<int, int>> stk;
  int regionid = 0;

  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      if (src.at<uchar>(i, j) > 0 && dst.at<uchar>(i, j) == 0) {
        regionid++;
        dst.at<uchar>(i, j) = regionid;
        stk.push(make_pair(i, j));
      }

      while (!stk.empty()) {
        int x = stk.top().second, y = stk.top().first;
        stk.pop();
        // check if up neighbor is fg and unlabeled
        if (y > 0) {
          if (src.at<uchar>(y - 1, x) > 0 && dst.at<uchar>(y - 1, x) == 0) {
            dst.at<uchar>(y - 1, x) = regionid;
            stk.push(make_pair(y - 1, x));
          }
        }
        // check if left neighbor is fg and unlabeled
        if (x > 0) {
          if (src.at<uchar>(y, x - 1) > 0 && dst.at<uchar>(y, x - 1) == 0) {
            dst.at<uchar>(y, x - 1) = regionid;
            stk.push(make_pair(y, x - 1));
          }
        }

        // check if down neighbor is fg and unlabeled
        if (y < R - 1) {
          if (src.at<uchar>(y + 1, x) > 0 && dst.at<uchar>(y + 1, x) == 0) {
            dst.at<uchar>(y + 1, x) = regionid;
            stk.push(make_pair(y + 1, x));
          }
        }

        // check if right neighbor is fg and unlabeled
        if (x < C - 1) {
          if (src.at<uchar>(y, x + 1) > 0 && dst.at<uchar>(y, x + 1) == 0) {
            dst.at<uchar>(y, x + 1) = regionid;
            stk.push(make_pair(y, x + 1));
          }
        }
      }
    }
  }
}

/*
void two_pass_segmentation(const cv::Mat& src, cv::Mat& dst) {
  if (src.empty()) {
    cout << "[two_pass_segmentation] Error: src is empty" << endl;
    return;
  }

  // build the disjoint set

  int R = src.rows, C = src.cols;

  DisjointSet ds(R * C);

  if (src.size() != dst.size()) {
    src.copyTo(dst);
  }

  // first pass

  int region = 0;
  uchar* cur;

  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      if ((cur = &dst.at<uchar>(i, j)) > 0) {
        // check 4-connected neighbors
        int up = i == 0 ? 0 : dst.at<uchar>(i - 1, j);
        int left = j == 0 ? 0 : dst.at<uchar>(i, j - 1);

        // no neighbors
        if (up == 0 && left == 0) {
          region++;
          *cur = region;
        } else {
          // neighbors are known to be part of the same component
          if (up == left) {
            *cur = up;
          } else { // neighbors have different labels
            ds.make_union(up, left);
            *cur = up;
          }
        }
      }
    }
  }

  // second pass
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      if ()
    }
  }
}
*/

/*
void get_regions(const cv::Mat& regionmap,
                 vector<vector<pair<int, int>>>& regions, int min_size = 50)
                 {
  if (regionmap.empty()) {
    cout << "[get_regions] Error: regionmap is empty" << endl;
    return;
  }

  int R = regionmap.rows, C = regionmap.cols;
}
*/

void segment(const cv::Mat& src, cv::Mat& dst, cv::Mat& stats,
             cv::Mat& centroids, vector<cv::Vec3b>& colors,
             vector<Region>& regions) {
  if (src.empty()) {
    cout << "[clean_segmentation] Error: seg is empty" << endl;
    return;
  }

  cv::Mat tmp = cv::Mat::zeros(src.size(), CV_8UC3);

  int num_regions = stats.rows;
  int min_area = 1000;
  int max_area = src.rows * src.cols - src.cols;

  int R = src.rows, C = src.cols;

  // track regions within min/max area and not at edges of image
  for (int i = 0; i < num_regions; i++) {
    int* region_stats = stats.ptr<int>(i);
    int top = region_stats[cv::CC_STAT_TOP];
    int left = region_stats[cv::CC_STAT_LEFT];
    int right = left + region_stats[cv::CC_STAT_WIDTH];
    int bot = top + region_stats[cv::CC_STAT_HEIGHT];
    int area = region_stats[cv::CC_STAT_AREA];

    if (area > min_area && area < max_area && top > 0 && left > 0 &&
        right < C && bot < R) {
      regions.push_back(Region(i, stats, centroids, colors[i], src));
    }
  }

  // sort the regions by area (ascending)
  sort(regions.begin(), regions.end(),
       [](const Region& a, const Region& b) { return a.area < b.area; });

  // recolor the regions according to their sorted order
  for (int i = 0; i < regions.size(); i++) {
    regions[i].color[0] = colors[i][0];
    regions[i].color[1] = colors[i][1];
    regions[i].color[2] = colors[i][2];
  }

  // color the regions in dst image
  for (auto r : regions) {
    r.colorRegion(src, tmp);
  }

  tmp.copyTo(dst);
}

double compute_distance(const Features& f1, const Features& f2,
                        const Features& std) {
  double dist = 0;

  dist += ((f1.minBBRatio - f2.minBBRatio) * (f1.minBBRatio - f2.minBBRatio)) /
          std.minBBRatio;
  // dist += ((f1.minBBArea - f2.minBBArea) * (f1.minBBArea - f2.minBBArea)) /
  //         std.minBBArea;
  dist += ((f1.maxBBPercentFilled - f2.maxBBPercentFilled) *
           (f1.maxBBPercentFilled - f2.maxBBPercentFilled)) /
          std.maxBBPercentFilled;
  // dist += ((f1.mu22a - f2.mu22a) * (f1.mu22a - f2.mu22a)) / std.mu22a;
  // Hu 7 is not used because it is invariant to skew
  // https://stackoverflow.com/questions/11379947/meaning-of-the-seven-hu-invariant-moments-function-from-opencv
  for (int i = 0; i < 6; i++) {
    dist += ((f1.hu[i] - f2.hu[i]) * (f1.hu[i] - f2.hu[i])) / std.hu[i];
  }

  return dist;
}

string nearest_neighbor(vector<pair<double, string>>& dists, bool sorted) {
  if (!sorted) {
    sort(dists.begin(), dists.end());
  }

  cout << "Nearest Neighbor Prediction: " << dists.front().second << endl;

  return dists.front().second;
}

string knn(vector<pair<double, string>>& dists, int n, bool sorted) {
  if (!sorted) {
    sort(dists.begin(), dists.end());
  }

  // get top k neighbors
  string topk[n];

  for (int i = 0; i < n; i++) {
    topk[i] = dists[i].second;
  }

  // find the most common label
  string pred = topk[n - 1];
  int count = 1;

  // Boyer-Moore majority vote algorithm
  for (int i = n - 2; i >= 0; i--) {
    if (topk[i] == pred) {
      count++;
    } else {
      count--;
    }

    if (count == 0) {
      pred = topk[i];
      count = 1;
    }
  }

  cout << "KNN Prediction (N=" << n << "): " << pred << endl;

  return pred;
}
