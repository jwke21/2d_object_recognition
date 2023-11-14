/**
 * Jake Van Meter
 * Fall 2023
 * CS 5330
 */

#include <opencv2/core.hpp>
#include <utility>
#include <vector>

#ifndef P3_UTIL_HPP
#define P3_UTIL_HPP

/**
 * A struct containing performance metrics for a classification system.
 */
struct Performance {
  int true_pos;
  int false_pos;
  int true_neg;
  int false_neg;
};

/**
 * A struct to hold the features of a region that will be used for
 * classification.
 */
struct Features {
  // height / width ratio
  double minBBRatio;
  // area of min bounding box
  // double minBBArea;
  // percent of the max bounding box filled by the region
  double maxBBPercentFilled;
  // Second moment about the central axis (orientation independent)
  double mu22a;
  // Hu invariants
  double hu[7];
};

class Region {
 public:
  /**
   * Constructor for Region.
   *
   * @param label the label of the region
   * @param stats the stats matrix from the connected components algorithm
   * @param centroids the centroids matrix from the connected components
   * algorithm
   * @param color the color of the region
   * @param img the image from which to get the region's points
   */
  Region(int label, const cv::Mat& stats, const cv::Mat& centroids,
         cv::Vec3b color, const cv::Mat& img);

  void getPointsFromImage(const cv::Mat& img);

  /**
   * Colors the region in the given destination image. Uses the region map
   * to determine whether a given pixel is part of the foreground or background.
   * Only foreground pixels will be colored.
   *
   * @param region_map the region map
   * @param dst the destination image
   */
  void colorRegion(const cv::Mat& region_map, cv::Mat& dst);
  void colorRegion(cv::Mat& dst);

  /**
   * Draws the maximum bounding box around the region in the given image.
   *
   * @param img the image that the bounding box will be drawn in
   */
  void drawMaxBoundingBox(cv::Mat& img);

  /**
   * Draws the minimum bounding box around the region in the given image.
   *
   * @param img the image that the bounding box will be drawn in
   */
  void drawMinBoundingBox(cv::Mat& img);

  /**
   * Draws the axis of least central moment around the region in the given
   * image.
   *
   * @param img the image that the axis will be drawn in
   */
  void drawAxisOfLeastCentralMoment(cv::Mat& img);

  /**
   * Draws a red circle around the centroid of the region in the given image.
   *
   * @param img the image to draw the centroid in
   */
  void drawCentroid(cv::Mat& img);

  /**
   * Draws the given label in the given image.
   *
   * @param img the image to draw the label in
   * @param label the label to draw
   */
  void drawLabel(cv::Mat& img, const std::string label);

  /**
   * Computes the features of the region in the given image. The features are
   * stored in the region's features struct.
   *
   * @param img the image to compute the features in
   */
  void computeFeatures(const cv::Mat& img);

  int label;
  int left;
  int top;
  int width;
  int height;
  int area;

  double cx;
  double cy;

  /**
   * The color of the region.
   */
  cv::Vec3b color;

  /**
   * The points in the region.
   */
  std::vector<cv::Point> points;

  /**
   * The moments of the region. Used for computing the features.
   */
  cv::Moments mu;

  /**
   * The features of the region. Used for classification.
   */
  Features* feats;
};

/**
 * Performs a binary threshold on the given source image and stores the result
 * in the given destination image. The image will be converted to grayscale
 * (not in-place) before performing the thresholding. The result will be an
 * image whose foreground pixels are set to 255 and whose background pixels are
 * set to 0.
 *
 * @param src the source image
 * @param dst the destination image
 * @param threshold the threshold value
 */
void bin_threshold(const cv::Mat& src, cv::Mat& dst, int threshold);

/**
 * Performs a grassfire transform on the given source image and stores the
 * result in the given destination image. The given source image is assumed to
 * be grayscale. This function is based off of Bruce Maxwell's in-class example.
 *
 * @param src the source image
 * @param dst the destination image
 */
void grassfire(const cv::Mat& src, cv::Mat& dst);

/**
 * Executes the region growing algorithm to segment the given source image into
 * regions and stores the resulting region map in the given destination image.
 * Based on the algorithm described during class by Bruce Maxwell.
 *
 * @param src the source image
 * @param dst the destination region map
 */
void region_growing_segmentation(const cv::Mat& src, cv::Mat& dst);

void two_pass_segmentation(const cv::Mat& src, cv::Mat& dst);

/**
 * Gets the regions from the given region map and stores them in the given
 * vector of vectors. Each region is represented as a vector of pairs, where
 * the pairs are coordinates of a pixel in the region.
 */
// void get_regions(const cv::Mat& regionmap,
//                  std::vector<std::vector<std::pair<int, int>>>& regions,
//                  int min_size = 50);

void segment(const cv::Mat& seg, cv::Mat& dst, cv::Mat& stats,
             cv::Mat& centroids, std::vector<cv::Vec3b>& colors,
             std::vector<Region>& regions);

/**
 * Computes the distance between two feature vectors. Uses a scaled Euclidean
 * distance metric.
 *
 * @param f1 the first feature vector
 * @param f2 the second feature vector
 * @param std the standard deviations of the features
 *
 * @return the distance between the two feature vectors
 */
double compute_distance(const Features& f1, const Features& f2,
                        const Features& std);

/**
 * Carries out the nearest neighbor algorithm on the given vector of distances.
 * Will sort the vector in-place if the sorted parameter is false.
 *
 * @param dists the vector of distances
 * @param sorted whether or not the vector is sorted
 *
 * @return the predicted label
 */
std::string nearest_neighbor(std::vector<std::pair<double, std::string>>& dists,
                             bool sorted = false);

/**
 * Carries out the k-nearest neighbor algorithm on the given vector of
 * distances. Will sort the vector in-place if the sorted parameter is false.
 *
 * @param dists the vector of distances
 * @param n the number of neighbors to consider
 * @param sorted whether or not the vector is sorted
 *
 * @return the predicted label
 */
std::string knn(std::vector<std::pair<double, std::string>>& dists, int n,
                bool sorted = false);

#endif  // P3_UTIL_HPP