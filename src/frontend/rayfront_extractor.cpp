#include "hydra/frontend/rayfront_extractor.h"

#include <config_utilities/config_utilities.h>

#include <limits>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace hydra {

// Local helpers
namespace {
struct AngleBin {
  int theta_bin;
  int phi_bin;
  bool operator==(const AngleBin& other) const {
    return theta_bin == other.theta_bin && phi_bin == other.phi_bin;
  }
};

struct AngleBinHash {
  std::size_t operator()(const AngleBin& k) const noexcept {
    return std::hash<int>()(k.theta_bin) ^ (std::hash<int>()(k.phi_bin) << 1);
  }
};
}  // namespace

RayFrontExtractor::RayFrontExtractor(const Config& config) : config(config) {}

void RayFrontExtractor::addRayFronts(const ActiveWindowOutput& input,
                                     std::vector<Frontier>& frontiers) {
  // Get the rayfront extraction range and fallback frontier size
  const auto& camera = input.sensor_data->getSensor();
  double sensor_range =
      (config.rayfront_range < 0) ? camera.max_range() : config.rayfront_range;
  double fixed_frontier_size = input.map().blockSize();

  // Extract labeled image from the input
  cv::Mat labels = input.sensor_data->label_image;

  // Extract depth image from the input
  // NOTE: Depth image doesnt always exist, range image does
  cv::Mat depth = input.sensor_data->depth_image;

  // Create a depth mask (and erode) based on sensor range
  // TODO: Does this work when sensor_range is infinite? (i.e. points out of cam range)
  cv::Mat mask;
  cv::threshold(depth, mask, sensor_range, 255.0, cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * config.erosion_kernel_size + 1, 2 * config.erosion_kernel_size + 1));

  cv::Mat eroded_mask;
  cv::erode(mask, eroded_mask, kernel);

  // Select the camera bearing rays corresponding to our eroded mask
  std::vector<cv::Point> candidate_rays_idx;
  cv::findNonZero(eroded_mask, candidate_rays_idx);

  int N = candidate_rays_idx.size();
  Eigen::MatrixXd ray_dir(N, 3);  // N x 3
  std::vector<uint32_t> ray_labels(N);
  for (int i = 0; i < N; ++i) {
    const auto& pt = candidate_rays_idx[i];
    ray_dir.row(i) = camera.getPixelBearing(pt.x, pt.y).cast<double>();
    ray_labels[i] = labels.at<uint32_t>(pt);
  }

  Eigen::Isometry3d world_T_camera = input.sensor_data->getSensorPose();
  Eigen::Vector3d ray_orig = world_T_camera.translation();  // 3 x 1

  // Make a matrix of frontier positions for calculations
  int M = frontiers.size();
  Eigen::MatrixXd frontier_orig(M, 3);  // M x 3
  for (int i = 0; i < M; ++i) frontier_orig.row(i) = frontiers[i].center;

  // Dot product from frontier to rayfront
  Eigen::MatrixXd frontier_vec =
      frontier_orig.rowwise() - ray_orig.transpose();  // (M x 3) - (3 x 1)^T = M x 3
  Eigen::MatrixXd dot_prod =
      frontier_vec * ray_dir.transpose();  // (M x 3) * (N x 3)^T = M x N

  // Distance to each frontier
  Eigen::VectorXd dist = frontier_vec.rowwise().norm();  // M x 1

  // Orthogonal distance
  Eigen::MatrixXd ortho_dist(M, N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      Eigen::RowVector3d closest =
          dot_prod(i, j) * ray_dir.row(j) + ray_orig.transpose();
      ortho_dist(i, j) = (closest - frontier_orig.row(i)).norm();
    }
  }

  // Cost matrix
  Eigen::MatrixXd ortho_norm = ortho_dist / ortho_dist.maxCoeff();  // M x N
  Eigen::VectorXd dist_norm = dist / dist.maxCoeff();               // M x 1
  Eigen::MatrixXd dist_norm_repeat = dist_norm.replicate(1, N);     // M x N

  Eigen::MatrixXd cost_matrix = (ortho_norm + dist_norm_repeat) / 2.0;  // M x N

  // TODO: Get this value from the frontiers if they have shape info, or estimate it?
  // NOTE: Make sure that frontier shape is extracted when frontiers are extracted, just
  // error if not
  double frontier_size = fixed_frontier_size;

  // Mask criteria to filter out frontiers
  // TODO: Add config options for some of these
  Eigen::ArrayXX<bool> mask_dot = (dot_prod.array() <= 0.0);               // M x N
  Eigen::ArrayXX<bool> mask_ortho = (ortho_dist.array() > frontier_size);  // M x N
  Eigen::ArrayXX<bool> mask_close =
      (dist.array() < 2.0 * frontier_size).replicate(1, N);  // M x N
  Eigen::ArrayXX<bool> mask_far =
      (dist.array() > 3.0 * sensor_range).replicate(1, N);  // M x N

  Eigen::ArrayXXi mask_sum = (mask_dot.cast<int>() + mask_ortho.cast<int>() +
                              mask_close.cast<int>() + mask_far.cast<int>());
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> frontier_mask =
      (mask_sum > 0).matrix();

  // Masks the cost matrix setting filtered frontiers to infinity
  cost_matrix = frontier_mask.select(
      Eigen::ArrayXXd::Constant(cost_matrix.rows(),
                                cost_matrix.cols(),
                                std::numeric_limits<double>::infinity()),
      cost_matrix.array());

  // Find minimum cost and index
  // Eigen::VectorXd min_cost(cost_matrix.cols());

  // Assign rays to frontiers
  double total_weight = 0.0;
  for (int j = 0; j < cost_matrix.cols(); ++j) {
    int idx;
    double cost = cost_matrix.col(j).minCoeff(&idx);

    // If minimum cost is finite, this ray gets assigned to a frontier
    if (std::isfinite(cost)) {
      double weight = 1.0 - cost;
      frontiers[idx].rayfronts.emplace_back(
          ray_dir.row(j).transpose(), ray_labels[j], weight);
      total_weight += weight;
    }
  }
  // No rays were assigned
  if (total_weight <= 0) return;

  // Normalize weights
  for (auto& f : frontiers) {
    for (auto& rf : f.rayfronts) {
      rf.weight /= total_weight;
    }
  }

  const double angle_bin_rad = config.angle_bin_deg * M_PI / 180.0;
  for (Frontier& frontier : frontiers) {
    if (frontier.rayfronts.size() <= 1) continue;

    std::unordered_map<AngleBin, std::vector<RayFront>, AngleBinHash> bins;

    // Group rays
    for (const RayFront& rf : frontier.rayfronts) {
      int theta_bin = static_cast<int>(std::floor(rf.theta / angle_bin_rad));
      int phi_bin = static_cast<int>(std::floor((rf.phi + M_PI) / angle_bin_rad));
      bins[{theta_bin, phi_bin}].push_back(rf);
    }

    std::vector<RayFront> merged;
    merged.reserve(bins.size());

    // Weight the rayfronts within their bins
    for (auto& kv : bins) {
      const auto& rayfronts_vec = kv.second;
      if (rayfronts_vec.empty()) continue;

      // Linear weighting of the direction, label with highest weight
      Eigen::Vector3d weighted_dir(0.0, 0.0, 0.0);
      std::unordered_map<uint32_t, double> label_weights;
      double sum_weight = 0.0;

      for (const auto& rf : rayfronts_vec) {
        weighted_dir += rf.weight * rf.direction;
        sum_weight += rf.weight;

        label_weights[rf.semantic_label] += rf.weight;
      }

      // Direction
      if (sum_weight > 0.0) weighted_dir /= sum_weight;
      weighted_dir.normalize();

      // Label
      uint32_t voted_label = 0;
      double max_label_weight = -1.0;
      for (const auto& kv : label_weights) {
        if (kv.second > max_label_weight) {
          voted_label = kv.first;
          max_label_weight = kv.second;
        }
      }

      merged.emplace_back(weighted_dir, voted_label, sum_weight);
    }

    frontier.rayfronts = std::move(merged);
  }
}

void declare_config(RayFrontExtractor::Config& config) {
  using namespace config;
  name("RayFrontExtractor::Config");
  field(config.erosion_kernel_size, "erosion_kernel_size");
  field(config.rayfront_range, "rayfront_range");
}

}  // namespace hydra