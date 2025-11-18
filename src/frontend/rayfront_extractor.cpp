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

template <typename T>
struct ConcatView {
  std::vector<T>& a;
  std::vector<T>& b;

  ConcatView(std::vector<T>& a_, std::vector<T>& b_) : a(a_), b(b_) {}

  size_t size() const { return a.size() + b.size(); }

  T& operator[](size_t i) {
    if (i < a.size()) return (a)[i];
    if (i - a.size() < b.size()) return b[i - a.size()];
    throw std::out_of_range("ConcatView index out of range");
  }

  const T& operator[](size_t i) const {
    if (i < a.size()) return (a)[i];
    if (i - a.size() < b.size()) return b[i - a.size()];
    throw std::out_of_range("ConcatView index out of range");
  }
};
}  // namespace

void declare_config(RayfrontExtractor::Config& config) {
  using namespace config;
  name("RayfrontExtractor::Config");
  field(config.erosion_kernel_size, "erosion_kernel_size");
  field(config.rayfront_range, "rayfront_range");
}

RayfrontExtractor::RayfrontExtractor(const Config& config)
    : config(config), sensor_range_(config.rayfront_range) {}

template <typename FrontierArrayLike>
void RayfrontExtractor::mergeRayfronts(FrontierArrayLike& frontiers) {
  if (frontiers.size() == 0) return;

  const double angle_bin_rad = config.angle_bin_deg * M_PI / 180.0;
  for (size_t i = 0; i < frontiers.size(); ++i) {
    Frontier& frontier = frontiers[i];
    if (frontier.rayfronts.size() <= 1) continue;

    std::unordered_map<AngleBin, std::vector<RayFront>, AngleBinHash> bins;

    // Bin rays by angle
    for (const RayFront& rf : frontier.rayfronts) {
      int theta_bin = static_cast<int>(std::floor(rf.theta / angle_bin_rad));
      int phi_bin = static_cast<int>(std::floor((rf.phi + M_PI) / angle_bin_rad));
      bins[{theta_bin, phi_bin}].push_back(rf);
    }

    std::vector<RayFront> merged;
    merged.reserve(bins.size());

    // Weight the rayfronts within their bins
    for (auto& bin_rayfronts : bins) {
      const auto& rayfronts_vec = bin_rayfronts.second;
      if (rayfronts_vec.empty()) continue;

      // Take the dominant label in this bin
      std::unordered_map<uint32_t, double> label_weights;
      for (const auto& rf : rayfronts_vec)
        label_weights[rf.semantic_label] += rf.weight;

      uint32_t voted_label = 0;
      double max_label_weight = -1.0;
      for (const auto& label_weight : label_weights) {
        if (label_weight.second > max_label_weight) {
          voted_label = label_weight.first;
          max_label_weight = label_weight.second;
        }
      }

      // Linear weighting of direction, highest weight origin for this label
      Eigen::Vector3d weighted_dir(0.0, 0.0, 0.0);
      double sum_weight = 0.0;
      Eigen::Vector3d dominant_origin;
      double max_weight = -1.0;

      for (const auto& rf : rayfronts_vec) {
        if (rf.semantic_label != voted_label) continue;

        weighted_dir += rf.weight * rf.direction;
        sum_weight += rf.weight;

        if (rf.weight > max_weight) {
          dominant_origin = rf.camera_origin;
          max_weight = rf.weight;
        }
      }

      if (sum_weight > 0.0) weighted_dir /= sum_weight;
      weighted_dir.normalize();

      merged.emplace_back(weighted_dir, dominant_origin, voted_label, sum_weight);
    }

    frontier.rayfronts = std::move(merged);
  }
}

template void RayfrontExtractor::mergeRayfronts(std::vector<Frontier>&);
template void RayfrontExtractor::mergeRayfronts(ConcatView<Frontier>&);

template <typename FrontierArrayLike>
bool RayfrontExtractor::assignRayfronts(const Eigen::MatrixXd& ray_dirs,
                                        const std::vector<uint32_t>& ray_labels,
                                        const Eigen::MatrixXd& camera_origins,
                                        FrontierArrayLike& frontiers) {
  // Guards
  if (sensor_range_ < 0.0) {
    LOG(ERROR) << "Sensor range must be set to a non-negative value in the rayfront "
                  "extractor config, or updated with setSensorRange.";
    return false;
  }

  if (ray_dirs.rows() != static_cast<Eigen::Index>(ray_labels.size()) ||
      (camera_origins.rows() != 1 && ray_dirs.rows() != camera_origins.rows())) {
    LOG(ERROR) << "ray_dirs, ray_labels, and camera_origins must have matching "
                  "dimensions, ray_dirs has "
               << ray_dirs.rows() << " rows, ray_labels has " << ray_labels.size()
               << " elements, and camera_origins has " << camera_origins.rows();
    return false;
  }

  int N = ray_labels.size();
  int M = frontiers.size();
  bool common_origin = camera_origins.rows() == 1;

  if (N == 0 || M == 0) return false;

  // Make a matrix of frontier positions for calculations
  Eigen::MatrixXd frontier_orig(M, 3);  // M x 3
  for (int i = 0; i < M; ++i) frontier_orig.row(i) = frontiers[i].center;

  // Dot product from frontier to rayfront and distance to each frontier
  Eigen::MatrixXd dot_prod(M, N);
  Eigen::MatrixXd dist(M, N);
  for (int j = 0; j < N; ++j) {
    Eigen::MatrixXd frontier_vec;  // M x 3
    if (common_origin)
      frontier_vec = frontier_orig.rowwise() - camera_origins.row(0);
    else
      frontier_vec = frontier_orig.rowwise() - camera_origins.row(j);

    dot_prod.col(j) =
        frontier_vec * ray_dirs.row(j).transpose();  // (M x 3) * (1 x 3)^T = M x 1

    dist.col(j) = frontier_vec.rowwise().norm();  // (M x 1)
  }

  // Orthogonal distance
  Eigen::MatrixXd ortho_dist(M, N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      Eigen::RowVector3d closest;
      if (common_origin)
        closest = dot_prod(i, j) * ray_dirs.row(j) + camera_origins.row(0);
      else
        closest = dot_prod(i, j) * ray_dirs.row(j) + camera_origins.row(j);
      ortho_dist(i, j) = (closest - frontier_orig.row(i)).norm();
    }
  }

  // Cost matrix
  Eigen::MatrixXd ortho_norm = ortho_dist;  // M x N
  Eigen::MatrixXd dist_norm = dist;         // M x N
  for (int i = 0; i < M; ++i) {
    double m1 = ortho_dist.row(i).maxCoeff();
    if (m1 > 0.0)
      ortho_norm.row(i) /= m1;
    else
      ortho_norm.row(i).setZero();

    double m2 = dist.row(i).maxCoeff();
    if (m2 > 0.0)
      dist_norm.row(i) /= m2;
    else
      dist_norm.row(i).setZero();
  }

  Eigen::MatrixXd cost_matrix = (ortho_norm + dist_norm) / 2.0;  // M x N

  // NOTE: Make sure that frontier shape is extracted when frontiers are extracted
  Eigen::VectorXd frontier_sizes(M);
  for (int i = 0; i < M; ++i) frontier_sizes(i) = frontiers[i].scale.maxCoeff();

  // Mask criteria to filter out frontiers
  // TODO: Add config options for some of these
  Eigen::ArrayXX<bool> mask_dot = (dot_prod.array() <= 0.0);  // M x N
  Eigen::ArrayXX<bool> mask_ortho =
      (ortho_dist.array() > frontier_sizes.replicate(1, N).array());  // M x N
  Eigen::ArrayXX<bool> mask_close =
      (dist.array() < 2.0 * frontier_sizes.replicate(1, N).array());     // M x N
  Eigen::ArrayXX<bool> mask_far = (dist.array() > 3.0 * sensor_range_);  // M x N

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

  // Assign rays to frontiers
  bool assigned = false;
  for (int j = 0; j < cost_matrix.cols(); ++j) {
    int idx;
    double cost = cost_matrix.col(j).minCoeff(&idx);

    // If minimum cost is finite, this ray gets assigned to a frontier
    if (std::isfinite(cost)) {
      double weight = 1.0 - cost;
      Eigen::Vector3d camera_origin =
          common_origin ? camera_origins.row(0) : camera_origins.row(j);
      frontiers[idx].rayfronts.emplace_back(
          ray_dirs.row(j).transpose(), camera_origin, ray_labels[j], weight);
      assigned = true;
    }
  }
  return assigned;
}

template bool RayfrontExtractor::assignRayfronts(const Eigen::MatrixXd&,
                                                 const std::vector<uint32_t>&,
                                                 const Eigen::MatrixXd&,
                                                 std::vector<Frontier>&);
template bool RayfrontExtractor::assignRayfronts(const Eigen::MatrixXd&,
                                                 const std::vector<uint32_t>&,
                                                 const Eigen::MatrixXd&,
                                                 ConcatView<Frontier>&);

void RayfrontExtractor::addRayfronts(const ActiveWindowOutput& input,
                                     std::vector<Frontier>& frontiers,
                                     std::vector<Frontier>& archived_frontiers) {
  ConcatView<Frontier> frontier_view(frontiers, archived_frontiers);
  if (frontier_view.size() == 0) return;

  // Get the rayfront extraction range
  const auto& camera = input.sensor_data->getSensor();
  setSensorRange(camera);

  // Extract labeled image from the input
  cv::Mat labels = input.sensor_data->label_image;

  // Extract depth image from the input
  // NOTE: Depth image doesnt always exist, range image does
  cv::Mat depth = input.sensor_data->depth_image;

  // Create a depth mask (and erode) based on sensor range
  // TODO: Does this work when sensor_range is infinite? (i.e. points out of cam
  // range)
  cv::Mat mask;
  cv::threshold(depth, mask, sensor_range_, 255.0, cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * config.erosion_kernel_size + 1, 2 * config.erosion_kernel_size + 1));

  cv::Mat eroded_mask;
  cv::erode(mask, eroded_mask, kernel);

  // Select the camera bearing rays corresponding to our eroded mask
  std::vector<cv::Point> candidate_rays_idx;
  cv::findNonZero(eroded_mask, candidate_rays_idx);

  // Camera pose
  Eigen::Isometry3d world_T_camera = input.sensor_data->getSensorPose();
  Eigen::Vector3d ray_orig = world_T_camera.translation();     // 3 x 1
  Eigen::Matrix3d world_R_camera = world_T_camera.rotation();  // 3 x 3

  int N = candidate_rays_idx.size();
  Eigen::MatrixXd ray_dir(N, 3);  // N x 3
  std::vector<uint32_t> ray_labels(N);
  for (int i = 0; i < N; ++i) {
    const auto& pt = candidate_rays_idx[i];
    Eigen::Vector3d dir_world =
        world_R_camera * camera.getPixelBearing(pt.x, pt.y).cast<double>();
    ray_dir.row(i) = dir_world.normalized().transpose();
    ray_labels[i] = labels.at<uint32_t>(pt);
  }

  // Return early if no rays are assigned
  Eigen::MatrixXd camera_origins(1, 3);
  camera_origins.row(0) = ray_orig.transpose();
  if (!assignRayfronts(ray_dir, ray_labels, camera_origins, frontier_view)) return;

  // Merge any rays within the same frontier
  mergeRayfronts(frontier_view);
}

}  // namespace hydra