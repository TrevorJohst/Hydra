#pragma once
#include <config_utilities/virtual_config.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spatial_hash/types.h>

#include "hydra/active_window/active_window_output.h"
#include "hydra/active_window/volumetric_window.h"
#include "hydra/common/dsg_types.h"
#include "hydra/frontend/frontier.h"
#include "hydra/frontend/rayfront_extractor.h"

namespace hydra {

class NearestNodeFinder;

class FrontierExtractor {
 public:
  struct Config {
    char prefix = 'f';
    double cluster_tolerance = .3;
    size_t min_cluster_size = 10;
    size_t max_cluster_size = 100000;
    double max_place_radius = 5;
    bool dense_frontiers = false;
    double frontier_splitting_threshold = 0.2;
    size_t point_threshold = 10;
    size_t culling_point_threshold = 10;
    double recent_block_distance = 25;
    double minimum_relative_z = -0.2;
    double maximum_relative_z = 1;
    bool compute_frontier_shape = false;
    bool extract_rayfronts = false;
    RayFrontExtractor::Config rayfront_config;
  } const config;

  explicit FrontierExtractor(const Config& config);

  void updateRecentBlocks(const Eigen::Vector3d& current_position, double block_size);

  void detectFrontiers(const ActiveWindowOutput& input,
                       DynamicSceneGraph& graph,
                       const NodeIdSet& active_places);

  void addFrontiers(uint64_t timestamp_ns, DynamicSceneGraph& graph);

  void setArchivedPlaces(const std::vector<NodeId>& archived_places);

 private:
  NodeSymbol next_node_id_;
  std::vector<std::pair<NodeId, BlockIndex>> nodes_to_remove_;

  TsdfLayer::Ptr tsdf_;
  std::vector<NodeId> archived_places_;
  spatial_hash::IndexSet just_archived_blocks_;
  spatial_hash::IndexSet recently_archived_blocks_;
  std::unique_ptr<VolumetricWindow> map_window_;

  std::unique_ptr<NearestNodeFinder> place_finder_;
  std::vector<Frontier> frontiers_;
  std::vector<Frontier> archived_frontiers_;

  // Helper functions.
  void updateTsdf(const ActiveWindowOutput& msg);

  void populateDenseFrontiers(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr archived_cloud,
                              const TsdfLayer& layer);

  void computeSparseFrontiers(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              const TsdfLayer& layer,
                              std::vector<Frontier>& frontiers) const;

  inline static const auto registration_ =
      config::RegistrationWithConfig<FrontierExtractor, FrontierExtractor, Config>(
          "voxel_clustering");
};

Eigen::Vector3d frontiersToCenters(const std::vector<Eigen::Vector3f>& positions);

void declare_config(FrontierExtractor::Config& conf);

}  // namespace hydra
