#pragma once
#include <spatial_hash/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "hydra/common/dsg_types.h"

namespace hydra {

struct Rayfront {
 public:
  Rayfront(){};
  Rayfront(const Eigen::Vector3f& d,
           const Eigen::Vector3f& o,
           uint32_t label,
           double weight)
      : direction(d),
        camera_origin(o),
        theta(std::acos(d.z() / d.norm())),
        phi(std::atan2(d.y(), d.x())),
        weight(weight),
        semantic_label(label){};

 public:
  Eigen::Vector3f direction;
  Eigen::Vector3f camera_origin;
  double theta;
  double phi;
  double weight;
  // TODO (trejohst): make a helper struct to switch between open/closed semantics
  uint32_t semantic_label;
};

struct Frontier {
 public:
  Frontier(){};
  Frontier(Eigen::Vector3d c,
           Eigen::Vector3d s,
           Eigen::Quaterniond o,
           size_t n,
           spatial_hash::BlockIndex b)
      : center(c),
        scale(s),
        orientation(o),
        num_frontier_voxels(n),
        block_index(b),
        has_shape_information(true) {}
  Frontier(Eigen::Vector3d c, size_t n, spatial_hash::BlockIndex b)
      : center(c),
        num_frontier_voxels(n),
        block_index(b),
        has_shape_information(false) {}

 public:
  Eigen::Vector3d center;
  Eigen::Vector3d scale;
  Eigen::Quaterniond orientation;
  size_t num_frontier_voxels = 0;
  spatial_hash::BlockIndex block_index;
  bool has_shape_information = false;
  std::vector<Rayfront> rayfronts;
};

}  // namespace hydra