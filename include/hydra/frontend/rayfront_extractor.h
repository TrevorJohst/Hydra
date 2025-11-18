#pragma once
#include <config_utilities/virtual_config.h>

#include "hydra/active_window/active_window_output.h"
#include "hydra/frontend/frontier.h"
#include "hydra/input/sensor.h"

namespace hydra {

class RayfrontExtractor {
 public:
  struct Config {
    int erosion_kernel_size = 3;
    double rayfront_range = -1.0;
    double angle_bin_deg = 30.0;
  } const config;

  explicit RayfrontExtractor(const Config& config);

  void mergeRayfronts(std::vector<Frontier>& frontiers);

  bool assignRayfronts(const Eigen::MatrixXd& ray_dirs,
                       const std::vector<uint32_t>& ray_labels,
                       const Eigen::Vector3d& camera_origin,
                       std::vector<Frontier>& frontiers);

  void addRayfronts(const ActiveWindowOutput& input, std::vector<Frontier>& frontiers);

  void setSensorRange(const hydra::Sensor& camera) {
    // Config takes priority when explicitly set by user
    sensor_range_ =
        (config.rayfront_range < 0.0) ? camera.max_range() : config.rayfront_range;
  };

 private:
  double sensor_range_;
};

void declare_config(RayfrontExtractor::Config& config);

}  // namespace hydra