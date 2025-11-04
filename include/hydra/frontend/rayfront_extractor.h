#pragma once
#include <config_utilities/virtual_config.h>

#include "hydra/active_window/active_window_output.h"
#include "hydra/frontend/frontier.h"

namespace hydra {

class RayFrontExtractor {
 public:
  struct Config {
    int erosion_kernel_size = 3;
    double rayfront_range = -1.0;
    double angle_bin_deg = 30.0;
  } const config;

  explicit RayFrontExtractor(const Config& config);

  void addRayFronts(const ActiveWindowOutput& input, std::vector<Frontier>& frontiers);
};

void declare_config(RayFrontExtractor::Config& config);

}  // namespace hydra