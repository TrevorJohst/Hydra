#pragma once
#include <config_utilities/virtual_config.h>

#include "hydra/active_window/active_window_output.h"
#include "hydra/frontend/frontier.h"
#include "hydra/input/sensor.h"

namespace hydra {

class RayfrontExtractor {
 public:
  struct Config {
    double image_scale = 0.5;
    int erosion_kernel_size = 3;
    double rayfront_range = -1.0;
    double angle_bin_deg = 30.0;
  } const config;

  explicit RayfrontExtractor(const Config& config);

  /**
   * @brief Merge rayfronts within the passed frontiers.
   * @note Only bins within semantic class. If frontiers have different extraction
   * origins, preserves the origin of highest weight
   * @param frontiers `std::vector<Frontier>` containing the frontiers to be merged
   */
  template <typename FrontierArrayLike>
  void mergeRayfronts(FrontierArrayLike& frontiers);

  /**
   * @brief Assigns rayfronts to their best frontier based on the cost function.
   * @param rayfronts Vector of length N containing rayfronts to re-assign
   * @param frontiers `std::vector<Frontier>` of length M with the candidate frontiers
   * @return True if any rays were assigned to a frontier
   */
  template <typename FrontierArrayLike>
  bool assignRayfronts(const std::vector<hydra::Rayfront>& rayfronts,
                       FrontierArrayLike& frontiers);

  /**
   * @brief Assigns rayfronts to their best frontier based on the cost function.
   * @param ray_dirs Nx3 matrix where each row is one candidate ray
   * @param ray_labels Vector of length N mapping the nth ray to its semantic label
   * @param camera_origins Nx3 OR 1x3 matrix where each row is the origin of extraction
   * @param frontiers `std::vector<Frontier>` of length M with the candidate frontiers
   * @return True if any rays were assigned to a frontier
   */
  template <typename FrontierArrayLike>
  bool assignRayfronts(const Eigen::MatrixXd& ray_dirs,
                       const std::vector<uint32_t>& ray_labels,
                       const Eigen::MatrixXd& camera_origins,
                       FrontierArrayLike& frontiers);

  /**
   * @brief Add rayfronts to frontiers for a given active window frame
   * @param input The `ActiveWindowOutput` for this frame
   * @param frontiers Vector of candidate frontiers
   * @param archived_frontiers Additional vector of candidate frontiers (can be empty)
   */
  void addRayfronts(const ActiveWindowOutput& input,
                    std::vector<Frontier>& frontiers,
                    std::vector<Frontier>& archived_frontiers);

  /**
   * @brief Set the rayfront extraction sensor range for a camera config
   * @param camera The camera config to extract the sensor range from
   */
  void setSensorRange(const hydra::Sensor& camera) {
    // Config takes priority when explicitly set by user
    sensor_range_ =
        (config.rayfront_range < 0.0) ? camera.max_range() : config.rayfront_range;
  };

  // Helper to handle disjointed frontier vectors
  struct FrontierConcatView {
    std::vector<Frontier>& a;
    std::vector<Frontier>& b;

    FrontierConcatView(std::vector<Frontier>& a_, std::vector<Frontier>& b_)
        : a(a_), b(b_) {}

    size_t size() const { return a.size() + b.size(); }

    Frontier& operator[](size_t i) {
      if (i < a.size()) return (a)[i];
      if (i - a.size() < b.size()) return b[i - a.size()];
      throw std::out_of_range("FrontierConcatView index out of range");
    }

    const Frontier& operator[](size_t i) const {
      if (i < a.size()) return (a)[i];
      if (i - a.size() < b.size()) return b[i - a.size()];
      throw std::out_of_range("FrontierConcatView index out of range");
    }
  };

 private:
  double sensor_range_;
};

void declare_config(RayfrontExtractor::Config& config);

}  // namespace hydra