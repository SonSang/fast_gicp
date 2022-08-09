#include <chrono>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/diff/diff_fast_gicp.hpp>
#include <fast_gicp/diff/diff_fast_gicp_st.hpp>

// benchmark for fast_gicp registration methods
template <typename Registration>
void test_diff(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  // fast_gicp reuses calculated covariances if an input cloud is the same as the previous one
  // to prevent this for benchmarking, force clear source and target clouds
  reg.clearTarget();
  reg.clearSource();
  reg.setInputTarget(target);
  reg.setInputSource(source);

  std::vector<float> target_weights = std::vector<float>(target->size(), 1.0);
  std::vector<float> source_weights = std::vector<float>(source->size(), 1.0);
  reg.setTargetWeights(target_weights);
  reg.setSourceWeights(source_weights);

  reg.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

  std::cout << "single:" << single << "[msec] " << std::flush;

  // backward
  auto t3 = std::chrono::high_resolution_clock::now();
  reg.backward();
  auto t4 = std::chrono::high_resolution_clock::now();
  double backwardt = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e6;
  std::cout << "single backward:" << backwardt << "[msec] " << std::flush;
}

/**
 * @brief main
 */
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "usage: gicp_align target_pcd source_pcd" << std::endl;
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (pcl::io::loadPCDFile(argv[1], *target_cloud)) {
    std::cerr << "failed to open " << argv[1] << std::endl;
    return 1;
  }
  if (pcl::io::loadPCDFile(argv[2], *source_cloud)) {
    std::cerr << "failed to open " << argv[2] << std::endl;
    return 1;
  }

  std::cout << "target:" << target_cloud->size() << "[pts] source:" << source_cloud->size() << "[pts]" << std::endl;

  std::cout << "--- diff fgicp_st ---" << std::endl;
  fast_gicp::DiffFastGICPSingleThread<pcl::PointXYZ, pcl::PointXYZ> diff_fgicp_st;
  test_diff(diff_fgicp_st, target_cloud, source_cloud);

  return 0;
}
