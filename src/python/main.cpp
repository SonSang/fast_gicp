#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <boost/filesystem.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

// @sanghyun: add PCL ICP methods
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/search/kdtree.h>

// @sanghyun: add differentiable WGICP methods
#include <fast_gicp/wgicp/d_wgicp.hpp>

namespace py = pybind11;

fast_gicp::NeighborSearchMethod search_method(const std::string& neighbor_search_method) {
  if(neighbor_search_method == "DIRECT1") {
    return fast_gicp::NeighborSearchMethod::DIRECT1;
  } else if (neighbor_search_method == "DIRECT7") {
    return fast_gicp::NeighborSearchMethod::DIRECT7;
  } else if (neighbor_search_method == "DIRECT27") {
    return fast_gicp::NeighborSearchMethod::DIRECT27;
  } else if (neighbor_search_method == "DIRECT_RADIUS") {
    return fast_gicp::NeighborSearchMethod::DIRECT_RADIUS;
  }

  std::cerr << "error: unknown neighbor search method " << neighbor_search_method << std::endl;
  return fast_gicp::NeighborSearchMethod::DIRECT1;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr eigen2pcl(const Eigen::Matrix<double, -1, 3>& points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->resize(points.rows());

  for(int i=0; i<points.rows(); i++) {
    cloud->at(i).getVector3fMap() = points.row(i).cast<float>();
  }
  return cloud;
}

Eigen::Matrix<double, -1, 3> downsample(const Eigen::Matrix<double, -1, 3>& points, double downsample_resolution) {
  auto cloud = eigen2pcl(points);

  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  voxelgrid.setInputCloud(cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  voxelgrid.filter(*filtered);

  Eigen::Matrix<float, -1, 3> filtered_points(filtered->size(), 3);
  for(int i=0; i<filtered->size(); i++) {
    filtered_points.row(i) = filtered->at(i).getVector3fMap();
  }

  return filtered_points.cast<double>();
}

Eigen::Matrix4d align_points(
  const Eigen::Matrix<double, -1, 3>& target,
  const Eigen::Matrix<double, -1, 3>& source,
  const std::string& method,
  double downsample_resolution,
  int k_correspondences,
  double max_correspondence_distance,
  double voxel_resolution,
  int num_threads,
  const std::string& neighbor_search_method,
  double neighbor_search_radius,
  const Eigen::Matrix4f& initial_guess
) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = eigen2pcl(target);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = eigen2pcl(source);

  if(downsample_resolution > 0.0) {
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.setInputCloud(target_cloud);
    voxelgrid.filter(*filtered);
    target_cloud.swap(filtered);

    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*filtered);
    source_cloud.swap(filtered);
  }

  std::shared_ptr<fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>> reg;

  if(method == "GICP") {
    std::shared_ptr<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>> gicp(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>);
    gicp->setMaxCorrespondenceDistance(max_correspondence_distance);
    gicp->setCorrespondenceRandomness(k_correspondences);
    gicp->setNumThreads(num_threads);
    reg = gicp;
  } else if (method == "VGICP") {
    std::shared_ptr<fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setResolution(voxel_resolution);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method));
    vgicp->setNumThreads(num_threads);
    reg = vgicp;
  } else if (method == "VGICP_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>> vgicp(new fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>);
    vgicp->setCorrespondenceRandomness(k_correspondences);
    vgicp->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    vgicp->setResolution(voxel_resolution);
    reg = vgicp;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else if (method == "NDT_CUDA") {
#ifdef USE_VGICP_CUDA
    std::shared_ptr<fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>> ndt(new fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>);
    ndt->setResolution(voxel_resolution);
    ndt->setNeighborSearchMethod(search_method(neighbor_search_method), neighbor_search_radius);
    reg = ndt;
#else
    std::cerr << "error: you need to build fast_gicp with BUILD_VGICP_CUDA=ON" << std::endl;
    return Eigen::Matrix4d::Identity();
#endif
  } else {
    std::cerr << "error: unknown registration method " << method << std::endl;
    return Eigen::Matrix4d::Identity();
  }

  reg->setInputTarget(target_cloud);
  reg->setInputSource(source_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
  reg->align(*aligned, initial_guess);

  return reg->getFinalTransformation().cast<double>();
}

using LsqRegistration = fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
using FastGICP = fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>;
using FastVGICP = fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
#ifdef USE_VGICP_CUDA
using FastVGICPCuda = fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>;
using NDTCuda = fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
#endif

// @sanghyun
using PCLICP = pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>;
// using PCLICP_Po2Pl = pcl::IterativeClosestPointWithNormals<pcl::PointXYZ, pcl::PointXYZ>;
using PCLGICP = pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>; 
using PCLKDTree = pcl::search::KdTree<pcl::PointXYZ>;

using dLsqRegistration = wgicp::dLsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
using dFastGICP = wgicp::dFastGICP<pcl::PointXYZ, pcl::PointXYZ>;
using dWGICP = wgicp::dWGICP<pcl::PointXYZ, pcl::PointXYZ>;

PYBIND11_MODULE(pygicp, m) {
  m.def("downsample", &downsample, "downsample points");

  m.def("align_points", &align_points, "align two point sets",
    py::arg("target"),
    py::arg("source"),
    py::arg("method") = "GICP",
    py::arg("downsample_resolution") = -1.0,
    py::arg("k_correspondences") = 15,
    py::arg("max_correspondence_distance") = std::numeric_limits<double>::max(),
    py::arg("voxel_resolution") = 1.0,
    py::arg("num_threads") = 0,
    py::arg("neighbor_search_method") = "DIRECT1",
    py::arg("neighbor_search_radius") = 1.5,
    py::arg("initial_guess") = Eigen::Matrix4f::Identity()
  );

  py::class_<LsqRegistration, std::shared_ptr<LsqRegistration>>(m, "LsqRegistration")
    .def("set_input_target", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
    .def("set_input_source", [] (LsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
    .def("swap_source_and_target", &LsqRegistration::swapSourceAndTarget)
    .def("get_final_hessian", &LsqRegistration::getFinalHessian)
    .def("get_final_transformation", &LsqRegistration::getFinalTransformation)
    .def("get_fitness_score", [] (LsqRegistration& reg, const double max_range) { return reg.getFitnessScore(max_range); })
    .def("align",
      [] (LsqRegistration& reg, const Eigen::Matrix4f& initial_guess) { 
        pcl::PointCloud<pcl::PointXYZ> aligned;
        reg.align(aligned, initial_guess);
        return reg.getFinalTransformation();
      }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
    )
  ;

  py::class_<FastGICP, LsqRegistration, std::shared_ptr<FastGICP>>(m, "FastGICP")
    .def(py::init())
    .def("set_num_threads", &FastGICP::setNumThreads)
    .def("set_correspondence_randomness", &FastGICP::setCorrespondenceRandomness)
    .def("set_max_correspondence_distance", &FastGICP::setMaxCorrespondenceDistance)
  ;

  py::class_<FastVGICP, FastGICP, std::shared_ptr<FastVGICP>>(m, "FastVGICP")
    .def(py::init())
    .def("set_resolution", &FastVGICP::setResolution)
    .def("set_neighbor_search_method", [](FastVGICP& vgicp, const std::string& method) { vgicp.setNeighborSearchMethod(search_method(method)); })
  ;

#ifdef USE_VGICP_CUDA
  py::class_<FastVGICPCuda, LsqRegistration, std::shared_ptr<FastVGICPCuda>>(m, "FastVGICPCuda")
    .def(py::init())
    .def("set_resolution", &FastVGICPCuda::setResolution)
    .def("set_neighbor_search_method",
      [](FastVGICPCuda& vgicp, const std::string& method, double radius) { vgicp.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_correspondence_randomness", &FastVGICPCuda::setCorrespondenceRandomness)
  ;

  py::class_<NDTCuda, LsqRegistration, std::shared_ptr<NDTCuda>>(m, "NDTCuda")
    .def(py::init())
    .def("set_neighbor_search_method",
      [](NDTCuda& ndt, const std::string& method, double radius) { ndt.setNeighborSearchMethod(search_method(method), radius); }
      , py::arg("method") = "DIRECT1", py::arg("radius") = 1.5
    )
    .def("set_resolution", &NDTCuda::setResolution)
  ;
#endif

  // @sanghyun
  py::class_<PCLICP, std::shared_ptr<PCLICP>>(m, "PCLICP")
    .def(py::init())
    .def("set_max_correspondence_distance", &PCLICP::setMaxCorrespondenceDistance)
    .def("set_input_target", [] (PCLICP& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
    .def("set_input_source", [] (PCLICP& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
    .def("get_final_transformation", &PCLICP::getFinalTransformation)
    .def("align",
      [] (PCLICP& reg, const Eigen::Matrix4f& initial_guess) { 
        pcl::PointCloud<pcl::PointXYZ> aligned;
        reg.align(aligned, initial_guess);
        return reg.getFinalTransformation();
      }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
    )
  ;

  // @sanghyun TODO: Po2Pl needs normal information, see this page: 
  // https://github.com/tttamaki/ICP-test/blob/master/src/icp3_with_normal_iterative_view.cpp

  // py::class_<PCLICP_Po2Pl, std::shared_ptr<PCLICP_Po2Pl>>(m, "PCLICP_Po2Pl")
  //   .def(py::init())
  //   .def("set_max_correspondence_distance", &PCLICP_Po2Pl::setMaxCorrespondenceDistance)
  //   .def("set_input_target", [] (PCLICP_Po2Pl& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
  //   .def("set_input_source", [] (PCLICP_Po2Pl& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
  //   .def("get_final_transformation", &PCLICP_Po2Pl::getFinalTransformation)
  //   .def("align",
  //     [] (PCLICP_Po2Pl& reg, const Eigen::Matrix4f& initial_guess) { 
  //       pcl::PointCloud<pcl::PointXYZ> aligned;
  //       reg.align(aligned, initial_guess);
  //       return reg.getFinalTransformation();
  //     }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
  //   )
  // ;

  py::class_<PCLGICP, std::shared_ptr<PCLGICP>>(m, "PCLGICP")
    .def(py::init())
    .def("set_max_correspondence_distance", &PCLGICP::setMaxCorrespondenceDistance)
    .def("set_input_target", [] (PCLGICP& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
    .def("set_input_source", [] (PCLGICP& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
    .def("get_final_transformation", &PCLGICP::getFinalTransformation)
    .def("align",
      [] (PCLGICP& reg, const Eigen::Matrix4f& initial_guess) { 
        pcl::PointCloud<pcl::PointXYZ> aligned;
        reg.align(aligned, initial_guess);
        return reg.getFinalTransformation();
      }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
    )
  ;

  py::class_<PCLKDTree, std::shared_ptr<PCLKDTree>>(m, "PCLKDTree")
    .def(py::init())
    .def("set_input_cloud", [] (PCLKDTree& tree, const Eigen::Matrix<double, -1, 3>& points) { tree.setInputCloud(eigen2pcl(points)); })
    .def("knn", 
      [] (PCLKDTree& tree, const Eigen::Matrix<double, -1, 3>& points, int k, int num_threads) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_points = eigen2pcl(points);
        Eigen::MatrixXi k_indices_m(pcl_points->size(), k);
        
        std::vector<int> k_indices(k);
        std::vector<float> k_sq_dists(k);

        #pragma omp parallel for num_threads(num_threads) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
        for (int i = 0; i < pcl_points->size(); i++) {
          pcl::PointXYZ& pt = pcl_points->at(i);
          tree.nearestKSearch(pt, k, k_indices, k_sq_dists);

          for (int j = 0; j < k; j++)
            k_indices_m(i, j) = k_indices[j];
        }

        return k_indices_m;
      }
    )
  ;

  py::class_<dLsqRegistration, std::shared_ptr<dLsqRegistration>>(m, "dLsqRegistration")
    .def("set_input_target", [] (dLsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputTarget(eigen2pcl(points)); })
    .def("set_input_source", [] (dLsqRegistration& reg, const Eigen::Matrix<double, -1, 3>& points) { reg.setInputSource(eigen2pcl(points)); })
    .def("swap_source_and_target", &dLsqRegistration::swapSourceAndTarget)
    .def("get_final_hessian", &dLsqRegistration::getFinalHessian)
    .def("get_final_transformation", &dLsqRegistration::getFinalTransformation)
    .def("get_fitness_score", [] (dLsqRegistration& reg, const double max_range) { return reg.getFitnessScore(max_range); })
    .def("align",
      [] (dLsqRegistration& reg, const Eigen::Matrix4f& initial_guess) { 
        pcl::PointCloud<pcl::PointXYZ> aligned;
        reg.align(aligned, initial_guess);
        return reg.getFinalTransformation();
      }, py::arg("initial_guess") = Eigen::Matrix4f::Identity()
    )
  ;

  py::class_<dFastGICP, dLsqRegistration, std::shared_ptr<dFastGICP>>(m, "dFastGICP")
    .def(py::init())
    .def("set_num_threads", &dFastGICP::setNumThreads)
    .def("set_correspondence_randomness", &dFastGICP::setCorrespondenceRandomness)
    .def("set_max_correspondence_distance", &dFastGICP::setMaxCorrespondenceDistance)
    .def("set_source_covariances", [] (dFastGICP& reg, const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
      reg.setSourceCovariances(covs);
    })
    .def("set_target_covariances", [] (dFastGICP& reg, const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
      reg.setTargetCovariances(covs);
    })
  ;

  py::class_<dWGICP, dFastGICP, std::shared_ptr<dWGICP>>(m, "dWGICP")
    .def(py::init())
    .def("set_source_weights", [](dWGICP& wgicp, const std::vector<float>& weights){
      wgicp.setSourceWeights(weights);
    })
    .def("set_target_weights", [](dWGICP& wgicp, const std::vector<float>& weights){
      wgicp.setTargetWeights(weights);
    })
  ;

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}