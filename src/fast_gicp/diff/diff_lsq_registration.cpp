#include <fast_gicp/diff/diff_lsq_registration.hpp>
#include <fast_gicp/diff/impl/diff_lsq_registration_impl.hpp>

template class fast_gicp::DiffLsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::DiffLsqRegistration<pcl::PointXYZI, pcl::PointXYZI>;
template class fast_gicp::DiffLsqRegistration<pcl::PointNormal, pcl::PointNormal>;
