#include <fast_gicp/wgicp/d_lsq_registration.hpp>
#include <fast_gicp/wgicp/impl/d_lsq_registration_impl.hpp>

template class wgicp::dLsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;
template class wgicp::dLsqRegistration<pcl::PointXYZI, pcl::PointXYZI>;
template class wgicp::dLsqRegistration<pcl::PointNormal, pcl::PointNormal>;
