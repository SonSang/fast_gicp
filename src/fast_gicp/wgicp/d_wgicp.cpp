#include <fast_gicp/wgicp/d_wgicp.hpp>
#include <fast_gicp/wgicp/impl/d_wgicp_impl.hpp>

template class wgicp::dWGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class wgicp::dWGICP<pcl::PointXYZI, pcl::PointXYZI>;
template class wgicp::dWGICP<pcl::PointNormal, pcl::PointNormal>;
