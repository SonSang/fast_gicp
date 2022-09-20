#include <fast_gicp/wgicp/d_fast_gicp.hpp>
#include <fast_gicp/wgicp/impl/d_fast_gicp_impl.hpp>

template class wgicp::dFastGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class wgicp::dFastGICP<pcl::PointXYZI, pcl::PointXYZI>;
template class wgicp::dFastGICP<pcl::PointNormal, pcl::PointNormal>;
