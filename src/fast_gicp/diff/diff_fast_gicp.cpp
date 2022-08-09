#include <fast_gicp/diff/diff_fast_gicp.hpp>
#include <fast_gicp/diff/impl/diff_fast_gicp_impl.hpp>

template class fast_gicp::DiffFastGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::DiffFastGICP<pcl::PointXYZI, pcl::PointXYZI>;
template class fast_gicp::DiffFastGICP<pcl::PointNormal, pcl::PointNormal>;
