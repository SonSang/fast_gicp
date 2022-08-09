#include <fast_gicp/diff/diff_fast_gicp_st.hpp>
#include <fast_gicp/diff/impl/diff_fast_gicp_st_impl.hpp>

template class fast_gicp::DiffFastGICPSingleThread<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::DiffFastGICPSingleThread<pcl::PointXYZI, pcl::PointXYZI>;
template class fast_gicp::DiffFastGICPSingleThread<pcl::PointNormal, pcl::PointNormal>;
