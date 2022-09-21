#ifndef WGICP_D_WGICP_HPP
#define WGICP_D_WGICP_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>
#include <fast_gicp/wgicp/d_lsq_registration.hpp>
#include <fast_gicp/wgicp/d_fast_gicp.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace wgicp {

/**
 * @brief Fast GICP algorithm optimized for multi threading with OpenMP
 */
template<typename PointSource, typename PointTarget>
class dWGICP : public dFastGICP<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<dWGICP<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const dWGICP<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<dWGICP<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const dWGICP<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

  using wgicp::dFastGICP<PointSource, PointTarget>::num_threads_;
  using wgicp::dFastGICP<PointSource, PointTarget>::k_correspondences_;
  using wgicp::dFastGICP<PointSource, PointTarget>::regularization_method_;
  using wgicp::dFastGICP<PointSource, PointTarget>::source_kdtree_;
  using wgicp::dFastGICP<PointSource, PointTarget>::target_kdtree_;
  using wgicp::dFastGICP<PointSource, PointTarget>::source_covs_;
  using wgicp::dFastGICP<PointSource, PointTarget>::target_covs_;
  using wgicp::dFastGICP<PointSource, PointTarget>::mahalanobis_;
  using wgicp::dFastGICP<PointSource, PointTarget>::correspondences_;
  using wgicp::dFastGICP<PointSource, PointTarget>::sq_distances_;
  
  

public:
  dWGICP();
  virtual ~dWGICP() override;

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  virtual void setSourceWeights(const std::vector<float>& weights);
  virtual void setTargetWeights(const std::vector<float>& weights);

  const std::vector<float>& getSourceWeights() const {
    return source_weights_;
  }

  const std::vector<float>& getTargetWeights() const {
    return target_weights_;
  }

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  virtual void setInputSourceCov(const PointCloudSourceConstPtr& cloud);
  virtual void setInputTargetCov(const PointCloudTargetConstPtr& cloud);
  
protected:
  virtual void update_correspondences(const Eigen::Isometry3d& trans);
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;
protected:
  std::vector<float> source_weights_;
  std::vector<float> target_weights_;

  // KDTrees used for computing covariances only; this is needed for rejection
  std::shared_ptr<pcl::search::KdTree<PointSource>> source_kdtree_cov_;
  std::shared_ptr<pcl::search::KdTree<PointTarget>> target_kdtree_cov_;
};
}  // namespace wgicp

#endif