#ifndef WGICP_D_WGICP_IMPL_HPP
#define WGICP_D_WGICP_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>

namespace wgicp {

template <typename PointSource, typename PointTarget>
dWGICP<PointSource, PointTarget>::dWGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 20;
  reg_name_ = "dWGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  regularization_method_ = fast_gicp::RegularizationMethod::PLANE;
  source_kdtree_.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree_.reset(new pcl::search::KdTree<PointTarget>);

  source_kdtree_cov_.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree_cov_.reset(new pcl::search::KdTree<PointTarget>);
}

template <typename PointSource, typename PointTarget>
dWGICP<PointSource, PointTarget>::~dWGICP() {}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree_.swap(target_kdtree_);
  source_kdtree_cov_.swap(target_kdtree_cov_);
  source_covs_.swap(target_covs_);
  source_weights_.swap(target_weights_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
  source_weights_.clear();
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
  target_weights_.clear();
}


template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  source_kdtree_->setInputCloud(cloud);
  source_kdtree_cov_->setInputCloud(cloud);
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  target_kdtree_->setInputCloud(cloud);
  target_kdtree_cov_->setInputCloud(cloud);
  target_covs_.clear();
}


template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::setInputSourceCov(const PointCloudSourceConstPtr& cloud) {
  source_kdtree_cov_->setInputCloud(cloud);
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::setInputTargetCov(const PointCloudTargetConstPtr& cloud) {
  target_kdtree_cov_->setInputCloud(cloud);
  target_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::setSourceWeights(const std::vector<float>& weights) {
  source_weights_ = weights;
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::setTargetWeights(const std::vector<float>& weights) {
  target_weights_ = weights;
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (source_covs_.size() != input_->size()) {
    dFastGICP<PointSource, PointTarget>::calculate_covariances(input_, *source_kdtree_cov_, source_covs_);
  }
  if (target_covs_.size() != target_->size()) {
    dFastGICP<PointSource, PointTarget>::calculate_covariances(target_, *target_kdtree_cov_, target_covs_);
  }

  dLsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
void dWGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());
  assert(source_weights_.size() == input_->size());
  assert(target_weights_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.resize(input_->size());
  sq_distances_.resize(input_->size());
  mahalanobis_.resize(input_->size());

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();

    target_kdtree_->nearestKSearch(pt, 1, k_indices, k_sq_dists);

    sq_distances_[i] = k_sq_dists[0];
    correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;

    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = source_covs_[i];
    const auto& cov_B = target_covs_[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    mahalanobis_[i] = RCR.inverse();
    mahalanobis_[i](3, 3) = 0.0f;

    Eigen::Matrix4d WeightMatrix = Eigen::Matrix4d::Identity() * source_weights_[i] * target_weights_[target_index];
    mahalanobis_[i] = mahalanobis_[i] * WeightMatrix;
  }
}

}  // namespace wgicp

#endif
