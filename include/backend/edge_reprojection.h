#ifndef MYSLAM_BACKEND_VISUALEDGE_H
#define MYSLAM_BACKEND_VISUALEDGE_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"

namespace myslam {
namespace backend {

/**
 * This edge is visual reprojection error and it has three contexs, the contexs are:
 * the InveseDepth of landmarks、the camera pose at the time that this landmark is observed first time: T_World_From_Body1，
 * and the corresponding mearsurement of Camera pose: T_World_From_Body2。
 * Note: the order of verticies_contex must be InveseDepth、T_World_From_Body1、T_World_From_Body2.
 */
class EdgeReprojection : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojection(const Vec3 &pts_i, const Vec3 &pts_j)
        : Edge(2, 4, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose", "VertexPose"}) {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }

    /// Return the type of the information
    virtual std::string TypeInfo() const override { return "EdgeReprojection"; }

    /// Compute the residual
    virtual void ComputeResidual() override;

    /// Compute the jacobi
    virtual void ComputeJacobians() override;

//    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    //Translation imu from camera
//    Qd qic;
//    Vec3 tic;

    //measurements
    Vec3 pts_i_, pts_j_;
};

/**
* This edge is visual reprojection error and it has two contexs, the contexs are:
* landmark's XYZ in world coordinante system、the camera pose which can observe the landmark: T_World_From_Body1
* Note: the order of verticies_contex must be XYZ、T_World_From_Body1。
*/
class EdgeReprojectionXYZ : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojectionXYZ(const Vec3 &pts_i)
        : Edge(2, 2, std::vector<std::string>{"VertexXYZ", "VertexPose"}) {
        obs_ = pts_i;
    }

    /// Return the type of the information
    virtual std::string TypeInfo() const override { return "EdgeReprojectionXYZ"; }

    /// Compute the residual
    virtual void ComputeResidual() override;

    /// Compute the jacobi
    virtual void ComputeJacobians() override;

    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    //Translation imu from camera
    Qd qic;
    Vec3 tic;

    //measurements
    Vec3 obs_;
};

/**
 * Just a example for computing the reprojects' pose
 */
class EdgeReprojectionPoseOnly : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojectionPoseOnly(const Vec3 &landmark_world, const Mat33 &K) :
        Edge(2, 1, std::vector<std::string>{"VertexPose"}),
        landmark_world_(landmark_world), K_(K) {}

    /// Return the type of the information
    virtual std::string TypeInfo() const override { return "EdgeReprojectionPoseOnly"; }

    /// Compute the residual
    virtual void ComputeResidual() override;

    /// Compute the jacobi
    virtual void ComputeJacobians() override;

private:
    Vec3 landmark_world_;
    Mat33 K_;
};

}
}

#endif
