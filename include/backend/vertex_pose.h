#ifndef MYSLAM_BACKEND_POSEVERTEX_H
#define MYSLAM_BACKEND_POSEVERTEX_H

#include <memory>
#include "vertex.h"

namespace myslam {
namespace backend {

/**
 * Pose vertex
 * parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
 * optimization is perform on manifold, so update is 6 DoF, left multiplication
 *
 * pose is represented as Twb in VIO case
 */
class VertexPose : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPose() : Vertex(7, 6) {}

    /// Addition, can be overrided
    /// default is vector addition
    virtual void Plus(const VecX &delta) override;

    std::string TypeInfo() const {
        return "VertexPose";
    }

    /**
     * need to maintain [H|b] matrix's type like following data structur
     * p: pose, m:mappoint
     * 
     *     Hp1_p2    
     *     Hp2_p2    Hp2_m1    Hp2_m2    Hp2_m3     |    bp2
     *                         
     *                         Hm2_m2               |    bm2
     *                                   Hm2_m3     |    bm3
     * 1. If this Camera is source camera，then maintain the vHessionSourceCamera；
     * 2. If this Camera is measurement camera, then maintain the vHessionMeasurementCamera；
     * 3. always maitain m_HessionDiagonal；
     */
};

}
}

#endif
