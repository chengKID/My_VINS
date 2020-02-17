#ifndef MYSLAM_BACKEND_VERTEX_H
#define MYSLAM_BACKEND_VERTEX_H

#include "eigen_types.h"

namespace myslam {
namespace backend {
extern unsigned long global_vertex_id;
/**
 * @brief vertex corresponds to parameter block
 * the variable is stored as VecX，the dimention should be defined with constructor
 */
class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * Constructor
     * @param num_dimension 
     * @param local_dimension，if it's -1, then it has the same dimention as the local dimention
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);

    virtual ~Vertex();

    /// Return the dimention of the variable
    int Dimension() const;

    /// Return the dimention of the local variable
    int LocalDimension() const;

    /// Return the id of this contex
    unsigned long Id() const { return id_; }

    /// Retrun the parameters
    VecX Parameters() const { return parameters_; }

    /// Return the reference of the parameters
    VecX &Parameters() { return parameters_; }

    /// Set the parameters
    void SetParameters(const VecX &params) { parameters_ = params; }

    // Backup and reset the parameters, it's useful at the time when the estimation is bad
    void BackUpParameters() { parameters_backup_ = parameters_; }
    void RollBackParameters() { parameters_ = parameters_backup_; }

    /// Addition，can be overrided
    /// default is vedtor addition
    virtual void Plus(const VecX &delta);

    /// Return the derived class's name, it can be implement in the derived class
    virtual std::string TypeInfo() const = 0;

    int OrderingId() const { return ordering_id_; }

    void SetOrderingId(unsigned long id) { ordering_id_ = id; };

    /// Fix the estimator
    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }

    /// Check whether this point is fixed
    bool IsFixed() const { return fixed_; }

protected:
    VecX parameters_;   // current variable
    VecX parameters_backup_; // back up the parameters at every step of iteration，用于回滚
    int local_dimension_;   // dimention of local parameters
    unsigned long id_;  // contex's id

    /// ordering id是在problem中排序后的id，用于寻找雅可比对应块
    /// ordering id带有维度信息，例如ordering_id=6则对应Hessian中的第6列
    /// start from 0
    unsigned long ordering_id_ = 0;

    bool fixed_ = false;    // ob it's fixed
};

}
}

#endif
