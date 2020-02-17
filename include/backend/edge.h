#ifndef MYSLAM_BACKEND_EDGE_H
#define MYSLAM_BACKEND_EDGE_H

#include <memory>
#include <string>
#include "eigen_types.h"
#include <eigen3/Eigen/Dense>
#include "loss_function.h"

namespace myslam {
namespace backend {

class Vertex;

/**
 * With edge we can compute the residual, residual = prediction - measurement，the dimention is defined in function
 * Cost function is residual*information*residual，it is a number, which can be minimised at the back-end
 */
class Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * constructor, automaticaly allocate the space for jacobi
     * @param residual_dimension 残差维度
     * @param num_verticies 顶点数量
     * @param verticies_types 顶点类型名称, if not provide, we don't check the type
     */
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string> &verticies_types = std::vector<std::string>());

    virtual ~Edge();

    /// Return id
    unsigned long Id() const { return id_; }

    /**
     * Set the vertex
     * @param vertex corresponding vertex object
     */
    bool AddVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

    /**
     * Set some contexs
     * @param vertices set the sequence by reference oder
     * @return
     */
    bool SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices) {
        verticies_ = vertices;
        return true;
    }

    /// Return the first vertex
    std::shared_ptr<Vertex> GetVertex(int i) {
        return verticies_[i];
    }

    /// Return all vertexs
    std::vector<std::shared_ptr<Vertex>> Verticies() const {
        return verticies_;
    }

    /// Return the number of all vertexs
    size_t NumVertices() const { return verticies_.size(); }

    /// Return the type, this would be implemented in derived class
    virtual std::string TypeInfo() const = 0;

    /// Compute the residual, this would be implemented in derived class
    virtual void ComputeResidual() = 0;

    /// Compute jacobi, this would be implemented in derived class
    /// This back-end don't allow automatic compute the derivation, this jacobi must be implemented in derived class
    virtual void ComputeJacobians() = 0;

//    ///计算该edge对Hession矩阵的影响，由子类实现
//    virtual void ComputeHessionFactor() = 0;

    /// Compute the squared error, it should multiply informatin matrix
    double Chi2() const;
    double RobustChi2() const;

    /// Return residual
    VecX Residual() const { return residual_; }

    /// Return jacobi
    std::vector<MatXX> Jacobians() const { return jacobians_; }

    /// Set information matrix
    void SetInformation(const MatXX &information) {
        information_ = information;
        // sqrt information
        sqrt_information_ = Eigen::LLT<MatXX>(information_).matrixL().transpose();
    }

    /// Return information matrix
    MatXX Information() const {
        return information_;
    }

    MatXX SqrtInformation() const {
        return sqrt_information_;
    }

    void SetLossFunction(LossFunction* ptr){ lossfunction_ = ptr; }
    LossFunction* GetLossFunction(){ return lossfunction_;}
    void RobustInfo(double& drho, MatXX& info) const;

    /// Set observation
    void SetObservation(const VecX &observation) {
        observation_ = observation;
    }

    /// Return observation
    VecX Observation() const { return observation_; }

    /// Check whether all edge information have been seted
    bool CheckValid();

    int OrderingId() const { return ordering_id_; }

    void SetOrderingId(int id) { ordering_id_ = id; };

protected:
    unsigned long id_;  // edge id
    int ordering_id_;   //edge id in problem
    std::vector<std::string> verticies_types_;  // all vertexs info. used for debug
    std::vector<std::shared_ptr<Vertex>> verticies_; // the corresponding vertex
    VecX residual_;                 // residual
    std::vector<MatXX> jacobians_;  // jacobi，the dimention is residual x vertex[i]
    MatXX information_;             // information matrix
    MatXX sqrt_information_;
    VecX observation_;              // observations

    LossFunction *lossfunction_;
};

}
}

#endif
