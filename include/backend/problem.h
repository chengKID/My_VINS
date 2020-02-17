#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

#include <iostream>
#include <fstream>

#include <pthread.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>

typedef unsigned long ulong;

namespace myslam {
namespace backend {

typedef unsigned long ulong;
//    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

// TODO: HOMEWORKE Multi-thread && SSE
struct ThreadsStruct {
    HashEdge sub_edges;
    MatXX sub_H;
    VecX sub_b;
};

class Problem {
public:

    /**
     * The type of problem
     * The problem like SLAM is still a general problem
     *
     * If it's a SLAM problem, then pose and landmark are different and Hessian Matrix is sparse
     * SLAM only allow sertain Vertex and Edge
     * If it's a general problem, then the hessian matrix is dense, if user don't define vertex as marginalized
     */
    enum class ProblemType {
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    /** 
     * //TODO: HOMEWORK implement Dog-Leg algorithm
     * 
     * \brief type of the step to take 
     * */
    enum {
        STEP_UNDEFINED,
        STEP_SD, STEP_GN, STEP_DL
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem(ProblemType problemType);

    ~Problem();

    bool AddVertex(std::shared_ptr<Vertex> vertex);

    /**
     * remove a vertex
     * @param vertex_to_remove
     */
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    bool AddEdge(std::shared_ptr<Edge> edge);

    bool RemoveEdge(std::shared_ptr<Edge> edge);

    /**
     * Get the edges, which are defined as outlier during the optimisation, so we can remove those outliers
     * @param outlier_edges
     */
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>> &outlier_edges);

    /**
     * Solve the problem
     * @param iterations
     * @return
     */
    bool Solve(int iterations = 10);

    /// Marginalize a frame and landmark, whose host is exactly this frame
    bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>> &landmarkVerticies);

    bool Marginalize(const std::shared_ptr<Vertex> frameVertex);
    bool Marginalize(const std::vector<std::shared_ptr<Vertex> > frameVertex,int pose_dim);

    MatXX GetHessianPrior(){ return H_prior_;}
    VecX GetbPrior(){ return b_prior_;}
    VecX GetErrPrior(){ return err_prior_;}
    MatXX GetJtPrior(){ return Jt_prior_inv_;}

    void SetHessianPrior(const MatXX& H){H_prior_ = H;}
    void SetbPrior(const VecX& b){b_prior_ = b;}
    void SetErrPrior(const VecX& b){err_prior_ = b;}
    void SetJtPrior(const MatXX& J){Jt_prior_inv_ = J;}

    void ExtendHessiansPriorSize(int dim);

    //test compute prior
    void TestComputePrior();

    // TODO: HOMEWORK store the lm's computation cost in txt
    double comput_time = 0.0;
private:

    /// Implement the Solve function for general problem
    bool SolveGenericProblem(int iterations);

    /// Implement the Solve function for SLAM problem
    bool SolveSLAMProblem(int iterations);

    /// Set the contex's ordering_index
    void SetOrdering();

    /// set ordering for new vertex in slam problem
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    /// Construct the big Hessian Matirx
    void MakeHessian();

    /// Solve the SBA by using schur complement
    void SchurSBA();

    /// Solve linear equation
    void SolveLinearSystem();

    /// Update the states
    void UpdateStates();

    void RollbackStates(); // When the residual increase after the update, redo

    /// Compute and update the Prior
    void ComputePrior();

    /// Determine ob the contex is Pose contex
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    /// Determine ob the contex is landmark contex
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    /// After input contex, we must change the dimention of hessian matrix
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    /// Check whether the ordering is right
    bool CheckOrdering();

    void LogoutVectorSize();

    /// Get the edge, which connects the contex
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    /// Levenberg
    /// Compute the initial Lambda
    void ComputeLambdaInitLM();

    /// The diagonal of Hessian matrix add/subtract Lambda
    void AddLambdatoHessianLM();

    void RemoveLambdaHessianLM();

    /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
    bool IsGoodStepInLM();

    /// PCG 迭代线性求解器
    VecX PCGSolver(const MatXX &A, const VecX &b, int maxIter);

    double currentLambda_;
    double currentChi_;
    double stopThresholdLM_;    // LM 迭代退出阈值条件
    double ni_;                 //控制 Lambda 缩放大小

    ProblemType problemType_;

    /// Entire information matrix
    MatXX Hessian_;
    VecX b_;
    VecX delta_x_;

    /// The prior part
    MatXX H_prior_;
    VecX b_prior_;
    VecX b_prior_backup_;
    VecX err_prior_backup_;

    MatXX Jt_prior_inv_;
    VecX err_prior_;

    /// SBA's Pose
    MatXX H_pp_schur_;
    VecX b_pp_schur_;
    // Heesian's Landmark and pose part
    MatXX H_pp_;
    VecX b_pp_;
    MatXX H_ll_;
    VecX b_ll_;

    /// all vertices
    HashVertex verticies_;

    /// all edges
    HashEdge edges_;

    /// through vertex id search the edge
    HashVertexIdToEdge vertexToEdge_;

    /// Ordering related
    ulong ordering_poses_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;        // based on ordering, pose顶点
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // based on ordering, landmark顶点

    // verticies need to marg. <Ordering_id_, Vertex>
    HashVertex verticies_marg_;

    bool bDebug = false;
    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;

    // TODO: HOMEWORK Dog-Leg algorithm
    VecX _auxVector;   ///< auxilary vector used to perform multiplications or other stuff
    double _normJG;
    double _normG;

    VecX hsd;         ///< steepest decent step
    VecX hgn;         ///< gaussian newton step
    VecX hdl;         ///< final dogleg step

    double _delta = 1.0; //100      ///< trust region
    double alpha;
    double beta;
    int _lastStep;                ///< type of the step taken by the algorithm
    bool stop;
};

}
}

#endif
