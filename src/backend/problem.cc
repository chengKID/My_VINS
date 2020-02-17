#include <eigen3/Eigen/Dense>
#include <iomanip>
#include "backend/problem.h"
#include "utility/tic_toc.h"

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

namespace myslam {
namespace backend {
void Problem::LogoutVectorSize() {
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

Problem::Problem(ProblemType problemType) :
    problemType_(problemType) {
    LogoutVectorSize();
    verticies_marg_.clear();
}

Problem::~Problem() {
//    std::cout << "Problem IS Deleted"<<std::endl;
    global_vertex_id = 0;
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if (IsPoseVertex(vertex)) {
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }
    return true;
}

void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) {
    if (IsPoseVertex(v)) {
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        ordering_poses_ += v->LocalDimension();
    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}

void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {

    int size = H_prior_.rows() + v->LocalDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->LocalDimension()).setZero();
    H_prior_.rightCols(v->LocalDimension()).setZero();
    H_prior_.bottomRows(v->LocalDimension()).setZero();

}
void Problem::ExtendHessiansPriorSize(int dim)
{
    int size = H_prior_.rows() + dim;
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(dim).setZero();
    H_prior_.rightCols(dim).setZero();
    H_prior_.bottomRows(dim).setZero();
}

bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPose") ||
            type == string("VertexSpeedBias");
}

bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ") ||
           type == string("VertexInverseDepth");
}

bool Problem::AddEdge(shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }

    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {

        // This edge still exist, it hasn't been removed
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // Here this contex's edge should be removed.
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }

    if (IsPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->Id());
    else
        idx_landmark_vertices_.erase(vertex->Id());

    vertex->SetOrderingId(-1);      // used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());

    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
    //check if the edge is in map_edges_
    if (edges_.find(edge->Id()) == edges_.end()) {
        // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
        return false;
    }

    edges_.erase(edge->Id());
    return true;
}

bool Problem::Solve(int iterations) {
    bool use_update_lmorg = false;
    bool use_update_lm_1 = true;
    bool use_update_dl = false;

    // LM with ORIGIN Update Strategy =======================================================================
    if (use_update_lmorg) {
        if (edges_.size() == 0 || verticies_.size() == 0)
        {
            std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
            return false;
        }

        TicToc t_solve;
        // Counting the dimention of changed variables，preparing for constructing the Hessian matrix
        SetOrdering();
        // search throught the edges, constructs the Hessian matrix
        MakeHessian();
        // LM initial
        ComputeLambdaInitLM();

        // LM iterative
        bool stop = false;
        int iter = 0;
        double last_chi_ = 1e20;
        while (!stop && (iter < iterations))
        {
            std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
            bool oneStepSuccess = false;
            int false_cnt = 0;
            while (!oneStepSuccess && false_cnt < 10) // try Lambda, until sucess
            {
                SolveLinearSystem();

                // Update variables
                UpdateStates();
                // determine whether current step is possible and LM's lambda, we also need to check chi2
                oneStepSuccess = IsGoodStepInLM();
                // follow-up work
                if (oneStepSuccess)
                {
                    // at the new point construct hessian
                    MakeHessian();
                    false_cnt = 0;
                }
                else
                {
                    false_cnt++;
                    RollbackStates(); // reset
                }
            }
            iter++;

            if (last_chi_ - currentChi_ < 1e-5)
            {
                std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
                stop = true;
            }
            last_chi_ = currentChi_;
        }
        // TODO: HOMEWORK write time cost in txt
        comput_time = t_solve.toc();

        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
        std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
        t_hessian_cost_ = 0.;
        return true;
    }
    // END LM with ORIGIN Update Strategy =======================================================================




    // TODO: LM with New Update Strategy 1 =======================================================================
    if (use_update_lm_1) {
        if (edges_.size() == 0 || verticies_.size() == 0)
        {
            std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
            return false;
        }

        TicToc t_solve;
        // Counting the dimention of changed variables，preparing for constructing the Hessian matrix
        SetOrdering();
        // search throught the edges, constructs the Hessian matrix
        MakeHessian();
        // LM initial
        ComputeLambdaInitLM();

        // LM iterative
        bool stop = false;
        int iter = 0;
        while (!stop && (iter < iterations))
        {
            std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
            bool oneStepSuccess = false;
            int false_cnt = 0;
            while (!oneStepSuccess) // try Lambda, until sucess
            {
                SolveLinearSystem();

                // stop condition 1: delta_x_ is to small
                if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10)
                {
                    stop = true;
                    break;
                }

                // update the states
                UpdateStates();
                // determine whether current step is possible and LM's lambda, we also need to check chi2
                oneStepSuccess = IsGoodStepInLM();
                // follow-up work
                if (oneStepSuccess)
                {
                    // at the new point construct hessian
                    MakeHessian();
                    false_cnt = 0;
                }
                else
                {
                    false_cnt++;
                    RollbackStates(); // reset
                }
            }
            iter++;

            if (sqrt(currentChi_) <= stopThresholdLM_)
                stop = true;
        }
        // TODO: HOMEWORK write time cost in txt
        comput_time = t_solve.toc();

        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
        std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
        t_hessian_cost_ = 0.;
        return true;
    }
    // END LM with New Update Strategy 1 =======================================================================




    // TODO: HOMEWORK Dog-Leg algorithm
    if (use_update_dl)
    {
        if (edges_.size() == 0 || verticies_.size() == 0) {
            std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
            return false;
        }

        TicToc t_solve;
        // Counting the dimention of changed variables，preparing for constructing the Hessian matrix
        SetOrdering();
        // search throught the edges, constructs the Hessian matrix
        MakeHessian();
        // compute currentChi_
        currentChi_ = 0.0;
        for (auto edge : edges_) {
            currentChi_ += edge.second->RobustChi2();
        }
        if (err_prior_.rows() > 0)
            currentChi_ += err_prior_.squaredNorm();
        currentChi_ *= 0.5;
        //stopThresholdLM_ = 1e-10 * sqrt(currentChi_);          // stop condition: the error reduce 1e-6 times

        // TODO: HOMEWORK Dog-Leg iterative
        stop = false;
        int iter = 0;
        while (!stop && (iter < iterations)) {
            std::cout << "iter: " << iter << " , chi= " << currentChi_ << std::endl;
            // Dog-Leg algorithm compute alpha
            // compute gradient and rescale
            _normG = b_.squaredNorm();

            // accelarate computing b^T * H * b
            int poses_size = ordering_poses_;
            int landmarks_size = ordering_landmarks_;
            MatXX _Hll = Hessian_.block(poses_size, poses_size, landmarks_size, landmarks_size);
            MatXX _Hpl = Hessian_.block(0, poses_size, poses_size, landmarks_size);
            MatXX _Hpp = Hessian_.block(0, 0, poses_size, poses_size);
            VecX _bpp = b_.segment(0, poses_size);
            VecX _bll = b_.segment(poses_size, landmarks_size);
            //_normJG = b_.transpose() * Hessian_ * b_;
            auto tmp1 = _bpp.transpose() * _Hpp * _bpp;
            auto tmp2 = _bpp.transpose() * _Hpl * _bll;
            auto tmp3 = _bll.transpose() * _Hll * _bll;
            _normJG = tmp1[0] + 2 * tmp2[0] + tmp3[0];
            alpha = _normG / _normJG;

            if (_normJG > 0) {
                // gradient descent step
                hsd = b_ * alpha;
            } else {
                // gradient descent step
                hsd = b_ * (_delta / sqrt(_normG));
            }

            // Gredient descent step and gauss newton step
            double hsdNorm = hsd.norm();
            double hgnNorm;
            bool oneStepSuccess = false;
            bool solvedGaussNewton = false;
            int false_cnt = 0;
            while (!oneStepSuccess) // try Lambda, untill success
            { 
                if (alpha*hsdNorm >= _delta) {   //if (hsdNorm >= _delta) {
                    hdl = hsd * (_delta / hsdNorm);
                    _lastStep = STEP_SD;
                }
                else {
                    if (!solvedGaussNewton) {
                        solvedGaussNewton = true;
                        SolveLinearSystem();

                        hgn = delta_x_;
                        hgnNorm = hgn.norm();
                    }

                    if (hgnNorm <= _delta) {
                        hdl = hgn;
                        _lastStep = STEP_GN;
                    } else {
                        _auxVector.setZero();
                        VecX a = alpha * hsd; 
                        _auxVector = hgn - a; 
                        double c = a.dot(_auxVector);                   
                        double bmaSquaredNorm = _auxVector.squaredNorm();
                        if (c <= 0.)                                   
                            beta = (-c + sqrt(c * c + bmaSquaredNorm * (_delta * _delta - a.squaredNorm()))) / bmaSquaredNorm;
                        else {
                            double hsdSqrNorm = a.squaredNorm();        
                            beta = (_delta * _delta - hsdSqrNorm) / (c + sqrt(c * c + bmaSquaredNorm * (_delta * _delta - hsdSqrNorm)));
                        }
                        assert(beta >= 0. && beta <= 1. && "Error while computing beta");

                        hdl = a + beta * (hgn - a);                      
                        _lastStep = STEP_DL;
                        assert(hdl.norm() < _delta + 1e-5 && "Computed step does not correspond to the trust region");
                    }
                }

                // Check update, if too small, then return
                if (hdl.norm() <= 1e-6 || false_cnt > 10) {
                    stop = true;
                    break;
                }
                
                // update states
                // update vertex
                UpdateStates();
                oneStepSuccess = IsGoodStepInLM();
                // follow-up work
                if (oneStepSuccess) {
                    // at the new point construct hessian
                    MakeHessian();
                    false_cnt = 0;
                } else {
                    false_cnt++;
                    // reset
                    // roll back states
                    RollbackStates();
                }
            }
            iter++;
        }
        // TODO: HOMEWORK write time cost in txt
        comput_time = t_solve.toc();

        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
        std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
        t_hessian_cost_ = 0.;
        return true;
    }
}

bool Problem::SolveGenericProblem(int iterations) {
    return true;
}

void Problem::SetOrdering() {

    // reset the counting
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    // Note:: verticies_ is map's type, the sequence is the same as id's order
    for (auto vertex: verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();  // the total dimention of all variables

        if (problemType_ == ProblemType::SLAM_PROBLEM)    // If it's a slam problem, we need to differ pose and landmark's dimentions，following the ordering
        {
            AddOrderingSLAM(vertex.second);
        }

    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // Here we need to add landmark's ordering with pose's nubmer，so that we can always keep landmark at back, and pose at front
        ulong all_pose_dimension = ordering_poses_;
        for (auto landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->SetOrderingId(
                landmarkVertex.second->OrderingId() + all_pose_dimension
            );
        }
    }

//    CHECK_EQ(CheckOrdering(), true);
}

bool Problem::CheckOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        for (auto v: idx_pose_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }

        for (auto v: idx_landmark_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}

// TODO: HOMEWORK Multi-thread
static inline double CalcDotProduct(__m128d x, __m128d y) 
{
    __m128d mul_res;
    mul_res = _mm_mul_pd(x, y);
    
    __m128 undef = _mm_undefined_ps();
    __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(mul_res));
    __m128d shuf = _mm_castps_pd(shuftmp);
    return _mm_cvtsd_f64(_mm_add_sd(mul_res, shuf));
}

MatXX MatrixMultiSSE(MatXX left_mat, MatXX right_mat) {
    int left_rows = left_mat.rows();
    int left_cols = left_mat.cols();
    int right_rows = right_mat.rows();
    int right_cols = right_mat.cols();
    MatXX product_mat(left_rows, right_cols);
    
    for (int i = 0; i < left_rows; i++) 
    {
        double ith_left_row[left_cols];
        Eigen::Map<VecX>(ith_left_row, left_cols) = left_mat.row(i);
        __m128d ith_left_sse = _mm_load_pd(ith_left_row);

        for (int j = 0; j < right_cols; j++)
        {
            //double result;
            double jth_right_col[right_rows];
            Eigen::Map<VecX>(jth_right_col, right_rows) = right_mat.col(j);
            __m128d jth_right_sse = _mm_load_pd(jth_right_col);

            product_mat(i, j) = CalcDotProduct(ith_left_sse, jth_right_sse);
        }
    }
    return product_mat;
}

MatXX MatrixAddSSE(MatXX left_mat, MatXX right_mat) {
    int left_rows = right_mat.rows();
    int left_cols = right_mat.cols();
    MatXX add_mat(left_rows, left_cols);

    for (int i = 0; i < left_cols; i++)
    {
        double result[left_rows];
        double ith_left_col[left_rows];
        double ith_right_col[left_rows];

        Eigen::Map<VecX>(ith_left_col, left_rows) = left_mat.col(i);
        Eigen::Map<VecX>(ith_right_col, left_rows) = right_mat.col(i);
        __m128d ith_left_sse = _mm_load_pd(ith_left_col);
        __m128d ith_right_sse = _mm_load_pd(ith_right_col);
        __m128d add_sse = _mm_add_pd(ith_left_sse, ith_right_sse);
        _mm_store_pd(result, add_sse);
        add_mat.col(i) = Eigen::Map<VecX>(result, left_rows);
    }
    return add_mat;
}

MatXX VectorSubSSE(VecX left_vec, VecX right_vec) {
    int left_rows = left_vec.size();
    VecX sub_vec(VecX::Zero(left_rows));
    
    double result[left_rows];
    double left_array[left_rows];
    double right_array[left_rows];
    Eigen::Map<VecX>(left_array, left_rows) = left_vec;
    Eigen::Map<VecX>(right_array, left_rows) = right_vec;
    __m128d left_array_sse = _mm_load_pd(left_array);
    __m128d right_array_sse = _mm_load_pd(right_array);
    __m128d sub_sse = _mm_sub_pd(left_array_sse, right_array_sse);
    _mm_store_pd(result, sub_sse);
    sub_vec = Eigen::Map<VecX>(result, left_rows);

    return sub_vec;
}

VecX MatVecMultiSSE(MatXX left_mat, VecX right_vec) {
    int left_rows = left_mat.rows();
    int left_cols = left_mat.cols();
    int right_rows = right_vec.size();
    VecX product_vec(VecX::Zero(left_rows));
    
    for (int i = 0; i < left_rows; i++) 
    {
        double ith_left_row[left_cols];
        Eigen::Map<VecX>(ith_left_row, left_cols) = left_mat.row(i);
        __m128d ith_left_sse = _mm_load_pd(ith_left_row);

        double right_array[right_rows];
        Eigen::Map<VecX>(right_array, right_rows) = right_vec;
        __m128d right_array_sse = _mm_load_pd(right_array);

        product_vec(i) = CalcDotProduct(ith_left_sse, right_array_sse);
    }
    return product_vec;    
}

VecX MatDoubleMultiSSE(VecX vector_, double a_) {
    __m128d a_sse = _mm_load_pd(&a_);
    int vec_size = vector_.size();
    VecX mul_vec(VecX::Zero(vec_size));
    for (int i = 0; i < vec_size; i++) 
    {
        double result = 0.0;
        double vec_array[vec_size];
        Eigen::Map<VecX>(vec_array, vec_size) = vector_;
        __m128d ith_vec_sse = _mm_load_pd(vec_array);
        __m128d product_sse = _mm_mul_pd(a_sse, ith_vec_sse);
        _mm_store_pd(&result, product_sse);
        mul_vec(i) = result;
    }
}

// TODO: HOMEWORK Multi-thread
void* HessianConstruct(void* threadsstruct) {
    ThreadsStruct* p = ((ThreadsStruct*) threadsstruct);
    // USE OPENMP to calculate with multi-thread
    //Eigen::setNbThreads(2);
    //#pragma omp parallel for
    for (auto &edge : p->sub_edges) {
        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();

        // TODO:: robust cost
        auto jacobians = edge.second->Jacobians();
        auto verticies = edge.second->Verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian has not other information，so its jacobi is 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            // Robust kernel function will change the residual and information matrix，if robust cost function has not been seted，then return the original
            double drho;
            MatXX robustInfo(edge.second->Information().rows(),edge.second->Information().cols());
            edge.second->RobustInfo(drho,robustInfo);

            // TODO: Accelarate using SSE
            MatXX JtW = jacobian_i.transpose() * robustInfo; 
            //MatXX T_jacobian = jacobian_i.transpose();
            //MatXX JtW = MatrixMultiSSE(T_jacobian, robustInfo);

            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                // TODO: Accelarate using SSE
                MatXX hessian = JtW * jacobian_j;
                //MatXX hessian = MatrixMultiSSE(JtW, jacobian_j);

                // Add all the information matrix
                // TODO: Accelarate using SSE
                p->sub_H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                MatXX tmp_H = p->sub_H.block(index_i, index_j, dim_i, dim_j);
                //MatXX tmp = p->sub_H.block(index_i, index_j, dim_i, dim_j);
                //MatXX tmp_H = MatrixAddSSE(tmp, hessian);
                //p->sub_H.block(index_i, index_j, dim_i, dim_j) = tmp_H;

                if (j != i) {
                    // Symmetric lower triangle matrix
                    // p->sub_H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    p->sub_H.block(index_j, index_i, dim_j, dim_i) = tmp_H.transpose();
                }
            }
            // TODO: Accelarate using SSE
            p->sub_b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge.second->Information() * edge.second->Residual();
            /*MatXX tmp_J = drho * jacobian_i.transpose();
            MatXX jaco_info_mat = MatrixMultiSSE(tmp_J, edge.second->Information());
            VecX JRhoR = MatVecMultiSSE(jaco_info_mat, edge.second->Residual());
            VecX tmp_b = p->sub_b.segment(index_i, dim_i);
            p->sub_b.segment(index_i, dim_i) = VectorSubSSE(tmp_b, JRhoR);*/          
        }
    }
    //Eigen::setNbThreads(0);
    return threadsstruct;    
}

void Problem::MakeHessian() {
    bool use_lm = true;
    bool use_dl = false;
    bool use_multi_thread = false;
    bool use_sse = false;

    // TODO: ORIGIN LM ===================================================================
    if (use_lm) {
        TicToc t_h;
        // Direkt contruct the big H matrix
        ulong size = ordering_generic_;
        MatXX H(MatXX::Zero(size, size));
        VecX b(VecX::Zero(size));

        // TODO:: accelate, accelate, accelate
        //#ifdef USE_OPENMP
        //#pragma omp parallel for
        //#endif
        //Eigen::initParallel();
        //omp_set_num_threads(4);
        Eigen::setNbThreads(4);
        for (auto &edge : edges_)
        {
            edge.second->ComputeResidual();
            edge.second->ComputeJacobians();

            // TODO:: robust cost
            auto jacobians = edge.second->Jacobians();
            auto verticies = edge.second->Verticies();
            assert(jacobians.size() == verticies.size());
            for (size_t i = 0; i < verticies.size(); ++i)
            {
                auto v_i = verticies[i];
                if (v_i->IsFixed())
                    continue; // Hessian has not other information，so its jacobi is 0

                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->OrderingId();
                ulong dim_i = v_i->LocalDimension();

                // Robust kernel function will change the residual and information matrix，if robust cost function has not been seted，then return the original
                double drho;
                MatXX robustInfo(edge.second->Information().rows(), edge.second->Information().cols());
                edge.second->RobustInfo(drho, robustInfo);

                MatXX JtW = jacobian_i.transpose() * robustInfo;
                for (size_t j = i; j < verticies.size(); ++j)
                {
                    auto v_j = verticies[j];
                    if (v_j->IsFixed())
                        continue;

                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->OrderingId();
                    ulong dim_j = v_j->LocalDimension();

                    assert(v_j->OrderingId() != -1);
                    MatXX hessian = JtW * jacobian_j;

                    // Add all the information matrix
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // Symmetric lower triangle matrix
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * edge.second->Information() * edge.second->Residual();
            }
        }
        Hessian_ = H;
        b_ = b;
        t_hessian_cost_ += t_h.toc();

        if (H_prior_.rows() > 0)
        {
            MatXX H_prior_tmp = H_prior_;
            VecX b_prior_tmp = b_prior_;

            /// Search all of the POSE's contexs，then set the corresponding prior's dimention to 0 .  fix outer parameters, SET PRIOR TO ZERO
            /// landmark dosen't have prior
            for (auto vertex : verticies_)
            {
                if (IsPoseVertex(vertex.second) && vertex.second->IsFixed())
                {
                    int idx = vertex.second->OrderingId();
                    int dim = vertex.second->LocalDimension();
                    H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx, dim).setZero();
                    //std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
            b_.head(ordering_poses_) += b_prior_tmp;
        }
        delta_x_ = VecX::Zero(size); // initial delta_x = 0_n;
        //Eigen::setNbThreads(0);
    }
    // END ORIGIN LM ===================================================================




    // TODO: HOMEWORK Dog-Leg ===================================================================
    if (use_dl) {
        TicToc t_h;
        // Direkt contruct the big H matrix
        ulong size = ordering_generic_;
        MatXX H(MatXX::Zero(size, size));
        VecX b(VecX::Zero(size));

        for (auto &edge : edges_) {
            edge.second->ComputeResidual();
            edge.second->ComputeJacobians();

            // TODO:: robust cost
            auto jacobians = edge.second->Jacobians();
            auto verticies = edge.second->Verticies();
            assert(jacobians.size() == verticies.size());

            for (size_t i = 0; i < verticies.size(); ++i) {
                auto v_i = verticies[i];
                if (v_i->IsFixed()) continue; // Hessian has not other information，so its jacobi is 0

                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->OrderingId();
                ulong dim_i = v_i->LocalDimension();

                // Robust kernel function will change the residual and information matrix，if robust cost function has not been seted，then return the original
                double drho;
                MatXX robustInfo(edge.second->Information().rows(), edge.second->Information().cols());
                edge.second->RobustInfo(drho, robustInfo);

                MatXX JtW = jacobian_i.transpose() * robustInfo;
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto v_j = verticies[j];
                    if (v_j->IsFixed()) continue;

                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->OrderingId();
                    ulong dim_j = v_j->LocalDimension();

                    assert(v_j->OrderingId() != -1);
                    MatXX hessian = JtW * jacobian_j;

                    // Add all the information matrix
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // Symmetric lower triangle matrix
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * edge.second->Information() * edge.second->Residual();
            }
        }
        Hessian_ = H;
        b_ = b;
        t_hessian_cost_ += t_h.toc();

        if (H_prior_.rows() > 0) {
            MatXX H_prior_tmp = H_prior_;
            VecX b_prior_tmp = b_prior_;

            /// Search all of the POSE's contexs，then set the corresponding prior's dimention to 0 .  fix outer parameters, SET PRIOR TO ZERO
            /// landmark dosen't have prior
            for (auto vertex : verticies_) {
                if (IsPoseVertex(vertex.second) && vertex.second->IsFixed())
                {
                    int idx = vertex.second->OrderingId();
                    int dim = vertex.second->LocalDimension();
                    H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx, dim).setZero();
                }
            }
            Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
            b_.head(ordering_poses_) += b_prior_tmp;
        }

        delta_x_ = VecX::Zero(size); // initial delta_x = 0_n;
        //Eigen::setNbThreads(0);
    }
    // END HOMEWORK Dog-Leg ============================================================================




    // TODO: HOMEWORKE Multi-thread ===========================================================================
    if (use_multi_thread) {
        TicToc t_h;
        // Direkt contruct the big H matrix
        ulong size = ordering_generic_;
        MatXX H(MatXX::Zero(size, size));
        VecX b(VecX::Zero(size));

        // creating thread && split data
        const int NUM_THREADS = 4; // default threads
        pthread_t tids[NUM_THREADS];
        ThreadsStruct threadsstruct[NUM_THREADS];
        int i = 0;
        for (auto it : edges_) {
            threadsstruct[i].sub_edges.insert(it);
            i++;
            i = i % NUM_THREADS;
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            threadsstruct[i].sub_H = MatXX::Zero(size, size);
            threadsstruct[i].sub_b = VecX::Zero(size);

            int ret = pthread_create(&tids[i], NULL, HessianConstruct, (void *)&(threadsstruct[i]));
            if (ret != 0) {
                std::cerr << "pthread_create error!" << std::endl;
                exit(1);
            }
        }

        for (int i = NUM_THREADS - 1; i >= 0; i--) {
            pthread_join(tids[i], NULL);
            H += threadsstruct[i].sub_H;
            b += threadsstruct[i].sub_b;
        }

        Hessian_ = H;
        b_ = b;
        t_hessian_cost_ += t_h.toc();

        if (H_prior_.rows() > 0) {
            MatXX H_prior_tmp = H_prior_;
            VecX b_prior_tmp = b_prior_;

            /// Search all of the POSE's contexs，then set the corresponding prior's dimention to 0 .  fix outer parameters, SET PRIOR TO ZERO
            /// landmark dosen't have prior
            for (auto vertex : verticies_) {
                if (IsPoseVertex(vertex.second) && vertex.second->IsFixed())
                {
                    int idx = vertex.second->OrderingId();
                    int dim = vertex.second->LocalDimension();
                    H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx, dim).setZero();
                }
            }
            Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
            b_.head(ordering_poses_) += b_prior_tmp;
        }

        delta_x_ = VecX::Zero(size); // initial delta_x = 0_n;
    }
    // END HOMEWORKE Multi-thread && SSE ============================================================================




    // TODO: HOMEWORKE SSE ===================================================================
    if (use_sse) {
        TicToc t_h;
        // Direkt contruct the big H matrix
        ulong size = ordering_generic_;
        MatXX H(MatXX::Zero(size, size));
        VecX b(VecX::Zero(size));

        // TODO:: accelate, accelate, accelate
        for (auto &edge : edges_) {
            edge.second->ComputeResidual();
            edge.second->ComputeJacobians();

            // TODO:: robust cost
            auto jacobians = edge.second->Jacobians();
            auto verticies = edge.second->Verticies();
            assert(jacobians.size() == verticies.size());
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto v_i = verticies[i];
                if (v_i->IsFixed()) continue;

                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->OrderingId();
                ulong dim_i = v_i->LocalDimension();

                // Robust kernel function will change the residual and information matrix，if robust cost function has not been seted，then return the original
                double drho;
                MatXX robustInfo(edge.second->Information().rows(), edge.second->Information().cols());
                edge.second->RobustInfo(drho, robustInfo);

                // TODO: Accelarate using SSE
                MatXX T_jacobian = jacobian_i.transpose();
                MatXX JtW = MatrixMultiSSE(T_jacobian, robustInfo);
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto v_j = verticies[j];
                    if (v_j->IsFixed()) continue;

                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->OrderingId();
                    ulong dim_j = v_j->LocalDimension();

                    assert(v_j->OrderingId() != -1);
                    // TODO: Accelarate using SSE
                    MatXX hessian = MatrixMultiSSE(JtW, jacobian_j);

                    // Add all the information matrix
                    // TODO: Accelarate using SSE
                    MatXX tmp = H.block(index_i, index_j, dim_i, dim_j).eval();
                    MatXX tmp_H = MatrixAddSSE(tmp, hessian);
                    H.block(index_i, index_j, dim_i, dim_j).noalias() = tmp_H;

                    if (j != i) {
                        // Symmetric lower triangle matrix
                        //H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                        H.block(index_j, index_i, dim_j, dim_i).noalias() = tmp_H.transpose();
                    }
                }
                // TODO: Accelarate using SSE
                MatXX tmp_J = drho * jacobian_i.transpose();
                MatXX jaco_info_mat = MatrixMultiSSE(tmp_J, edge.second->Information());
                VecX JRhoR = MatVecMultiSSE(jaco_info_mat, edge.second->Residual());
                VecX tmp_b = b.segment(index_i, dim_i);
                b.segment(index_i, dim_i).noalias() = VectorSubSSE(tmp_b, JRhoR);
            }
        }
        Hessian_ = H;
        b_ = b;
        t_hessian_cost_ += t_h.toc();

        if (H_prior_.rows() > 0) {
            MatXX H_prior_tmp = H_prior_;
            VecX b_prior_tmp = b_prior_;

            /// Search all of the POSE's contexs，then set the corresponding prior's dimention to 0 .  fix outer parameters, SET PRIOR TO ZERO
            /// landmark dosen't have prior
            for (auto vertex : verticies_)
            {
                if (IsPoseVertex(vertex.second) && vertex.second->IsFixed())
                {
                    int idx = vertex.second->OrderingId();
                    int dim = vertex.second->LocalDimension();
                    H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx, dim).setZero();
                    //std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
            b_.head(ordering_poses_) += b_prior_tmp;
        }
        delta_x_ = VecX::Zero(size); // initial delta_x = 0_n;
    }
    // END HOMEWORKE SSE ===================================================================
}

/*
 * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
 */
void Problem::SolveLinearSystem() {


    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        // PCG solver
        MatXX H = Hessian_;
        for (size_t i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }
        // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        delta_x_ = H.ldlt().solve(b_);

    } else {

        //TicToc t_Hmminv;
        // step1: schur marginalization --> Hpp, bpp
        int reserve_size = ordering_poses_;
        int marg_size = ordering_landmarks_;
        MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_.segment(0, reserve_size);
        VecX bmm = b_.segment(reserve_size, marg_size);

        // Hmm is a diagonal matrix, its inverse can be directly use to compute the inverse，if it's inverse depth，the diagonal has dimention 1，so it's the reciprocal
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto landmarkVertex : idx_landmark_vertices_) {
            int idx = landmarkVertex.second->OrderingId() - reserve_size;
            int size = landmarkVertex.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();

            /*// TODO: HOMEWORK for landmark direct inverse the depth
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size);
            for (int i = idx; i < size+1; i++)
            {
                Hmm_inv(i, i) = 1 / Hmm(i, i);
            }*/
        }

        MatXX tempH = Hpm * Hmm_inv;
        H_pp_schur_ = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
        b_pp_schur_ = bpp - tempH * bmm;

        // step2: solve Hpp * delta_x = bpp
        VecX delta_x_pp(VecX::Zero(reserve_size));

        // TODO: HOMEWORK Dog-Leg donot need this
        for (ulong i = 0; i < ordering_poses_; ++i) {
            H_pp_schur_(i, i) += currentLambda_;              // LM Method
        }

        // TicToc t_linearsolver;
        delta_x_pp =  H_pp_schur_.ldlt().solve(b_pp_schur_);//  SVec.asDiagonal() * svd.matrixV() * Ub;    
        delta_x_.head(reserve_size) = delta_x_pp;
        // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

        // step3: solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
        VecX delta_x_ll(marg_size);
        delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
        delta_x_.tail(marg_size) = delta_x_ll;
        //std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;
    }

}

void Problem::UpdateStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->BackUpParameters();    // store the last estimation

        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // update prior
    if (err_prior_.rows() > 0) {
        // BACK UP b_prior_
        b_prior_backup_ = b_prior_;
        err_prior_backup_ = err_prior_;

        /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
        /// \delta x = Computes the linearized deviation from the references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       // update the error_prior
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);

//        std::cout << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
//        std::cout << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
    }

}

void Problem::RollbackStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->RollBackParameters();
    }

    // Roll back prior_
    if (err_prior_.rows() > 0) {
        b_prior_ = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}

/// LM
void Problem::ComputeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

    for (auto edge: edges_) {
        currentChi_ += edge.second->RobustChi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.squaredNorm();
    currentChi_ *= 0.5;

    stopThresholdLM_ = 1e-10 * sqrt(currentChi_);          // stop condition is: error reduces 1e-6 times
    //stopThresholdLM_ = 1e-10 * currentChi_;

    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }

    maxDiagonal = std::min(5e10, maxDiagonal);

    // TODO: HOMEWORK store the Hessian before update
    // For ORIGIN_Update: tau = 1e-5;   For Update_1: tau = 1e-5;   For Update_2: tau = 1e-5;   For Update_3: 1e-8; 
    double tau = 1e-5; 
    currentLambda_ = tau * maxDiagonal;
}

void Problem::AddLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        /*// TODO: HOMEWORK store the Hessian before update
        curr_Hessian_(i, i) = Hessian_(i, i);*/
        Hessian_(i, i) += currentLambda_;
    }
}

void Problem::RemoveLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: it should not be reduced, the duplication of addition and reduction can affect the precise. So we should always keep the pre. lambda，it should be directly assigned here.
    for (ulong i = 0; i < size; ++i) {
        /*// TODO: HOMEWORK direct equal to previous hessian
        Hessian_(i, i) = curr_Hessian_(i, i);*/
        Hessian_(i, i) -= currentLambda_;
    }
}

bool Problem::IsGoodStepInLM() {
    bool use_lm_org = true;
    bool use_lm_1 = false;
    bool use_lm_2 = false;
    bool use_dl = false;

    // LM with ORIGIN Update Strategy ========================================
    if (use_lm_org) {
        double scale = 0;
        scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
        scale += 1e-6; // make sure it's non-zero :)

        // recompute residuals after update state
        double tempChi = 0.0;
        for (auto edge : edges_)
        {
            edge.second->ComputeResidual();
            tempChi += edge.second->RobustChi2();
        }
        if (err_prior_.size() > 0)
            tempChi += err_prior_.squaredNorm();
        tempChi *= 0.5; // 1/2 * err^2

        double rho = (currentChi_ - tempChi) / scale;
        if (rho > 0 && isfinite(tempChi)) // last step was good, the error is reduceing
        {
            double alpha = 1. - pow((2 * rho - 1), 3);
            alpha = std::min(alpha, 2. / 3.);
            double scaleFactor = (std::max)(1. / 3., alpha);
            currentLambda_ *= scaleFactor;
            ni_ = 2;
            currentChi_ = tempChi;
            return true;
        }
        else
        {
            currentLambda_ *= ni_;
            ni_ *= 2;
            return false;
        }
    }
    // END LM with ORIGIN Update Strategy ========================================




    // TODO: HOMEWORK robuster LM strategy ========================================
    // recompute residuals after update state
    if (use_lm_1)
    {
        double tempChi = 0.0;
        for (auto edge : edges_)
        {
            edge.second->ComputeResidual();
            tempChi += edge.second->RobustChi2();
        }
        if (err_prior_.size() > 0)
            tempChi += err_prior_.squaredNorm();
        tempChi *= 0.5; // 1/2 * err^2

        double frac_numerator = b_.transpose() * delta_x_;
        double alpha = frac_numerator / ((tempChi - currentChi_) / 2. + 2. * frac_numerator);

        //delta_x_ *= alpha;

        double scale = 0;
        // scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
        scale = 0.5 * (delta_x_ * alpha).transpose() * (currentLambda_ * (delta_x_ * alpha) + b_);
        scale += 1e-6; // make sure it's non-zero :)
        double rho = (currentChi_ - tempChi) / scale;
        if (rho > 0 && isfinite(tempChi))
        {
            currentLambda_ = std::max(currentLambda_ / (1. + alpha), 1e-7); // 1e-6: faster; 1e-7: accurater
            currentChi_ = tempChi;

            RollbackStates();
            delta_x_ *= alpha;
            UpdateStates();
            return true;
        }
        else
        {
            currentLambda_ += abs(tempChi - currentLambda_) / (2. * alpha);
            return false;
        }
    }
    // END HOMEWORK robuster LM strategy ========================================




    // TODO: HOMEWORK LM with Marquardt Update strategy ========================================
    if (use_lm_2) {
        double scale = 0;
        scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
        scale += 1e-6; // make sure it's non-zero :)

        // recompute residuals after update state
        double tempChi = 0.0;
        for (auto edge : edges_)
        {
            edge.second->ComputeResidual();
            tempChi += edge.second->RobustChi2();
        }
        if (err_prior_.size() > 0)
            tempChi += err_prior_.squaredNorm();
        tempChi *= 0.5; // 1/2 * err^2

        double rho = (currentChi_ - tempChi) / scale;
        if (rho > 0.75 && isfinite(tempChi))
        {
            currentLambda_ /= 3.;
            currentChi_ = tempChi;
            return true;
        }
        else if (rho <= 0.75 && rho >= 0.25 && isfinite(tempChi))
        {
            currentChi_ = tempChi;
            return true;
        }
        else if (rho < 0.25 && rho > 0 && isfinite(tempChi))
        {
            currentChi_ = tempChi;
            currentLambda_ *= 2.;
            return true;
        }
        else
        {
            currentLambda_ *= 2.;
            return false;
        }
    }
    // END HOMEWORK LM with Marquardt Update strategy ==========================================




    // TODO: HOMEWORK Dog-Leg update strategy ==========================================
    // compute the linear gain, there is not dampf factor
    if (use_dl) {
        double scale = 0.0;
        if (hdl == hgn)
            scale = currentChi_;
        else if (hdl == -_delta*b_.normalized())
            scale = _delta * b_.norm() - _delta * _delta / 2 / alpha;
        else
            scale = 0.5*alpha*(1-beta)*(1-beta)*b_.squaredNorm() + beta*(2-beta)*currentChi_;
        
        //scale = 0.5 * hdl.transpose() * b_;
        scale += 1e-6; // make sure it's non-zero :)

        // recompute residuals after update state
        double tempChi = 0.0;
        for (auto edge : edges_) {
            edge.second->ComputeResidual();
            tempChi += edge.second->RobustChi2();
        }
        if (err_prior_.size() > 0)
            tempChi += err_prior_.squaredNorm();
        tempChi *= 0.5; // 1/2 * err^2

        // update trust region based on the step quality
        double rho = (currentChi_ - tempChi) / scale;
        if (rho > 0) {
            currentChi_ = tempChi;
            stop = (sqrt(currentChi_) <= 1e-10 || b_.norm() <+ 1e-10);

            if (rho > 0.75) {
                _delta = std::max<double>(_delta, 3 * hdl.norm());
                return true;
            } else if (rho < 0.25) {
                _delta *= 0.5;
                stop = (_delta <= 1e-8);
                return true;
            }
        }
        else {
            _delta *= 0.5;
            return false;
        }
    }
    // END HOMEWORK Dog-Leg update strategy ==========================================
}

/** @brief conjugate gradient with perconditioning
 *
 *  the jacobi PCG method
 *
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    return x;
}

/*
 *  marg all edges who are connected with frame: imu factor, projection factor
 *  If some landmark connects with this frame，but there is no marg, 那就把改edge先去掉
 *
 */
bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex> > margVertexs, int pose_dim) {

    SetOrdering();
    /// 找到需要 marg 的 edge, margVertexs[0] is frame, its edge contained pre-intergration
    std::vector<shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

    std::unordered_map<int, shared_ptr<Vertex>> margLandmark;
    // Construct Hessian, the order of pose dosen't change，landmark's oder should be reseted
    int marg_landmark_size = 0;
//    std::cout << "\n marg edge 1st id: "<< marg_edges.front()->Id() << " end id: "<<marg_edges.back()->Id()<<std::endl;
    for (size_t i = 0; i < marg_edges.size(); ++i) {
//        std::cout << "marg edge id: "<< marg_edges[i]->Id() <<std::endl;
        auto verticies = marg_edges[i]->Verticies();
        for (auto iter : verticies) {
            if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
                iter->SetOrderingId(pose_dim + marg_landmark_size);
                margLandmark.insert(make_pair(iter->Id(), iter));
                marg_landmark_size += iter->LocalDimension();
            }
        }
    }
//    std::cout << "pose dim: " << pose_dim <<std::endl;
    int cols = pose_dim + marg_landmark_size;
    /// construct Hessian with H = H_marg + H_pp_prior
    MatXX H_marg(MatXX::Zero(cols, cols));
    VecX b_marg(VecX::Zero(cols));
    int ii = 0;
    for (auto edge: marg_edges) {
        edge->ComputeResidual();
        edge->ComputeJacobians();
        auto jacobians = edge->Jacobians();
        auto verticies = edge->Verticies();
        ii++;

        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            double drho;
            MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
            edge->RobustInfo(drho,robustInfo);

            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

                assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());
                // add all information matrix
                H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
                if (j != i) {
                    // Sysmmetric lower trigule matrix
                    H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                }
            }
            b_marg.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
        }

    }
        std::cout << "edge factor cnt: " << ii <<std::endl;

    /// marg landmark
    int reserve_size = pose_dim;
    if (marg_landmark_size > 0) {
        int marg_size = marg_landmark_size;
        MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_marg.segment(0, reserve_size);
        VecX bmm = b_marg.segment(reserve_size, marg_size);

        // Hmm is a diagonal matrix, its inverse can be directly use to compute the inverse，if it's inverse depth，the diagonal has dimention 1，so it's the reciprocal
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto iter: margLandmark) {
            int idx = iter.second->OrderingId() - reserve_size;
            int size = iter.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        MatXX tempH = Hpm * Hmm_inv;
        MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
        bpp = bpp - tempH * bmm;
        H_marg = Hpp;
        b_marg = bpp;
    }

    VecX b_prior_before = b_prior_;
    if(H_prior_.rows() > 0)
    {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }

    /// marg frame and speedbias
    int marg_dim = 0;

    // The bigger index move first
    for (int k = margVertexs.size() -1 ; k >= 0; --k)
    {

        int idx = margVertexs[k]->OrderingId();
        int dim = margVertexs[k]->LocalDimension();
//        std::cout << k << " "<<idx << std::endl;
        marg_dim += dim;
        // move the marg pose to the Hmm bottown right
        // move row i to the button of the matrix
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

        // move col i to the right side of matrix
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
        Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
        b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
        b_marg.segment(reserve_size - dim, dim) = temp_b;
    }

    double eps = 1e-8;
    int m2 = marg_dim;
    int n2 = reserve_size - marg_dim;   // marg pose
    Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                              saes.eigenvectors().transpose();

    Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
    Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
    Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
    Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
    Eigen::VectorXd brr = b_marg.segment(0, n2);
    Eigen::MatrixXd tempB = Arm * Amm_inv;
    H_prior_ = Arr - tempB * Amr;
    b_prior_ = brr - tempB * bmm2;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd(
            (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    err_prior_ = -Jt_prior_inv_ * b_prior_;

    MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    H_prior_ = J.transpose() * J;
    MatXX tmp_h = MatXX( (H_prior_.array().abs() > 1e-9).select(H_prior_.array(),0) );
    H_prior_ = tmp_h;

    // std::cout << "my marg b prior: " <<b_prior_.rows()<<" norm: "<< b_prior_.norm() << std::endl;
    // std::cout << "    error prior: " <<err_prior_.norm() << std::endl;

    // remove vertex and remove edge
    for (size_t k = 0; k < margVertexs.size(); ++k) {
        RemoveVertex(margVertexs[k]);
    }

    for (auto landmarkVertex: margLandmark) {
        RemoveVertex(landmarkVertex.second);
    }

    return true;

}

}
}






