#include <iostream>
#include <random>
#include "backend/problem.h"

using namespace myslam::backend;
using namespace std;

// The contex of curve model, template parameters： the dimention of optimisers and data type
class CurveFittingVertex: public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingVertex(): Vertex(3) {}  // abc: three paramters， Vertex is 3D
    virtual std::string TypeInfo() const { return "abc"; }
};

// error model, template parameters： the dimention of observation，type and contex's type
class CurveFittingEdge: public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x, double y ): Edge(1,1, std::vector<std::string>{"abc"}) {
        x_ = x;
        y_ = y;
    }
    // compute the error of the curve model
    virtual void ComputeResidual() override
    {
        Vec3 abc = verticies_[0]->Parameters();  // estimated paramters
        residual_(0) = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) ) - y_;  // contruct the residual
    }

    // Compute the jacobi of residual to variables
    virtual void ComputeJacobians() override
    {
        Vec3 abc = verticies_[0]->Parameters();
        double exp_y = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) );

        Eigen::Matrix<double, 1, 3> jaco_abc;  // the residual is 1D，states are 3，so it's a 1x3 jacobi matrix
        jaco_abc << x_ * x_ * exp_y, x_ * exp_y , 1 * exp_y;
        jacobians_[0] = jaco_abc;
    }
    /// Return the edge's type info.
    virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }
public:
    double x_,y_;  // x 值， y 值为 _measurement
};

int main()
{
    double a=1.0, b=2.0, c=1.0;         // true parameters
    int N = 100;                          // measurements
    double w_sigma= 1.;                 // Noise: Sigma

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.,w_sigma);

    // Construct problem
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM);
    shared_ptr< CurveFittingVertex > vertex(new CurveFittingVertex());

    // Set the initial estimator: parameters a, b, c
    vertex->SetParameters(Eigen::Vector3d (0.,0.,0.));
    // Insert the parameters into the least square problem
    problem.AddVertex(vertex);

    // construct N times observation
    for (int i = 0; i < N; ++i) {

        double x = i/100.;
        double n = noise(generator);
        // 观测 y
        double y = std::exp( a*x*x + b*x + c ) + n;
//        double y = std::exp( a*x*x + b*x + c );

        // the correspoing residual functions
        shared_ptr< CurveFittingEdge > edge(new CurveFittingEdge(x,y));
        std::vector<std::shared_ptr<Vertex>> edge_vertex;
        edge_vertex.push_back(vertex);
        edge->SetVertex(edge_vertex);

        // insert the edge into the problem
        problem.AddEdge(edge);
    }

    std::cout<<"\nTest CurveFitting start..."<<std::endl;
    /// use LM to solve the problem
    problem.Solve(30);

    std::cout << "-------After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->Parameters().transpose() << std::endl;
    std::cout << "-------ground truth: " << std::endl;
    std::cout << "1.0,  2.0,  1.0" << std::endl;

    // std
    return 0;
}


