//
// Created by gaoxiang19 on 11/10/18.
//

#ifndef MYSLAM_LOSS_FUNCTION_H
#define MYSLAM_LOSS_FUNCTION_H

#include "eigen_types.h"

namespace myslam {
namespace backend {

    /**
     * compute the scaling factor for a error:
     * The error is e^T Omega e
     * The output rho is
     * rho[0]: The actual scaled error value
     * rho[1]: First derivative of the scaling function
     * rho[2]: Second derivative of the scaling function
     *
     * LossFunction is a bese class, variaus other loss function could be derived from this base class
     */
    class LossFunction {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual ~LossFunction() {}

    //    virtual double Compute(double error) const = 0;
        virtual void Compute(double err2, Eigen::Vector3d& rho) const = 0;
    };

    /**
     * 平凡的Loss，with nothing to do
     * use nullptr as loss function has the same effect
     *
     * TrivalLoss(e) = e^2
     */
    class TrivalLoss : public LossFunction {
    public:
        virtual void Compute(double err2, Eigen::Vector3d& rho) const override
        {
            // TODO:: whether multiply 1/2
            rho[0] = err2;
            rho[1] = 1;
            rho[2] = 0;
        }
    };

    /**
     * Huber loss
     *
     * Huber(e) = e^2                      if e <= delta
     * huber(e) = delta*(2*e - delta)      if e > delta
     */
    class HuberLoss : public LossFunction {
    public:
        explicit HuberLoss(double delta) : delta_(delta) {}

        virtual void Compute(double err2, Eigen::Vector3d& rho) const override;

    private:
        double delta_;

    };

    /*
     * Cauchy loss
     *
     */
    class CauchyLoss : public LossFunction
    {
    public:
        explicit CauchyLoss(double delta) : delta_(delta) {}

        virtual void Compute(double err2, Eigen::Vector3d& rho) const override;

    private:
        double delta_;
    };

    class TukeyLoss : public LossFunction
    {
    public:
        explicit TukeyLoss(double delta) : delta_(delta) {}

        virtual void Compute(double err2, Eigen::Vector3d& rho) const override;

    private:
        double delta_;
    };
}
}

#endif //MYSLAM_LOSS_FUNCTION_H
