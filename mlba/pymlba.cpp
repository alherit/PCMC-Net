#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "mlba.hpp"

typedef std::vector<Eigen::Ref<const Eigen::MatrixXd>> alt_vec;

namespace py = pybind11;



std::vector<Eigen::VectorXd>
predict(alt_vec alternatives, const double m, const double lambda1, const double lambda2, const double i0){
  MLBA::Parameter p(m, lambda1, lambda2, i0);
  MLBA::Model model;
  std::vector<Eigen::VectorXd> ret;
  for(alt_vec::const_iterator it=alternatives.begin();it != alternatives.end(); it++){
    std::vector<double> probs;
    model.predict(probs,*it,p);
    Eigen::VectorXd eig_probs = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(probs.data(), probs.size());
    ret.push_back(eig_probs);
  }
  return ret;

}


std::vector<Eigen::MatrixXd>
test2(Eigen::Ref<const Eigen::MatrixXd> x, Eigen::Ref<const Eigen::VectorXd> as){
  std::vector<Eigen::MatrixXd> matrices;
  for(unsigned int k = 0; k < as.size(); k++){
    Eigen::MatrixXd ys = x * as[k];
    matrices.push_back(ys);
  }
  return matrices;
}


PYBIND11_MODULE(pymlba, m) {
  m.doc() = "mlba"; // optional module docstring

  m.def("predict", &predict, "predict with mlba");
  m.def("test2", &test2, "mult");
}
