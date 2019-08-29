#ifndef BEST_FIT_ALL_
#define BEST_FIT_ALL_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "fit_hexa_control_nn.hpp"

#include <robot_dart/robot_dart_simu.hpp>
//#include <robot_dart/control/hexa_control.hpp>
#include "hexa_control.hpp"

#ifdef GRAPHIC
#include <robot_dart/graphics/graphics.hpp>
#endif


#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/constraint/ConstraintSolver.hpp>

#include <modules/nn2/mlp.hpp>
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>

#include <modules/nn2/gen_dnn_ff.hpp>

#include "desc_hexa.hpp"


namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitAll, Stat) {
    public:
      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        _best = *std::max_element(ea.pop().begin(), ea.pop().end(), fit::compare_max());

        //std::cout << "pop size: " << ea.pop().size() << std::endl;

        this->_create_log_file(ea, "bestfit.dat");
        if (ea.dump_enabled())
          (*this->_log_file) << ea.gen() << " " << ea.nb_evals() << " " << _best->fit().value() << std::endl;

        //change it to depend from params 
        if (_cnt%Params::pop::dump_period == 0){ //save model

          typedef boost::archive::binary_oarchive oa_t;

          std::cout << "writing...model" << std::endl;
          //const std::string fmodel = std::string("/git/sferes2/exp/tmp/model_") + std::to_string(_cnt) + ".bin";
          const std::string fmodel = ea.res_dir() + "/model_" + std::to_string(_cnt) + ".bin";
          {
          std::ofstream ofs(fmodel, std::ios::binary);
          oa_t oa(ofs);
          //oa << model;
          oa << *_best;
          }
          std::cout << "model written" << std::endl;}
        _cnt += 1;
      	
	if (_cnt == Params::pop::nb_gen - 1){
          test_and_save(ea);
        }
	}

      void show(std::ostream& os, size_t k) {
        _best->develop();
        _best->show(os);
        _best->fit().set_mode(fit::mode::view);
        _best->fit().eval(*_best);
      }
      const boost::shared_ptr<Phen> best() const {
        return _best;
      }
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(_best);
      }

      template<typename E>
      void test_and_save(const E& ea){

        int cnt = 0;

        std::cout << "starting test and save" << std::endl;

        const std::string filename = ea.res_dir() + "/dict_models.txt";
        std::ofstream dict_file;
        dict_file.open(filename);

        for( auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {

          std::vector<double> results(3);
          results = test_model(*it); //simulate and obtain fitness and behavior descriptors

          std::cout << "test unitaire - fitness: " << results[0] << " behavior descriptor: " << results[1] << " - " << results[2] << " - " << results[3] << std::endl;

          dict_file << "final_model_" + std::to_string(cnt) << "  " << results[0] << "  " << results[1] << "  " << results[2] << "  " << results[3] << "\n"; //save simulation results in dictionary file

          typedef boost::archive::binary_oarchive oa_t;
          const std::string fmodel = ea.res_dir() + "/final_model_" + std::to_string(cnt) + ".bin";
          {
          std::ofstream ofs(fmodel, std::ios::binary);

          if (ofs.fail()){
            std::cout << "wolla ca s'ouvre pas" << std::endl;}

          oa_t oa(ofs);
          //oa << model;
          oa << **it;
          } //save model

          cnt ++;
        }

        dict_file.close();

        std::cout << std::to_string(cnt) + " models saved" << std::endl;
      }

      template<typename T>
      std::vector<double> test_model(T& model){
	
	Eigen::Vector3d target = model.get_target(); // TODO: test type
	std::vector<double> result(4); //fit / bd 1,2,3
  	result = simulate(target, model);}

      template<typename Model>
      std::vector<double> simulate(Eigen::Vector3d& target, Model& model)
  {

    auto g_robot=global::global_robot->clone();
    g_robot->skeleton()->setPosition(5, 0.15);


    double ctrl_dt = 0.015;
    g_robot->add_controller(std::make_shared<robot_dart::control::HexaControlNN<Model>>());
    //std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->set_h_params(std::vector<double>(1, ctrl_dt));
    std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->setModel(model); //TODO : understand why do we use a static pointer cast
    std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->setTarget(target);
    robot_dart::RobotDARTSimu simu(0.005); //creation d'une simulation

#ifdef GRAPHIC
    simu.set_graphics(std::make_shared<robot_dart::graphics::Graphics>(simu.world(), 640, 480, false));
#endif

    simu.world()->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    simu.add_floor();
    simu.add_robot(g_robot);

    simu.add_descriptor(std::make_shared<robot_dart::descriptor::HexaDescriptor>(robot_dart::descriptor::HexaDescriptor(simu)));
    simu.add_descriptor(std::make_shared<robot_dart::descriptor::DutyCycle>(robot_dart::descriptor::DutyCycle(simu)));

    simu.run(5);

    //_body_contact = std::static_pointer_cast<robot_dart::descriptor::DutyCycle>(simu.descriptor(1))->body_contact(); //should be descriptor 1
    std::vector<Eigen::VectorXf> traj;
    traj = std::static_pointer_cast<robot_dart::descriptor::HexaDescriptor>(simu.descriptor(0))->traj;
    //_on_back = std::static_pointer_cast<robot_dart::descriptor::HexaDescriptor>(simu.descriptor(0))->on_back();
    g_robot.reset();
    
    std::vector<double> results = get_fit_bd(_traj, target);

   return(results);

     }     

  std::vector<double> get_fit_bd(std::vector<Eigen::VectorXf> & traj, Eigen::Vector3d & target)
  {

    int size = traj.size();


    double dist = 0;
    std::vector<double> zone_exp(3);
    std::vector<double> res(3);
    std::vector<double> results(4);

    Eigen::VectorXf pos_init = traj[0];
    //std::cout << "init done" << std::endl;

    for (int i = 0; i < size; i++)
      {

        //std::cout << "traj " << i << " : " << _traj[i][0] << " - " << _traj[i][1] << std::endl;
        //std::cout << "fit" << std::endl;
        if (sqrt((target[0]-_traj[i][0])*(target[0]-_traj[i][0]) + (target[1]-_traj[i][1])*(target[1]-_traj[i][1])) < 0.02){
          dist -= sqrt((target[0]-_traj[i][0])*(target[0]-_traj[i][0]) + (target[1]-_traj[i][1])*(target[1]-_traj[i][1]));}

        else {
          dist -= (log(1+i)) + sqrt((target[0]-_traj[i][0])*(target[0]-_traj[i][0]) + (target[1]-_traj[i][1])*(target[1]-_traj[i][1]));}

        //std::cout << "bd" << std::endl;
        res = get_zone(pos_init, target, traj[i]); //TODO : check if get zone accepts vector with different sizes
        zone_exp[0] = zone_exp[0] + res[0];
        zone_exp[1] = zone_exp[1] + res[1];
        zone_exp[2] = zone_exp[2] + res[2];
      }
//std::cout << "fit 1" << std::endl;
    if (sqrt((target[0]-_traj.back()[0])*(target[0]-_traj.back()[0]) + (target[1]-_traj.back()[1])*(target[1]-_traj.back()[1])) < 0.05){
          dist = 1.0 + dist/10000;} // -> 1 (TODO : check division by 500)

    else {
          dist = dist/10000; // -> 0
        }
    //std::cout << "fit 2" << std::endl;


    //int sum_zones = abs(zone_exp[0]) + abs(zone_exp[1]) + abs(zone_exp[2]);
    int sum_zones = size; //always the same number of time steps

    //std::cout << "sum results: " << sum_zones << std::endl;

    results[0] = dist;
    results[1] = zone_exp[0]/sum_zones;
    results[2] = zone_exp[1]/sum_zones;
    results[3] = zone_exp[2]/sum_zones;

    //std::cout << "final results: " << results[0] << " - " << results[1] << " - " << results[2] << " - " << results[3] << std::endl;

    return results;
  }

std::vector<double> get_zone(Eigen::VectorXf start, Eigen::Vector3d target, Eigen::VectorXf pos){


      std::vector<double> desc_add (3);

      Eigen::Vector3d middle;
      middle[0] = (start[0]+target[0])/2;
      middle[1] = (start[1]+target[1])/2;
      middle[2] = 1;

      std::vector<double> distances (3);
      distances = {0,0,0};

//std::cout << "get zone 1" << std::endl;

      distances[0] = sqrt((start[0] - pos[0])*(start[0] - pos[0]) + (start[1] - pos[1])*(start[1] - pos[1]));

      distances[1] = sqrt((target[0] - pos[0])*(target[0] - pos[0]) + (target[1] - pos[1])*(target[1] - pos[1]));

      distances[2] = sqrt((middle[0] - start[0])*(middle[0] - start[0]) + (middle[1] - start[1])*(middle[1] - start[1]));


      Eigen::Vector3d vO2_M_R0; //vector 02M in frame R0; (cf sketch on page 4)
      vO2_M_R0[0] = pos[0] - start[0];
      //vO2_M_R0[0] = pos[0];
      vO2_M_R0[1] = pos[1] - start[1];
      //vO2_M_R0[1] = pos[1];
      vO2_M_R0[2] = 1;

      Eigen::Vector3d vMid_M_R0; //vector Middle_M in frame R0;
      vMid_M_R0[0] = pos[0] - middle[0];
      vMid_M_R0[1] = pos[1] - middle[1];
      vMid_M_R0[2] = 1;

//Eigen::Matrix3d T; //translation matrix
      //T << 1,0,-start[0],0,1,-start[1],0,0,1; //translation matrix

      Eigen::Vector3d vO2_T;
      vO2_T[0] = target[0] - start[0];
      vO2_T[1] = target[1] - start[1];
      vO2_T[2] = 1;

      double theta = atan2(vO2_T[1], vO2_T[0]) - atan2(1, 0);

      if (theta > M_PI){
          theta -= 2*M_PI;
      }
      else if (theta <= -M_PI){
          theta += 2*M_PI;
      }

      Eigen::Matrix3d R;
      R << cos(theta), sin(theta), 0, -sin(theta), cos(theta), 0, 0, 0, 1; //rotation matrix

      Eigen::Vector3d vO2_M_R1; //vector 02M in frame R1;
      //vO2_M_R1 = T*vO2_M_R0;  
      vO2_M_R1 = R*vO2_M_R0;

      Eigen::Vector3d vMid_M_R1; //vector Middle_M in frame R1;
      vMid_M_R1 = R*vMid_M_R0;


      if (vO2_M_R1[0] < 0){ //negative zone (cf sketch on page 3)
          if (distances[0] < 0.1 || distances[1] < 0.1 || (abs(vMid_M_R1[0]) < 0.1 && abs(vMid_M_R1[1]) < distances[2])) {
              return {-1, 0, 0};
          }
          if ((distances[0] < 0.2 || distances[1] < 0.2 || (abs(vMid_M_R1[0]) < 0.2 && abs(vMid_M_R1[1]) < distances[2])) && (distances[0] >= 0.1 || distances[1] >= 0.1 || (abs(vMid_M_R1[0]) >= 0.1 && abs(vMid_M_R1[1]) < distances[2]))){
              return {0, -1, 0};
          }
          else {
              return {0,0,-1};
          }

     }

      else{ //positive zone
          if (distances[0] < 0.1 || distances[1] < 0.1 || (abs(vMid_M_R1[0]) < 0.1 && abs(vMid_M_R1[1]) < distances[2])) {
              return {1, 0, 0};
          }
          if ((distances[0] < 0.2 || distances[1] < 0.2 || (abs(vMid_M_R1[0]) < 0.2 && abs(vMid_M_R1[1]) < distances[2])) && (distances[0] >= 0.1 || distances[1] >= 0.1 || (abs(vMid_M_R1[0]) >= 0.1 && abs(vMid_M_R1[1]) < distances[2]))){
              return {0, 1, 0};
          }
          else {
              return {0,0,1};
          }
      }
  }




    protected:
      int _cnt = 0; //not sure if it is useful
      boost::shared_ptr<Phen> _best;
    };
  }
}
#endif
