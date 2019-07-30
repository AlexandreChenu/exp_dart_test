#ifndef ___FIT_HEXA_CONTROL_NN_HPP__
#define ___FIT_HEXA_CONTROL_NN_HPP__

#include <iostream>
#include <Eigen/Core>

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



namespace global{
  std::shared_ptr<robot_dart::Robot> global_robot;
}

void load_and_init_robot()
{
  std::cout<<"INIT Robot"<<std::endl;
  global::global_robot = std::make_shared<robot_dart::Robot>("exp/example_dart_exp/ressources/hexapod_v2.urdf");
  global::global_robot->set_position_enforced(true);
  //global::global_robot->set_position_enforced(true);
  //global_robot->skeleton()->setPosition(1,100* M_PI / 2.0);
  
  global::global_robot->set_actuator_types(dart::dynamics::Joint::SERVO);
  global::global_robot->skeleton()->enableSelfCollisionCheck();
  std::cout<<"End init Robot"<<std::endl;
}



FIT_QD(Fit_hexa_control_nn)
{
public:
  Fit_hexa_control_nn(){  }
  
  typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > Mat;

  template<typename Indiv>
    void eval(Indiv& ind)
  {

    //INITIALISATION
    Eigen::Vector3d target;
    target = {8.0, 0.0,0.0}; 

    simulate(target, ind); //simulate robot behavior for given nn (ind) and target

    std::vector<double> res(4);
    res = get_fit_bd(_traj);

    this->_value = res[0]; //save fitness value
    
    // descriptor is the final position of the robot. 
    std::vector<double> desc(3);
 
    desc[0] = res[1];
    desc[1] = res[2];
    desc[2] = res[3];

    this->set_desc(desc); //save behavior descriptor

    if(desc[0]<0 || desc[0]>1 ||desc[1]<0 || desc[1]>1)
      this->_dead=true; //if something is wrong, we kill this solution. 
    
  }
  
  template<typename Model>
  void simulate(Eigen::Vector3d& target, Model& model) 
  {
    auto g_robot=global::global_robot->clone();
    g_robot->skeleton()->setPosition(5, 0.15);

    double ctrl_dt = 0.015;
    g_robot->add_controller(std::make_shared<robot_dart::control::HexaControlNN<Model>>());
    std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->set_h_params(std::vector<double>(1, ctrl_dt));

    std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->setModel(model); //TODO : understand why do we use a static pointer cast
    std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->setTarget(target);

    
    robot_dart::RobotDARTSimu simu(0.005); //creation d'une simulation

#ifdef GRAPHIC
    simu.set_graphics(std::make_shared<robot_dart::graphics::Graphics>(simu.world()));
#endif

    simu.world()->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    simu.add_floor();
    simu.add_robot(g_robot);

    simu.add_descriptor(std::make_shared<robot_dart::descriptor::HexaDescriptor>(robot_dart::descriptor::HexaDescriptor(simu)));
  
    simu.run(5);

    g_robot.reset();

    _traj=std::static_pointer_cast<robot_dart::descriptor::HexaDescriptor>(simu.descriptor(0))->traj;
  }


  std::vector<double> get_fit_bd(std::vector<Eigen::VectorXf> & traj, Eigen::Vector3d & target)
  {

    int size = traj.size();
    double dist = 0;
    std::vector<double> zone_exp(3);
    std::vector<double> res(3);
    std::vector<double> results(4);

    Eigen::VectorXf pos_init = traj[0];


    for (int i = 0; i < size; i++)
      {
        if (sqrt((target[0]-_traj.back()[0])*(target[0]-_traj.back()[0]) + (target[1]-_traj.back()[1])*(target[1]-_traj.back()[1])) < 0.02){
          dist -= sqrt((target[0]-_traj.back()[0])*(target[0]-_traj.back()[0]) + (target[1]-_traj.back()[1])*(target[1]-_traj.back()[1]));}

        else {
          dist -= (log(1+i)) + sqrt((target[0]-_traj.back()[0])*(target[0]-_traj.back()[0]) + (target[1]-_traj.back()[1])*(target[1]-_traj.back()[1]));}

        res = get_zone(pos_init, target, traj[i]); //TODO : check if get zone accepts vector with different sizes
        zone_exp[0] = zone_exp[0] + res[0];
        zone_exp[1] = zone_exp[1] + res[1];
        zone_exp[2] = zone_exp[2] + res[2];
      }

    if (sqrt((target[0]-_traj.back()[0])*(target[0]-_traj.back()[0]) + (target[1]-_traj.back()[1])*(target[1]-_traj.back()[1])) < 0.05){
          dist = 1.0 + dist/500;} // -> 1 (TODO : check division by 500)

    else {
          dist = dist/500; // -> 0
        }

    int sum_zones = zone_exp[0] + zone_exp[1] + zone_exp[2];

    results[0] = dist;
    results[1] = zone_exp[0]/sum_zones;
    results[2] = zone_exp[1]/sum_zones;
    results[3] = zone_exp[2]/sum_zones;

    return results;
  }

  std::vector<double> get_zone(Eigen::Vector3d start, Eigen::Vector3d target, Eigen::Vector3d pos){
      
      
      std::vector<double> desc_add (3);
      
      Eigen::Vector3d middle;
      middle[0] = (start[0]+target[0])/2;
      middle[1] = (start[1]+target[1])/2;
      middle[2] = 1;
      
      std::vector<double> distances (3);
      distances = {0,0,0};
      
      distances[0] = sqrt(square(start.array() - pos.array()).sum()); //R1 (cf sketch on page 3)
      distances[1] = sqrt(square(target.array() - pos.array()).sum()); //R2
      distances[2] = sqrt(square(middle.array() - pos.array()).sum()); //d
      
      double D;
      D = *std::min_element(distances.begin(), distances.end()); //get minimal distance
      
      Eigen::Vector3d vO2_M_R0; //vector 02M in frame R0; (cf sketch on page 4)
      //vO2_M_R0[0] = pos[0] - start[0];
      vO2_M_R0[0] = pos[0];
      //vO2_M_R0[1] = pos[1] - start[1];
      vO2_M_R0[1] = pos[1];
      vO2_M_R0[2] = 1;
      
      Eigen::Matrix3d T; //translation matrix
      T << 1,0,-start[0],0,1,-start[1],0,0,1; //translation matrix
      
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
      vO2_M_R1 = T*vO2_M_R0;  
      vO2_M_R1 = R*vO2_M_R1;
      
      
      if (vO2_M_R1[0] < 0){ //negative zone (cf sketch on page 3)
          if (D < 0.2) {
              return {-1, 0, 0};
          }
          if (D >= 0.2 && D < 0.4){
              return {0, -1, 0};
          }
          else {
              return {0,0,-1};
          }
      }
      
      else{ //positive zone
          if (D < 0.2) {
              return {1, 0, 0};
          }
          if (D >= 0.2 && D < 0.4){
              return {0, 1, 0};
          }
          else {
              return {0,0,1};
          }
      }
  }

  
private:
  std::vector<double> _ctrl;
  std::vector<Eigen::VectorXf> _traj;

  
};



#endif
