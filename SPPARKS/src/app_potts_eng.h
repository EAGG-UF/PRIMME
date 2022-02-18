/* -------------------------------------------------------
AppPotts_eng class header for abnormal grain growth
--
Read-Shockley implementation developed by Efrain Hernandez-Rivera (2017--2018)
US Army Research Laboratory
--
THIS SOFTWARE IS MADE AVAILABLE ON AN "AS IS" BASIS
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, NEITHER
EXPRESSED OR IMPLIED
------------------------------------------------------- */

#ifdef APP_CLASS
AppStyle(potts/eng,AppPotts_eng)

#else

#ifndef SPK_APP_POTTS_ENG_H
#define SPK_APP_POTTS_ENG_H

#include <map>
#include "app_potts.h"

namespace SPPARKS_NS {

class AppPotts_eng : public AppPotts {
  public:

    AppPotts_eng(class SPPARKS *, int, char **);
    ~AppPotts_eng();
    void init_app();
    void grow_app();
    void input_app(char *, int, char **);
    void site_event_rejection(int, class RandomPark *);
    double site_energy(int);

  protected:
    double *phi1,*phi2,*Phi; //pointer to 3 rotation angles
    int *spin;
    double thetam; //High-low angle divider
    double Jij; //Interaction energy
    int Osym; //Symmetry Operator flag
    int matrixID;
    // double **symquat; //Symmetry Operator in quaternion space

    //Mobility = Mm * [1 - exp(-B * {theta/theham}^n)]
    // double nmob; //Mobility exponential power, n
    // double bmob; //Mobility exponential scaling, B

    // //Get misorientation angle from quaternions
    // double quaternions(const double qi[4], const double qj[4]);
    //
    // //Multiplication between quaternion vectors
    // void quat_mult(const double qi[4], const double qj[4], double q[4]);
    //
    // //Define the symmetry operator based on symmetry flag
    // void symmat(double ***);
    //
    // //Convert symmetry operator into quaternion space
    // void mat2quat(const double O[3][3], double q[4]);
    // void euler2quat(int i, double q[4]);

    //map to store misorientations
    std::map<std::pair<int,int>, double> misos;
    std::map<std::pair<int,int>, double> energy;

  private:
   double pfraction;
   int multi,nthresh;
};

}

#endif
#endif
