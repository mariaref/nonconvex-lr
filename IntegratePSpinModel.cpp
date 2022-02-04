#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <getopt.h>


#include<iostream>
#include <sstream>

#include <string.h>
#include <stdexcept>
#include <unistd.h>

const char * usage = R"USAGE(
This is 2-p-spin_fixed_grid++, a tool to solve the spiked tensor model dynamical equation

usage: *.exe [-h] [--g G]

optional arguments:
  -h, -?                show this help message and exit
  --p P                 the variance of the matrix channel
  --Deltap               the variance of the tensor channel
  --Cbar0                initial value of magnetisation
  --Nmax                 number of points in the discretization of the grid
  --tmax                 maximum simulation time
  --lr                   which learning rate to use
)USAGE";

// This structure contains the physical parameters of the system.
typedef struct _psys {
  float p,Delta2,Deltap,Tg,alpha,Cbar0;  // system parameters
  float W22_0,Wp2_0,r2_0,rp_0;      // auxiliary variables
  float W22,Wp2,r2,rp;          // auxiliary variables
  float *T2, *Tp;              // mismatching parameters
} psys;


//	The function computes the equation for mu.
float compute_mu(float* lr, float T, float*mu,  float**C,float** R, int t, float Deltap, float h, int p);
//	The function computes the equation for C.
float compute_C(float* lr, float T, int t, int l,float*mu,  float**C,float** R, float Deltap, float h, int p);
//	The function computes the equation for R.
float compute_R(float* lr, float T, int t, int l,float*mu,  float**C,float** R, float Deltap, float h, int p);

//	The function produces the output file concerning the Lagnange multiplier.
float print_mu(float** C,float** R,float* mu,float* H, int Nmax,float h, FILE* mu_file);

//	The function computes the lr at all steps.
void lr_policy(float beta, float lr0, float h, int Nmax, float* lr, bool crossover);

//	The function propagates the initial values of C,R,Cbar in time up to a given time.
int main (int argc, char *argv[]) {	
    
  // Assign the parameters of the system.
  double p=3.;
  float Deltap=1.;      // the variance of the tensor channel
  int Nmax=1000;        // number of points in the discretization of the grid
  float tmax=100.;        // maximum simulation time         // learning rate
  float T = 1.;           //temperature
  float beta = 0. ; // lr schedule to use
  float lr0 = 1.;
  bool crossover = false; 
  // parse command line options using getopt
  int c;
  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"p",         required_argument, 0, 'p'},
    {"T",         required_argument, 0, 'T'},
    {"Deltap",          required_argument, 0, 'f'},
    {"Nmax",          required_argument, 0, 'n'},
    {"tmax",         required_argument, 0, 't'},
    {"lr0",         required_argument, 0, 'l'},
    {"beta",         required_argument, 0, 'b'},
    {"crossover",         required_argument, 0, 'c'},
    {0, 0, 0, 0}
  };
  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;
    c = getopt_long(argc, argv, "p:T:f:n:t:l:b:c:",
                    long_options, &option_index);
    /* Detect the end of the options. */
    if (c == -1) {
      break;
    }
    switch (c) {
      case 0:
        break;
      case 'p':
        p = atof(optarg);
        break;
      case 'T':
        T = atof(optarg);
        break;
      case 'f':
        Deltap = atof(optarg);
        break;
      case 'n':
        Nmax = atoi(optarg);
        break;
      case 't':
        tmax = atof(optarg);
        break;
      case 'b':
        beta = atof(optarg);
      break;
      case 'l':
        lr0 = atof(optarg);
      break;
      case 'c':
        crossover = (bool)(atof(optarg));
      break;
            
      case '?':
        std::cout << usage << std::endl;
        return 0;
      default:
        abort ();
    }
  }
  
	float h ;
  float **C, **R, *mu, *H, *lr;
	int i,t,l;
	char suffix[100], directory[100]="";
  char muname_file[100];
  FILE* mu_file;

  if(crossover){
    std::cout<<"using cross over"<<std::endl;}
  h = ((float) tmax) /((float)Nmax)  ; // time step size
  std::cout<<"h :"<<h<<std::endl;
  
	// Use protocol? true - false.
	float C_policy = 100., tau_policy = 70.;
  
  fprintf(stderr, "Dynamics for (2+%.0f)-spin planted with beta %.2f lr0 %.2f and Deltap %.4f (alpha %.4f) temperature %.2f \n",p,beta,lr0,Deltap, 0., T);
	fprintf(stderr, "Maximum simulation time %.0f using %d steps of size %.2e\n",tmax,Nmax,h);
    
    
	sprintf(suffix, "P%.0f_beta-%.2f_lr0-%.2f_T%.2f_t-%.2f_Nmax-%d",p,beta,lr0, T,tmax,Nmax);
  fprintf(stderr, "Saving in P%.0f_beta-%.2f_lr0-%.2f_T%.2f_t-%.2f_Nmax-%d.txt \n",p,beta,lr0, T,tmax,Nmax);
	sprintf(muname_file,"%smu_%s.txt",directory,suffix);

	mu_file=fopen(muname_file,"w");
	C=(float**)malloc((Nmax+1)*sizeof(float*));
	R=(float**)malloc((Nmax+1)*sizeof(float*));
	mu=(float*)malloc((Nmax+1)*sizeof(float));
  H=(float*)malloc((Nmax+1)*sizeof(float));
  
  lr=(float*)malloc((Nmax+1)*sizeof(float));
  lr_policy(beta,lr0,h,Nmax, lr, crossover);
  
	for(i=0; i<=Nmax;i++){
		C[i]=(float*)malloc((i+1)*sizeof(float));
		R[i]=(float*)malloc((i+1)*sizeof(float));
	}
	
	// inizialization.
	C[0][0]=1;
	R[0][0]=0;

	// integration loop.
	for(i=0; i<=Nmax-1; i++){
    
    t=i;
    C[t+1][t+1]=1;
    R[t+1][t+1]=0;
    
    mu[t] = compute_mu(lr, T, mu, C, R, t, Deltap, h, p);
    H[t] = 1./p * ( lr[t] * T - mu[t] );
    
    std::ostringstream msg;
    msg << h*t <<","<< H[t] << ", " << mu[t] << ", " << lr[t] ;
    
    std::string msg_str = msg.str();
    msg_str = msg_str.substr(0, msg_str.length());
    fprintf(mu_file, "%s\n", msg_str.c_str());
    fflush(mu_file);

    // compute the two times observables.
		for(l=0;l<=t && t<Nmax;l++){
			C[t+1][l]=compute_C(lr, T, t, l, mu, C, R, Deltap ,h, p);
			R[t+1][l]=compute_R(lr, T, t, l, mu, C, R, Deltap ,h, p);
		}

		// print progress.
		if(i%100==0){
			fprintf(stderr, "Iteration %d of %d\n",i,Nmax);
            fprintf(stderr, "time %.2f mu %.2f H %.2f \n", i * h , mu[i], H[i]);
		}
		R[t+1][t]=1;
    
	}
  fclose(mu_file);
	free(C); free(R); free(mu);
  
  
  
		
	return 0;	
}

float compute_mu(float *lr, float T, float*mu, float**C,float** R, int t, float Deltap, float h, int p){
  int l;
  float auxp;
    float munew;
    
  auxp  = 0.5*pow(C[t][0],p-1)*R[t][0]*lr[0];
  for(l=1;l<=t-1;l++){
    auxp  += pow(C[t][l],p-1)*R[t][l]*lr[l];
  }
    auxp += 0.5*pow(C[t][t],p-1)*R[t][t]*lr[t];
    auxp *= h * p ;
    
    munew = T*lr[t] + auxp;
  return munew;
}


float compute_C(float *lr, float T, int t, int l,float*mu, float**C,float** R, float Deltap, float h, int p ){
  int m, n;
  float auxp1,auxp2;
  float Cnew;

  auxp1 = .5*pow(C[t][0],p-2)*R[t][0]*C[l][0]*lr[t]*lr[0];
  // C is filled by the algorithm only when t'<t, in this case we want
  // to swith l and m, so we have to order C.
  for(m=1; m<=l-1;m++){
    auxp1 += pow(C[t][m],p-2)*R[t][m]*C[l][m]*lr[t]*lr[m];
  }
  for(m=l; m<=t-1;m++){
    auxp1 += pow(C[t][m],p-2)*R[t][m]*C[m][l]*lr[t]*lr[m];
  }
  auxp1 += .5*pow(C[t][t],p-2)*R[t][t]*C[t][l]*lr[t]*lr[t];
  auxp1 *=  (p-1) * pow(h, 2) ;
  

  auxp2 = .5*pow(C[t][0],p-1)*R[l][0]*lr[t]*lr[0];
  for(n=1;n<=l-1;n++){
    auxp2 += pow(C[t][n],p-1)*R[l][n]*lr[t]*lr[n];
  }
  auxp2 += .5*pow(C[t][l],p-1)*R[l][l]*lr[t]*lr[t];
  auxp2 *= pow(h, 2) ;
  
  Cnew = C[t][l] * (1 - mu[t] * lr[t] * h);
//  Cnew += 2 * T * h * R[l][t] * lr[t] * lr[t];
  Cnew += auxp1;
  Cnew += auxp2;
  return Cnew;
}

float compute_R(float *lr, float T, int t, int l,float*mu, float**C,float** R, float Deltap, float h, int p){
  int m;
  float kroneker;
  float auxp;
  
  float Rnew;
  
  auxp = .5*pow( C[t][l], p-2) * R[t][l] * R[l][l] * lr[t] * lr[0];
  for(m=l+1; m<=t-1;m++){
    auxp += pow(C[t][m], p-2) * R[t][m] * R[m][l] * lr[t] * lr[m];
  }
  auxp += .5*pow(C[t][t],p-2)*R[t][t]*R[t][l] * lr[t] * lr[t] ;
  auxp *= pow(h, 2) * (p-1);
  
  Rnew  = R[t][l] * ( 1 - h * lr[t] * mu[t] ); // mu term
//  Rnew += h *  kroneker * lr[t];
  Rnew += auxp;
  
  return Rnew;
}


float print_mu(float ** C,float ** R,float *mu, float *H, int Nmax, float h, FILE* mu_file){
	int i;
	for(i=0; i<=Nmax;i++){
    fprintf(mu_file, "%f \t %f \t %f\n",h*(float)i, H[i]  , mu[i]); // format time, Cbar(t) , mu(t)
	}
	return 1.1;
}

//  The function modifies the learning rate according to a given learning rate schedule.
void lr_policy(float beta, float lr0, float h, int Nmax, float *lr, bool crossover){
  double t; int i;
  if (crossover) { 
    float tCrossOver = pow(Nmax, 2./3.);
    bool set = false;
    for(i=0; i<=Nmax-1; i++){
      t = h*(float)i;
      
      if(t>tCrossOver){ 
          if(not set){
              set = true;
              tCrossOver = t;
          }
          lr[i] = lr0 / pow( (t-tCrossOver) / h, beta);
      }
      else{ lr[i] = lr0; }
    }
  }
  else{
      for(i=0; i<=Nmax-1; i++){
          t = h*(float)i;
          if(t>0){ lr[i] = lr0 / pow(t, beta); }
          else{    lr[i] = lr0; }
    }
  }
  return;
}
