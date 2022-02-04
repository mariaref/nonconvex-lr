#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <time.h>
#include <string.h>
#include <getopt.h>
#include <iostream>
#include <unistd.h>
#include <stdexcept>
#include <sstream>

const char * usage = R"USAGE(
This is 2-p-spin_fixed_grid++, a tool to solve the spiked tensor model dynamical equation

usage: *.exe [-h] [--g G]

optional arguments:
  -h, -?                show this help message and exit
  --p P                 the variance of the matrix channel
  --Delta2               Noise of the two-spin interaction
  --Deltap               the variance of the tensor channel
  --Cbar0                initial value of magnetisation
  --Nmax                 number of points in the discretization of the grid
  --tmax                 maximum simulation time
  --T                    temperature
  --protocol             decrease with t^-beta after some time
  --tp                   time to start decreasing the learning rate

)USAGE";


// This structure contains the physical parameters of the system.
typedef struct _psys {
  float p,Delta2,Deltap,Tg,alpha,Cbar0;  // system parameters
  float T;
  long double *lr;   // learning rate
} psys;


//  The function computes the equation for mu.
float compute_mu(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h);
//  The function computes the equation for energy.
float compute_e(float*e, float*Cbar, float**C,float** R, int t, psys *w,float h);
//  The function computes the equation for Cbar.
float compute_Cbar(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h);
//  The function computes the equation for C.
float compute_C(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h);
//  The function computes the equation for R.
float compute_R(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h);

//  The function produces the output file concerning correlation and response function.
float print(float** C,float** R,float* mu,float *Cbar,int Nmax,float h,FILE* correlation_file,FILE* response_file,FILE* mu_file,FILE* time_file);
//  The function produces the output file concerning the Lagnange multiplier.
std::string status(float t, float lr, float Cbar, float mu, float H );
// defines the learning rate protocol
float lr_policy(psys *w, float beta, float lr0, float h, int Nmax, int t0);

//  The function propagates the initial values of C,R,Cbar in time up to a given time.
int main (int argc, char *argv[]) {
    
  // Assign the parameters of the system.
  double p=3.;
  float Delta2=0.5;      // the variance of the matrix channel
  float Deltap=.1;      // the variance of the tensor channel
  float Cbar0 = 0.0000000001;    // initial value of Cbar
  float h = 0.1;        // number of points in the discretization of the grid
  float tmax=100.;        // maximum simulation time
  float T=0.;        // maximum simulation time
  float beta = 0. ;  // lr schedule to use
  float lr0 = 1.;    // initial learning rate
  int protocol = 0; // decrease with t^-beta after some time
  float tp = -1; // decrease with t^-beta after some time
  std::string output_dir = "data_tp";  // output directory
  
  int c;
  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"p",            required_argument, 0, 'p'},
    {"Delta2",       required_argument, 0, 'd'},
    {"Deltap",       required_argument, 0, 'f'},
    {"Cbar0",        required_argument, 0, 'c'},
    {"h",            required_argument, 0, 'n'},
    {"tmax",         required_argument, 0, 't'},
    {"T",            required_argument, 0, 'h'},
    {"lr0",          required_argument, 0, 'l'},
    {"beta",         required_argument, 0, 'b'},
    {"tp",           required_argument, 0, 'r'},
    {"protocol",     no_argument, &protocol,  1},
    {"output_dir",   required_argument, 0, 'k'},
    {0, 0, 0, 0}
  };
  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;
    c = getopt_long(argc, argv, "p:d:f:c:n:t:h:l:b:r:k:",
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
      case 'd':
        Delta2 = atof(optarg);
        break;
      case 'f':
        Deltap = atof(optarg);
        break;
      case 'c':
        Cbar0 = atof(optarg);
        break;
      case 'n':
        h = atof(optarg);
        break;
      case 't':
        tmax = atof(optarg);
        break;
      case 'r':
        tp = atof(optarg);
        break;
      case 'h':
        T = atof(optarg);
        break;
      case 'b':
        beta = atof(optarg);
        break;
      case 'l':
        lr0 = atof(optarg);
        break;
      case 'k':
        output_dir  = std::string(optarg);
        break;
      case '?':
        std::cout << usage << std::endl;
        return 0;
      default:
        abort ();
    }
  }
  
  psys w;
  
  int Nmax;
  float **C, **R, *Cbar, *mu, *H;
  int i,t,l;
  
  // Assign the parameters of the system.
  w.p= p ;
  w.Delta2= Delta2;      // the variance of the matrix channel
  w.Deltap=Deltap;      // the variance of the tensor channel
  w.Cbar0 = Cbar0;    // initial value of Cbar
  w.T = T;            // temperature
  h /= w.Delta2;
  Nmax = (int)( ( (float)tmax)/ ( h * w.Delta2) ) ;
  float mjump = 0.01; // value at which the jump is determined
  
  char* log_fname;
  char* output_dir_desc;
  asprintf(&output_dir_desc, "%s/", output_dir.c_str());
  
  asprintf(&log_fname, "%smu_P%.0f_T%.4f_beta%.2f_lr0%.2f_Delta2-%.4f_DeltaP-%.4f_Cbar-%.2e_t-%.2f_Nmax-%d%s_tp%.1f.txt",output_dir_desc,w.p,T, beta, lr0,w.Delta2,w.Deltap,w.Cbar0,tmax,Nmax, (protocol ? "_protocol" : ""), tp);
  
  std::cout << "# saving in "<< log_fname<<std::endl;
  
  FILE* logfile = fopen(log_fname, "w");
  
  fprintf(stderr, "# Dynamics for (2+%.0f)-spin planted with Delta2 %.4f and Deltap %.4f (alpha %.4f)\n",w.p,w.Delta2,w.Deltap,w.alpha);
  fprintf(stderr, "# System initialized to %.2e magnetization\n",w.Cbar0);
  fprintf(stderr, "# Maximum simulation time %.0f using %d steps of size %.2e\n",tmax,Nmax,h);
  fprintf(logfile, "# Dynamics for (2+%.0f)-spin planted with Delta2 %.4f and Deltap %.4f (alpha %.4f)\n",w.p,w.Delta2,w.Deltap,w.alpha);
  fprintf(logfile, "# System initialized to %.2e magnetization\n",w.Cbar0);
  fprintf(logfile, "# Maximum simulation time %.0f using %d steps of size %.2e\n",tmax,Nmax,h);
  
  C=(float**)malloc((Nmax+1)*sizeof(float*));
  R=(float**)malloc((Nmax+1)*sizeof(float*));
  Cbar=(float*)malloc((Nmax+1)*sizeof(float));
  mu=(float*)malloc((Nmax+1)*sizeof(float));
  H=(float*)malloc((Nmax+1)*sizeof(float));
  w.lr=(long double*)malloc((Nmax+1)*sizeof(long double));
  for(i=0; i<=Nmax;i++){
    C[i]=(float*)malloc((i+1)*sizeof(float));
    R[i]=(float*)malloc((i+1)*sizeof(float));
  }
  
  // inizialization.
  C[0][0]=1;
  R[0][0]=0;
  Cbar[0]=w.Cbar0;
  
  // learning rate
  if(protocol or tp>0){
    lr_policy(&w, 0, lr0, h, Nmax, 0); // if protocol or tp the begining is at constant learning rate
  }
  else{
      lr_policy(&w, beta, lr0, h, Nmax, 0);
  }
  
  fprintf(logfile, "# format time , learning rate , magnetisation, mu, energy\n");
  // integration loop.
  bool valid_tp = tp>0;
  
  for(i=0; i<=Nmax-1; i++){
    
    if(isnan(Cbar[i])){break;} // if is nan simply stops
    
    t=i;
    
    // changes the leaning rate
    if( (protocol and Cbar[i] > mjump) or (valid_tp and i*h>tp) ){
      std::cout<<"applying protocol!!"<<std::endl;
      lr_policy(&w, beta, lr0, h, Nmax, i);
      protocol = false;
      valid_tp = false;
    }
    
    compute_mu(mu, Cbar, C, R, t, &w, h);
    compute_e(H, Cbar, C, R, t, &w, h);
    C[t+1][t+1]=1;
    R[t+1][t+1]=0;
    Cbar[t+1]=compute_Cbar(mu, Cbar, C, R, t, &w, h);
    // compute the two times observables.
    for(l=0;l<=t && t<Nmax;l++){
      C[t+1][l]=compute_C(t,l,mu,Cbar,C,R,&w,h);
      R[t+1][l]=compute_R(t,l,mu,Cbar,C,R,&w,h);
    }

    
    if( (i%20==0 and i!=0) or i==1 ){
      fprintf(stderr, "Iteration %d of %d , time %g of %g \n",i,Nmax, i*h, tmax);
      std::string msg = status(h*(float)i, w.lr[i], Cbar[i], mu[i], H[i] );
      std::cout << msg << std::endl;
      fprintf(logfile, "%s\n", msg.c_str());
      fflush(logfile);
    }
    R[t+1][t]=1; // comes from ito causality
  }
  free(C); free(R); free(mu);
    
  return 0;
}

//---------------------------------------------------

float compute_mu(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h){
  
  int l;
  float auxp,aux2;
  
  aux2  = w->lr[0]*C[t][0]*R[t][0];
  auxp  = w->lr[0]*0.5*w->p*pow(C[t][0],w->p-1)*R[t][0];
  for(l=1;l<=t-1;l++){
    aux2  += w->lr[l]*2.*C[t][l]*R[t][l];
    auxp  += w->lr[l]*w->p*pow(C[t][l],w->p-1)*R[t][l];
  }
  aux2 += w->lr[t]*C[t][t]*R[t][t]; aux2/=w->Delta2;
  auxp += w->lr[t]*0.5*w->p*pow(C[t][t],w->p-1)*R[t][t]; auxp/=w->Deltap;
  
  mu[t] = pow(Cbar[t],w->p)/w->Deltap+Cbar[t]*Cbar[t]/w->Delta2;
  mu[t]+= h*(auxp+aux2);
  mu[t]+= w->lr[t]*w->T;
  
  return 0;
  
}

//---------------------------------------------------

float compute_e(float*e, float*Cbar, float**C,float** R, int t, psys *w,float h){
  
  int l;
  float auxp,aux2;
  float ep, e2;
  
  aux2  = w->lr[0]*C[t][0]*R[t][0];
  auxp  = w->lr[0]*0.5*w->p*pow(C[t][0],w->p-1)*R[t][0];
  for(l=1;l<=t-1;l++){
    aux2  += w->lr[l]*2.*C[t][l]*R[t][l];
    auxp  += w->lr[l]*w->p*pow(C[t][l],w->p-1)*R[t][l];
  }
  aux2 += w->lr[t]*C[t][t]*R[t][t]; aux2/=w->Delta2;
  auxp += w->lr[t]*0.5*w->p*pow(C[t][t],w->p-1)*R[t][t]; auxp/=w->Deltap;
  
  ep = pow(Cbar[t],w->p)/w->Deltap + h*(auxp);
  ep *= -1./w->p;
  
  e2 = Cbar[t]*Cbar[t]/w->Delta2 + h*(aux2);
  e2 *= -1./2;
  
  e[t] = ep + e2;
  return 0;
  
}

//---------------------------------------------------

float compute_Cbar(float*mu, float*Cbar, float**C,float** R, int t, psys *w,float h){

  float Cbar_new;
  float aux2,auxp;
  int m;

  aux2=w->lr[0]*0.5*R[t][0]*Cbar[0];
  auxp=w->lr[0]*0.5*(w->p-1)*R[t][0]*pow(C[t][0],w->p-2)*Cbar[0];
  for(m=1;m<=t-1;m++){
    aux2+=w->lr[m]*R[t][m]*Cbar[m];
    auxp+=w->lr[m]*(w->p-1)*R[t][m]*pow(C[t][m],w->p-2)*Cbar[m];
  }
  aux2+=w->lr[t]*0.5*R[t][t]*Cbar[t]; aux2/=w->Delta2;
  auxp+=w->lr[t]*0.5*(w->p-1)*R[t][t]*pow(C[t][t],w->p-2)*Cbar[t]; auxp/=w->Deltap;

  Cbar_new = Cbar[t]+h*w->lr[t]*(-mu[t]*Cbar[t]+Cbar[t]/w->Delta2+pow(Cbar[t],w->p-1)/w->Deltap);
  Cbar_new+= h*h*w->lr[t]*(aux2+auxp);

  return Cbar_new;
}

//-----------------------------------------------------

float compute_C(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h){

  int m,n;
  float auxp1,auxp2,aux21,aux22;
  float Cnew;

  aux21 = w->lr[0]*.5*R[t][0]*C[l][0];
  auxp1 = w->lr[0]*.5*(w->p-1)*pow(C[t][0],w->p-2)*R[t][0]*C[l][0];
  // C is filled by the algorithm only when t'<t, in this case we want
  // to swith l and m, so we have to order C.
  for(m=1; m<=l-1;m++){
    aux21 += w->lr[m]*R[t][m]*C[l][m];
    auxp1 += w->lr[m]*(w->p-1)*pow(C[t][m],w->p-2)*R[t][m]*C[l][m];
  }
  for(m=l; m<=t-1;m++){
    aux21 += w->lr[m]*R[t][m]*C[m][l];
    auxp1 += w->lr[m]*(w->p-1)*pow(C[t][m],w->p-2)*R[t][m]*C[m][l];
  }
  aux21 += w->lr[t]*.5*R[t][t]*C[t][l]; aux21/=w->Delta2;
  auxp1 += w->lr[t]*.5*(w->p-1)*pow(C[t][t],w->p-2)*R[t][t]*C[t][l]; auxp1/=w->Deltap;

  aux22 = w->lr[0]*.5*C[t][0]*R[l][0];
  auxp2 = w->lr[0]*.5*pow(C[t][0],w->p-1)*R[l][0];
  for(n=1;n<=l-1;n++){
    aux22 += w->lr[n]*C[t][n]*R[l][n];
    auxp2 += w->lr[n]*pow(C[t][n],w->p-1)*R[l][n];
  }
  aux22 += w->lr[t]*.5*C[t][l]*R[l][l]; aux22/=w->Delta2;
  auxp2 += w->lr[t]*.5*pow(C[t][l],w->p-1)*R[l][l]; auxp2/=w->Deltap;

  Cnew = C[t][l]+h*w->lr[t]*(-mu[t]*C[t][l]+Cbar[l]*Cbar[t]/w->Delta2+Cbar[l]*pow(Cbar[t],w->p-1)/w->Deltap);
  Cnew+= h*w->lr[t]*((aux21+aux22)*h+(auxp1+auxp2)*h);
//  Cnew+= 2*w->T* w->lr[t]* w->lr[t]*R[l,t]; by causality is always 0 because integrating l<=t
  return Cnew;
  
}

//-----------------------------------------------------

float compute_R(int t, int l,float*mu, float*Cbar, float**C,float** R, psys *w,float h){
  int m;
  float auxp,aux2;
  float Rnew;
  
  aux2 = w->lr[l]*.5*R[t][l]*R[l][l];
  auxp = w->lr[l]*.5*(w->p-1)*pow(C[t][l],w->p-2)*R[t][l]*R[l][l];
  for(m=l+1; m<=t-1;m++){
    aux2 += w->lr[m]*R[t][m]*R[m][l];
    auxp += w->lr[m]*(w->p-1)*pow(C[t][m],w->p-2)*R[t][m]*R[m][l];
  }
  aux2 += w->lr[t]*.5*R[t][t]*R[t][l]; aux2/=w->Delta2;
  auxp += w->lr[t]*.5*(w->p-1)*pow(C[t][t],w->p-2)*R[t][t]*R[t][l]; auxp/=w->Deltap;

  Rnew = R[t][l]+h*w->lr[t]*(-mu[t]*R[t][l]+h*aux2+h*auxp);
  //  Rnew+= w->lr[t]*kroneker; --> always 0 by ito's convention
  return Rnew;
  
}

//-----------------------------------------------------
float print(float ** C,float ** R,float *mu,float *Cbar,int Nmax, float h, FILE* correlation_file, FILE* response_file, FILE* mu_file,FILE* time_file){
  int i,j;
  
  for(i=0; i<=Nmax;i++){
    
    // if(i%10==0){
    if(1){
      for(j=0;j<Nmax-i; j++){
        
        fprintf(correlation_file, "%f ", C[i+j][j]);
        fprintf(response_file, "%f ", R[i+j][j]);
        fprintf(time_file,"%f", h*(float)(i+j));
      }
      
      fprintf(correlation_file,"\n");
      fprintf(response_file,"\n");
      fprintf(time_file,"\n");
    }
  }
  return 1.1;
}

float print_mu(float ** C,float ** R,float *mu,float *Cbar,int Nmax, float h, FILE* mu_file){
  int i;
  
  for(i=0; i<=Nmax;i++){
    fprintf(mu_file, "%f , %f , %f \n",h*(float)i, Cbar[i], mu[i]); // format time, Cbar(t) , mu(t)
  }
  return 1.1;
}


//  The function modifies the learning rate according to a given learning rate schedule.
float lr_policy(psys *w, float beta, float lr0, float h, int Nmax, int i0){
  double t; int i;
  for(int i=0; i<i0; ++i){
    w->lr[i] = lr0;
  }
  for(i = i0; i < Nmax;++i){
    t = h*((double)(i - i0));
    w->lr[i] = lr0 / pow(t+1, beta);
  }
  return 0;
}

std::string status(float t, float lr, float Cbar, float mu, float H ){
  std::ostringstream msg;
  msg << t << ", " << lr << ", " << Cbar << ", " << mu << ", " << H ;
  std::string msg_str = msg.str();
  return msg_str.substr(0, msg_str.length()-2);
}

