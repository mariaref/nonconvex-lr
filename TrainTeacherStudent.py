import numpy as np
import math
from math import sqrt, pi
import torch

from torch import relu as relu
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

test_batch_size = int(1e4) ## defaut variable - can be changed

#Create a Student Network of K Nodes
class TwoLayerNN(nn.Module):
    def __init__(self,K,D,act_function,std0, both):
        """Two Layers FCC Network:
        - K :          hidden nodes
        - D :          input dimensions
        -act_function: activation function 
        -std0:         std of the initial weights """
        print("Creating a FC with D: %d, K: %d"%(D,K) )
        super(TwoLayerNN, self).__init__()
        self.D=D
        self.gname=act_function
        if act_function=="erf":
            self.g = lambda x : torch.erf(x/math.sqrt(2))
        if act_function=="relu":
            self.g = lambda x : torch.relu(x)
        if act_function=="lin":
            self.g = lambda x : x
        self.K=K

        self.fc1 = nn.Linear(D, K, bias=False)
        self.fc2 = nn.Linear(K, 1, bias=False)
        
        self.both = both
        self.fc2.weight.requires_grad =  both
        
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        self.fc1.weight.data *= std0
        self.fc2.weight.data *= std0
        
    def forward(self, x):
        # input to hidden
        x = self.g(self.fc1(x)/sqrt(self.D) )
        x = self.fc2(x)
        return x

def lr_policy(step, beta, h, time0):
    step0 = time0/h
    if step  <= step0 :
        return 1
    else:
        dt = ((step - step0))**(-beta)
        return min(1, dt)
    
def compute_loss(loader, student, criterion):
    loss = 0
    for ii, (xs, ys) in enumerate(loader):
        loss += criterion(ys, student(xs))
    return loss/len(loader)
    
def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs += [param_group['lr']]
    return lrs
    
### FIle written in order to run just the training for varey long ##
def main(args, logfname):

    # random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # various usefull boleans
    protocol = args.time_protocol>0 or args.beta>0
    train_online = args.online ==1
    train_batch  = not train_online
    train_both_layers = args.both == 1
    
    
    ## Creates a FC
    teacher=TwoLayerNN(args.teacher_dimension,args.input_dimension,args.activation_function, args.std0, (args.both==1))
    teacher.fc2.weight.requires_grad = False
    teacher.fc1.weight.requires_grad = False
    
    ## Creates a FC
    student=TwoLayerNN(args.hidden_dimension,args.input_dimension,args.activation_function, args.std0,(args.both==1))
        
    if args.save_initial:
        torch.save(
            {'teacher': teacher.state_dict(), 
             'student': student.state_dict()
            },
            logfname[:-4] + '.pyT' )
    if args.prefix:
        weights = torch.load(args.prefix)
        student.load_state_dict(weights['student'])
        teacher.load_state_dict(weights['teacher'])
        
    ## Loss
    criterion = lambda x,y: 0.5 * nn.MSELoss()(x,y)
    
    ## optimiser
    params = []
    if args.optimizer == "sgd":
        params += [{'params': student.fc1.parameters()}] # If we train the last layer, ensure its learning rate scales correctly
        if train_both_layers:
            params += [{'params': student.fc2.parameters()}]
        optimizer = optim.SGD(params, lr=args.learning_rate, weight_decay = args.weight_decay )
    elif args.optimizer == "adam":
        params += [{'params': student.fc1.parameters()}]
        if train_both_layers:
            params += [{'params': student.fc2.parameters()}]
        optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay = args.weight_decay )
        
    step = 0
    dstep = 1./args.input_dimension # can be changed
    
    # learning rate schedule
    if protocol:
        lambda1 = lambda step : lr_policy(step, args.beta, dstep, args.time_protocol)
        if args.both == 0:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
        elif args.both == 1:
            lambda2 = lambda step : lr_policy(step, args.beta, dstep, args.time_protocol)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
            
    
    ## creates a test set
    test_xs = torch.randn(test_batch_size, args.input_dimension)
    test_ys = teacher(test_xs)
    
    if train_batch:
        train_xs = torch.randn(test_batch_size, args.input_dimension)
        train_ys = teacher(train_xs) 
        train_ys+= args.std * torch.randn(train_ys.shape)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_xs, train_ys), batch_size = args.batch_size, shuffle = True)
        
    # when to print?
    end = torch.log10(torch.tensor([1. * args.steps])).item()
    steps_to_print = list(torch.logspace(-2, end, steps=100))
    
    
    print("I am going to print for %d steps"%len(steps_to_print))    
    print("saving in %s"%logfname)
    logfile = open(logfname, "w", buffering=1)
    
    iteration = 0    
    while len(steps_to_print)>0:
        
                
        if step >= steps_to_print[0].item() or step == 0:
            with torch.no_grad():
                student.eval()
                if train_online or step == 0:
                    egtrain = 0;
                else: 
                    egtrain =  compute_loss(train_loader, student, criterion).item() # train error
                egtest =  criterion(student(test_xs), test_ys).item() # generalisation error w.r.t. the noiseless teacher
                lr =  (scheduler.get_last_lr()[0] if protocol else args.learning_rate)
                msg = ("%g, %g, %g, %g" % (step, egtest, egtrain, lr))
                for lr in get_lr(optimizer):
                    msg += ", %.5f"%lr
                msg = msg[:-1]
                print(msg)
                logfile.write(msg + "\n")
                
                if step>0:
                    steps_to_print.pop(0)
        
        # training
        if train_online:
            xs  = torch.randn(1, args.input_dimension)
            ys  = teacher(xs)
            ys += args.std * torch.randn(ys.shape)
        else: 
            xs , ys  = iter(train_loader).next()
            
        student.train()
        preds = student(xs)
        loss = criterion( ys , preds)
        
        student.zero_grad()
        loss.backward()
        optimizer.step()
        if protocol:
            scheduler.step()
  
        step += dstep
        #############
        
    print("Bye-bye")  
    
    

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-activation_function", "--activation_function", type=str,default="erf",
                    help="activation function of the NN")
    
    parser.add_argument('-hidden_dimension', '--hidden_dimension', metavar='hidden_dimension', type=int, default=2,
                        help="Student hidden layer width")
    parser.add_argument("-teacher_dimension", "--teacher_dimension", type=int, default=2,
                        help="Teacher hidden layer width")
    parser.add_argument("-both", "--both", type=int, default=True,
                        help="train both layers")
    
    parser.add_argument('-input_dimension', '--input_dimension', metavar='input_dimension', type=int, default=500,
                        help="input dimensions default = 500")
    parser.add_argument('-std0', '--std0', type=float, default=1.,
                        help="initial weights samples i.i.d. N(0, std0^2)")
    parser.add_argument('-std', '--std', type=float, default=0.,
                        help="output noise in the labels")
    
    
    parser.add_argument("--learning_rate", type=float, default=.5,
                        help="learning rate constant")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        help="optimizer  sgd or adam")
    
    parser.add_argument("--beta", type=float, default=0,
                        help="how to decay the learning rate")
    parser.add_argument("--time_protocol", type=float, default=0,
                        help="whether to keep constant and decay as 1/t")
    parser.add_argument("--online", type=int, default=0,
                        help="train online or not")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="batch_size") 
    
    parser.add_argument('-steps', '--steps', type=int, default=int(1e5),
                        help="steps of simulations. Default 1e5")
    parser.add_argument('-weight_decay', '--weight_decay', type=float, default=0,
                        help="weight_decay used in training")
      
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random seed for reproducibility. Default=0")
    parser.add_argument("-output_directory", "--output_directory", type=str, default="",
                        help="Where to save the data. Default=output_directory.")
    parser.add_argument('-comment', '--comment', type=str, default="",
                        help="A nice Comment to add to the logfilename")
    
    parser.add_argument('-save_initial', '--save_initial', type=int, default=None,
                        help="save initial weights")
    parser.add_argument('-prefix', '--prefix', type=str, default=None,
                        help="load the weights from")

    args = parser.parse_args()
    
    logfname = "GMM%s%s%s_%s_std0%.2f_std%.2f_M%d_K%d%s_D%d_lr%.2f_bs%d_wd%.2f_s%d_beta%g%s.dat"%( \
                         "_"+args.comment if args.comment!="" else "",\
                         "_online" if args.online==1 else "",\
                         "_adam" if args.optimizer == "adam" else "",\
                          args.activation_function, \
                          args.std0,\
                          args.std, \
                          args.teacher_dimension,\
                          args.hidden_dimension,\
                          ("_both" if args.both==1 else ""),\
                          args.input_dimension,\
                          args.learning_rate, \
                          args.batch_size, \
                          args.weight_decay, \
                          args.seed, \
                          args.beta,\
                          "_tp%g"%args.time_protocol if args.time_protocol>0 else ""  )
    if args.output_directory != "": logfname = args.output_directory  + "/" + logfname

    args = parser.parse_args()
    main(args, logfname)
