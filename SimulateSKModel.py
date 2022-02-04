import numpy as np
import matplotlib
from math import factorial, sqrt
import sys 
import torch
import torch.nn as nn
import torchvision
import argparse


#p-spin gradient descent, p=3
def calc_loss(s,J,p):
    N = len(s)
    if p==2:
        return - .5 * np.dot(np.dot(J,s),s)/N
    elif p==3:
        return - .5 * np.dot(np.dot(np.dot(J,s),s),s)/N

def psm_gd(N, iterations, lr, seed, noise, temperature, schedule_args, p=2, init_s=None, init_iteration=1, fname = None, args = None, data = None):
    signal = args.signal==1        
    if seed is not None:
        np.random.seed(seed)
    scale = np.sqrt(0.5*factorial(p)/N**(p-1))
    if p==2:
        J = np.random.normal(scale=scale,size=N*N).reshape(N,N)
        for i in range(N):
            for j in range(i,N):
                J[i,j] = J[j,i]
        J *= noise
        if signal:
            J += np.outer(np.ones(N), np.ones(N))/N
     
    w, v = np.linalg.eigh(J)
    topv = v[:,-1]
    gap = w[-1]-w[-2]
    gs = - .5 * w[-1] 
    print('GS:',gs, 'max eigval:', w[-1], 'gap:', gap)
    
    
    
    if init_s is None:
        s = np.random.normal(size=N)
        s = s * np.sqrt(N/np.sum(s**2))
    else:
        s = init_s
        
    if lr is None: ## fixes the initial learning rate to ensure eta_0 > 1 / (delta * 4)
        lr = 1 / (gap * 3.8) # which is then twice the right value
        
    grad = np.zeros(N)
    losses = []
    mags = []
    lrs = []
    sphericals = []
    
    iteration = init_iteration
    k = 0
    
    orig_lr = lr
    
    end = np.log10(np.asarray([1. * iterations])).item()
    steps_to_print = (np.logspace(-2, end, 100))
    steps_to_print = list(np.unique(np.int_(steps_to_print)))
    print("saving for %d steps"%len(steps_to_print))
    
    try:
        while iteration<iterations:
            if schedule_args['schedule']=='step':
                if iteration in schedule_args['decay_steps']: lr/=schedule_args['decay_amount']
            elif schedule_args['schedule']=='power':
                lr = orig_lr / (iteration-init_iteration+1)**schedule_args['exponent']
            elif schedule_args['schedule']=='log':
                lr = orig_lr / np.log(iteration+1)
    
            if p==2:
                grad = - np.dot(J,s)
            # else:
            #     grad = - np.dot(np.dot(J,s),s) - np.dot(np.dot(s,J).T,s) - np.dot(s,np.dot(J,s))
            #s += - lr * (grad + 2 * noise * np.random.normal(size=N))
            s += - lr * (grad + sqrt(2 * temperature) * np.random.normal(size=N))
            s = s * np.sqrt(N/np.sum(s**2))
    
            loss = calc_loss(s,J,p) - gs
            spherical = temperature * lr - 2 * (loss+gs)
            
            mag = abs(np.dot(s,topv)/N**.5)
            losses.append(loss)
            mags.append(mag)
            lrs.append(lr)
            sphericals.append(spherical)
            if iteration>steps_to_print[0]:
                print('{0:.1f}% done, loss = {1:.8f}, lr={2:.8f}'.format(iteration/float(iterations)*100.,loss, lr))
                print("saving")
                dict_ = {"args" : args, \
                                "losses" : np.array(losses), \
                                "mags" : np.array(mags), \
                                "s" : s, \
                                "lrs" : np.array(lrs), \
                                "sphericals" : np.array(sphericals),\
                                 "gap" : gap, "w" : w, "v" : v, "s": s}
                if data is None:
                    torch.save(dict_,  fname)
                    
                else: 
                    datainternal = {}
                    for name, el in data.keys():
                        datainternal[name + "1"] = el
                    for name, el in dict_.keys():
                        datainternal[name + "2"] = el
                    torch.save(datainternal,  fname)
                steps_to_print.pop(0)
                
            iteration +=1
            
    except KeyboardInterrupt:
        print("caught keyboard interupt")
        dict_ = {"args" : args, \
                        "losses" : np.array(losses), \
                        "mags" : np.array(mags), \
                        "s" : s, \
                        "lrs" : np.array(lrs), \
                        "sphericals" : np.array(sphericals),\
                         "gap" : gap, "w" : w, "v" : v, "s": s}
        if data is None:
            torch.save(dict_,  fname)
            
        else: 
            datainternal = {}
            for name, el in data.keys():
                datainternal[name + "1"] = el
            for name, el in dict_.keys():
                datainternal[name + "2"] = el
            torch.save(datainternal,  fname)
            
        return np.array(losses), np.array(mags), np.array(lrs), np.array(sphericals), s, gap, w, v
    
    return np.array(losses), np.array(mags), np.array(lrs), np.array(sphericals), s, gap, w, v

def get_fname(N, seed, temperature, iterations, lr, exponent, noise, tl):
    fname = f"fig4_N%d_s%d_T%.1f_iter%d_lr%.4f_exp%.2f_noise%.1f"%(N, seed, temperature, iterations, (lr if lr is not None else -1), exponent, noise)
    if tl is not None:
        fname += f"_tl{tl}"
    return fname+".pyT"

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int,default=3000)
    parser.add_argument("--seed", type=int,default=1)
    parser.add_argument("--temperature", type=float,default=3)
    parser.add_argument("--iterations", type=int,default=10000)
    parser.add_argument("--lr", type=float,default=None)
    parser.add_argument("--exponent", type=float,default=0.)
    parser.add_argument("--noise", type=float,default = 1)
    parser.add_argument("--output_directory", type=str,default="")
    parser.add_argument("--comment", type=str,default="")
    parser.add_argument("--tl", type=int,default=None)
    parser.add_argument("--signal", type=int,default=0)
    args = parser.parse_args()
    
    # for the learning rate scheduling
    protocol = args.tl is not None
    

    fname = get_fname(args.N, args.seed, args.temperature, args.iterations, args.lr, args.exponent, args.noise, args.tl)
    if args.output_directory != "":
        fname = args.output_directory  + "/%s"%("" if args.comment=="" else args.comment)  + ("" if args.signal==0 else "_signal") + fname
    
    print(f"saving in {fname}")
    if not protocol:
        schedule_args = {'schedule':'power', 'exponent':args.exponent}
        losses, mags, s, lrs,sphericals, gap, w, v = psm_gd(args.N, \
                                                            args.iterations, \
                                                            args.lr, \
                                                            schedule_args=schedule_args, \
                                                            p=2, \
                                                            noise=args.noise, \
                                                            temperature=args.temperature, \
                                                            seed=args.seed,\
                                                            fname = fname, \
                                                            args = args)
        data = {"args" : args, "losses" : losses, "mags" : mags, "s" : s, "lrs" : lrs, "sphericals" : sphericals,\
                "gap" : gap, "w" : w, "v" : v, "s": s}
    
    elif protocol:
        schedule_args = {'schedule':'power', 'exponent':0.0}
        schedule_args2 = {'schedule':'power', 'exponent':args.exponent}

        losses1, mags1, lrs1,sphericals1, s1, gap1, w1, v1 = psm_gd(args.N, \
                                                args.tl,  \
                                                args.lr, \
                                                seed=args.seed, \
                                                noise=args.noise, \
                                                temperature=args.temperature, \
                                                schedule_args = schedule_args,\
                                                fname = fname, \
                                                args = args)
        
        
        data = {"args" : args, "losses1" : losses1, "mags1" : mags1, "lrs1" : lrs1, "sphericals1" : sphericals1,\
                "s1":s1 , "gap1" : gap1, "w1":w1, "v1": v1, "s1": s1 }
        losses2, mags2, lrs2,sphericals2, s2, gap2,w2, v2 = psm_gd(args.N, \
                                                args.iterations, \
                                                args.lr, \
                                                seed=args.seed, \
                                                noise=args.noise, \
                                                temperature=args.temperature, \
                                                schedule_args=schedule_args2, \
                                                init_s = s1, \
                                                init_iteration = args.tl, \
                                                fname = fname, \
                                                args = args)
        
        data.update({"losses2" : losses2, "mags2" : mags2,"lrs2" : lrs2 ,"sphericals2":sphericals2,\
                     "s2":s2 , "gap2" : gap2, "w2":w2, "v2":v2, "s2": s2 }  )
    
    print(f"finished!")
    torch.save(data,  fname)
    
