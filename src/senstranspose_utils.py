from src.assoc_utils_np import *
from numpy.random import shuffle 
from src.data_utils import *
from src.assoc_utils_np import *

def pseudotrain_Wgp(ca1book, gbook, Npatts):
    ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
    return np.einsum('ij, ljk -> lik', gbook[:,:Npatts], ca1inv[:,:Npatts,:]) 

def pseudotrain_Wpg(gbook, ca1book, Npatts):
    ginv = np.linalg.pinv(gbook[:,:Npatts])
    return np.einsum('ijk, kl -> ijl', ca1book[:,:,:Npatts], ginv[:Npatts,:])


def corrupt_p_1(codebook,p=0.1):
  if p==0.:
    return codebook
  rand_indices = np.sign(np.random.uniform(size=codebook.shape)- p )
  return np.multiply(codebook,rand_indices)


# # nearest neighbour
def cleanup(s, sbook):
    sclean = np.zeros_like(s)
    for i in range(len(sclean)):  #runs
        idx = np.argmax(s[i].T@sbook)
        sclean[i,:,0] = sbook[:,idx]
    return sclean


# nearest neighbour
# def cleanup(s, sbook):
#   a,b,c = s.shape
#   sr = np.reshape(s, (a,c,b))
#   idx = np.argmax(sr@sbook, axis=2)
#   sclean = sbook[:,idx]
#   x,y,z = sclean.shape
#   sclean = np.reshape(sclean, (y,x,z))
#   return sclean


def train_sensory(pbook, sbook, Npatts):
    return (1/Npatts)*np.einsum('ij, klj -> kil', sbook[:,:Npatts], pbook[:,:,:Npatts]) 

def dynamics_gs_random_sparse_p(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh):
    Ng = np.sum(np.square(lambdas)) 
    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]
    m = len(lambdas)
    s = sinit
    p = nonlin(Wps@s, thresh=0)
    for i in range(Niter):
        gin = Wgp@p
        g = module_wise_NN_2d(gin, module_gbooks, module_sizes)  # modular net
        #g = topk_binary(gin, m) 
        #p = nonlin(Wpg@g, thresh)
        p=Wpg@g
        # if i%2 == 0:
        #   p = np.sign(Wpg@g) 
        # else:
        #   p = np.sign(Wps@s) 
    #s = topk_binary(sin, sparsity)
    sin = Wsp@p
    s = np.sign(sin)    
    scup = cleanup(s, sbook)
    # errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errpc = np.linalg.norm(p-ptrue, axis=(1,2))/Np
    errgc = np.linalg.norm(g-gtrue, axis=(1,2)) /Ng
    errsens = np.linalg.norm(s-strue, axis=(1,2)) /Ns #(2*sparsity);
    errsenscup = np.linalg.norm(scup-strue, axis=(1,2))/Ns

    # errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*m);
    errsensl1 = np.sum(np.abs(s-strue), axis=(1,2))/(2*Ns) #(2*sparsity);
    # errsenscup = np.sum(np.abs(scup-strue), axis=(1,2))/(2*Ns)
    return errpc, errgc, errsens, errsenscup, errsensl1 

def dynamics_gs_vectorized_patts(sinit,Niter, sbook, pbook, gbook, Wgp, Wpg,Wsp,Wps,lambdas,sparsity,thresh,Npatts):
    Ns = sbook.shape[0]
    Np = pbook.shape[1]
    Ng = gbook.shape[0]
    
    # mean_p_norm = np.mean(np.linalg.norm(pbook[0],axis=0))
    # noise_val=1.
    # print("using p noise")#; mean_p_norm="+str(mean_p_norm))
    
    pin = nonlin(Wps@sinit[:,:,:Npatts], thresh=0)
    # pnoise = noise_val*mean_p_norm*np.random.normal(0,1,pin.shape)/np.sqrt(Np)
    
    # pin = pin+pnoise
    p = np.copy(pin)
    for i in range(Niter):
        gin = Wgp@p
        g = gridCAN_2d(gin,lambdas)
        p = nonlin(Wpg@g, thresh)
    pout = np.copy(p)
    gout = np.copy(g)
    sout = np.sign(Wsp@p)
    
    strue=sbook[:,:Npatts]
    ptrue=pbook[:,:,:Npatts]
    gtrue=gbook[:,:Npatts]
    
    s_l1_err = np.average(abs(sout - strue))/2
    
    s_l2_err = np.linalg.norm(sout-strue,axis=(1,2))/(Ns)
    p_l2_err = np.linalg.norm(pout-ptrue,axis=(1,2))/(Np)
    g_l2_err = np.linalg.norm(sout-strue,axis=(1,2))/(Ng)
    
    # print(strue.shape)
    # print(sout.shape)
    struenormed=(strue/np.linalg.norm(strue,axis=0));
    soutnormed=np.einsum('ijk,ik->ijk',sout,1/np.linalg.norm(sout,axis=1));
    # soutnormed=(sout/np.linalg.norm(sout,axis=1));
    dots = np.einsum('ijk,jk->ik',soutnormed,struenormed)
        
    #scup = cleanup(s, sbook)
    errpc = p_l2_err
    errgc = g_l2_err
    errsens = s_l2_err
    errsenscup = np.nan*np.zeros_like(errsens)#np.linalg.norm(scup-strue, axis=(1,2))/Ns

    errsensl1 = s_l1_err
    
    return errpc, errgc, errsens, errsenscup, errsensl1 

def dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh):
    Ng = np.sum(np.square(lambdas)) 
    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]
    m = len(lambdas)
    s = sinit
    p = nonlin(Wps@s, thresh=0)
    for i in range(Niter):
        gin = Wgp@p
        g = module_wise_NN_2d(gin, module_gbooks, module_sizes)  # modular net
        #g = topk_binary(gin, m) 
        p = nonlin(Wpg@g, thresh)
        # if i%2 == 0:
        #   p = np.sign(Wpg@g) 
        # else:
        #   p = np.sign(Wps@s) 
    #s = topk_binary(sin, sparsity)
    sin = Wsp@p
    s = np.sign(sin)    
    scup = cleanup(s, sbook)
    # errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errpc = np.linalg.norm(p-ptrue, axis=(1,2))/Np
    errgc = np.linalg.norm(g-gtrue, axis=(1,2)) /Ng
    errsens = np.linalg.norm(s-strue, axis=(1,2)) /Ns #(2*sparsity);
    errsenscup = np.linalg.norm(scup-strue, axis=(1,2))/Ns

    # errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*m);
    errsensl1 = np.sum(np.abs(s-strue), axis=(1,2))/(2*Ns) #(2*sparsity);
    # errsenscup = np.sum(np.abs(scup-strue), axis=(1,2))/(2*Ns)
    return errpc, errgc, errsens, errsenscup, errsensl1 

def dynamics_gs_vectorized_patts_cts(sinit,Niter, sbook, pbook, gbook, Wgp, Wpg,Wsp,Wps,lambdas,sparsity,thresh,Npatts):
    Ns = sbook.shape[0]
    Np = pbook.shape[1]
    Ng = gbook.shape[0]
    
    # mean_p_norm = np.mean(np.linalg.norm(pbook[0],axis=0))
    # noise_val=1.5
    # print("using p noise")#; mean_p_norm="+str(mean_p_norm))
    
    pin = nonlin(Wps@sinit[:,:,:Npatts], thresh=0)
    # pnoise = noise_val*mean_p_norm*np.random.normal(0,1,pin.shape)/np.sqrt(Np)
    
    # pin = pin+pnoise
    p = np.copy(pin)
    for i in range(Niter):
        gin = Wgp@p
        g = gridCAN_2d(gin,lambdas)
        p = nonlin(Wpg@g, thresh)
    pout = np.copy(p)
    gout = np.copy(g)
    sout = Wsp@p
    
    strue=sbook[:,:Npatts]
    ptrue=pbook[:,:,:Npatts]
    gtrue=gbook[:,:Npatts]
    
    s_l1_err = np.average(abs(sout - strue))/2
    
    s_l2_err = np.linalg.norm(sout-strue,axis=(1,2))/(Ns)
    p_l2_err = np.linalg.norm(pout-ptrue,axis=(1,2))/(Ns)
    g_l2_err = np.linalg.norm(sout-strue,axis=(1,2))/(Ns)
    
    # print(strue.shape)
    # print(sout.shape)
    struenormed=(strue/np.linalg.norm(strue,axis=0));
    soutnormed=np.einsum('ijk,ik->ijk',sout,1/np.linalg.norm(sout,axis=1));
    # soutnormed=(sout/np.linalg.norm(sout,axis=1));
    dots = np.einsum('ijk,jk->ik',soutnormed,struenormed)
        
    #scup = cleanup(s, sbook)
    errpc = p_l2_err
    errgc = g_l2_err
    errsens = dots.mean(axis=1) #s_l2_err
    errsenscup = np.nan*np.zeros_like(errsens)#np.linalg.norm(scup-strue, axis=(1,2))/Ns

    errsensl1 = s_l1_err
    
    return errpc, errgc, errsens, errsenscup, errsensl1 
    
def dynamics_gs_cts(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh):
    Ng = np.sum(np.square(lambdas)) 
    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]
    m = len(lambdas)
    s = sinit
    p = nonlin(Wps@s, thresh=0)
    for i in range(Niter):
        gin = Wgp@p
        g = module_wise_NN_2d(gin, module_gbooks, module_sizes)  # modular net
        #g = topk_binary(gin, m) 
        p = nonlin(Wpg@g, thresh)
        # if i%2 == 0:
        #   p = np.sign(Wpg@g) 
        # else:
        #   p = np.sign(Wps@s) 
    #s = topk_binary(sin, sparsity)
    sin = Wsp@p
    s=np.copy(sin)
    #s = np.sign(sin)    
    scup = cleanup(s, sbook)
    # errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errpc = np.linalg.norm(p-ptrue, axis=(1,2))/Np
    errgc = np.linalg.norm(g-gtrue, axis=(1,2)) /Ng
    #errsens = np.linalg.norm(s-strue, axis=(1,2)) /Ns #(2*sparsity);
    errsens = np.einsum('ij,j->i',s[:,:,0]/np.linalg.norm(s,axis=1),(strue/np.linalg.norm(strue.squeeze())).squeeze())
    errsenscup = np.linalg.norm(scup-strue, axis=(1,2))/Ns

    # errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*m);
    errsensl1 = np.sum(np.abs(s-strue), axis=(1,2))/(2*Ns) #(2*sparsity);
    # errsenscup = np.sum(np.abs(scup-strue), axis=(1,2))/(2*Ns)
    return errpc, errgc, errsens, errsenscup, errsensl1 
    
def dynamics_gs_linear_p(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh):
    Ng = np.sum(np.square(lambdas)) 
    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]
    m = len(lambdas)
    s = sinit
    p = nonlin(Wps@s, thresh=0)
    for i in range(Niter):
        gin = Wgp@p
        g = module_wise_NN_2d(gin, module_gbooks, module_sizes)  # modular net
        #p = nonlin(Wpg@g, thresh)
        p=Wpg@g
    sin = Wsp@p
    s = np.sign(sin)    
    scup = cleanup(s, sbook)
    # errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errpc = np.linalg.norm(p-ptrue, axis=(1,2))/Np
    errgc = np.linalg.norm(g-gtrue, axis=(1,2)) /Ng
    errsens = np.linalg.norm(s-strue, axis=(1,2)) /Ns #(2*sparsity);
    errsenscup = np.linalg.norm(scup-strue, axis=(1,2))/Ns

    # errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*m);
    errsensl1 = np.sum(np.abs(s-strue), axis=(1,2))/(2*Ns) #(2*sparsity);
    # errsenscup = np.sum(np.abs(scup-strue), axis=(1,2))/(2*Ns)
    return errpc, errgc, errsens, errsenscup, errsensl1 


def dynamics_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    g = gtrue
    p = np.sign(Wpg@g)
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        #s = np.sign(sin)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        if i%2 == 0:
            p = np.sign(Wpg@g) 
        else:
            p = np.sign(Wps@s)      
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 


def dynamics_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    p = pinit
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        #s = np.sign(sin)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net 
        if i%2 == 0:
            p = np.sign(Wpg@g) 
        else:
            p = np.sign(Wps@s) 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np) #np.sum(np.abs(pinit-ptrue), axis=(1,2));
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens    

def senstrans_gs_vectorized_patts(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    err_sensl1 = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    c = 0.60     # connection probability
    prune = int((1-c)*Np*Ng)
    mask = np.ones((Np, Ng))
    mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
    Wpg = np.multiply(mask, Wpg)

    thresh=0.5
    #thresh=-5
    
    print('thresh='+str(thresh))
    pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)
    Wgp = train_gcpc(pbook, gbook, Npos)

    k=0
    # print("Wsp and Wps Hebbian")
    for Npatts in tqdm(Npatts_lst):
        #print("k=",k)

        # Learning patterns 
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        # Wsp = train_sensory(pbook, sbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)
        # Wps = np.einsum('ijk->ikj',Wsp)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        sum_senscup = 0 
        sum_sensl1 = 0
        srep = np.repeat(sbook[None,:],nruns,axis=0)
        sinit = corrupt_p_1(srep, p=pflip)
        
        #For CTS
        # print(pflip)
        # sbook_std = np.std(sbook.flatten())
        # sinit = srep + np.random.normal(0,1,srep.shape)*pflip*sbook_std
        # print(srep.shape)
        # print(sinit.shape)
        
        
        err_pc[k],err_gc[k],err_sens[k],_,err_sensl1[k] = dynamics_gs_vectorized_patts(sinit,Niter, sbook, pbook, gbook, Wgp, Wpg,Wsp,Wps,lambdas,sparsity,thresh,Npatts)
        # err_pc[k],err_gc[k],err_sens[k],_,err_sensl1[k] = dynamics_gs_vectorized_patts_cts(sinit,Niter, sbook, pbook, gbook, Wgp, Wpg,Wsp,Wps,lambdas,sparsity,thresh,Npatts)

        k += 1   
    return err_pc, err_gc, err_sens, err_senscup, err_sensl1  


def senstrans_gs(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    err_sensl1 = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    c = 0.60     # connection probability
    prune = int((1-c)*Np*Ng)
    mask = np.ones((Np, Ng))
    mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
    Wpg = np.multiply(mask, Wpg)

    thresh=0.5
    #thresh=-5
    
    print('thresh='+str(thresh))
    pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)
    Wgp = train_gcpc(pbook, gbook, Npos)

    k=0
    for Npatts in tqdm(Npatts_lst):
        #print("k=",k)

        # Learning patterns 
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        sum_senscup = 0 
        sum_sensl1 = 0
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            srep = np.zeros((nruns, *strue.shape))
            srep[:,:,:] = strue  #(nruns,Ns,1)
            sinit = corrupt_p(Ns, pflip, srep, nruns)   #srep   # make corrupted sc pattern
            errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs_cts(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh) 
            # errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                # Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh) 


            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
            sum_senscup += errsenscup
            sum_sensl1 += errsensl1      
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        err_senscup[k,:] = sum_senscup/Npatts
        err_sensl1[k,:] = sum_sensl1/Npatts
        k += 1   
    return err_pc, err_gc, err_sens, err_senscup, err_sensl1  

def senstrans_gs_linear_p(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    err_sensl1 = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    c = 0.60     # connection probability
    prune = int((1-c)*Np*Ng)
    mask = np.ones((Np, Ng))
    mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
    Wpg = np.multiply(mask, Wpg)

    thresh=0.5
    #pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)
    pbook = np.einsum('ijk,kl->ijl', Wpg, gbook)  # (nruns, Np, Npos)
    Wgp = train_gcpc(pbook, gbook, Npos)

    k=0
    for Npatts in tqdm(Npatts_lst):
        #print("k=",k)

        # Learning patterns 
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        sum_senscup = 0 
        sum_sensl1 = 0
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            srep = np.zeros((nruns, *strue.shape))
            srep[:,:,:] = strue  #(nruns,Ns,1)
            sinit = corrupt_p(Ns, pflip, srep, nruns)   #srep   # make corrupted sc pattern
            # errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                # Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh) 
            errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs_linear_p(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh)                                                


            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
            sum_senscup += errsenscup
            sum_sensl1 += errsensl1      
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        err_senscup[k,:] = sum_senscup/Npatts
        err_sensl1[k,:] = sum_sensl1/Npatts
        k += 1   
    return err_pc, err_gc, err_sens, err_senscup, err_sensl1  


def senstrans_gs_random_sparse_p(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    print("random sparse p")
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    err_sensl1 = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    c = 0.60     # connection probability
    prune = int((1-c)*Np*Ng)
    mask = np.ones((Np, Ng))
    mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
    Wpg = np.multiply(mask, Wpg)

    thresh=0.5
    pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)
    for i in range(nruns):
        pbook_i = pbook[i].flatten()
        np.random.shuffle(pbook_i)
        pbook[i] = pbook_i.reshape((Np,Npos))
    #Wgp = train_gcpc(pbook, gbook, Npos)

    k=0
    for Npatts in tqdm(Npatts_lst):
        #print("k=",k)
        #Wgp = train_gcpc(pbook, gbook, Npatts)
        
        Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)
        Wpg = pseudotrain_Wpg(gbook, pbook, Npatts)
        
        # Learning patterns 
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        sum_senscup = 0 
        sum_sensl1 = 0
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            srep = np.zeros((nruns, *strue.shape))
            srep[:,:,:] = strue  #(nruns,Ns,1)
            sinit = corrupt_p(Ns, pflip, srep, nruns)   #srep   # make corrupted sc pattern
            # errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                # Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh) 
            errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs_random_sparse_p(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh) 

            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
            sum_senscup += errsenscup
            sum_sensl1 += errsensl1      
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        err_senscup[k,:] = sum_senscup/Npatts
        err_sensl1[k,:] = sum_sensl1/Npatts
        k += 1   
    return err_pc, err_gc, err_sens, err_senscup, err_sensl1  

def senstrans_gs_linear_p_spiral(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    err_sensl1 = -1*np.ones((len(Npatts_lst), nruns))

    x,y=make_spiral(np.prod(lambdas))
    locs1=np.prod(lambdas)*y+x
    #permutation = np.random.permutation(Npos)
    gbook = gbook[:,locs1]

    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    c = 0.60     # connection probability
    prune = int((1-c)*Np*Ng)
    mask = np.ones((Np, Ng))
    mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
    Wpg = np.multiply(mask, Wpg)

    thresh=0.5
    #pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)
    pbook = np.einsum('ijk,kl->ijl', Wpg, gbook)  # (nruns, Np, Npos)
    Wgp = train_gcpc(pbook, gbook, Npos)

    k=0
    for Npatts in tqdm(Npatts_lst):
        #print("k=",k)

        # Learning patterns 
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        sum_senscup = 0 
        sum_sensl1 = 0
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            srep = np.zeros((nruns, *strue.shape))
            srep[:,:,:] = strue  #(nruns,Ns,1)
            sinit = corrupt_p(Ns, pflip, srep, nruns)   #srep   # make corrupted sc pattern
            # errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                # Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh) 
            errpc, errgc, errsens, errsenscup, errsensl1 = dynamics_gs_linear_p(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns, thresh)                                                


            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
            sum_senscup += errsenscup
            sum_sensl1 += errsensl1      
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        err_senscup[k,:] = sum_senscup/Npatts
        err_sensl1[k,:] = sum_sensl1/Npatts
        k += 1   
    return err_pc, err_gc, err_sens, err_senscup, err_sensl1  



def senstrans_gg(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        #Wsp = train_sensory(pbook, sbook, Npatts)
        #Wps = np.transpose(Wsp, axes=(0,2,1))
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)
    

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]     # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = dynamics_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens      


def senstrans_gp(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M))                       # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))   # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        #Wsp = train_sensory(pbook, sbook, Npatts)
        #Wps = np.transpose(Wsp, axes=(0,2,1))
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern

            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            errpc, errgc, errsens = dynamics_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, 
                                            lambdas, Wsp, Wps, sparsity, gtrue, strue, Np)   
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens

 
def make_spiral(lambda_prod):
    Npos=int(lambda_prod**2)
    x,y=[(lambda_prod-1)//2],[(lambda_prod-1)//2]
    stepsize=1
    sgn=1
    flag=0
    for i in range(Npos):
        if flag==0:
            for j in range(stepsize):
                x.append(x[-1]+sgn)
                y.append(y[-1])
            for j in range(stepsize):
                x.append(x[-1])
                y.append(y[-1]+sgn)
            sgn=-sgn
            stepsize=stepsize+1
            if stepsize==lambda_prod:
                flag=1
        else:
            for j in range(stepsize-1):
                x.append(x[-1]+sgn)
                y.append(y[-1])
            break
    return np.array(x),np.array(y)