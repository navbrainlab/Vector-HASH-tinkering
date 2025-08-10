import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from src.assoc_utils_np import train_gcpc, pseudotrain_Wgp
from src.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, module_wise_NN_2d
from src.seq_utils import *
from numpy.random import randint
from scipy import stats
plt.style.use('./src/presentation.mplstyle')
plt.rcParams['figure.autolayout'] = False

lesion_size = 80
# test = np.load(f"MTT/Wgp_and_Wsp_both_reinforced_niter=10_patts=20_interleaved/Wgp_and_Wsp_both_reinforced_recons_imagenet_sensory_lesion_size={lesion_size}_niter=10.npy", allow_pickle=True)

# test = np.load(f"MTT/baseline_noreinforcement_interleaved/baseline_noreinforcement_recons_imagenet_sensory_lesion_size={lesion_size}_niter=10.npy", allow_pickle=True)

test = np.load(f"MTT/Wgp_Wsp_both_reinforced_recons_imagenet_sensory_lesion_size={lesion_size}_niter=10_Npatts=1000.npy", allow_pickle=True)
print(test.shape)

lesion = (lesion_size/400)*100

idx = 3
plt.figure(0)
plt.title("In the repeated set")
plt.imshow(test[0,:,idx].reshape(60,60), cmap="gray")

# plt.figure(1)
# sbook = np.load("BW_miniimagenet_3600_60_60_full_rank.npy")  # 3600 x 60 x 60
# sbook_flattened = sbook.reshape(3600, 3600)
# sbook_flattened = sbook_flattened.T
# bw_mean = np.mean(sbook_flattened.flatten())
# sbook_flattened = sbook_flattened - bw_mean
# indices = np.concatenate([np.arange(i, 3600, 600) for i in range(600)])
# sbook_flattened = sbook_flattened[:,indices]
# plt.imshow(sbook_flattened[:,idx].reshape(60,60), cmap="gray")
# plt.savefig(f"MTT/paper_figures_interleaved_Npatts=600/original_image_{idx}.pdf", format="pdf")

# plt.savefig(f"MTT/paper_figures_interleaved_Npatts=600/lesion_{lesion}%_image_{idx}.pdf", format="pdf")
# plt.close()

plt.show()



idx = 30 #30
plt.figure(2)
plt.imshow(test[0,:,idx].reshape(60,60), cmap="gray")
plt.title("Not in the repeated set")
# plt.savefig(f"MTT/paper_figures_interleaved_Npatts=600/lesion_{lesion}%_image_{idx}.pdf", format="pdf")
# plt.close()


# plt.figure(3)
# sbook = np.load("BW_miniimagenet_3600_60_60_full_rank.npy")  # 3600 x 60 x 60
# sbook_flattened = sbook.reshape(3600, 3600)
# sbook_flattened = sbook_flattened.T
# bw_mean = np.mean(sbook_flattened.flatten())
# sbook_flattened = sbook_flattened - bw_mean
# indices = np.concatenate([np.arange(i, 3600, 600) for i in range(600)])
# sbook_flattened = sbook_flattened[:,indices]
# # plt.title("Not In the repeated set")
# plt.imshow(sbook_flattened[:,idx].reshape(60,60), cmap="gray")

#plt.savefig(f"MTT/paper_figures_interleaved_Npatts=600/original_image_{idx}.pdf", format="pdf")

plt.show()

exit()




nruns=2
Np = 400
lambdas = [3,4,5] 
Ng = np.sum(np.square(lambdas))
Npos = np.prod(lambdas)
gbook = gen_gbook_2d(lambdas, Ng, Npos)
gbook.shape     # (Ng, Npos, Npos)

module_sizes = np.square(lambdas)
module_gbooks = [np.eye(i) for i in module_sizes]

Wpg = randn(nruns, Np, Ng)
c = 0.60     # connection probability
prune = int((1-c)*Np*Ng)
mask = np.ones((Np, Ng))
mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
Wpg = np.multiply(mask, Wpg)


thresh = 0.5
pbook = nonlin(np.einsum('ijk,klm->ijlm', Wpg, gbook), thresh=thresh)  # (nruns, Np, Npos, Npos) 

gbook_flattened = gbook.reshape(Ng, Npos*Npos)  
pbook_flattened = pbook.reshape(nruns, Np, Npos*Npos)
Wgp = train_gcpc(pbook_flattened, gbook_flattened, Npos*Npos)
# Wgp = train_gcpc(pbook_flattened, gbook_flattened, 600)


Ns = 3600
x,y = Npos, Npos
Npatts = 1000  #x*y

#sbook = np.sign(randn(Ns, x, y))
sbook = np.load("BW_miniimagenet_3600_60_60_full_rank.npy")  # 3600 x 60 x 60
sbook_flattened = sbook.reshape(Ns, x*y)
sbook_flattened = sbook_flattened.T
bw_mean = np.mean(sbook_flattened.flatten())
sbook_flattened = sbook_flattened - bw_mean
indices = np.concatenate([np.arange(i, 3600, 600) for i in range(600)])
sbook_flattened = sbook_flattened[:,indices]
sbook_flattened = sbook_flattened[:,:Npatts]

sbookinv = np.linalg.pinv(sbook_flattened)
sbookinv.shape
Wps = pbook_flattened[:,:,:Npatts]@sbookinv
Wps = np.squeeze(Wps)

pbookinv = np.linalg.pinv(pbook_flattened[:,:,:Npatts])
pbookinv.shape
Wsp = sbook_flattened[:,:Npatts]@pbookinv
Wsp = np.squeeze(Wsp)

########################################################################################################
## MTT

print("Starting MTT")

niter = 10 #2,5,10

# Here ak is the source (s) and yk is destination (p) in the reconstruction Wps (s->p)
def pseudotrain_2d_iterative_step(W, theta, ak, yk):
    ak = ak[:, None]
    yk = yk[:, None]
    bk = ((theta@ak) /(1+ak.T@theta@ak)).T    
    theta = theta - theta@ak@bk
    W = W + (yk - W@ak)@bk
    return W, theta


def reinforce_wsp(patt_reinf):
    Wsp_trace = np.copy(Wsp)
    epsilon = 0.01
    theta = (1/epsilon**2)*np.eye(Np, Np)

    for i in range(niter):
        for j in range(patt_reinf):
            s = sbook_flattened[:,j]
            for run in range(nruns):
                p = pbook_flattened[run,:,j]
                Wsp_trace[run], theta = pseudotrain_2d_iterative_step(Wsp_trace[run], theta, p, s)       
    return Wsp_trace


sbook_recons = np.zeros((Ns, x, y))
lesion_size_list = np.asarray(np.arange(0,0.9,0.1)*Np, dtype = 'int')
patt_reinf_list = [20] #, 100, 200]                      

place_error_lesion_nr = np.zeros((len(lesion_size_list), len(patt_reinf_list)))  #Non-reinforced patterns
place_error_lesion_r = np.zeros((len(lesion_size_list), len(patt_reinf_list)))   #Reinforced patterns
pel_nr_std = np.zeros((len(lesion_size_list), len(patt_reinf_list), 2))  #Non-reinforced patterns
pel_r_std = np.zeros((len(lesion_size_list), len(patt_reinf_list), 2))   #Reinforced patterns


grid_error_nr = np.zeros((len(lesion_size_list), len(patt_reinf_list)))  #Non-reinforced patterns
grid_error_r = np.zeros((len(lesion_size_list), len(patt_reinf_list)))   #Reinforced patterns
ge_nr_std = np.zeros((len(lesion_size_list), len(patt_reinf_list), 2))  #Non-reinforced patterns
ge_r_std = np.zeros((len(lesion_size_list), len(patt_reinf_list), 2))   #Reinforced patterns


sens_error_nr = np.zeros((len(lesion_size_list), len(patt_reinf_list)))  #Non-reinforced patterns
sens_error_r = np.zeros((len(lesion_size_list), len(patt_reinf_list)))   #Reinforced patterns
se_nr_std = np.zeros((len(lesion_size_list), len(patt_reinf_list), 2))  #Non-reinforced patterns
se_r_std = np.zeros((len(lesion_size_list), len(patt_reinf_list), 2))   #Reinforced patterns

print("i: ", len(lesion_size_list))
print("j: ", len(patt_reinf_list))

i = 0
for lesion_size in lesion_size_list:
    lesion_cells = np.random.choice(Np, size=lesion_size, replace=False)

    j = 0
    for patt_reinf in patt_reinf_list:
        print(i,j)

        Wgp_trace = np.copy(Wgp)
        traces = (1/Np)*np.einsum('ij, klj -> kil', gbook_flattened[:,:patt_reinf], pbook_flattened[:,:,:patt_reinf])
        for _ in range(niter):
            Wgp_trace += traces
          
        
        #Wsp_trace = np.copy(Wsp)
        Wsp_trace = reinforce_wsp(patt_reinf)
        
        p_bookflat = nonlin(Wps@sbook_flattened, thresh=0)
        p_bookflat[:,lesion_cells,:] = 0
        g_in = Wgp_trace@p_bookflat
        g_bookflat = np.zeros((nruns, Ng, Npatts))
        for k in range(Npatts):
            g_bookflat[:,:,k] = module_wise_NN_2d(g_in[:,:,k,None], module_gbooks, module_sizes)[0,:,0]


        errorg = np.linalg.norm(g_bookflat-gbook_flattened[:,:Npatts], axis=1)/Ng
        errorg = np.mean(errorg, axis=0)

        p_bookflatclean = nonlin(Wpg@g_bookflat, thresh)

        p_bookflatclean[:,lesion_cells,:] = 0
        pbook_flattened1 = np.copy(pbook_flattened)
        pbook_flattened1[:,lesion_cells,:] = 0
        errorpl = np.linalg.norm(p_bookflatclean-pbook_flattened1[:,:,:Npatts], axis=1)/(Np-lesion_size)
        errorpl = np.mean(errorpl, axis=0)

        # s_bookflatrecall = np.sign(Wsp_trace@p_bookflatclean)
        s_bookflatrecall = Wsp_trace@p_bookflatclean
        np.save(f"MTT/Wgp_Wsp_both_reinforced_recons_imagenet_sensory_lesion_size={lesion_size}_niter={niter}_Npatts={Npatts}.npy", s_bookflatrecall)
        errors = np.linalg.norm(s_bookflatrecall-sbook_flattened, axis=1)/Ns
        errors = np.mean(errors, axis=0)
        
        place_error_lesion_nr[i,j] = np.mean(errorpl[patt_reinf:])   # mean error across patterns
        place_error_lesion_r[i,j] = np.mean(errorpl[:patt_reinf])  
        pel_nr_std[i,j,0] = np.std(errorpl[patt_reinf:])    # std-dev across patterns
        pel_r_std[i,j,0] = np.std(errorpl[:patt_reinf])
        pel_nr_std[i,j,1] = stats.sem(errorpl[patt_reinf:])
        pel_r_std[i,j,1] = stats.sem(errorpl[:patt_reinf])

        
        grid_error_nr[i,j] = np.mean(errorg[patt_reinf:])
        grid_error_r[i,j] = np.mean(errorg[:patt_reinf])
        ge_nr_std[i,j,0] = np.std(errorg[patt_reinf:])
        ge_r_std[i,j,0] = np.std(errorg[:patt_reinf])
        ge_nr_std[i,j,1] = stats.sem(errorg[patt_reinf:])
        ge_r_std[i,j,1] = stats.sem(errorg[:patt_reinf])
        
        sens_error_nr[i,j] = np.mean(errors[patt_reinf:])
        sens_error_r[i,j] = np.mean(errors[:patt_reinf])
        se_nr_std[i,j,0] = np.std(errors[patt_reinf:])
        se_r_std[i,j,0] = np.std(errors[:patt_reinf])
        se_nr_std[i,j,1] = stats.sem(errors[patt_reinf:])
        se_r_std[i,j,1] = stats.sem(errors[:patt_reinf])
             
                      
        j=j+1
    i=i+1   
########################################################################################################
## Data

print("Saving Data")
data = {
		"place_error_lesion_nr": place_error_lesion_nr,
		"place_error_lesion_r": place_error_lesion_r,
		"pel_nr_std": pel_nr_std,
		"pel_r_std": pel_r_std,
		"grid_error_nr": grid_error_nr,
		"grid_error_r": grid_error_r,
		"ge_nr_std" : ge_nr_std,
		"ge_r_std" : ge_r_std,
		"sens_error_nr": sens_error_nr,
		"sens_error_r": sens_error_r,
		"se_nr_std": se_nr_std,
		"se_r_std": se_r_std
}

filename = f"Wgp_Wsp_both_reinforced_niter={niter}_Npatts={Npatts}"
np.save(f"MTT/{filename}.npy", data)

########################################################################################################
## plots

fraction_lesion_size_list = np.divide(lesion_size_list,Np)*100
#Npatts = Npos*Npos

print("Saving Figures")

fig = plt.figure()
plt.errorbar(fraction_lesion_size_list, place_error_lesion_nr[:,0], yerr=pel_nr_std[:,0,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[0])/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, place_error_lesion_nr[:,1], yerr=pel_nr_std[:,1,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[1])/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, place_error_lesion_nr[:,2], yerr=pel_nr_std[:,2,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[2])/Npatts,3)}")

plt.errorbar(fraction_lesion_size_list, place_error_lesion_r[:,0], yerr=pel_r_std[:,0,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[0]/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, place_error_lesion_r[:,1], yerr=pel_r_std[:,1,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[1]/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, place_error_lesion_r[:,2], yerr=pel_r_std[:,2,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[2]/Npatts,3)}")

plt.ylabel("Place code L2 error \n after lesioning", fontsize=18)
plt.xlabel("% of place cells lesioned", fontsize=18)
plt.legend(bbox_to_anchor=[1, 1])


fig.savefig(f"MTT/{filename}_place_std.svg")
fig.savefig(f"MTT/{filename}_place_std.png")

####################################################

fig = plt.figure()
plt.errorbar(fraction_lesion_size_list, grid_error_nr[:,0], yerr=ge_nr_std[:,0,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[0])/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, grid_error_nr[:,1], yerr=ge_nr_std[:,1,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[1])/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, grid_error_nr[:,2], yerr=ge_nr_std[:,2,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[2])/Npatts,3)}")

plt.errorbar(fraction_lesion_size_list, grid_error_r[:,0], yerr=ge_r_std[:,0,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[0]/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, grid_error_r[:,1], yerr=ge_r_std[:,1,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[1]/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, grid_error_r[:,2], yerr=ge_r_std[:,2,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[2]/Npatts,3)}")

plt.ylabel("Grid code L2 error \n after lesioning", fontsize=18)
plt.xlabel("% of place cells lesioned", fontsize=18)
plt.legend(bbox_to_anchor=[1, 1])


fig.savefig(f"MTT/{filename}_grid_std.svg")
fig.savefig(f"MTT/{filename}_grid_std.png")

####################################################

fig = plt.figure()
plt.errorbar(fraction_lesion_size_list, sens_error_nr[:,0], yerr=se_nr_std[:,0,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[0])/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, sens_error_nr[:,1], yerr=se_nr_std[:,1,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[1])/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, sens_error_nr[:,2], yerr=se_nr_std[:,2,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[2])/Npatts,3)}")

plt.errorbar(fraction_lesion_size_list, sens_error_r[:,0], yerr=se_r_std[:,0,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[0]/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, sens_error_r[:,1], yerr=se_r_std[:,1,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[1]/Npatts,3)}")
# plt.errorbar(fraction_lesion_size_list, sens_error_r[:,2], yerr=se_r_std[:,2,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[2]/Npatts,3)}")

plt.ylabel("Sensory code L2 error \n after lesioning", fontsize=18)
plt.xlabel("% of place cells lesioned", fontsize=18)
plt.legend(bbox_to_anchor=[1, 1])


fig.savefig(f"MTT/{filename}_sensory_std.svg")
fig.savefig(f"MTT/{filename}_sensory_std.png")
