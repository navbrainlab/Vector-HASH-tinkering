import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./src/presentation.mplstyle')
plt.rcParams['figure.autolayout'] = False


Np = 400
lambdas = [3,4,5] 
Npos = np.prod(lambdas)

########################################################################################################
## MTT

lesion_size_list = np.asarray(np.arange(0,0.9,0.1)*Np, dtype = 'int')
patt_reinf_list = [20, 100, 200]                      

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


# Note that all the arrays for std-deviation are 2 dimensional with std-dev in first and sem in second dimension


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

filename = f"Wgp_only_reinforced_niter={niter}"
#np.save(f"MTT/{filename}.npy", data)

########################################################################################################
## plots

fraction_lesion_size_list = np.divide(lesion_size_list,Np)*100
Npatts = Npos*Npos

print("Saving Figures")

fig = plt.figure()
plt.errorbar(fraction_lesion_size_list, place_error_lesion_nr[:,0], yerr=pel_nr_std[:,0,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[0])/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, place_error_lesion_nr[:,1], yerr=pel_nr_std[:,1,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[1])/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, place_error_lesion_nr[:,2], yerr=pel_nr_std[:,2,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[2])/Npatts,3)}")

plt.errorbar(fraction_lesion_size_list, place_error_lesion_r[:,0], yerr=pel_r_std[:,0,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[0]/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, place_error_lesion_r[:,1], yerr=pel_r_std[:,1,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[1]/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, place_error_lesion_r[:,2], yerr=pel_r_std[:,2,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[2]/Npatts,3)}")

plt.ylabel("Place code L2 error \n after lesioning", fontsize=18)
plt.xlabel("% of place cells lesioned", fontsize=18)
plt.legend(bbox_to_anchor=[1, 1])


fig.savefig(f"MTT/{filename}_place_std.svg")
fig.savefig(f"MTT/{filename}_place_std.png")

####################################################

fig = plt.figure()
plt.errorbar(fraction_lesion_size_list, grid_error_nr[:,0], yerr=ge_nr_std[:,0,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[0])/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, grid_error_nr[:,1], yerr=ge_nr_std[:,1,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[1])/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, grid_error_nr[:,2], yerr=ge_nr_std[:,2,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[2])/Npatts,3)}")

plt.errorbar(fraction_lesion_size_list, grid_error_r[:,0], yerr=ge_r_std[:,0,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[0]/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, grid_error_r[:,1], yerr=ge_r_std[:,1,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[1]/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, grid_error_r[:,2], yerr=ge_r_std[:,2,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[2]/Npatts,3)}")

plt.ylabel("Grid code L2 error \n after lesioning", fontsize=18)
plt.xlabel("% of place cells lesioned", fontsize=18)
plt.legend(bbox_to_anchor=[1, 1])


fig.savefig(f"MTT/{filename}_grid_std.svg")
fig.savefig(f"MTT/{filename}_grid_std.png")

####################################################

fig = plt.figure()
plt.errorbar(fraction_lesion_size_list, sens_error_nr[:,0], yerr=se_nr_std[:,0,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[0])/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, sens_error_nr[:,1], yerr=se_nr_std[:,1,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[1])/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, sens_error_nr[:,2], yerr=se_nr_std[:,2,0], fmt='o--', label=f"# patt nr = {round((Npatts-patt_reinf_list[2])/Npatts,3)}")

plt.errorbar(fraction_lesion_size_list, sens_error_r[:,0], yerr=se_r_std[:,0,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[0]/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, sens_error_r[:,1], yerr=se_r_std[:,1,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[1]/Npatts,3)}")
plt.errorbar(fraction_lesion_size_list, sens_error_r[:,2], yerr=se_r_std[:,2,0], fmt='o--', label=f"# patt r = {round(patt_reinf_list[2]/Npatts,3)}")

plt.ylabel("Sensory code L2 error \n after lesioning", fontsize=18)
plt.xlabel("% of place cells lesioned", fontsize=18)
plt.legend(bbox_to_anchor=[1, 1])


fig.savefig(f"MTT/{filename}_sensory_std.svg")
fig.savefig(f"MTT/{filename}_sensory_std.png")
