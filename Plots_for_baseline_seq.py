def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None,label='',lw=1.):
  ax = ax if ax is not None else plt.gca()
  if color is None:
    color = next(ax._get_lines.prop_cycler)['color']
  if np.isscalar(yerr) or len(yerr) == len(y):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr
  ax.plot(x, y, color=color,label=label,lw=lw)
  ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill,lw=0.)


nruns=12

results_dir="continuum_results_seq"
#-------------------------
filename = "stdhopfield__mutualinfo_N=708_noise=0.0_nruns="+str(nruns)+".pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_stdhop = data["MI"]
m_stdhop = data["m"]
Npatts_lst_stdhop = data["Npatts_list"]
alpha_stdhop = Npatts_lst_stdhop/708
i_stdhop = alpha_stdhop*MI_stdhop
#-------------------------
filename = "pinvhopfield__mutualinfo_N=708_noise=0.05_nruns="+str(nruns)+".pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_pinvhop = data["MI"]
m_pinvhop = data["m"]
Npatts_lst_pinvhop = data["Npatts_list"]
alpha_pinvhop = Npatts_lst_pinvhop/708
i_pinvhop = alpha_pinvhop*MI_pinvhop
#--------------------------
filename = "sparseconnhopfield__mutualinfo_N=7071_noise=0.0_gamma=0.01_nruns="+str(nruns)+".pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_spconhop = data["MI"]
m_spconhop = data["m"]
Npatts_lst_spco = data["Npatts_list"]
gamma = data["gamma"]
N_size = data["N"]
alpha_spco = Npatts_lst_spco/(gamma*N_size)
i_spco = alpha_spco*MI_spconhop
#--------------------------
filename = "./autoencoder_miniimagenet_seq_overlaps/seq_w_sgn_overlap_list_Np_275_Ns_900_Ng_38_nruns_6_good_runs.npy"
#filename="final_noisy_overlap_list_Np_300_Ns_816_Ng_18.npy"
m_auto = np.load(f"{filename}").T
a = (1+m_auto)/2
b = (1-m_auto)/2
S = - a * np.log2(a) - b * np.log2(b)
MI_auto = 1 - S 
Npatts_lst_auto = np.load('./autoencoder_miniimagenet_seq_overlaps/Npatts_list_seq.npy')
Ns = 900#816
Np = 275#300
Ng_ = 38#18
alpha_auto = ( (Npatts_lst_auto*Ns) ) / ( (2*Ns*Np)+(2*Np*Ng_) )
#alpha_auto = Npatts_lst_auto/816
i_auto = alpha_auto*MI_auto
#--------------------------
filename = "sparsehopfield__mutualinfo_N=708_noise=0.0_p=0.2_nruns="+str(nruns)+".pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_spashop = data["MI"]
m_spashop = data["m"]
Npatts_lst_spashop = data["Npatts_list"]
q = data["q"].mean()
alpha_spashop = (-q*np.log2(q) - (1-q)*np.log2(1-q))*Npatts_lst_spashop/708
i_spashop = Npatts_lst_spashop*MI_spashop/708
#-------------------------
filename = "boundedhopfield__mutualinfo_N=708_noise=0.0_bound=0.3_nruns="+str(nruns)+".pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_bndhop = data["MI"]
m_bndhop = data["m"]
Npatts_lst_bndhop = data["Npatts_list"]
alpha_bndhop = Npatts_lst_bndhop/708
i_bndhop = alpha_bndhop*MI_bndhop
#------------------------------
filename = "./MI_235_Np275.npy"
MI_VH = np.load(f"{filename}")[None,]
Npatts_lst_VH = np.arange(1,901,10)
Ns = 900#816
Np = 275#300
Ng_ = 38#18
alpha_VH = ( (Npatts_lst_VH*Ns) ) / ( (2*Ns*Np)+(2*Np*Ng_) )
#alpha_auto = Npatts_lst_auto/816
i_VH = alpha_VH*MI_VH

#-----------------------------------

errorfill(alpha_stdhop,i_stdhop.mean(axis=0),i_stdhop.std(axis=0), label='std hop',lw=3.);
errorfill(alpha_pinvhop,i_pinvhop.mean(axis=0),i_pinvhop.std(axis=0), label='pinv',lw=3.);
errorfill(alpha_spco,i_spco.mean(axis=0),i_spco.std(axis=0), label='sp co',lw=3.);
errorfill(alpha_auto,i_auto.mean(axis=0),i_auto.std(axis=0), label='auto',lw=3.);
errorfill(alpha_spashop,i_spashop.mean(axis=0),i_spashop.std(axis=0), label='spa hop',lw=3.);
errorfill(alpha_bndhop,i_bndhop.mean(axis=0),i_bndhop.std(axis=0), label='bnd hop',lw=3.);
errorfill(alpha_VH,i_VH.mean(axis=0),i_VH.std(axis=0), label='VH',lw=3.);
plt.legend()
plt.xlabel('input info/synapse');
plt.ylabel('MI/synapse')
plt.ylim(0,0.7)
plt.xlim(xmin=0)
# plt.savefig('seq_info_rate_all_models.pdf')