# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: lciw
#     language: python
#     name: lciw
# ---

# %% [markdown]
# # Kinetic energy analysis

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import jcpy.signal as jsig
from matplotlib.dates import DateFormatter
from munch import munchify
import utils
import os
from scipy import stats
from scipy.signal import spectrogram
import xskillscore as xs

def interval_to_mid(intervals):
    return np.array([v.mid for v in intervals])


# %% [markdown]
# Load datasets.

# %%
AS = xr.open_dataset("../data/ABLE_sentinel_mooring_2018.nc")
AS.coords["depth_adcp"] = (AS.distance.dims, (AS.depth_ADCP.mean().values - AS.distance).data, dict(long_name="depth"))
DD = xr.open_dataset("../data/downstream_deep_mooring_2018.nc")

SMB = munchify(utils.loadmat(os.path.expanduser("../data/runoff_extended_2018.mat")))
SMB = xr.Dataset(dict(low_scen=(['time'], SMB.low_scen), middle_scen=(['time'], SMB.middle_scen), high_scen=(['time'], SMB.high_scen)), dict(time=(['time'], utils.datenum_to_datetime(SMB.time))))


# %% [markdown]
# Define some useful functions for estimating cutoff frequency and time for moving averages. 

# %%
def Fc(ts, N):
    """
    Parameters 
    ----------
        ts : float
            Sampling period [s]
        N : float
            Points in window.
            
            
    Notes
    -----
    -3dB frequency response of moving average: https://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        
    """
    return 0.442946470689452340308369 / np.sqrt(N**2 - 1) / ts

def Tcmin(ts, N):
    """
    Parameters 
    ----------
        ts : float
            Sampling period [s]
        N : float
            Points in window.
        
    """
    return 1 / Fc(ts, N) / 60


def NFc(Tc, ts):
    """
    Parameters 
    ----------
        ts : float
            Sampling period in units of time e.g. seconds.
        Tc : float
            Cut off period. Same units as ts.
        
    """
    a = 0.442946470689452340308369
    return round((1 + (a**2 * Tc**2)/ts**2)**0.5)


# %% [markdown]
# # MN - near mooring
#
# Define parameters for moving average filtering, depth cut offs, minimum good data threshold and pitch/roll variance thresholds for iceberg detection. 

# %%
dt = (AS.time[1] - AS.time[0]).data.astype("timedelta64[s]").astype(float)
frac_min = 0.9
nhigh = NFc(60*5, int(dt))
nlow = NFc(60*30, int(dt))
nminhigh = int(frac_min*nhigh)
nminlow = int(frac_min*nlow)
npitch = 7
max_pitch_var = 0.05
dmin = 20
dmax = 120
nhours = 1 # Time binning
min_good = 0.9 # Depth fraction allowed

print(f"High cut off = {Tcmin(10, nhigh):1.2f} min")
print(f"Low cut off = {Tcmin(10, nlow):1.2f} min")
print(f"High cut off = {1440/Tcmin(10, nhigh):1.2f} cpd")
print(f"Low cut off = {1440/Tcmin(10, nlow):1.2f} cpd")
print(f"n High = {nhigh}")
print(f"n Low = {nlow}")

pitch_var = AS.pitch.rolling(time=npitch, center=True).var()

# Remove iceberg events
bad = pitch_var > max_pitch_var
contig = jsig.contiguous_regions(bad)
idx0 = contig[:, 0]
idx1 = contig[:, 1]
# go 2 min earlier
idx0 -= int(60*2/dt)
# go 20 min later
idx1 = idx0 + int(60*20/dt)

# Make mask
for i in range(idx0.size):
    bad[idx0[i]:idx1[i]] = True
    
good = ~bad


# %%
AS_ = AS.isel(distance=(AS.depth > dmin) & (AS.depth < dmax))

ul = AS_.u.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
uh = AS_.u.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
ub = uh - ul
wl = AS_.vv.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
wh = AS_.vv.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
wb = wh - wl
vl = AS_.v.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
vh = AS_.v.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
vb = vh - vl

KEl = ul**2 + vl**2
KEb = ub**2 + vb**2 + wb**2


# %%
wb.to_netcdf("../data/MN_band_pass_w.nc")

# %%
# time1 = np.datetime64("2018-09-06T15:00")
# tslice = slice(time1, time1 + np.timedelta64(3600*3, 's'))

time1 = np.datetime64("2018-09-06T12:00")
tslice = slice(time1, time1 + np.timedelta64(3600*12, 's'))

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 5))

ub.sel(time=tslice).plot(ax=axs[0], x="time", y="depth", yincrease=False)
vb.sel(time=tslice).plot(ax=axs[1], x="time", y="depth", yincrease=False)
wb.sel(time=tslice).plot(ax=axs[2], x="time", y="depth", yincrease=False)

# %% [markdown]
# Depth average kinetic energy

# %%
nall = KEl.isnull() | KEb.isnull()

nmax = nall.distance.size
ngood = nmax - nall.sum("distance")
frac_good = ngood/nmax
good = (frac_good > min_good)
# dz = (AS.distance[1] - AS.distance[0]).data

KEl_ave = (KEl.sum("distance")/ngood).where(good)
KEb_ave = (KEb.sum("distance")/ngood).where(good)

KEl_re = KEl_ave.resample(time=f"{nhours}H", skipna=True).mean()
KEb_re = KEb_ave.resample(time=f"{nhours}H", skipna=True).mean()

KEl_re.plot(marker=".", linestyle="")
KEb_re.plot(marker=".", linestyle="")

# %% [markdown]
# What is velocity like during those odd spikes?

# %%
# time1 = np.datetime64("2018-09-06T15:00")
# tslice = slice(time1, time1 + np.timedelta64(3600*3, 's'))

time1 = np.datetime64("2018-09-06T12:00")
tslice = slice(time1, time1 + np.timedelta64(3600*12, 's'))

fig, axs = plt.subplots(7, 1, sharex=True, figsize=(10, 16))

KEl_re.sel(time=tslice).plot(marker=".", linestyle="", ax=axs[0])
KEb_re.sel(time=tslice).plot(marker=".", linestyle="", ax=axs[0])
ul.sel(time=tslice).plot(ax=axs[1], x="time", y="depth", yincrease=False, add_colorbar=False)
vl.sel(time=tslice).plot(ax=axs[2], x="time", y="depth", yincrease=False, add_colorbar=False)
ub.sel(time=tslice).plot(ax=axs[3], x="time", y="depth", yincrease=False, add_colorbar=False)
vb.sel(time=tslice).plot(ax=axs[4], x="time", y="depth", yincrease=False, add_colorbar=False)
wb.sel(time=tslice).plot(ax=axs[5], x="time", y="depth", yincrease=False, add_colorbar=False)
AS.a2.sel(time=tslice).plot(ax=axs[6], x="time", y="depth", yincrease=False, add_colorbar=False)

# %% [markdown]
# A qualitative look at discharge and energy. 

# %%
bins = np.arange(0, 0.02, 0.001)

fig, axs = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw=dict(width_ratios=[3, 1]))

KEl_re.plot(ax=axs[0])
KEb_re.plot(ax=axs[0])

axt = axs[0].twinx()
SMB.middle_scen.sel(time=slice(KEl_re.time[0], KEl_re.time[-1])).plot(ax=axt, color="k")

axs[1].hist(KEl_re, bins, density=True, orientation="horizontal", histtype="step")
axs[1].hist(KEb_re, bins, density=True, orientation="horizontal", histtype="step")

# %% [markdown]
# Figure for paper.

# %%
bins = np.arange(0, 0.02, 0.001)

fig, ax = plt.subplots(1, 1, figsize=(3, 2))

ax.plot(KEl_re.time, KEl_re, "C0.", ms=2.5, lw=0.5, label="low freq")
ax.plot(KEb_re.time, KEb_re, "C1.", ms=2.5, lw=0.5, label="wave band")

ax.legend(ncol=2, loc=(0, 1.03), fontsize=7)
ax.set_ylabel("Depth ave. KE [J kg$^{-1}$]")

date_form = DateFormatter("%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel("Sept 2018")

fig.savefig("../figures/short_term_KE.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
fig.savefig("../figures/short_term_KE.png", dpi=300, bbox_inches="tight", pad_inches=0.01)


# with emperical distribution function below

# bins = np.arange(0, 0.02, 0.001)

# fig, axs = plt.subplots(1, 2, figsize=(3, 2), gridspec_kw=dict(width_ratios=[5, 1], wspace=0))
# axs[1].set_yticks([])
# axs[1].set_xticks([])

# axs[0].plot(KEl_re.time, KEl_re, "C9.", lw=2, label="low freq")
# axs[0].plot(KEb_re.time, KEb_re, "C0.", lw=2, label="wave band")

# axs[0].legend(ncol=2, loc=(0, 1.03), fontsize=7)
# axs[0].set_ylabel("Depth ave. KE\n[J kg$^{-1}$]")

# date_form = DateFormatter("%d")
# axs[0].xaxis.set_major_formatter(date_form)
# axs[0].set_xlabel("Sept 2018")

# _ = axs[1].hist(KEl_re, bins, density=True, orientation="horizontal", histtype="step",  color="C9", lw=2)
# _ = axs[1].hist(KEb_re, bins, density=True, orientation="horizontal", histtype="step",  color="C0", lw=2)

# fig.savefig("../figures/short_term_KE.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
# fig.savefig("../figures/short_term_KE.png", dpi=300, bbox_inches="tight", pad_inches=0.01)

# %% [markdown]
# ## Error analysis
#
# Error on the energy fraction.

# %%
bootnum = 500
np.random.seed(482990)

good = np.isfinite(KEl_re) & np.isfinite(KEb_re) #& (SMBb.middle_scen > 10)

KElr = KEl_re[good].data
KEbr = KEb_re[good].data

ndat = KEbr.size
idxs = np.arange(ndat)

KEl_mean = np.zeros((bootnum))
KEb_mean = np.zeros((bootnum))

for i in range(bootnum):
    idxs_ = np.random.choice(idxs, ndat)
    KEl_mean[i] = KElr[idxs_].mean()
    KEb_mean[i] = KEbr[idxs_].mean()
    
KE_frac = KEb_mean/(KEb_mean + KEl_mean)

fig, ax = plt.subplots()
_ = ax.hist(KEl_mean)

fig, ax = plt.subplots()
_ = ax.hist(KEb_mean)

fig, ax = plt.subplots()
_ = ax.hist(KE_frac)

print(f"KE low mean {KEl_mean.mean():.2e} J kg-1")
print(f"KE wave mean {KEb_mean.mean():.2e} J kg-1")
print(f"KE std mean {KEb_mean.std():.2e} J kg-1")
print(f"KE frac mean {KE_frac.mean():.2%}")
print(f"KE frac std {KE_frac.std():.2%}")
print(f"KE frac 90% error {2*KE_frac.std():.2%}")

# %% [markdown]
# Error on energy fraction for times when the plume is not over the mooring.

# %%
bootnum = 500
np.random.seed(482990)

good = np.isfinite(KEl_re) & np.isfinite(KEb_re) & (KEl_re < 0.005)

KElr = KEl_re[good].data
KEbr = KEb_re[good].data

ndat = KEbr.size
idxs = np.arange(ndat)

KEl_mean = np.zeros((bootnum))
KEb_mean = np.zeros((bootnum))

for i in range(bootnum):
    idxs_ = np.random.choice(idxs, ndat)
    KEl_mean[i] = KElr[idxs_].mean()
    KEb_mean[i] = KEbr[idxs_].mean()
    
KE_frac = KEb_mean/(KEb_mean + KEl_mean)

fig, ax = plt.subplots()
_ = ax.hist(KEl_mean)

fig, ax = plt.subplots()
_ = ax.hist(KEb_mean)

fig, ax = plt.subplots()
_ = ax.hist(KE_frac)

print(f"KE low mean {KEl_mean.mean():.2e}")
print(f"KE wave mean {KEb_mean.mean():.2e}")
print(f"KE wave std {KEb_mean.std():.2e}")
print(f"KE frac mean {KE_frac.mean():.2%}")
print(f"KE frac std {KE_frac.std():.2%}")

# %% [markdown]
# ## Mean speed of waves and impact on melt rates

# %%
rand_seed = 9885333
nsamples = 5000
nit = 100

spdl = np.sqrt(vl**2)
spdtot = np.sqrt((vb + vl)**2 + wb**2)
spdlm = spdl.mean("time")
spdtotm = spdtot.mean("time")

np.random.seed(rand_seed)
spdl_rs = xs.resample_iterations_idx(spdl, nit, dim="time", dim_max=nsamples).mean("time")
np.random.seed(rand_seed)
spdtot_rs = xs.resample_iterations_idx(spdtot, nit, dim="time", dim_max=nsamples).mean("time")

pctl = np.percentile(spdl_rs, [5, 95], axis=1)
pcttot = np.percentile(spdtot_rs, [5, 95], axis=1)
pct_ratio = np.percentile((100*(spdtot_rs/spdl_rs - 1)), [5, 95], axis=1) 

# %%
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(2, 2), gridspec_kw=dict(width_ratios=[2, 1.25]))
axs[0].plot(spdlm*100, spdl.depth, label="low freq")
axs[0].fill_betweenx(spdl.depth, pctl[0, :]*100, pctl[1, :]*100, color="C0", alpha=0.2)
axs[0].plot(spdtotm*100, spdl.depth, "k", label="low freq + wave")
axs[0].fill_betweenx(spdl.depth, pcttot[0, :]*100, pcttot[1, :]*100, color="k", alpha=0.2)

axs[0].invert_yaxis()
axs[1].plot((spdtotm/spdlm - 1)*100, spdtot.depth, "k")
axs[1].fill_betweenx(spdl.depth, pct_ratio[0, :], pct_ratio[1, :], color="k", alpha=0.2)

axs[0].set_xlabel("Speed\n[cm s$^{-1}$]")#, fontsize=fontsize)
axs[1].set_xlabel("Increase in\nmelt rate [%]")#, fontsize=fontsize)
axs[0].set_ylabel("Depth [m]")#, fontsize=fontsize)

axs[0].legend(ncol=2, loc=(-0.3, 1.03), fontsize=7)

fig.savefig("../figures/melting2.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
fig.savefig("../figures/melting2.png", dpi=300, bbox_inches="tight", pad_inches=0.01)

# %% [markdown]
# ## Correlations with discharge

# %%
min_good_frac = 0.9
nhours = 24
time_resample = f"{nhours}H"

dmin = 30
dmax = 100

KEl = ul**2 + vl**2
KEb = ub**2 + vb**2 + wb**2

KEl = KEl.isel(distance=(KEl.depth > dmin) & (KEl.depth < dmax))
KEb = KEb.isel(distance=(KEb.depth > dmin) & (KEb.depth < dmax))

nall = KEl.isnull() | KEb.isnull()
nmax = nall.distance.size
ngood = nmax - nall.sum("distance")
frac_good = ngood/nmax

good = (frac_good > min_good_frac)

KEl_ave = (KEl.sum("distance")/ngood).where(good)
KEb_ave = (KEb.sum("distance")/ngood).where(good)

KEl_re = KEl_ave.resample(time=time_resample, skipna=True).mean()
KEb_re = KEb_ave.resample(time=time_resample, skipna=True).mean()

time_bins = np.hstack((KEb_re.time, KEb_re.time[-1] + np.timedelta64(nhours, "h"))) - np.timedelta64(30*nhours, "m")

SMBb = SMB.groupby_bins("time", time_bins).mean()

SMBb["time_bins"] = interval_to_mid(SMBb.time_bins.values).astype("datetime64[s]")
SMBb = SMBb.rename({"time_bins": "time"})

fig, ax = plt.subplots()
ax.plot(SMBb.middle_scen, KEb_re, '.')

fig, ax = plt.subplots()
axt = ax.twinx()
ax.semilogy(SMBb.time, SMBb.middle_scen,)
axt.semilogy(KEl_re.time, KEl_re, '.')
axt.semilogy(KEb_re.time, KEb_re, '.')

print(f"Correlation between IW KE and discharge: {xr.corr(SMBb.middle_scen, KEb_re).data}")
print(f"Correlation between IW KE and LF KE: {xr.corr(KEl_re, KEb_re).data}")
print(f"Correlation between LF KE and discharge: {xr.corr(KEl_re, SMBb.middle_scen).data}")

# %% [markdown]
# Correlation between discharge and wave energy.

# %%
bootnum = 500
np.random.seed(56642)

good = np.isfinite(SMBb.middle_scen) & np.isfinite(KEb_re) #& (SMBb.middle_scen > 10)

SMBr = SMBb.middle_scen[good].data
KEr = KEb_re[good].data

ndat = KEr.size
idxs = np.arange(ndat)

rvals = np.zeros((bootnum))
pvals = np.zeros((bootnum))

for i in range(bootnum):
    idxs_ = np.random.choice(idxs, ndat)
    rvals[i], pvals[i] = stats.pearsonr(SMBr[idxs_], KEr[idxs_])

fig, ax = plt.subplots()
_ = ax.hist(rvals)

fig, ax = plt.subplots()
_ = ax.hist(rvals**2)

print(f"R mean {rvals.mean()}")
print(f"R std {rvals.std()}")
print(f"R^2 mean {(rvals**2).mean()}")
print(f"R^2 std {(rvals**2).std()}")

# %% [markdown]
# Correlation between low freq energy and wave energy.

# %%
bootnum = 500
np.random.seed(56642)

good = np.isfinite(KEl_re) & np.isfinite(KEb_re) #& (SMBb.middle_scen > 10)

KElr = KEl_re[good].data
KEbr = KEb_re[good].data

ndat = KEbr.size
idxs = np.arange(ndat)

rvals = np.zeros((bootnum))
pvals = np.zeros((bootnum))

for i in range(bootnum):
    idxs_ = np.random.choice(idxs, ndat)
    rvals[i], pvals[i] = stats.pearsonr(KElr[idxs_], KEbr[idxs_])

fig, ax = plt.subplots()
_ = ax.hist(rvals)

fig, ax = plt.subplots()
_ = ax.hist(rvals**2)

print(f"R mean {rvals.mean()}")
print(f"R std {rvals.std()}")
print(f"R^2 mean {(rvals**2).mean()}")
print(f"R^2 std {(rvals**2).std()}")

# %% [markdown]
# ## Spectral analysis

# %%
dt = (AS.time[1] - AS.time[0]).data.astype("timedelta64[s]").astype(float)
frac_min = 0.9
nhigh = int(60*3/dt)
nlow = int(60*30/dt)
nminhigh = int(frac_min*nhigh)
nminlow = int(frac_min*nlow)
npitch = 7
max_pitch_var = 0.05
dmin = 20
dmax = 120
nhours = 1 # Time binning
min_good = 0.9 # Depth fraction allowed

pitch_var = AS.pitch.rolling(time=npitch, center=True).var()

# Remove iceberg events
bad = pitch_var > max_pitch_var
contig = jsig.contiguous_regions(bad)
idx0 = contig[:, 0]
idx1 = contig[:, 1]
# go 2 min earlier
idx0 -= int(60*2/dt)
# go 20 min later
idx1 = idx0 + int(60*20/dt)

# Make mask
for i in range(idx0.size):
    bad[idx0[i]:idx1[i]] = True
    
good = ~bad

time1 = np.datetime64("2018-09-06T15:00")
tslice = slice(time1, time1 + np.timedelta64(3600*3, 's'))

# %%
AS_ = AS.isel(distance=(AS.depth > dmin) & (AS.depth < dmax))

ul = AS_.u.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
uh = AS_.u.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
ub = uh - ul
wl = AS_.vv.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
wh = AS_.vv.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
wb = wh - wl
vl = AS_.v.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
vh = AS_.v.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
vb = vh - vl

KEl = ul**2 + vl**2
KEb = ub**2 + vb**2 + wb**2


# %% [markdown]
# Energy in the different frequency bands. 

# %%
nperseg = 2**11  # 2**9 is about 1.5 hours for a time step of 10 seconds
noverlap = nperseg/2
window = "hann"
fs = 86400./10.  # in cpd
clevs = np.linspace(-7, -4, 7)
scaling = "density"
dmin = 30
dmax = 100.
cmap = "autumn"
padded = False

fig, ax = plt.subplots()
ax.set_ylim(1e-7, 1e-4)

freqs, time, Suu = spectrogram(AS.u, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Svv = spectrogram(AS.v, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Sww = spectrogram(AS.vv, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
SKE_AS =  0.5*(Suu + Svv + Sww) # 0.5*Sww
logKE_AS = np.log10(np.nanmean(SKE_AS.real, axis=-1).T)


use = (AS.depth_adcp > dmin) & (AS.depth_adcp < dmax)
ax.loglog(freqs[1:], np.nanmean(10**logKE_AS[use, 1:], axis=0), label="unfiltered")

freqs, time, Suu = spectrogram(ul, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Svv = spectrogram(vl, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Sww = spectrogram(wl, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
SKE_AS =  0.5*(Suu + Svv + Sww) # 0.5*Sww
logKE_AS = np.log10(np.nanmean(SKE_AS.real, axis=-1).T)


use = (wl.depth_adcp > dmin) & (wl.depth_adcp < dmax)
ax.loglog(freqs[1:], np.nanmean(10**logKE_AS[use, 1:], axis=0), label="low pass")

freqs, time, Suu = spectrogram(uh, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Svv = spectrogram(vh, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Sww = spectrogram(wh, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
SKE_AS =  0.5*(Suu + Svv + Sww) # 0.5*Sww
logKE_AS = np.log10(np.nanmean(SKE_AS.real, axis=-1).T)


use = (wl.depth_adcp > dmin) & (wl.depth_adcp < dmax)
ax.loglog(freqs[1:], np.nanmean(10**logKE_AS[use, 1:], axis=0), label="high pass")

freqs, time, Suu = spectrogram(ub, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Svv = spectrogram(vb, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
freqs, time, Sww = spectrogram(wb, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, axis=0)
SKE_AS =  0.5*(Suu + Svv + Sww) # 0.5*Sww
logKE_AS = np.log10(np.nanmean(SKE_AS.real, axis=-1).T)


use = (wl.depth_adcp > dmin) & (wl.depth_adcp < dmax)
ax.loglog(freqs[1:], np.nanmean(10**logKE_AS[use, 1:], axis=0), label="band pass")
ax.legend()

ax.set_xlabel("Frequency [cpd]")
ax.set_ylabel("Spectral density [m$^2$ s$^{-2}$ cpd$^{-1}$]")

# %% [markdown]
# # MD - downstream mooring
#
# Repeat analysis above for the downstream mooring.

# %%
dt = (DD.time[1] - DD.time[0]).data.astype("timedelta64[s]").astype(float)
frac_min = 0.9
nhigh = NFc(60*5, int(dt))
nlow = NFc(60*30, int(dt))
nminhigh = int(frac_min*nhigh)
nminlow = int(frac_min*nlow)
npitch = 7
max_pitch_var = 0.05
dmin = 20
dmax = 120
nhours = 1 # Time binning
min_good = 0.9 # Depth fraction allowed

print(f"High cut off = {Tcmin(10, nhigh):1.2f} min")
print(f"Low cut off = {Tcmin(10, nlow):1.2f} min")
print(f"High cut off = {1440/Tcmin(10, nhigh):1.2f} cpd")
print(f"Low cut off = {1440/Tcmin(10, nlow):1.2f} cpd")
print(f"n High = {nhigh}")
print(f"n Low = {nlow}")

pitch_var = DD.pitch.isel(instrument=1).rolling(time=npitch, center=True).var()

# Remove iceberg events
bad = pitch_var > max_pitch_var
contig = jsig.contiguous_regions(bad)
idx0 = contig[:, 0]
idx1 = contig[:, 1]
# go 2 min earlier
idx0 -= int(60*2/dt)
# go 20 min later
idx1 = idx0 + int(60*20/dt)

# Make mask
for i in range(idx0.size):
    bad[idx0[i]:idx1[i]] = True
    
good = ~bad

time1 = np.datetime64("2018-09-06T15:00")
tslice = slice(time1, time1 + np.timedelta64(3600*3, 's'))



# %%
DD_ = DD.isel(depth_adcp=(DD.depth_adcp > dmin) & (DD.depth_adcp < dmax))

ul = DD_.u.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
uh = DD_.u.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
ub = uh - ul
wl = DD_.w.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
wh = DD_.w.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
wb = wh - wl
vl = DD_.v.where(good).rolling(time=nlow, min_periods=nminlow, center=True).mean()
vh = DD_.v.where(good).rolling(time=nhigh, min_periods=nminhigh, center=True).mean()
vb = vh - vl

KEl = ul**2 + vl**2
KEb = ub**2 + vb**2 + wb**2

# %%
wb.to_netcdf("../data/MD_band_pass_w.nc")

# %%
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 5))

ub.sel(time=tslice).plot(ax=axs[0], x="time", y="depth_adcp", yincrease=False)
vb.sel(time=tslice).plot(ax=axs[1], x="time", y="depth_adcp", yincrease=False)
wb.sel(time=tslice).plot(ax=axs[2], x="time", y="depth_adcp", yincrease=False)

# %%
bootnum = 500
np.random.seed(482990)

nall = KEl.isnull() | KEb.isnull()

nmax = nall.distance.size
ngood = nmax - nall.sum("depth_adcp")
frac_good = ngood/nmax
good = (frac_good > min_good)
# dz = (AS.distance[1] - AS.distance[0]).data

KEl_ave = (KEl.sum("depth_adcp")/ngood).where(good)
KEb_ave = (KEb.sum("depth_adcp")/ngood).where(good)

KEl_re = KEl_ave.resample(time=f"{nhours}H", skipna=True).mean()
KEb_re = KEb_ave.resample(time=f"{nhours}H", skipna=True).mean()

good = np.isfinite(KEl_re) & np.isfinite(KEb_re) #& (SMBb.middle_scen > 10)

KElr = KEl_re[good].data
KEbr = KEb_re[good].data

ndat = KEbr.size
idxs = np.arange(ndat)

KEl_mean = np.zeros((bootnum))
KEb_mean = np.zeros((bootnum))

for i in range(bootnum):
    idxs_ = np.random.choice(idxs, ndat)
    KEl_mean[i] = KElr[idxs_].mean()
    KEb_mean[i] = KEbr[idxs_].mean()
    
KE_frac = KEb_mean/(KEb_mean + KEl_mean)

fig, ax = plt.subplots()
_ = ax.hist(KEl_mean)

fig, ax = plt.subplots()
_ = ax.hist(KEb_mean)

fig, ax = plt.subplots()
_ = ax.hist(KE_frac)

print(f"KE low mean {KEl_mean.mean():.2e} J kg-1")
print(f"KE wave mean {KEb_mean.mean():.2e} J kg-1")
print(f"KE std mean {KEb_mean.std():.2e} J kg-1")
print(f"KE frac mean {KE_frac.mean():.2%}")
print(f"KE frac std {KE_frac.std():.2%}")
print(f"KE frac 90% error {2*KE_frac.std():.2%}")
