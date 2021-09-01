from particella_cerchio import *

# importo il file analisi_errori.py comune a tutti i moduli
import os
import sys
from scipy.optimize import curve_fit

from inspect import getfile, currentframe
current_dir = os.path.dirname(os.path.abspath(getfile(currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from analisi_errori import *

save_fig = True
write_txt = False

'''
# SIMULAZIONE (PARTE 1) - COS'E' il WINDING NUMBER
print('Parte 1:')

# carico dati da file
simul_path = 'DATA/winding_number.dat'
with open(simul_path, 'r') as file:
    beta = float(next(file))
    chi = float(next(file))
    N = int(next(file))
    winding_number = np.array([int(Q) for Q in next(file).split()])
    y = np.array([float(phi) for phi in next(file).split()])

# ora calcolo la suscettibilità
susc = suscettibility(winding_number, beta, chi)
print('suscettibility a beta =', beta, ': susc =', susc)

# calcolo il tempo di autocorrelazione e plotto le funzioni di autocorrel
print('tempo di autocorrelazione della suscettibilità =',
      autocorr_time_definizione(winding_number**2))

# disegno il valore del campo
plot_me(y, beta, chi, save_fig=save_fig)
'''


# SIMULAZIONE (PARTE 2) - CRITICAL SLOWING DOWN
print('\nParte 2 (critical slowing down):')
# ripeto la simulazione della parte 1, ma variando N, mostrando il critical slowing down
# nel limite al continuo (T fissa), è più difficile cambiare Q

winding_numbers = []
# carico dati da file
simul_path = 'DATA/critical_slowing_down.dat'
with open(simul_path, 'r') as file:
    beta = float(next(file))
    chi = float(next(file))
    N_list = [int(N) for N in next(file).split()]
    for N in N_list:
        winding_numbers.append(np.array([int(Q) for Q in next(file).split()]))

winding_numbers = np.array(winding_numbers)

'''
# PARAMETRI dei grafici
max_correl = 4000  # massima funzione di correlazione a k vicini plottata
size = 14

i_0 = 2
i_1 = 3
i_2 = 4

O = winding_numbers**2

figsize = (6.3, 9.5)

fig, [ax_corr, ax_int] = plt.subplots(2, 1, figsize=figsize, sharex=True)

ax_corr.set_title('Autocorrelation function', size=size)
ax_corr.scatter(np.arange(max_correl), [correlation_function(O[i_0], k)
                                        for k in range(max_correl)], label=r"$N={}$".format(N_list[i_0]))
ax_corr.scatter(np.arange(max_correl), [correlation_function(O[i_1], k)
                                        for k in range(max_correl)], label=r"$N={}$".format(N_list[i_1]))
ax_corr.scatter(np.arange(max_correl), [correlation_function(O[i_2], k)
                                        for k in range(max_correl)], label=r"$N={}$".format(N_list[i_2]))

ax_corr.set_ylabel(r"$C(k)$", size=size)
ax_corr.tick_params(labelsize=size)
ax_corr.legend(frameon=False, fontsize='x-large')

ax_int.set_title('Autocorrelation function integrata', size=size)
ax_int.set_ylabel(r"$\Sigma_k\,\, C(k)$", size=size)
ax_int.set_xlabel(r"$k$", size=size)
ax_int.tick_params(labelsize=size)

integral_O_sum = 0.
integral_O_list = []
for k in range(max_correl):
    integral_O_sum += correlation_function(O[i_0], k)
    integral_O_list.append(integral_O_sum)
ax_int.scatter(np.arange(max_correl), integral_O_list,
               color='C0', label=r"$N={}$".format(N_list[i_0]))
ax_int.hlines(y=autocorr_time_definizione(O[i_0]), xmin=0.,
              xmax=max_correl, color='C0', linestyle='--')

integral_O_sum = 0.
integral_O_list = []
for k in range(max_correl):
    integral_O_sum += correlation_function(O[i_1], k)
    integral_O_list.append(integral_O_sum)
ax_int.scatter(np.arange(max_correl), integral_O_list,
               color='C1', label=r"$N={}$".format(N_list[i_1]))
ax_int.hlines(y=autocorr_time_definizione(O[i_1]), xmin=0.,
              xmax=max_correl, color='C1', linestyle='--')

integral_O_sum = 0.
integral_O_list = []
for k in range(max_correl):
    integral_O_sum += correlation_function(O[i_2], k)
    integral_O_list.append(integral_O_sum)
ax_int.scatter(np.arange(max_correl), integral_O_list,
               color='C2', label=r"$N={}$".format(N_list[i_2]))
ax_int.hlines(y=autocorr_time_definizione(O[i_2]), xmin=0.,
              xmax=max_correl, color='C2', linestyle='--')
ax_int.legend(frameon=False, fontsize='x-large', loc='upper left')

fig.tight_layout(pad=3.0)

if save_fig is True:
    fig.savefig("figure/correl.jpg")
'''

# mostro una MC chain, dove estraggo Q, 'measures' volte, N è la divisione temporale

figsize = (6.3, 9.5)

size = 14

i_0 = 0
i_1 = 4
i_2 = 8

fig_Q, [ax_Q, ax_bin] = plt.subplots(2, 1, figsize=figsize, sharex=False)


ax_Q.set_xlabel(r"MCMC sweeps [x10$^5$]", size=size)
ax_Q.set_ylabel(r"$Q$", size=size)
ax_Q.tick_params(labelsize=size)
ax_Q.set_title('MCMC for the winding number', size=size)

ax_Q.set_ylim(-5.5, 7.5)


ax_bin.set_xlabel(r"$Q$", size=size)
ax_bin.set_ylabel(r"norm. counts", size=size)
ax_bin.tick_params(labelsize=size)
ax_bin.set_title('Histogram for the winding number', size=size)
ax_bin.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
ax_bin.set_xticklabels([-4, -3, -2, -1, 0, 1, 2, 3, 4])

stop = int(1e6)

ax_Q.plot(np.arange(len(winding_numbers[i_0][:stop:])) / 1e5, winding_numbers[i_0]
          [:stop:], label="N={}".format(N_list[i_0]), alpha=0.6, color='C0')
ax_Q.plot(np.arange(len(winding_numbers[i_1][:stop:])) / 1e5, winding_numbers[i_1]
          [:stop:], label="N={}".format(N_list[i_1]), alpha=0.9, color='C1')
ax_Q.plot(np.arange(len(winding_numbers[i_2][:stop:])) / 1e5,
          winding_numbers[i_2][:stop:], label="N={}".format(N_list[i_2]), color='C2')

bins = np.arange(-4.5, 4.5, 0.2)

from matplotlib.transforms import ScaledTranslation

trans0 = ax_bin.transData + ScaledTranslation(0, 0, fig_Q.dpi_scale_trans)
trans1 = ax_bin.transData + \
    ScaledTranslation(+7 / 72, 0, fig_Q.dpi_scale_trans)
trans2 = ax_bin.transData + \
    ScaledTranslation(-7 / 72, 0, fig_Q.dpi_scale_trans)


ax_bin.hist(winding_numbers[i_2], bins=bins,  label="N={}".format(
    N_list[i_2]), alpha=1., density=True, color='C2', transform=trans2)
ax_bin.hist(winding_numbers[i_1], bins=bins, label="N={}".format(
    N_list[i_1]), alpha=0.9, density=True, color='C1', transform=trans1)
ax_bin.hist(winding_numbers[i_0], bins=bins, label="N={}".format(
    N_list[i_0]), alpha=0.6, density=True, color='C0', transform=trans0)

Q_x = np.linspace(-5, 5, 1000)
ax_bin.plot(Q_x, 8 * np.exp(-Q_x**2 / 2 / (beta)) /
            np.sqrt(2 * np.pi * (beta)), color='C3')
ax_Q.legend(frameon=False, fontsize='x-large', loc='upper center', ncol=3)
ax_bin.legend(frameon=False, fontsize='x-large', loc='upper right', ncol=1)

fig_Q.tight_layout(pad=2.0)
if save_fig is True:
    fig_Q.savefig("figure/winding_number.jpg")


# CRITICAL SLOWING DOWN
#
#


stop_N = 6
'''
if write_txt == True:
    suscett = []  # suscettibilità media
    autocorr_time = []  # tempo di autocorrelazione
    print("starting csd")
    for i in range(len(N_list[:stop_N:])):
        print(i)
        # aggiorno le liste dei risultati:
        # suscettibilità
        suscett.append(suscettibility(winding_numbers[i]**2, beta, chi))
        # tempo di autocorrelazione della suscettibilità
        autocorr_time.append(autocorr_time_definizione(winding_numbers[i]**2))
    np.savetxt("DATA/susc_tau_csd.dat", (suscett, autocorr_time))
'''
suscett, autocorr_time = np.loadtxt("DATA/susc_tau_csd.dat")


figsize = (6.3, 9.5)

size = 14


fig_csd, [ax_csd, ax_tail] = plt.subplots(2, 1, figsize=figsize, sharex=True)


# ax_csd.set_xlabel(r"N", size=size)
ax_csd.set_ylabel(r"$\tau$", size=size)
ax_csd.tick_params(labelsize=size)
ax_csd.set_title('Critical slowing down', size=size)

# ax_csd.set_ylim(-5.5,7.5)

ax_tail.set_xlabel(r"N", size=size)
ax_tail.set_ylabel(r"$\tau$", size=size)
ax_tail.tick_params(labelsize=size)
ax_tail.set_title('Tailor method', size=size)


# plotto la suscettibilità al variare di N
fig_susc, [ax_susc, ax_susc_tail] = plt.subplots(
    2, 1, figsize=figsize, sharex=True)

# ax_csd.set_xlabel(r"N", size=size)
ax_susc.set_ylabel(r"$\chi=\langle Q^2 \rangle$", size=size)
ax_susc.tick_params(labelsize=size)
ax_susc.set_title('$\chi$ for standard Metropolis', size=size)

# ax_csd.set_ylim(-5.5,7.5)

ax_susc_tail.set_xlabel(r"N", size=size)
ax_susc_tail.set_ylabel(r"$\chi=\langle Q^2 \rangle$", size=size)
ax_susc_tail.tick_params(labelsize=size)
ax_susc_tail.set_title('$\chi$ for tailor method', size=size)

ax_susc_tail.set_ylim((0.4, 1.6))
ax_susc.scatter(N_list[:stop_N:], suscett)

# plotto il tempo di autocorrelazione della suscettibilità al variare di N
ax_csd.scatter(N_list[:stop_N:], autocorr_time)
ax_csd.set_yscale('log')


def line(x, a, b):
    return a * x + b


color = 'gray'

popt, pcov = curve_fit(line, N_list[:stop_N:], np.log10(autocorr_time))
ax_csd.plot(N_list[:stop_N:], 10**popt[1] *
            10**(np.asarray(N_list[:stop_N:]) * popt[0]), color=color, linestyle='--')

print("a=", popt[0])

# SIMULAZIONE (PARTE 3) - TAYLOR METHOD
# ripeto la parte 2 (presa dati a vari N), usando taylor method
# mi calcolo il tempo di autocorrelazione della suscettibilità
print('\nParte 3 (taylor method):')

winding_number = []
# carico dati da file
simul_path = 'DATA/taylor.dat'
with open(simul_path, 'r') as file:
    beta = float(next(file))
    chi = float(next(file))
    N_list_taylor = [int(N) for N in next(file).split()]
    for N in N_list_taylor:
        winding_number.append(np.array([int(Q) for Q in next(file).split()]))

winding_number = np.array(winding_number)
remove_list_taylor = []


if write_txt == True:

    print("startin tail")
    # Array con i risultati
    suscett_taylor = []  # suscettibilità media
    autocorr_time_taylor = []  # tempo di autocorrelazione

    for i in range(len(N_list_taylor)):
        print(i)
        # aggiorno le liste dei risultati:
        # suscettibilità
        suscett_taylor.append(suscettibility(winding_number[i]**2, beta, chi))
        # tempo di autocorrelazione della suscettibilità
        autocorr_time_taylor.append(
            autocorr_time_definizione(winding_number[i]**2))

        np.savetxt("DATA/susc_tau_tail.dat",
                   (suscett_taylor, autocorr_time_taylor))

suscett_taylor, autocorr_time_taylor = np.loadtxt("DATA/susc_tau_tail.dat")


# plotto la suscettibilità al variare di N
ax_susc_tail.scatter(N_list_taylor, suscett_taylor,
                     color='C2', label='Tailor method')
ax_susc_tail.scatter(N_list[:stop_N:], suscett,
                     color='C0', alpha=0.3, label='Standard Metropolis')

ax_susc_tail.legend(frameon=False, fontsize='x-large',
                    loc='upper left', ncol=1)


# plotto il tempo di autocorrelazione della suscettibilità al variare di N

ax_tail.scatter(N_list[:stop_N:], autocorr_time,
                color='C0', alpha=0.3, label='Standard Metropolis')
ax_tail.scatter(N_list_taylor, autocorr_time_taylor,
                color='C2', label='Tailor method')


ax_tail.legend(frameon=False, fontsize='x-large', loc='upper left', ncol=1)

ax_tail.set_yscale('log')


fig_csd.tight_layout(pad=2.0)
if save_fig is True:
    fig_csd.savefig("figure/tailor_csd.jpg")


fig_susc.tight_layout(pad=2.0)
if save_fig is True:
    fig_susc.savefig("figure/susc_tailor_csd.jpg")
