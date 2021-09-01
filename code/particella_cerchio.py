import numpy as np
import random
from matplotlib import pyplot as plt
from numba import jit

'''
Questo file contiene varie definizioni utili per fare la simulazioni e analizzare i dati:
    - dist(x, y) restituisce la distanza fra 2 punti sul cerchio
    - taylor_move agisce sul campo con il metodo del ricucimento 1 volta
    - update_metropolis aggiorno il campo facendo una spazzata su tutti i tempi, usando algoritmo metropolis locale
    - winding_number dato il campo calcola il umero di avvolgimenti
    - suscettibility dato il campo calcola la suscettibilità
    - plot_me plotta il campo
'''

# FUNZIONI (usate per la simulazione)


@jit
def dist(x, y):
    '''definisco la distanza fra 2 punti sul cerchio'''
    if (x - y) > 0.5:
        return x - y - 1
    elif(x - y) < -0.5:
        return x - y + 1
    else:
        return x - y


@jit
def taylor_move(y, eta, epsilon):
    '''metodo del ricucimento:
    cerco 2 punti tali che delta_y = 0.5, prendo la parte in mezzo e la incollo 'da fuori'
    '''
    N = len(y)
    # scelgo casualmento un punto i
    i_init = random.randrange(N)
    # trovo i_end come il più piccolo i tale che |x_i_end - x_i_init - 0.5|mod(1/2) < epsilon
    for i in range(N):
        i_end = i_init - i
        if np.absolute(dist(y[i_init], y[i_end] + 0.5)) < epsilon:
            break
        if i == N - 1:
            return
    # propose the update consisting of the change x_i → [2x_0 − x_i]mod(1 / 2) in [i_init, i_end]
    # accept/reject it with a Metropolis test:
    # l'azione cambia solo per come si collega i_end:
    phi_prova = dist(2 * y[i_init], y[i_end])

    prob = dist(y[i_end - 1], phi_prova)**2 - \
        dist(y[i_end - 1], y[i_end])**2
    # condizioni al bordo periodiche (%N)
    # mi ricordo che la prob è exp(-S_D), con i giusti prefattori
    prob = np.exp(-1. / (2 * eta) * prob)
    # vedo se accetto il cambio
    rand = random.random()  # numero fra 0 e 1
    if (rand < prob):  # accettanza del metropolis
        # se aceetto cambio tutti i punti intermedi
        for i in range(i_end, i_init):
            y[i] = dist(2 * y[i_init], y[i])
    return


@jit
def update_metropolis(y, eta, delta_metr):
    '''aggiorno il vettore y facendo una spazzata su tutti i tempi, usando algoritmo metropolis locale'''
    N = len(y)
    # provo a cambiare il valore di un singolo sito, l'accettanza dipende solo dal valore del campo nei tempo adiacenti
    for i in range(N):
        # provo a cambiare il campo al punto i-esimo in modo casuale di circa delta_metr
        phi_prova = dist(y[i] + delta_metr *
                         (1 - 2 * random.random()), 0)  # valore di prova
        # e riporto il valore fra -0.5 e 0.5 (sono nel cerchio)

        # calcolo la probabilità di accettare il cambio del campo nel punto i (theta = 0)
        # la prob dipende dal'azione discretizzata
        prob = dist(y[(i + 1) % N], phi_prova)**2 + \
            dist(phi_prova, y[i - 1])**2 - \
            dist(y[(i + 1) % N], y[i])**2 - dist(y[i], y[i - 1])**2
        # condizioni al bordo periodiche (%N)
        # mi ricordo che la prob è exp(-S_D), con i giusti prefattori
        prob = np.exp(- 1. / (2 * eta) * prob)
        # estraggo la mia variabile random
        rand = random.random()
        # se accetto il cambio, cambio il campo con campo_prova
        if (rand < prob):
            y[i] = phi_prova
    return


# visto che non è banale aggiungere il theta-term, misuro F(theta) tramite il suo sviluppo in serie di Taylor
# a temperatura nulla, l'unico termine non nullo è quello al secondo ordine: topological suscettibility
# F^(2)= (<Q^2> - <Q>^2) / beta*h
# a temperatura grande prendo solo i primi termini Q=0, Q=1, Q=-1


@jit
def winding_number(y):
    ''' dato un campo y, conto il suo winding number '''
    counter = 0
    for i in range(len(y)):
        # se due punti sono molto distanti (si chiudono 'per fuori')
        if np.abs(y[i] - y[i - 1]) > 0.5:
            # controllo se passo in senso anti-orario
            if y[i] > y[i - 1]:
                counter -= 1
            # o in senso orario
            else:
                counter += 1
    return counter


@jit
def suscettibility(Q, beta, chi=1):
    ''' sviluppo in Taylor calcolato a theta = 0
        - per beta grande: tende a 1 '''
    return (np.mean(Q**2) - np.mean(Q)**2) / (beta / chi)


def plot_me(y, beta, chi=1, name=None, save_fig=False):
    ''' plotta il campo y'''
    N = len(y)
    eta = beta / N

    size = 14
    color = 'C0'

    # SCELTA ZERO:  dov'è centrato in media il campo?
    # assegno ad ogni valore del campo un numero complesso (il suo angolo),
    # medio i numeri complessi e tiro fuori l'angolo medio
    x = np.exp(2 * np.pi * 1j * y)
    time_bh = beta / chi / N * np.arange(N)
    angolo_medio = np.angle(x.mean())  # angolo medio in radianti da -pi a pi
    # ora proietto sul piano e sposto tutti gli elementi
    for i in range(N):
        y[i] -= angolo_medio / np.pi
        if y[i] > 0.5:
            y[i] -= 1
        if y[i] < -0.5:
            y[i] += 1

    # WINDING NUMBER CUT
    # taglio l'array in tanti array quante volte passo da -0.5 a 0.5 'da fuori'
    fig, ax = plt.subplots(1, 1, figsize=(8 / 1.2, 6 / 1.2))

    j = 0  # variabile dummy per tenere conto da dove devo plottare
    for i in range(1, N):
        # se il campo in due punto successivi è molto diverso: passo 'da fuori'
        if np.abs(y[i] - y[i - 1]) > 0.5:
            # plotto da j fino a i-1
            ax.plot(y[j:i], time_bh[j:i], color=color)

            # ora faccio la 'connessione esterna' a mano, (sorry per il casino)
            # inanzitutto calcolo l'altezza di un punto intermedio da disegnare
            # sia a -0.5 che a 0.5, per fare avere la stessa pendenza alla curva
            i_sx = i
            i_dx = i - 1
            if y[i] > y[i - 1]:
                i_sx = i - 1
                i_dx = i
            delta_x_sx = 0.5 + min(y[i], y[i - 1])
            delta_x_dx = 0.5 - max(y[i], y[i - 1])
            pendenza = eta * chi / (delta_x_sx + delta_x_dx)
            delta_y_sx = pendenza * delta_x_sx
            if y[i] > y[i - 1]:
                time_pto_intermedio = time_bh[i - 1] + delta_y_sx
            elif y[i] < y[i - 1]:
                time_pto_intermedio = time_bh[i] - delta_y_sx
            # ora tiriamo le linee fra i punti distanti al pto intermedio (sx e dx)
            ax.plot([y[i_sx], -0.5], [time_bh[i_sx],
                                      time_pto_intermedio], color=color)
            ax.plot([y[i_dx], +0.5], [time_bh[i_dx],
                                      time_pto_intermedio], color=color)

            # ripeto ripartendo mettendo j = punto appena fatto il giro
            j = i

    # plotto quello che rimane dopo l'ultimo taglio, tranne l'ultimo piccolo trattino
    ax.plot(y[j:N], time_bh[j:N], color=color)
    # aggiungo l'ultimissimo tratto tenendo a mente le condizioni al bordo periodiche
    ax.plot([y[N - 1], y[0]], [time_bh[N - 1], beta / chi], color=color)

    # bellurie
    if name is None:
        ax.set_title('Cammino con Q={}'.format(winding_number(y)), size=size)
    else:
        ax.set_title(name, size=size)

    ax.set_xlabel(r"$x$", size=size)
    ax.set_ylabel(r"$\tau$", size=size)
    ax.tick_params(labelsize=size)
    ax.set_xticks([-0.5, -0.2, 0., 0.2, 0.5])
    ax.set_xticklabels([-0.5, -0.2, 0., 0.2, 0.5])

    ax.set_yticks([0., 1., 2., 3., 4.])
    ax.set_yticklabels([0., "", "", "", r"$\beta\hbar$"])

    # fisso i limiti per visualizzare bene il campo
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, beta / chi)

    fig.tight_layout()
    if save_fig is True:
        fig.savefig("figure/path_example.jpg")

    return
