import matplotlib

from .utils import *
from .sample_generator import *

import numpy as np
import math
import os

import torch
from tqdm.auto import tqdm
from functools import partialmethod


def denoise_step(sample, H=3, dn1=1., dn2=1.):
    def get_denoise_value(idx):
        start_idx, end_idx = get_neighbor_idx(len(sample), idx, H)
        idxs = np.arange(start_idx, end_idx)
        weight_sample = sample[idxs]

        weights = np.array(list(map(lambda j: bilateral_filter(
            j, idx, sample[j], sample[idx], dn1, dn2), idxs)))
        return np.sum(weight_sample * weights)/np.sum(weights)

    idx_list = np.arange(len(sample))
    denoise_sample = np.array(list(map(get_denoise_value, idx_list)))
    return denoise_sample


def seasonality_extraction(sample, season_len=10, K=2, H=5, ds1=50., ds2=1.):
    sample_len = len(sample)
    idx_list = np.arange(sample_len)

    def get_season_value(idx):
        idxs = get_season_idx(sample_len, idx, season_len, K, H)
        if idxs.size == 0:
            return sample[idx]

        weight_sample = sample[idxs]
        weights = np.array(list(map(lambda j: bilateral_filter(
            j, idx, sample[j], sample[idx], ds1, ds2), idxs)))
        season_value = np.sum(weight_sample * weights)/np.sum(weights)
        return season_value

    seasons_tilda = np.array(list(map(get_season_value, idx_list)))
    return seasons_tilda


def adjustment(sample, relative_trends, seasons_tilda, season_len):
    num_season = int(len(sample)/season_len)
    trend_init = np.mean(seasons_tilda[:season_len*num_season])

    trends_hat = relative_trends + trend_init
    seasons_hat = seasons_tilda - trend_init
    remainders_hat = sample - trends_hat - seasons_hat
    return [trends_hat, seasons_hat, remainders_hat]


def check_converge_criteria(prev_remainders, remainders):
    diff = np.sqrt(np.mean(np.square(remainders-prev_remainders)))
    if diff < 1e-2:
        return True
    else:
        return False


def extract_trend(dt, sample, season_len, M, D, reg1, reg2, l1loss_func, opt, max_iter, trial):
    bar = tqdm(range(max_iter))
    bar.set_description(f'Trial {trial}')

    season_diff = sample[season_len:] - sample[:-season_len]
    g = season_diff.reshape(-1, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.from_numpy(g).float().to(device)

    for i in bar:
        opt.zero_grad()
        loss = l1loss_func(g, M @ dt) + \
            reg1 * torch.sum(torch.abs(dt)) + \
            reg2 * torch.sum(torch.abs(D@dt))
        loss.backward()
        opt.step()
    delta_trends = dt.detach().cpu().numpy()
    relative_trends = get_relative_trends(delta_trends)
    detrend_sample = sample-relative_trends

    return relative_trends, detrend_sample


def _RobustSTL(input, season_len, reg1, reg2, K, H, dn1, dn2, ds1, ds2, learning_rate, max_iter, max_trials, verbose):
    '''
    args:
    - reg1: first order regularization parameter for trend extraction
    - reg2: second order regularization parameter for trend extraction
    - K: number of past season samples in seasonaility extraction
    - H: number of neighborhood in seasonality extraction
    - dn1, dn2 : hyperparameter of bilateral filter in denoising step.
    - ds1, ds2 : hypterparameter of bilarteral filter in seasonality extraction step.
    - learning_rate: the Adam optimizer learning rate
    - max_iter: number of iterations for the Adam optimizer
    - max_trials: number of outer STL iterations
    - verbose: whether showing or hiding progress bar
    '''
    sample = input
    trial = 1
    patient = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_len = len(sample)
    trends = torch.zeros((sample_len)).float()
    seasons = torch.zeros((sample_len)).float()


    dt = torch.zeros((sample_len-1, 1)).float().to(device)
    dt.requires_grad = True
    #season_diff = sample[season_len:] - sample[:-season_len]
    # g = season_diff.reshape(-1, 1)
    # g = torch.from_numpy(g).float().to(device)
    M = get_toeplitz([sample_len-season_len, sample_len-1],
                     np.ones([season_len]))
    M = torch.from_numpy(M).float().to(device)
    D = get_toeplitz([sample_len-2, sample_len-1], np.array([1, -1]))
    D = torch.from_numpy(D).float().to(device)

    l1loss = torch.nn.L1Loss(reduction='sum').to(device)
    opt = torch.optim.Adam([dt], learning_rate)

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not verbose))

    while True:
        # step1: remove noise in input via bilateral filtering
        denoise_sample =\
            denoise_step(sample, H, dn1, dn2)

        # step2: trend extraction via LAD loss regression (using gradient descent)
        # relative_trends, detrend_sample = extract_trend(dt, sample, season_len, M, D,
        #                                                 g, reg1, reg2, l1loss, opt,
        #                                                 max_iter, trial)
        relative_trends, detrend_sample = extract_trend(dt, denoise_sample, season_len, M, D,
                                                        reg1, reg2, l1loss, opt,
                                                        max_iter, trial)

        # step3: seasonality extraction via non-local seasonal filtering
        seasons_tilda =\
            seasonality_extraction(detrend_sample, season_len, K, H, ds1, ds2)

        # step4: adjustment of trend and season
        trends_hat, seasons_hat, remainders_hat =\
            adjustment(sample, relative_trends, seasons_tilda, season_len)

        # step5: repreat step1 - step4 until remainders are converged
        if trial != 1:
            converge = check_converge_criteria(
                previous_remainders, remainders_hat)
            if converge or trial >= max_trials:
                return [input, trends, seasons, remainders_hat]

        trial += 1
        previous_remainders = remainders_hat[:]
        sample = remainders_hat
        trends += trends_hat
        seasons += seasons_hat

        samples = zip([input, trends, seasons, remainders_hat], ['sample', 'trend', 'seasonality', 'remainder'])

        for i, item in enumerate(samples):
            plt.subplot(4, 1, (i + 1))
            if i == 0:
                plt.plot(item[0], color='blue')
                plt.title(item[1])
                plt.subplot(4, 1, i + 2)
                plt.plot(item[0], color='blue')
            else:
                plt.plot(item[0], color='red')
                plt.title(item[1])
        plt.show()

    return [input, trends, seasons, remainders_hat]


def RobustSTL(input, season_len, reg1=10.0, reg2=0.5, K=2, H=5, dn1=1., dn2=1.,
              ds1=50., ds2=1., learning_rate=0.01, max_iter=100, max_trials=10,
              verbose=True):
    if np.ndim(input) < 2:
        return _RobustSTL(input, season_len, reg1, reg2, K, H, dn1, dn2, ds1,
                          ds2, learning_rate, max_iter, max_trials, verbose)

    elif np.ndim(input) == 2 and np.shape(input)[1] == 1:
        return _RobustSTL(input[:, 0], season_len, reg1, reg2, K, H, dn1, dn2,
                          ds1, ds2, learning_rate, max_iter, max_trials, verbose)

    elif np.ndim(input) == 2 or np.ndim(input) == 3:
        if np.ndim(input) == 3 and np.shape(input)[2] > 1:
            print(
                "[!] Valid input series shape: [# of Series, # of Time Steps] or [# of series, # of Time Steps, 1]")
            raise
        elif np.ndim(input) == 3:
            input = input[:, :, 0]
        num_series = np.shape(input)[0]

        input_list = [input[i, :] for i in range(num_series)]

        from pathos.multiprocessing import ProcessingPool as Pool
        p = Pool(num_series)

        def run_RobustSTL(_input):
            return _RobustSTL(_input, season_len, reg1, reg2, K, H, dn1, dn2, ds1, ds2)
        result = p.map(run_RobustSTL, input_list)

        return result
    else:
        print("[!] input series error")
        raise


if __name__ == '__main__':
    from sample_generator import *
    sample_list = sample_generation()
    result = RobustSTL(sample_list[0], 50,
                       reg1=10.0, reg2=0.5, K=2, H=5, ds1=10)
