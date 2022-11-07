from sklearn.linear_model import LinearRegression

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np 


def draw_hist(sample, x, y, mean, std, ax): 
    ax.hist(sample, density=True, bins=30, label='histogram')
    ax.plot(x, y, color='red', label='theoretical dist')
    offset = ax.get_ylim()[1]*0.1
    ax.vlines(mean, *ax.get_ylim(), color='black', label='mean')
    for i in range(1, 4):
        ax.arrow(mean, -i*offset, i*std, 0, width=0.01,head_length=std/5, color='green', label=f'{i}σ')
        #ax.arrow(mean, -i*offset, -i*var, 0, width=0.01, head_length=var/5, color='green')
    _ = ax.legend()
    
    
def pareto_rule(x):
    counts = Counter(x)
    del x
    counts = {k: v/sum(counts.values()) for k, v in counts.items()}
    #counts = [(k, counts.get(k, 0)) for k in range(1, max(counts)+1)]
    counts = Counter(counts).most_common()
    
    counts = sorted(counts, key=lambda x: -x[1])
    #print(counts)
    total_probability = 0
    counter = 0
    for k, probability in counts: 
        total_probability += probability
        counter += 1
        if total_probability > 0.8: 
            break
    print(f'There is {round(total_probability, 2)} probability in {round(100*counter/len(counts), 2)}% of randomly generated numbers')
    cumulative_probabilities = np.cumsum([count for k, count in counts])
    plt.bar(range(len(cumulative_probabilities)), cumulative_probabilities)
   
def MLE_alpha(x, x_min=None):
    if x_min is None: 
        x_min = x.min()
    return len(x)/np.sum(np.log(x/x_min))  + 1
    
def print_alpha(bins, hist, convolve=True):
    if convolve:
        bins = np.convolve(bins, np.ones(2)/2, mode='valid')
    slope, intercept = linear_regression(bins, hist)
    print(f"α={-slope}")
    
    
def linear_regression(x, y): 
    x = x.reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    return lr.coef_[0], lr.intercept_