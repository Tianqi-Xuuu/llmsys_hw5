import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name, ylabel):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

if __name__ == '__main__':
    # Data Parallel: training time
    single_mean, single_std = 48.63379514217377, 0.10550626386472832
    device0_mean, device0_std = 28.49230043888092, 0.5969966111801951
    device1_mean, device1_std = 27.562995648384096, 0.1516383381645642

    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn_time.png',
        'Training Time (s)')

    # Data Parallel: tokens per second
    single_mean, single_std = 82925.95375461315, 291.2216789780129

    device0_mean, device0_std = 81329.05220289997, 582.8478067179874
    device1_mean, device1_std = 81641.01233182676, 532.9606692484114

    double_mean = (device0_mean + device1_mean)
    double_std = np.sqrt((device0_std ** 2 + device1_std ** 2))

    plot([double_mean, single_mean],
        [double_std, single_std],
        ['Data Parallel', 'Single GPU'],
        'ddp_vs_rn_tokens.png',
        'Tokens / Second')

    # Pipeline/Model Parallel: training time
    pp_mean, pp_std = 40.00531303882599, 0.21180689334869385
    mp_mean, mp_std = 49.598854422569275, 0.0850290060043335

    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp_time.png',
        'Training Time (s)')

    # Pipeline/Model Parallel: tokens per second
    pp_mean, pp_std = 15998.323522169147, 84.70262939148415
    mp_mean, mp_std = 12903.561753167018, 22.12101554684432

    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp_tokens.png',
        'Tokens / Second')

    print("Plots saved: ddp_vs_rn_time.png, ddp_vs_rn_tokens.png, pp_vs_mp_time.png, pp_vs_mp_tokens.png")
