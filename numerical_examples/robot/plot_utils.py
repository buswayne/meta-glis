import numpy as np
import matplotlib.pyplot as plt

def plot_outs(outs, targets):
    clip = -2
    tgt = targets.ravel()
    tgt = np.clip(tgt, clip, None)

    fig, axes = plt.subplots(nrows=3, ncols=6,
                             figsize=(18, 9),
                             sharex=False, sharey=False)

    groups = ['Kd', 'Ki', 'Kp']
    offsets = [0, 7, 14]

    # Original bounds
    # bounds = {
    #     'Kd': (0, 50),
    #     'Ki': (0, 500),
    #     'Kp': (0, 400)
    # }

    bounds = {
        'Kd': (0, 500),
        'Ki': (0, 200),
        'Kp': (0, 15000)
    }

    def add_margin(lim, margin=0.05):
        lower, upper = lim
        range_ = upper - lower
        return (lower - margin * range_, upper + margin * range_)

    limits = {k: add_margin(v) for k, v in bounds.items()}

    for row, (group, base) in enumerate(zip(groups, offsets)):
        for col in range(6):
            i, j = base + col, base + col + 1
            ax = axes[row, col]

            x = outs[..., i].ravel()
            y = outs[..., j].ravel()

            sc = ax.scatter(x, y, c=tgt,
                            cmap='plasma_r',
                            s=20, alpha=0.3)

            ax.set_title(f'{group}{col + 1} vs {group}{col + 2}', fontsize=9)
            if col == 0:
                ax.set_ylabel(group)
            if row == 2:
                ax.set_xlabel(f'{group}{col + 1}')
            ax.grid(True, lw=0.3, alpha=0.4)

            # Set axis limits with margin
            ax.set_xlim(limits[group])
            ax.set_ylim(limits[group])

    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax,
                 label=f'objective value (≤ {str(clip)} shown as {str(clip)})')

    plt.show()


