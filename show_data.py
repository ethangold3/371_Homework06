import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as Colormap


def figure_size(data, width=6, margin=1, per_label_height=0.5):
    dims = len(data.x[0, :])
    assert dims in {1, 2}, 'can only plot in one or two dimensions'
    if dims == 1:
        n_labels = len(data.labels)
        height = margin + per_label_height * (n_labels - 1)
    else:
        height = width
    return (width, height), height / width


def axes_box(data, margins=(0.1, 0.25)):
    x, y = data.x, data.y
    dims = len(x[0, :])
    if dims == 1:
        box = [np.min(x) - margins[0], np.max(x) + margins[0]]
        box.extend([-margins[1], data.labels[-1] + margins[1]])
    else:
        box = []
        for i in range(dims):
            box.extend([np.min(x[:, i]) - margins[0], np.max(x[:, i]) + margins[0]])
    return box


def show_decision_regions(data, h, colors, box, samples):
    x, y, labels = data.x, data.y, data.labels
    n_labels = len(labels)
    dims = len(x[0, :])
    if dims == 1:
        x_grid = np.linspace(box[0], box[1], samples[0])
        y_grid = np.outer(np.ones(samples[1]), h(x_grid[:, None]))
    else:
        buckets = [np.linspace(box[0], box[1], samples[0]),
                   np.linspace(box[2], box[3], samples[1])]
        x_grid = np.array([a.ravel() for a in np.meshgrid(*buckets)]).T
        y_grid = h(x_grid).reshape((samples[1], samples[0]))
    c_map = Colormap.from_list('label_colors', colors[:n_labels], N=n_labels)
    plt.imshow(y_grid, origin='lower', extent=box, aspect='auto', cmap=c_map, alpha=0.35)
    return box


def show_data(data, clf=None, title=None, x_samples=300, ms=8, fs=12):
    box = axes_box(data)
    fig_size, aspect = figure_size(data)
    samples = (x_samples * np.array([1, aspect])).astype(int)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=fig_size)
    if clf is not None:
        show_decision_regions(data, clf, colors, box, samples)
    dims = len(data.x[0, :])
    for y in data.labels:
        x = data.x[data.y == y, :]
        abscissa = x[:, 0]
        ordinate = y * np.ones_like(abscissa) if dims == 1 else x[:, 1]
        plt.plot(abscissa, ordinate, '.', mfc=colors[y], mew=0., ms=ms, label=str(y))
    plt.xlim(box[:2])
    plt.ylim(box[2:])
    if dims == 1:
        plt.yticks(data.labels)
        plt.xlabel(r'$x$')
        plt.ylabel('label')
    if dims == 2:
        plt.legend(title='labels', fontsize=fs, title_fontsize=fs)
        plt.xlabel(r'$x_0$')
        plt.ylabel(r'$x_1$')
    if title is not None:
        plt.title(title)
    plt.tight_layout(pad=0.3)
    plt.draw()


def show_data_dict(d, prefix=None):
    for key, value in d.items():
        title = key if prefix is None else '.'.join((prefix, key))
        if isinstance(value, dict):
            show_data_dict(value, title)
        else:
            show_data(value, title=title)
