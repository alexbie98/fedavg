import matplotlib.pyplot as plt
import IPython


def view_10(img, label):
    """ view 10 labelled examples from tensor"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.set_title(label[i].cpu().numpy())
        ax.imshow(img[i][0], cmap="gray")
    IPython.display.display(fig)
    plt.close(fig)


def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
