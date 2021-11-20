import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import io
from PIL import Image

def fig2img ( fig):
    """
    	Convert matplotlib figure to numpy array

        :param fig (matplotlib.figure.Figure): figure to convert
        :returns (numpy.array): (HxWx3) numpy array of figure fig
    """
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    mplimage = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    return mplimage

def scatterBWImages(locs,imgs,xlims = (-3,3), ylims = (-3,3)):
    """
        Plots BW images in imgs at locations locs

        :param loc: Lx2 where locs(1,:) = (x,y) is the
            x,y position of image imgs(1)
        :param imgs: LxMxN numpy matrix of images where imgs(1,:) is image 1
        :param xlims: x axis limits
        :param ylims: y axis limits
        :returns: (figure, axis) handles tuple
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10) #Width,Height
    ## Scatter of images
    for i in range(len(locs)):
        plotBWImage(locs[i,0],locs[i,1],imgs[i,0,:],ax)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    buf = fig2img(fig)
    plt.close('all')

    return buf

def plotBWImage(x, y, im, ax, lims = (3,3)):
    """
        Plots image im on a x,y scatter plot

        :param x: x plot location
        :param y: y plot location
        :param im: bw image to be plotted, should be a 2D numpy matrix
        :param ax: axis to plot the image on
        :param lims: (xlim,ylim) tuple where |x| > xlim and |y| > ylim are not
                        plotted
        :returns: matplotlib.image.BboxImage handle to the plotted image
                    or None if x,y outside of lims
    """
    if np.abs(x) > lims[0] or np.abs(y) > lims[1]:
        return

    bb = Bbox.from_bounds(x,y,0.1,0.1)
    bb2 = TransformedBbox(bb,ax.transData)
    bbox_image = BboxImage(bb2,
                        norm = None,
                        origin=None,
                        clip_on=False)

    bbox_image.set_data(im)
    ax.add_artist(bbox_image)
