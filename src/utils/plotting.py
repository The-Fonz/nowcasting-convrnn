import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import HTML

import cartopy.crs as ccrs
import cartopy.feature


def plot_radar(h5radar, imgdata, figsize=(12,12), interval=200, linecolor='white', colormap=None, img_slice=None):
    """
    Plot KNMI .h5 radar file, or animate list of h5 files.
    :param h5file: one h5file with projection information (must be h5py.File instance)
    :param imgdata: a single array with img data (plot) or list (animation)
    :param figsize: Size of plot
    :param interval: If h5file is list, interval of animation frames
    :returns: matplotlib.Axes or IPython.display.HTML
    """    
    if type(imgdata) in (list,tuple):
        first_imgdata = imgdata[0]
        is_animation = True
    else:
        first_imgdata = imgdata
        is_animation = False
    
    stere = ccrs.Stereographic(central_latitude=90, true_scale_latitude=60)
    # TODO: Get projection info from h5
    # p = [s.split('=') for s in '+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378.14 +b=6356.75 +x_0=0 y_0=0'.replace('+','').split()]
    # stere_nl = ccrs.CRS(p)

    corners_lonlat = h5radar["geographic"].attrs["geo_product_corners"].reshape(4,2)
    # Ignore z coordinate added on transform
    bbox_stere = stere.transform_points(
                        ccrs.PlateCarree(),
                        corners_lonlat.T[0],
                        corners_lonlat.T[1])[:,:2]

    ext_stere = (bbox_stere[:,0].min(), bbox_stere[:,0].max(),
                 bbox_stere[:,1].min(), bbox_stere[:,1].max())

    # Init plot, set axes projection, set extent
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=stere)
    ax.set_extent(ext_stere, crs=stere)
    
    # Add high-res coastline
    # NOTE: Downloads dataset if required
    ax.coastlines('10m', color=linecolor)
    # Add high-res border
    ax.add_feature(cartopy.feature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m'),
                   edgecolor=linecolor, facecolor='none')
    ax.gridlines(crs=stere)

    # We need one extra, not sure why, but not with guaraud shading
    lons = np.linspace(ext_stere[0], ext_stere[1], first_imgdata.shape[1]+1)
    lats = np.linspace(ext_stere[3], ext_stere[2], first_imgdata.shape[0]+1)

    cmesh = ax.pcolormesh(lons, lats, first_imgdata, cmap=colormap)
    
    # If this is not an animation, we're done here
    if not is_animation:
        return ax

    def animate(i):
        cmesh.set_array(imgdata[i].ravel())
        # Must return Artists
        return cmesh.findobj()

    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(imgdata), interval=interval, blit=True)

    return HTML(anim.to_html5_video())


def plot_synthetic(canvas, interval=100, figsize=None):
    """
    Plot synthetic moving dataset
    :param canvas: images in (b, w, h) order
    :returns: matplotlib.Axes or IPython.display.HTML
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    p = ax.imshow(canvas[0])

    def animate(i):
        p.set_array(canvas[i])
        # Must return Artists
        return p.findobj()

    anim = animation.FuncAnimation(fig, animate, frames=len(canvas), interval=interval, blit=True)

    return HTML(anim.to_html5_video())


def plot_loss(losses, learning_rates=None, save_as=None, log_scale=True):
    """
    Plot loss and (optionally) learning rate.
    :param losses: Array of loss values
    :kwarg learning_rates: Optional array of learning rates
    :kwarg save_as: Optional filename to save plot as png
    :returns: 
    """
    fig, ax = plt.subplots()
    ax.set_title("Training loss per batch, lowest {:.5f}".format(min(losses)))
    ax.set_ylabel('MSE loss')
    ax.plot(losses, 'b-')
    if log_scale:
        ax.set_yscale('log')

    if learning_rates:
        ax2 = ax.twinx()
        ax2.plot(learning_rates, 'r--')
        if log_scale:
            ax2.set_yscale('log')
        ax2.set_ylabel('Learning rate')
    
    if not save_as:
        plt.show()
    else:
        plt.savefig(save_as)

    plt.close()
