from IPython.display import display
from PIL import Image

def display_rgb_image(batchimage):

    # we have the bgr and float32 dtype output, which every pixel's value is between 0 and 1
     
    display(Image.fromarray((np.concatenate(tuple(batchimage[x] for x in range(batchimage.shape[0])), 
                                                    axis=2)[::-1,:,:].transpose(1,2,0)*255).astype('uint8')))
def display_grey_image(batchimage):

    # squeeze to get grey image type 
    
    display(Image.fromarray(np.squeeze((np.concatenate(tuple(batchimage[x] for x in range(batchimage.shape[0])), 
                                                               axis=2).transpose(1,2,0)*255).astype('uint8'))))
                                                               
def show_recon_result(real, warped, recon, mask):

    display_rgb_image(real)
    display_rgb_image(warped)
    display_rgb_image(recon)
    display_grey_image(mask)
    
def show_swap_result(real, swaped, mask):

    display_rgb_image(real)
    display_rgb_image(swaped)
    display_grey_image(mask)
