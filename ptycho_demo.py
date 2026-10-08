import h5py
import numpy as np

# Use cupy if GPU & CUDA is available, otherwise use plain numpy
try:
  import cupy as cp
  xp = cp
except:
  xp = np

# Use matplotlib for plotting
try:
  import matplotlib
  matplotlib.use('TkAgg')  # bypass PyCharm SciView backend (incompatible with matplotlib>=3.10)
  import matplotlib.pyplot as plt
  plot_result = True
except:
  plot_result = False
    
class ptycho_recon_demo:
  def __init__(self,h5_filename,focus_to_sample_distance_m=800e-6,num_iterations = 50,algorithm = 'ePIE'):
    with h5py.File(h5_filename) as h:
      diffpattern = xp.array(h['diffpattern'][()],dtype=xp.uint32)
      self.diffamp = xp.array(xp.fft.fftshift(xp.sqrt(diffpattern),axes=(1,2)),dtype=xp.float32)
      self.points_um = h['points'][()]
      self.npoints,self.nx,self.ny = self.diffamp.shape
      
      self.wavelength_m = h['lambda_nm'][()]*1e-9
      self.pixel_size_m = h['lambda_nm'][()]*1e-9*h['z_m'][()]/(h['ccd_pixel_um'][()]*1e-6*self.nx)
      
    # For probe initialization
    self.focus_to_sample_distance_m = focus_to_sample_distance_m
    
    self.num_iterations = num_iterations
    self.algorithm = algorithm.lower()
      
  def calc_positions_and_dim(self):
    self.scan_positions = xp.zeros((2,self.npoints),xp.int32)
    # (x_low,y_low)
    
    self.scan_positions[0] = xp.round(self.points_um[0]*1e-6/self.pixel_size_m)
    self.scan_positions[1] = xp.round(self.points_um[1]*1e-6/self.pixel_size_m)
    
    self.scan_positions[0] -= xp.min(self.scan_positions[0])
    self.scan_positions[1] -= xp.min(self.scan_positions[1])
    
    self.obj_dim_x = int(xp.max(self.scan_positions[0]+self.nx))
    self.obj_dim_y = int(xp.max(self.scan_positions[1]+self.ny))
    
  def init_prb(self,focus_to_sample_distance_m):
    self.prb = xp.zeros((self.nx,self.ny),dtype=xp.complex64)
    self.prb[:] = xp.fft.fftshift(xp.fft.ifft2(xp.mean(self.diffamp,0)))*np.sqrt(self.nx*self.ny)
    self.prb[:] = propagate(self.prb,self.pixel_size_m,self.wavelength_m,focus_to_sample_distance_m)
    
  def init_arrays(self):
    self.prb = xp.ones((self.nx,self.ny),dtype=xp.complex64)
    self.obj = xp.ones((self.obj_dim_x,self.obj_dim_y),dtype=xp.complex64)
    
    self.psi = xp.zeros((self.npoints,self.nx,self.ny),dtype=xp.complex64)
    self.psi_tmp = xp.zeros((self.npoints,self.nx,self.ny),dtype=xp.complex64)
    
    self.p_tmp_complex = xp.zeros((self.nx,self.ny),dtype=xp.complex64)
    self.p_tmp_real = xp.zeros((self.nx,self.ny),dtype=xp.float32)
    
    self.o_tmp_complex = xp.zeros((self.obj_dim_x,self.obj_dim_y),dtype=xp.complex64)
    self.o_tmp_real = xp.zeros((self.obj_dim_x,self.obj_dim_y),dtype=xp.float32)
    
  
  def calc_exit_wave(self):
    for i in range(self.npoints):
      self.psi[i] = self.prb*\
        self.obj[self.scan_positions[0,i]:self.scan_positions[0,i]+self.nx,\
                self.scan_positions[1,i]:self.scan_positions[1,i]+self.ny]
      
  def decompose_new_obj(self):
    self.o_tmp_complex[:] = 0
    self.o_tmp_real[:] = 0.000001 # Avoid divide by zero
    
    for i in range(self.npoints):
      self.o_tmp_complex[self.scan_positions[0,i]:self.scan_positions[0,i]+self.nx,\
                self.scan_positions[1,i]:self.scan_positions[1,i]+self.ny]\
      +=\
      self.psi[i]*xp.conj(self.prb)
      
      self.o_tmp_real[self.scan_positions[0,i]:self.scan_positions[0,i]+self.nx,\
                self.scan_positions[1,i]:self.scan_positions[1,i]+self.ny]\
      +=\
      xp.abs(self.prb)**2
      
    self.obj = self.o_tmp_complex / self.o_tmp_real
    
  def decompose_new_prb(self):
    self.p_tmp_complex[:] = 0
    self.p_tmp_real[:] = 0.000001 # Avoid divide by zero
    
    for i in range(self.npoints):
      obj_crop = self.obj[self.scan_positions[0,i]:self.scan_positions[0,i]+self.nx,\
                self.scan_positions[1,i]:self.scan_positions[1,i]+self.ny]
      self.p_tmp_complex += self.psi[i]*xp.conj(obj_crop)
      
      self.p_tmp_real+=xp.abs(obj_crop)**2
      
    self.prb = self.p_tmp_complex / self.p_tmp_real
    
  def psi_replace_amplitude(self):
    self.psi[:] = self.diffamp[:]*xp.exp(1j*xp.angle(self.psi))
    
  def iterative_recon_epie(self):
    self.calc_exit_wave()
    
    self.psi[:] = xp.fft.fft2(self.psi,axes=(1,2))/np.sqrt(self.nx*self.ny)
    self.psi_replace_amplitude()
    self.psi[:] = xp.fft.ifft2(self.psi,axes=(1,2))*np.sqrt(self.nx*self.ny)
    
    self.decompose_new_prb()
    self.decompose_new_obj()
    
  def iterative_recon_dm(self):
    self.psi_tmp[:] = self.psi
  
    self.calc_exit_wave()
    
    self.psi_tmp[:] -= self.psi
    self.psi[:] -= self.psi_tmp
    
    self.psi[:] = xp.fft.fft2(self.psi,axes=(1,2))/np.sqrt(self.nx*self.ny)
    self.psi_replace_amplitude()
    self.psi[:] = xp.fft.ifft2(self.psi,axes=(1,2))*np.sqrt(self.nx*self.ny)
    
    self.psi[:] += self.psi_tmp
    
    self.decompose_new_prb()
    self.decompose_new_obj()
    
  def recon(self):
    self.calc_positions_and_dim()
    self.init_arrays()
    self.init_prb(self.focus_to_sample_distance_m)
    
    self.calc_exit_wave()
    
    if plot_result:
      fig, axs = plt.subplots(2, 2)
      
    for i in range(self.num_iterations):
    
      if self.algorithm == 'epie':
        self.iterative_recon_epie()
      elif self.algorithm == 'dm':
        self.iterative_recon_dm()
      else:
        raise NotImplementedError('Unknown algorithm')
      if plot_result and i%5==0:
        plot_complex_arrays(fig,axs,self.prb,self.obj,self.nx,f'Iteration {i}')
      
    


# propagate a wavefront using angular spectrum method
def propagate(array, pixel_size_m, wavelength_m, z_m):
    k = 2. * np.pi / wavelength_m

    nx, ny = array.shape

    spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(array))) / np.sqrt(nx*ny)

    dkx = 2.*np.pi / (nx * pixel_size_m)
    dky = 2.*np.pi / (ny * pixel_size_m)

    skx = dkx * nx / 2
    sky = dky * ny / 2

    kxx = xp.linspace(-skx, skx-dkx, nx)
    kyy = xp.linspace(-sky, sky-dky, ny)
    kx, ky = xp.meshgrid(kxx, kyy, indexing='ij')

    phase = xp.sqrt(k**2 - kx**2 - ky**2) * z_m
    spectrum *= xp.exp(1j*phase)

    array_prop = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(spectrum))) * np.sqrt(nx*ny)

    return array_prop


def plot_complex_arrays(fig,axs,prb, obj, crop_size, title):
    if xp.__name__ == 'cupy':
      prb = prb.get()
      obj = obj.get()

    # Probe amplitude
    axs[0, 0].imshow(np.abs(prb), cmap='viridis')
    axs[0, 0].set_title('Probe Amplitude')

    # Probe phase
    axs[0, 1].imshow(np.angle(prb), cmap='twilight')
    axs[0, 1].set_title('Probe Phase')

    obj_crop = obj[crop_size:-crop_size,crop_size:-crop_size]
    # Object amplitude
    im = axs[1, 0].imshow(np.abs(obj), cmap='viridis')
    axs[1, 0].set_title('Object Amplitude')
    im.set_clim([np.min(np.abs(obj_crop)),np.max(np.abs(obj_crop))])

    # Object phase
    im = axs[1, 1].imshow(np.angle(obj), cmap='twilight')
    axs[1, 1].set_title('Object Phase')
    im.set_clim([np.min(np.angle(obj_crop)),np.max(np.angle(obj_crop))])

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for title

    plt.draw()        # draw the figure
    plt.pause(0.001)  # pause to allow GUI event loop to process

if __name__=='__main__':
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    p = ptycho_recon_demo(os.path.join(_here, 'demo_81points.h5'), 800e-6, 50, 'dm')
    p.recon()
    plt.show()
