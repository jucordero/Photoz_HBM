from astropy.table import Table

truth = Table.read('DES_Y1_Buzzard/Chinchilla-0Y1a_v1.6_truth.16.fits', hdu=1)
obs = Table.read('DES_Y1_Buzzard/Chinchilla-0Y1a_v1.6_obs.16.fits', hdu=1)

ind = [(obs['MAG_G'] < 99) & (obs['MAG_R'] < 99) & (obs['MAG_I'] < 99) & (obs['MAG_Z'] < 99) & (obs['MAG_G'] > 0) & (obs['MAG_R'] > 0) & (obs['MAG_I'] > 0) & (obs['MAG_Z'] > 0)]

truth = truth[ind]
obs = obs[ind]
print(len(obs), len(truth))

Z = truth['Z']

G_obs_mag = obs['MAG_G']
R_obs_mag = obs['MAG_R']
I_obs_mag = obs['MAG_I']
Z_obs_mag = obs['MAG_Z']
G_obs_err_mag = obs['MAGERR_G']
R_obs_err_mag = obs['MAGERR_R']
I_obs_err_mag = obs['MAGERR_I']
Z_obs_err_mag = obs['MAGERR_Z']

G_obs_flux = 10**(-0.4*(obs['MAG_G']-30))
R_obs_flux = 10**(-0.4*(obs['MAG_R']-30))
I_obs_flux = 10**(-0.4*(obs['MAG_I']-30))
Z_obs_flux = 10**(-0.4*(obs['MAG_Z']-30))
G_obs_err_flux = obs['MAGERR_G']*G_obs_flux*0.921
R_obs_err_flux = obs['MAGERR_R']*R_obs_flux*0.921
I_obs_err_flux = obs['MAGERR_I']*I_obs_flux*0.921
Z_obs_err_flux = obs['MAGERR_Z']*Z_obs_flux*0.921

names = ('Z', 'MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z', 'MAGERR_G', 'MAGERR_R', 'MAGERR_I', 'MAGERR_Z', 'FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z', 'FLUXERR_G', 'FLUXERR_R', 'FLUXERR_I', 'FLUXERR_Z')

t = Table([Z, G_obs_mag, R_obs_mag, I_obs_mag, Z_obs_mag, G_obs_err_mag, R_obs_err_mag, I_obs_err_mag, Z_obs_err_mag, G_obs_flux, R_obs_flux, I_obs_flux, Z_obs_flux, G_obs_err_flux, R_obs_err_flux, I_obs_err_flux, Z_obs_err_flux], names = names)
t.write('DES_Y1_Buzzard/Chinchilla-0Y1a_v1.6_matched.16.fits', format = 'fits', overwrite=True)
