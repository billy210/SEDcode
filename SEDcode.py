import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyAstronomy.pyasl.asl.unred as fm_unred
from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.modeling import models
from astropy.coordinates import SkyCoord
import aplpy, atpy
import astropy.coordinates as coord
from astropy.io import ascii
from astropy import table
from astropy.io.votable import parse
import os
import seaborn as sns
from astropy import constants as const


#open all the needed files
datapath = '/home/billy/Mstars/scriptstopush'
tablefile = 'NGC2068mdwarflist.csv'


#####################################################
### Opens the VOtable containing the model spectra ##
#####################################################
# For the below code to open the votables from http://svo2.cab.inta-csic.es/theory/newov2/index.php
#
# we need to change the filetype from datafile to meta. so use the following sed code 
###CAREFULLY###
# Go to the folder outside the model folder, and run this as written. 
#
# *************If this is your first time running the code***********************
#      save a copy of the models in case you permabreak it with sed
#
#
#for filename in bt-nextgen-agss2009/*; do sed -i $filename -e s/datafile/meta/; done

def votable_to_pandas(votable_file):
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()





###################################
## Inputs needed for the code    ##
###################################

Rv=5.1 #nwe results for young star forming regions


#Optical photometry zero points in Jy Bessel(1979)
uzp=1810.0
bzp=4260.0
vzp=3640.0
rzp=3080.0
izp=2550.0

#IRAC and MIPS zero points in Jy
b1zp=280.9
b2zp=179.7
b3zp=115
b4zp=64.13
b5zp=7.17

#2mass zero points in Jy  
jzp=1594.0
hzp=1024.0
kzp=666.7
 
#zp=[uzp,bzp,vzp,rzp,izp,jzp,hzp,kzp,b1zp,b2zp,b3zp,b4zp,b5zp]
zp=[jzp,hzp,kzp]    #MUST MATCH LENGTH OF FILTERWHEEL TO WORK


#conversions
c=3.0E10        #cm/s
cang=3.0E18
cmicron = 3e14 #speed of light in micron/sec
micron=1.0E-4   #micron to anstrom 1 micron is 10^4 angstroms

jy=1e-23          #erg/s/cm2/Hz

#wavelengths
l1=0.3735   #U
l2=0.4443   #B
l3=0.5483   #V
l4=0.6855   #R
l5=0.86     #I
l6=1.235                #l6=1.22     #J
l7=1.662                #l7=1.63     #H
l8=2.159                #l8=2.22     #K
l9=3.6      #b1
l10=4.8     #b2
l11=5.6     #b3
l12=8.0     #b4
l13=24.0    #b5

#wavelen = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13]
wavelen=[l6,l7,l8]
  
#  Lum=Ls*Lsun     ;luminosity of the source in erg/s

#filterwheel=['U','B','V','R','I','J','H','K','B1','B2','B3','B4','B5']
filterwheel=['J','H','K'] #MUST MATCH LENGTH OF ZP TO WORK
fullfilters = ['J','H','K','3.6mag_paper2','4.5mag_paper2','5.8mag_paper2','8.0mag_paper2']



#########################################
###         Region Specifics           ##
#########################################


region = 'NGC2068' # Region name for giving the plot the correct title
if os.path.isdir(f'{datapath}/plots/{region}') == True:
    pass
else:
    os.mkdir(f'{datapath}/plots/{region}')


crossmatchtable = pd.read_csv(f'{datapath}/{tablefile}', sep=',', header = 0) #table of stars
d=415 #distance to source in pc
paper2 = 'megeath12' #paper/catalog source of Spitzer data


##########################################################################################################
## set up the initials such as Av, wavelengths used, and zeropoints which are listed above             ###
## filters are defined here as well allowing for plotting spitzer and jhk in different colors          ###
##########################################################################################################

outtable = pd.DataFrame() #outputs will be saved here


av = crossmatchtable['Av']
ebv = av/Rv                   #E_b-v needed for the dereddening function


wavelen = [l6,l7,l8,l9,l10,l11,l12]
wavelen = np.array(wavelen)
wvang = wavelen*1e4 #wavelengths converted to anstroms (since model has waves in angstroms)

zpjhk = [jzp,hzp,kzp]  #zeropoints for jhk fluxes
zpspit = [b1zp,b2zp,b3zp,b4zp] #zeropoints for spitzer fluxes
zptote = [jzp,hzp,kzp,b1zp,b2zp,b3zp,b4zp] #all zeropoints combined (total)

#myfilterwheel = ['J','H','K']
#spitzerwheel = ['3.6mag_paper2','4.5mag_paper2','5.8mag_paper2','8.0mag_paper2']

#fullfilters = ['J','H','K',f'3.6mag_{paper2}',f'4.5mag_{paper2}',f'5.8mag_{paper2}',f'8.0mag_{paper2}'] #Filters the code cycles through
fullfilters = [f'J',f'H',f'K',f'3.6mag_{paper2}',f'4.5mag_{paper2}',f'5.8mag_{paper2}',f'8.0mag_{paper2}'] 

for col in fullfilters:
    crossmatchtable[col][crossmatchtable[col] < 0] = np.nan



################################################################
###########Change the magnitudes to fluxes in Jy ###############
################################################################


#Need to convert JHK,spitzer columns from magnitudes into fluxes using zeropoint in Jys

for filter in fullfilters:      #select filter type
    crossmatchtable[filter] = zptote[fullfilters.index(filter)]/(10**(crossmatchtable[filter]/2.5)) 

#fluxes are now in Jy units
    
    
    
###############################################
#############Deredden the fluxes ##############
###############################################

#make sure wavelength from microns to angstroms as required by fm_unred input


#input = fluxes in Jy
#1 Jy = 10^-23 erg/s/cm2/Hz
#output = fluxes in erg/s/cm2/Hz

for filter in fullfilters:
    wvcol = np.squeeze(np.full((len(crossmatchtable[filter]),1), wvang[fullfilters.index(filter)]))
    crossmatchtable[f'dered{filter}'] = (fm_unred(wvcol, crossmatchtable[filter], ebv))*jy   #converts output from Jy to cgs
    crossmatchtable[filter] = crossmatchtable[filter]*jy         #puts non dereddened data in same cgs units as dereddened
    
    
    

#convert to lambda*F_lambda units (erg/s/cm2) needed for the alpha calculation
# To do this we multiplying by nu. This is because our flux above is F_nu, and nu*F_nu = lambda*F_lambda
# for reference nu=c/lambda

for filter in fullfilters:
    crossmatchtable[f'lfddered{filter}']=crossmatchtable[f'dered{filter}']*(cmicron/wavelen[fullfilters.index(filter)])
    crossmatchtable[f'lfd{filter}']=crossmatchtable[f'{filter}']*(cmicron/wavelen[fullfilters.index(filter)])
    
    
#get the fluxes in logspace not to plot, but for the alpha calc
for filter in fullfilters:
    crossmatchtable[f'loglfd{filter}']=np.log10(crossmatchtable[f'lfd{filter}']) 
    crossmatchtable[f'log{filter}']=np.log10(crossmatchtable[f'{filter}']) 
    

for row in range(len(crossmatchtable)):
    data = crossmatchtable
    title = data.loc[row,'ID']
    teff = data.loc[row,'Teff']    ######Teff to select model
    stteff =str(round(teff,-2))[:2]   #convert teff to string for file name
    stteffmodel =round(teff,-2)//100 #convert teff to string for model selection
    stlogg = 3.0 #str(round(lgng,-2))   #convert logg to string
    meta = '0.0' #nextgen needs the a here
    alpha = '0.0'
    if stteffmodel < 26:
        dir = '/home/billy/datafiles/bt-settl-agss/'
        name = dir+f'lte0{stteff}-{stlogg}-{meta}.BT-Settl.7.dat.xml'   
        model = votable_to_pandas(name)
    else:
        dir = '/home/billy/datafiles/bt-settl-agss/'
        name = dir+f'lte0{stteff}-{stlogg}-{meta}a+{alpha}.BT-Settl.7.dat.xml'   
        model = votable_to_pandas(name)
    #j = data.loc[row,f'lfdJ']   ##ergs/s/cm^2
    #h = data.loc[row,f'lfdH']
    #k = data.loc[row,f'lfdK']
    #s1 = data.loc[row,f'lfd3.6mag_{paper2}']
    #s2 = data.loc[row,f'lfd4.5mag_{paper2}']
    #s3 = data.loc[row,f'lfd5.8mag_{paper2}']
    #s4 = data.loc[row,f'lfd8.0mag_{paper2}']
    j = data.loc[row,f'lfdderedJ']   ##ergs/s/cm^2
    h = data.loc[row,f'lfdderedH']
    k = data.loc[row,f'lfdderedK']
    s1 = data.loc[row,f'lfddered3.6mag_{paper2}']
    s2 = data.loc[row,f'lfddered4.5mag_{paper2}']
    s3 = data.loc[row,f'lfddered5.8mag_{paper2}']
    s4 = data.loc[row,f'lfddered8.0mag_{paper2}']
    wj = wvang[0]              ##wavelengths in angstroms
    wh = wvang[1]
    wk = wvang[2]
    ws1 = wvang[3]
    ws2 = wvang[4]
    ws3 = wvang[5]
    ws4 = wvang[6]
    
    flamingosflux = np.array([j,h,k])    #JHK FLAMINGOS fluxes in erg/s/cm2
    spitzflux = np.array([s1,s2,s3,s4])  #spitzer fluxes in erg/s/cm2
    fwaves = np.array([wj,wh,wk])        # wavelength bands for JHK in angstroms
    swaves = np.array([ws1,ws2,ws3,ws4]) # wavelength bands for spitzer in angstroms
    model['lambdaFLUX'] = model['WAVELENGTH']* model['FLUX'] #Put the model in the same flux units as our values b/c "FLUX" unit="ERG/CM2/S/A"
    
    #Scale the model to bring the flux down to that of the measured values for those at photosphere (should exclude K)
    #scale1 is extrapolating the flux value at each of our band for the model spectra. need our waves in angstroms b/c the model waves are in A
    #scale1 = (np.interp(wj, model['WAVELENGTH'],model['lambdaFLUX'])+np.interp(wh, model['WAVELENGTH'],model['lambdaFLUX']))/2 #these are units ergs/s/cm^2
    scale1 = np.interp(wj, model['WAVELENGTH'],model['lambdaFLUX']) #these are units ergs/s/cm^2
    #scale2 = (j+h)/2 #these are units ergs/s/cm^2
    scale2 = j #these are units ergs/s/cm^2
    
    scaler = scale2/scale1
    
    scalewv = model['WAVELENGTH'] #Angstroms
    scalewv = scalewv/1e4 #model waves now in microns
    fwaves = fwaves/1e4 #convert JHK back to micron for plotting
    swaves = swaves/1e4 #convert spitzer back to micron for plotting
    scalefl = model['lambdaFLUX'] * scaler #Scale the model fluxes to match our photosphere measures
    

    
    logfluxes = np.log10(spitzflux)
    logwaves = np.log10(swaves)

    

    try:
        Mtype = data.loc[row,'Msubclass']
        #print(f'M{Mtype}')
        idx = np.isfinite(logwaves) & np.isfinite(logfluxes)
        coef = np.polyfit(logwaves[idx], logfluxes[idx], 1)
        alpha = coef[0]
        #print(f'{region} {title}')
        #print(f'spitzer fit = {alpha}')
        if alpha > 0.3:
            #print('Class I')
            data.loc[row,'class']= 'I'
            yso = 'I'
        elif (alpha < 0.3) & (alpha > -0.3):
            #print('Flat')
            data.loc[row,'class']= 'Flat'
            yso = 'Flat'
        elif (alpha < -0.3) & (alpha > -1.8):
            #print('Class II')
            data.loc[row,'class']= 'II'
            yso = 'II'
        elif (alpha < -1.8) & (alpha > -2.56):
            #print('Anemic')
            data.loc[row,'class']= 'Anemic'
            yso = 'Anemic'
        elif alpha < -2.56:
            #print('Class III')
            data.loc[row,'class']= 'III'
            yso = 'III'

    except:
        print(f'{title} is having trouble with polyfit, check inputs.')
        
    
    outtable.loc[row,'reg'] = region
    outtable.loc[row,'name'] = title
    outtable.loc[row,'SpT'] = Mtype
    outtable.loc[row,'lfdderedJ'] = j
    outtable.loc[row,'lfdderedH'] = h 
    outtable.loc[row,'lfdderedK'] = k 
    outtable.loc[row,f'lfddered3.6mag_{paper2}'] = s1
    outtable.loc[row,f'lfddered4.5mag_{paper2}'] = s2
    outtable.loc[row,f'lfddered5.8mag_{paper2}'] = s3
    outtable.loc[row,f'lfddered8.0mag_{paper2}'] = s4
    outtable.loc[row,'temp'] = teff
    outtable.loc[row,'alpha'] = alpha
    outtable.loc[row,'ysoclass'] = yso
    
    plt.figure(figsize=(6,7))
    plt.margins(0,0)
    plt.plot(scalewv,scalefl,zorder=1) #plot BT-Settl spectra in lambda vs lambda*F_lambda
    plt.scatter(fwaves,flamingosflux, color='magenta', marker='x', zorder=2, label='jhk') #plot jhk data in lambda vs lambda*F_lambda
    plt.scatter(swaves,spitzflux, color='orange', marker='x', zorder=3, label= 'Spitzer') #plot spitzer data in lambda vs lambda*F_lambda
    xmin = 0.1
    xmax = 100
    ymin = 1e-14
    ymax = 1e-9
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{region}-{title}-{yso}')
    plt.xlabel(f'Wavelength (microns)')
    plt.ylabel('Flux (erg/s/cm2)')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    #plt.savefig(f'{datapath}/plots/{region}/{title}.pdf')
    plt.savefig(f'{datapath}/plots/{region}/{title}sanity.png') #output plots of the results
    plt.clf()
    outtable.to_csv(f'{datapath}/plots/{region}/{region}tablesanity.csv')  #output table of final data
