import os
# =============================================================================
# code where I group the physical and geometric input for the delta in a class
# =============================================================================
# =============================================================================
# load general constants
# =============================================================================
from model.inpu.load_physics import phys_gen

# =============================================================================
# load physical constants
# =============================================================================
from model.inpu.load_physics import phys_RMD1 , phys_test1, phys_RMD2

# =============================================================================
# load forcing conditions
# =============================================================================
#from model.inpu.load_forcing_td import forc_RMD4, forc_RMD5,  forc_RMD_fromSOBEK, forc_RMD_fromcsv, forc_RMD_fromMo, forc_RMD_fromJesse
from model.inpu.load_forcing_td import forc_RMD20, forc_RMD_fromcsv_old
#from model.inpu.load_forcing_td import forc_test1, forc_test2

# =============================================================================
# load geometry
# =============================================================================
from model.inpu.load_geo_RMD import geo_RMD9, geo_RMD10, geo_RMD_game
#from model.inpu.load_geo_test import geo_test1
#from model.inpu.funn import geo_fun
# =============================================================================
# choose final numbers
# =============================================================================

#date_start, date_stop = '2008-01-01' , '2010-11-01'
'''
#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_fun()
#choose forcing.
forc_pars = forc_fun()
#choose physical constants
phys_pars = phys_test1()
'''
#choose physical constants
constants = phys_gen()
#choose geometry
#geo_pars = geo_RMD9()
geo_pars = geo_RMD_game()
#choose forcing.
#forc_pars = forc_RMD5()
#forc_pars = forc_RMD_fromJesse(33,18700,18750)
#forc_pars = forc_RMD20()
#forc_pars = forc_RMD_fromSOBEK('01-07-2021' , '01-08-2021')
# forc_pars = forc_RMD_fromcsv('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RM_data/MO_Q2122/','Q_daily_mean_Hag-Tie-Meg-Har-Gou_2021-2022_Bouke030524.csv',
#                         '01-01-2022', '31-12-2022')
# orc_pars = forc_RMD_fromMo('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RM_data/MO_Q2122/','Q_daily_mean_Hag-Tie-Meg-Har-Gou_2020.csv' )
#choose physical constants
phys_pars = phys_RMD2()

def set_forcing(scenario="2017"):
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    input_dir = os.path.join(file_dir, "model", "inpu", "forcing_files")
    if scenario == "2017":
        forc_pars = forc_RMD_fromcsv_old(input_dir,
                                         r'\forcing_2017_dummy.csv')
    elif scenario == "2018":
        forc_pars = forc_RMD_fromcsv_old(input_dir,
                                         r'\forcing_2018_dummy.csv')
    elif scenario == "2100le":
        forc_pars = forc_RMD_fromcsv_old(input_dir,
                                         r'\forcing_2100le_dummy.csv')
    elif scenario == "2100he":
        forc_pars = forc_RMD_fromcsv_old(input_dir,
                                         r'\forcing_2100he_dummy.csv')
    else:
        print("unknown scenario, no update to forcing conditions")
        return
    return forc_pars

'''
#choose physical constants
constants = phys_gen()
#choose geometry
geo_pars = geo_test1()
#choose forcing.
forc_pars = forc_test2()
#choose physical constants
phys_pars = phys_test1()
#'''