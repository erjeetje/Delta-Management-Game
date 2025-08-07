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
#from model.inpu.load_physics import phys_RMD1 , phys_test1
from model.inpu.load_physics import phys_RMD2

# =============================================================================
# load forcing conditions
# =============================================================================
from model.inpu.load_forcing_td import forc_RMD_game
#from model.inpu.load_forcing_td import forc_RMD4, forc_RMD5,  forc_RMD_fromSOBEK, forc_RMD_fromcsv, forc_RMD_fromMo, forc_RMD_fromJesse, forc_RMD20
#from model.inpu.load_forcing_td import forc_test1, forc_test2

# =============================================================================
# load geometry
# =============================================================================
#from model.inpu.load_geo_RMD import geo_RMD9, geo_RMD10
from model.inpu.load_geo_RMD import geo_RMD_game
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
geo_pars = geo_RMD_game()


#forc_pars = forc_RMD5()
#forc_pars = forc_RMD_fromJesse(33,18700,18750)
#forc_pars = forc_RMD20()
#forc_pars = forc_RMD_fromSOBEK('01-07-2021' , '01-08-2021')
# forc_pars = forc_RMD_fromcsv('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RM_data/MO_Q2122/','Q_daily_mean_Hag-Tie-Meg-Har-Gou_2021-2022_Bouke030524.csv',
#                         '01-01-2022', '31-12-2022')
# forc_pars = forc_RMD_fromMo('/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/RM_data/MO_Q2122/','Q_daily_mean_Hag-Tie-Meg-Har-Gou_2020.csv' )

#choose physical constants
phys_pars = phys_RMD2()

def set_forcing_beta(scenario="2017", timeseries_length=None):
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    input_dir = os.path.join(file_dir, "model", "inpu", "forcing_files")
    filename = '\\forcing_' + scenario + ".csv"

    #print(forc_pars)
    start_date = 0
    end_date = None
    if timeseries_length == "dummy":
        start_date = 224
        end_date = 233
    elif timeseries_length == "month":
        start_date = 212
        end_date = 257
    elif timeseries_length == "game_scenario":
        start_date = 212
        end_date = 257
    elif timeseries_length == "two_months":
        start_date = 213
        end_date = 271

    forc_pars = forc_RMD_game(input_dir, filename, start=start_date, end=end_date)
    print("next scenario is:", scenario)
    return forc_pars

def set_forcing(scenario="2017", timeseries_type="drought", timeseries_length=None):
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    input_dir = os.path.join(file_dir, "model", "inpu", "forcing_files")
    if timeseries_length is not None:
        filename = '\\forcing_' + scenario + "_" + timeseries_type + "_" + timeseries_length + ".csv"
    else:
        filename = '\\forcing_' + scenario + "_" + timeseries_type + ".csv"
    forc_pars = forc_RMD_game(input_dir, filename)
    print("next scenario is a", timeseries_type, "scenario of", scenario)
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