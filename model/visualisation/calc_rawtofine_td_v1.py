# =============================================================================
# this file converts the 'raw' output of the model to physical quantities like salinity
# for the output of the time-dependent model
# we keep it here relatively simple, so only salinity is calculated
# TODO: build in checks that the output is realistic. 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


def calc_output(self):
    
    #some preparations
    self.Tvec = np.zeros(self.T)
    for t in range(self.T-1): self.Tvec[t+1] = np.sum(self.DT[:t+1])/(3600*24)
    nnp = np.arange(1,self.M)*np.pi #n*pi

    #discharge time series for each channel
    Q_all = {}
    for key in self.ch_keys: Q_all[key] = []
    for t in range(self.T): 
        Qnow = self.Qdist_calc((self.Qriv[:,t], self.Qweir[:,t], self.Qhar[:,t], self.n_sea[:,t]))
        for key in self.ch_keys: Q_all[key].append(Qnow[key])
    for key in self.ch_keys: Q_all[key] = np.array(Q_all[key])
    
    #do the operations for every channel
    for key in self.ch_keys:
        # =============================================================================
        # x coordinates for ploting
        # =============================================================================
        self.ch_outp[key]['px'] = np.zeros(np.sum(self.ch_pars[key]['nxn']))
        self.ch_outp[key]['px'][0:self.ch_pars[key]['nxn'][0]] = -np.linspace(np.sum(self.ch_gegs[key]['L'][0:]), np.sum(self.ch_gegs[key]['L'][0+1:]), self.ch_pars[key]['nxn'][0])
        for i in range(1,len(self.ch_pars[key]['nxn'])): self.ch_outp[key]['px'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] =\
            -np.linspace(np.sum(self.ch_gegs[key]['L'][i:]), np.sum(self.ch_gegs[key]['L'][i+1:]), self.ch_pars[key]['nxn'][i])
        tot_L = np.sum(self.ch_gegs[key]['L'])

        #cross section
        self.ch_outp[key]['CS'] = self.ch_pars[key]['H']*self.ch_pars[key]['b']
        self.ch_outp[key]['dl'] = self.ch_pars[key]['dl'].copy()
        # =============================================================================
        # subtidal salinity      
        # =============================================================================
        self.ch_outp[key]['sss'] = self.out[:,self.ch_inds[key]['totx'][0]*self.M:self.ch_inds[key]['totx'][-1]*self.M]
        self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sss'][:,self.ch_inds[key]['isb']] * self.soc_sca
        self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sss'][:,self.ch_inds[key]['isn']].reshape((self.T,self.ch_pars[key]['di'][-1],self.N)) * self.soc_sca
        self.ch_outp[key]['sp_st'] = np.sum(self.ch_outp[key]['sn_st'][:,:,:,np.newaxis] * np.cos(np.pi*self.mm[:,np.newaxis]*self.z_nd),2)
        #subtidal salinity
        self.ch_outp[key]['s_st'] = self.ch_outp[key]['sp_st'] + self.ch_outp[key]['sb_st'][:,:,np.newaxis]

        #salinity gradient, required for transports. 
        self.ch_outp[key]['sb_st_x'] = np.zeros((self.T,self.ch_pars[key]['di'][-1])) + np.nan #
        for dom in range(len(self.ch_pars[key]['di'])-1):
            self.ch_outp[key]['sb_st_x'][:,self.ch_pars[key]['di'][dom]+1:self.ch_pars[key]['di'][dom+1]-1] = (self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom]+2:self.ch_pars[key]['di'][dom+1]]\
                 -self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom]:self.ch_pars[key]['di'][dom+1]-2])/(2*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom]+1:self.ch_pars[key]['di'][dom+1]-1]*self.Lsc)   
            #also do the boundaries
            self.ch_outp[key]['sb_st_x'][:,self.ch_pars[key]['di'][dom]] = (-3*self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom]] +4*self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom]+1] -self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom]+2]) / (2*self.Lsc*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom]])
            self.ch_outp[key]['sb_st_x'][:,self.ch_pars[key]['di'][dom+1]-1] =(3*self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom+1]-1] -4*self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom+1]-2] +self.ch_outp[key]['sb_st'][:,self.ch_pars[key]['di'][dom+1]-3]) / (2*self.Lsc*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom+1]-1])
           
        #tidal velocity and salinity
        self.ch_outp[key]['uti'] = self.ch_pars[key]['ut'].copy()
        self.ch_outp[key]['sti'] = np.zeros((self.T,self.ch_pars[key]['di'][-1],self.nz), dtype = complex) + np.nan
        for t in range(self.T):
            tid_out = self.tidal_salinity(key , self.ch_outp[key]['sss'][t])
            temp = tid_out['st'].copy()
            for dom in range(self.ch_pars[key]['n_seg']):
                temp[self.ch_pars[key]['di'][dom] : self.ch_pars[key]['di'][dom] + len(tid_out['stci_x=-L'][dom])]     += tid_out['stci_x=-L'][dom]
                temp[self.ch_pars[key]['di'][dom+1] - len(tid_out['stci_x=0'][dom]) : self.ch_pars[key]['di'][dom+1] ] += tid_out['stci_x=0'][dom]
            self.ch_outp[key]['sti'][t] = temp

        # =============================================================================
        # transport components        
        # =============================================================================
        self.ch_outp[key]['TQ'] = Q_all[key][:,np.newaxis]*self.ch_outp[key]['sb_st'] #transport by mean current        
        self.ch_outp[key]['TE'] = np.sum(self.ch_outp[key]['sn_st']*(2*Q_all[key][:,np.newaxis,np.newaxis]*self.ch_pars[key]['g2'][np.newaxis,:,np.newaxis]*np.cos(nnp)/nnp**2 \
                              + self.ch_outp[key]['CS'][np.newaxis,:,np.newaxis]*self.ch_pars[key]['alf'][np.newaxis,:,np.newaxis]*self.ch_outp[key]['sb_st_x'][:,:,np.newaxis]*(2*self.ch_pars[key]['g4'][np.newaxis,:,np.newaxis]\
                              * np.cos(nnp)/nnp**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),2) #transport by vertically sheared current
        self.ch_outp[key]['TD'] = self.ch_outp[key]['CS'] *- self.ch_pars[key]['Kh']*self.ch_outp[key]['sb_st_x'] #transport by horizontal diffusion
        self.ch_outp[key]['TT'] = 1/4 * np.real(np.mean(np.conj(self.ch_outp[key]['sti'])*self.ch_outp[key]['uti'] + self.ch_outp[key]['sti'] * np.conj(self.ch_outp[key]['uti']),2)) * self.ch_outp[key]['CS'] #transport by tides

        # =============================================================================
        # calculate velocities and water level with hourly timesteps
        # =============================================================================

        # hourly time vector
        dth = 3600  # seconds in an hour
        #Tvec_new = np.arange(0, np.sum(self.DT) + dth, dth) / 86400
        # this removes the last timestep, giving 24 hours * X days output, and not 24+1 hours output for the last day.
        Tvec_new = np.arange(0, np.sum(self.DT), dth) / 86400

        # water level

        # subtract amplitude and phase
        eta_abs, eta_ang = np.abs(self.ch_pars[key]['eta'][0, :, 0]), np.angle(self.ch_pars[key]['eta'][0, :, 0])

        # interpolate to hourly
        eta_series = np.zeros((len(Tvec_new), self.ch_pars[key]['di'][-1])) + np.nan
        for x in range(self.ch_pars[key]['di'][-1]):
            eta_series[:, x] = np.real(eta_abs[x] * np.exp(1j * (self.omega * Tvec_new * 86400 + eta_ang[x])))

        # velocities
        # calculate subtidal velocities.
        u_st = Q_all[key][:, np.newaxis, np.newaxis] * self.ch_pars[key]['bH_1'][np.newaxis, :, np.newaxis] \
               * (self.ch_pars[key]['g1'][np.newaxis, :, np.newaxis] + 1 + self.ch_pars[key]['g2'][np.newaxis, :,
                                                                           np.newaxis] * self.z_nd ** 2) \
               + self.ch_pars[key]['alf'][np.newaxis, :, np.newaxis] * self.ch_outp[key]['sb_st_x'][:, :, np.newaxis] \
               * (self.ch_pars[key]['g3'][np.newaxis, :, np.newaxis] + self.ch_pars[key]['g4'][np.newaxis, :,
                                                                       np.newaxis] * self.z_nd ** 2 + self.ch_pars[key][
                      'g5'] * self.z_nd ** 3)

        # subtidal, only interpolate to hourly time step
        u_st_h = np.zeros((len(Tvec_new), self.ch_pars[key]['di'][-1], self.nz)) + np.nan
        for x in range(self.ch_pars[key]['di'][-1]):
            for z in range(self.nz):
                u_st_h[:, x, z] = np.interp(Tvec_new, self.Tvec, u_st[:, x, z])

        # tidal, same method as for water level
        u_ti_h = np.zeros((len(Tvec_new), self.ch_pars[key]['di'][-1], self.nz)) + np.nan
        for x in range(self.ch_pars[key]['di'][-1]):
            for z in range(self.nz):
                ut_abs, ut_ang = np.abs(self.ch_pars[key]['ut'][0, x, z]), np.angle(self.ch_pars[key]['ut'][0, x, z])
                u_ti_h[:, x, z] = np.real(ut_abs * np.exp(1j * (self.omega * Tvec_new * 86400 + ut_ang)))


        # save total water depth
        self.ch_outp[key]['htot'] = eta_series + self.ch_pars[key]['H']
        # save total velocity
        self.ch_outp[key]['utot'] = u_ti_h + u_st_h

        """
        # adjust for river and sea domains
        if self.ch_gegs[key]['loc x=-L'][0] == 's':
            print("water level output correction for", key)
            print('loc x=-L: ', self.ch_gegs[key]['loc x=-L'])
            htot = htot[:, self.ch_pars[key]['di'][1] + 1:]
            utot = utot[:, self.ch_pars[key]['di'][1] + 1:,:]
        if self.ch_gegs[key]['loc x=0'][0] == 's':
            print("water level output correction for", key)
            print('loc x=0: ', self.ch_gegs[key]['loc x=0'])
            htot = htot[:, :self.ch_pars[key]['di'][1] - 1]
            utot = utot[:,:self.ch_pars[key]['di'][1]-1, :]
        if self.ch_gegs[key]['loc x=-L'][0] == 'r':
            print("water level output correction for", key)
            print('loc x=-L: ', self.ch_gegs[key]['loc x=-L'])
            htot = htot[:, self.ch_pars[key]['di'][1] + 1:]
            utot = utot[:, self.ch_pars[key]['di'][1] + 1:, :]
        if self.ch_gegs[key]['loc x=0'][0] == 'r':
            print("water level output correction for", key)
            print('loc x=0: ', self.ch_gegs[key]['loc x=0'])
            htot = htot[:, :self.ch_pars[key]['di'][1] - 1]
            utot = utot[:, :self.ch_pars[key]['di'][1] - 1, :]
        """


        # =============================================================================
        # remove sea and river domain
        # =============================================================================
        #remove sea domain
        if self.ch_gegs[key]['loc x=0'][0] == 's':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_sea]+self.length_sea
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_sea]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,:-self.nx_sea]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][:-self.nx_sea]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,:-self.nx_sea]
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,:-self.nx_sea]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_sea]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][:-self.nx_sea]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_sea]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_sea]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_sea]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][:-self.nx_sea]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][:-self.nx_sea]

            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:,:-self.nx_sea]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:,:-self.nx_sea]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:,:-self.nx_sea]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:,:-self.nx_sea]
            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])
            self.ch_outp[key]['CS'] = self.ch_outp[key]['CS'][:-self.nx_sea]
            self.ch_outp[key]['dl'] = self.ch_outp[key]['dl'][:-self.nx_sea]

            # added
            #self.ch_outp[key]['htot'] = self.ch_outp[key]['htot'][:, :self.ch_pars[key]['di'][1] - 1]
            #self.ch_outp[key]['utot'] = self.ch_outp[key]['utot'][:, :self.ch_pars[key]['di'][1] - 1, :]
            #htot = htot[:, :self.ch_pars[key]['di'][1] - 1]
            #utot = utot[:, :self.ch_pars[key]['di'][1] - 1, :]
            self.ch_outp[key]['htot'] = self.ch_outp[key]['htot'][:, :-self.nx_sea]
            self.ch_outp[key]['utot'] = self.ch_outp[key]['utot'][:, :-self.nx_sea, :]

        elif self.ch_gegs[key]['loc x=-L'][0] == 's':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_sea:]
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_sea:]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,self.nx_sea:]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][self.nx_sea:]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,self.nx_sea:]
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,self.nx_sea:]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_sea:]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][self.nx_sea:]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_sea:]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_sea:]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_sea:]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][self.nx_sea:]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][self.nx_sea:]
            self.ch_outp[key]['CS'] = self.ch_outp[key]['CS'][self.nx_sea:]
            self.ch_outp[key]['dl'] = self.ch_outp[key]['dl'][self.nx_sea:]
            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:,self.nx_sea:]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:,self.nx_sea:]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:,self.nx_sea:]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:,self.nx_sea:]

            tot_L = np.sum(self.ch_gegs[key]['L'][1:])

            #added
            #htot = htot[:, self.ch_pars[key]['di'][1] + 1:]
            #utot = utot[:, self.ch_pars[key]['di'][1] + 1:, :]
            self.ch_outp[key]['htot'] = self.ch_outp[key]['htot'][:, self.nx_sea:]
            self.ch_outp[key]['utot'] = self.ch_outp[key]['utot'][:, self.nx_sea:, :]



        #remove river domain
        if self.ch_gegs[key]['loc x=0'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_riv]+self.length_riv
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_riv]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,:-self.nx_riv]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][:-self.nx_riv]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,:-self.nx_riv]
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,:-self.nx_riv]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_riv]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][:-self.nx_riv]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_riv]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_riv]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_riv]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][:-self.nx_riv]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][:-self.nx_riv]
            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:,:-self.nx_riv]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:,:-self.nx_riv]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:,:-self.nx_riv]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:,:-self.nx_riv]

            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])
            self.ch_outp[key]['CS'] = self.ch_outp[key]['CS'][:-self.nx_riv]
            self.ch_outp[key]['dl'] = self.ch_outp[key]['dl'][:-self.nx_riv]

            #added
            #htot = htot[:, :self.ch_pars[key]['di'][1] - 1]
            #utot = utot[:, :self.ch_pars[key]['di'][1] - 1, :]
            self.ch_outp[key]['htot'] = self.ch_outp[key]['htot'][:, :-self.nx_riv]
            self.ch_outp[key]['utot'] = self.ch_outp[key]['utot'][:, :-self.nx_riv, :]

        elif self.ch_gegs[key]['loc x=-L'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_riv:]
            #self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_riv:]
            self.ch_outp[key]['s_st'] = self.ch_outp[key]['s_st'][:,self.nx_riv:]
            #self.ch_outp[key]['ss_st'] = self.ch_outp[key]['ss_st'][self.nx_riv:]
            self.ch_outp[key]['sb_st'] = self.ch_outp[key]['sb_st'][:,self.nx_riv:]
            self.ch_outp[key]['sn_st'] = self.ch_outp[key]['sn_st'][:,self.nx_riv:]
            #self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_riv:]
            #self.ch_outp[key]['sb_st_x'] = self.ch_outp[key]['sb_st_x'][self.nx_riv:]
            #self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_riv:]
            #self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_riv:]
            #self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_riv:]
            #self.ch_outp[key]['s_ti'] = self.ch_outp[key]['s_ti'][self.nx_riv:]
            #self.ch_outp[key]['s_ti_r'] = self.ch_outp[key]['s_ti_r'][self.nx_riv:]
            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:,self.nx_riv:]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:,self.nx_riv:]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:,self.nx_riv:]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:,self.nx_riv:]

            tot_L = np.sum(self.ch_gegs[key]['L'][1:])
            self.ch_outp[key]['CS'] = self.ch_outp[key]['CS'][self.nx_riv:]
            self.ch_outp[key]['dl'] = self.ch_outp[key]['dl'][self.nx_riv:]

            # added
            #htot = htot[:, self.ch_pars[key]['di'][1] + 1:]
            #utot = utot[:, self.ch_pars[key]['di'][1] + 1:, :]
            self.ch_outp[key]['htot'] = self.ch_outp[key]['htot'][:, self.nx_riv:]
            self.ch_outp[key]['utot'] = self.ch_outp[key]['utot'][:, self.nx_riv:, :]

        # added: reshape and transpose to get output shape (days, location, hours)
        n_days = len(self.Tvec)
        htot_shape = self.ch_outp[key]['htot'].shape
        utot_shape = self.ch_outp[key]['utot'].shape
        htot_shape_new = (n_days, int(htot_shape[0] / n_days)) + htot_shape[1:]
        utot_shape_new = (n_days, int(utot_shape[0] / n_days)) + utot_shape[1:]
        self.ch_outp[key]['htot'] = np.reshape(self.ch_outp[key]['htot'], htot_shape_new)
        self.ch_outp[key]['htot'] = np.transpose(self.ch_outp[key]['htot'], axes=(0, 2, 1))
        self.ch_outp[key]['utot'] = np.reshape(self.ch_outp[key]['utot'], utot_shape_new)
        self.ch_outp[key]['utot'] = np.transpose(self.ch_outp[key]['utot'], axes=(0, 2, 1, 3))

        if key in ["Breeddiep", "Maas", "Waal"]:
            print("htot shape after:", self.ch_outp[key]['htot'].shape)
            print("utot shape after:", self.ch_outp[key]['utot'].shape)

        # =============================================================================
        # prepare some plotting quantities, x and y coordinates in map plots
        # =============================================================================
        self.ch_outp[key]['plot d'] = np.zeros(len(self.ch_gegs[key]['plot x']))
        for i in range(1,len(self.ch_gegs[key]['plot x'])): self.ch_outp[key]['plot d'][i] = \
            self.ch_outp[key]['plot d'][i-1]+ ((self.ch_gegs[key]['plot x'][i]-self.ch_gegs[key]['plot x'][i-1])**2 + (self.ch_gegs[key]['plot y'][i]-self.ch_gegs[key]['plot y'][i-1])**2)**0.5
        self.ch_outp[key]['plot d'] = (self.ch_outp[key]['plot d']-self.ch_outp[key]['plot d'][-1])/self.ch_outp[key]['plot d'][-1]*tot_L
        self.ch_outp[key]['plot xs'] = np.interp(self.ch_outp[key]['px'],self.ch_outp[key]['plot d'],self.ch_gegs[key]['plot x'])
        self.ch_outp[key]['plot ys'] = np.interp(self.ch_outp[key]['px'],self.ch_outp[key]['plot d'],self.ch_gegs[key]['plot y'])

        self.ch_outp[key]['points'] = np.array([self.ch_outp[key]['plot xs'],self.ch_outp[key]['plot ys']]).T.reshape(-1, 1, 2)
        self.ch_outp[key]['segments'] = np.concatenate([self.ch_outp[key]['points'][:-1], self.ch_outp[key]['points'][1:]], axis=1)



#calc_output(delta)








