def raw_reader(filename):
    import os
    """reads the raw voltage data from MPMS .raw-files and converts it via fit to magnetic moments
    INPUT:
        filename: strinng, path of the file to be read
    """
    rawflag = True
    print('Opening ' + filename + '...')
    data_raw = np.genfromtxt(filename, skip_header=31, delimiter=',',\
                             usecols=[0, 2, 3, 4, 5, 7, 8])
    """
    0: time                -> data[:, 0]
    2: field in Oe         -> data[:, 1]
    3: starting temperature-> data[:, 2]
    4: endtemperature      -> data[:, 3]
    5: scan number         -> data[:, 4]
    7: z-position          -> data[:, 5]
    8: measured voltage    -> data[:, 6]
       sensitivity factor  -> data[:, 7]
       measurement number  -> data[:, 9]"""

    data_raw = np.c_[data_raw, np.ones((data_raw.shape[0], 2))]
    print('Raw data read.')
    if os.path.exists(filename.replace('raw', 'diag')):
        diagflag = True
        print('Diagnostic file ' + filename.replace('raw', 'diag')\
              + ' available. Gathering sensitivity information...')
        data_diag = np.genfromtxt(filename.replace('raw', 'diag'), skip_header=27, delimiter=',',\
                                  usecols=[0, 24, 25])
    else:
        print('No diagnostic file found. Sensitivities will be guessed - the magnetic moment will\
              not be quantitatively correct!')
        diagflag = False

    print('{0:g} data sets found'.format(np.unique(data_raw[:,0]).shape[0]))
    number = 0 #number of measurement
    for time in np.unique(data_raw[:, 0]):
        mask_diag = np.where(data_raw[:, 0] == time)
        for scan_number in np.unique(data_raw[mask_diag, 4]):
            mask_raw = np.where((data_raw[:, 0] == time) & (data_raw[:, 4] == scan_number)) #same time and correct scan number
            data_raw[mask_raw, 8] = number #plugging in the overall numberof this measurement
            if diagflag:
                diag_mask = np.where(data_diag[:, 0] == time)
                line = data_diag[diag_mask][scan_number -1]
                data_raw[mask_raw, 7] = _sens_quotient(line[1:]) #plugging in the sensitivity factor

            """----------Calculate the fit----------"""
            #getting the average field (should be constant)
            field = data_raw[mask_raw][0, 1]
    
            #getting the average temperature (should be constant)
            temperature = data_raw[mask_raw][0, 2]
    
            #getting the average sensitivity factor (should be constant)
            sf = data_raw[mask_raw, 7].mean()
    
            #getting the positions and reshaping the array for simplicity
            z = data_raw[mask_raw, 5]
            z = z.reshape(z.shape[1])
#    
#            #getting the voltages and reshaping the array for simplicity
            U = data_raw[mask_raw,6]
            U = U.reshape(U.shape[1])
            
            #get estimates for the fir parameter
            p0 = init_params( np.c_[z, U] )

            p = fit(p0, z, U)#leastsq(einzelresiduen, p0, args = (z, U), full_output = 1)[0]
            moment = _UtoM(p[1])/sf*1e-3 #dividing by the sensitivity factor and bring the moment to SI-units
            new_data = np.atleast_2d([time, field, temperature, moment])

            try:
                data = np.r_[ data, new_data ]
            except:
                data = new_data
            
            number += 1
    return data, data_raw
    
def _UtoM(voltage):
    """Converts a voltage reading into a magnetic moment (sensitivity factor still has to be applied)
    the multipliers are for a specific instrument only and have to be changed accordingly 
    
    input: float or array

    returns: same type as input
   
    """ 
    rso_cal_fac   = 1.0140
    long_reg_fac  = 1.825
    squid_cal_fac = 7741.139
    corr_fac      = 0.9125
    moment = voltage * rso_cal_fac * long_reg_fac / squid_cal_fac / corr_fac
    return moment

def _sens_quotient(arg):
    """
    input: range code and gain code    
    """
#    moment = np.array([.000125,.00025,.000625,.00125,.0025,.00625,.0125,.025,.0625,.125, .25,.625,1.25])
#    factor = np.array([     10,     5,      2,     1,   .5,    .2,   .1, .05,  .02, .01,.005,.002,.0001])
    range_factors = [1., 10., 100., 1000.]
    gain_factors  = [1.,  2.,   5.,   10.]            
    return gain_factors[int(arg[1])] / range_factors[int(arg[0])]

def init_params(data):
    """calculates initial parameters for a given raw dataset
    the fitting function uses number of data points as time
    estimate the background"""
    B1 = (data[-1,1] - data[0,1]) / data.shape[0]
    B0 = (data[:,1]).mean()
    #
    centerind = abs(data[:,1] - drift((0,0,B0,B1), data[:, 0])).argmax()

    center = data[centerind, 0]
    cleandata = (data[:,1] - B0 - B1*np.arange(data.shape[0]))
    amplitude = (max(cleandata) - min(cleandata))/2.66 #magic number from geometry of SQUID
    
    return np.array([center, amplitude, B0, B1])

def voltage(p, z):
    """Calculates the signal of a diploe in the MPMS-coils"""
    lam = 1.519 #in cm
    R=0.97       #in cm
    sum1 = 2*(R**2 +         (z - p[0])**2)**(-3/2.)                                
    sum2 =  -(R**2 + ( lam + (z - p[0]))**2)**(-3/2.)
    sum3 =  -(R**2 + (-lam + (z - p[0]))**2)**(-3/2.)
    return p[1]*(sum1+sum2+sum3)

def drift(p, z):
    """approximates the signal drift in time as linear function of the index"""
    t = np.arange(z.shape[0]) #the drift should be given by time but z is not monotonous in time
    return p[2] + p[3]*t

def fit(p0, z, U):
    """performs a least square fit to get the correct parameters to describe the signal of the MPMS"""
    def einzelresiduen(p, z, U):
        sample_signal     = voltage(p ,z)
        drift_signal      = drift(p,z)
        return U - sample_signal - drift_signal

    p = leastsq(einzelresiduen, p0, args = (z, U), full_output = 1)[0]
    return p
                
def _cleanup(self, kind = 'gaussian'):
    factors = np.array([5, 4, 2.5, 2, 1])    
    decimals = np.array([1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3])
    factors = ( factors[:,None] * decimals ).reshape( decimals.shape[0] * factors.shape[0], )
    """tries to clean up the fitted data if there has been no diag-file"""
    if kind == "gaussian":
        def kernel(point, data, h):
            eff_diff = np.dot( data - point , np.array([1./1000, 1]) ) #the effective different between the points - field differences should be scaled down
            return np.exp(-.5*( eff_diff / h )**2)
        
        def localfit(point, data, h):
            values = kernel(point[:-1], data[:,:-1], h)
            values /=np.sum(values)
            return np.dot(values.reshape(values.shape[0],), data[:,-1])
        
        for index, point in enumerate(self.data[ :, 1: ]):
            if index > 0:
                newpoints = factors * point[-1]
                prediction = localfit(point, self.data[:index, 1:], .5)
                idx = abs( newpoints - prediction ).argmin()
                self.data[index, 3] = newpoints[idx]
#                    self.data[]

    
    
#def rawdata(self, n):
#    """gives the rawdata of the n-th measurement"""
#    """should be callable with certain field and temperature parameters"""
#    """starting with n for now. n as non-zero index"""
#    try:
#        mask = np.where(self.data_raw[:,8] == n)
#        return self.data_raw[mask]
#    except:
#        print 'No raw data available'

class chooseablepoints:
    def __init__(self, df, abscissa = 'temperature'):
        self.flag = False
        """
        Input:
            df: dataframe with magnetic field, temperature, magnetic moment
            """
        self.df = df

        self.abscissa = abscissa
        self.ordinate = 'magnetic moment'
        """initialize the fitparameter"""
        self.p = [0, 0, 0, 0]
        pnames = ['z0', 'V', 'B0', 'B1']

        """prepare the figure and axes---"""
        self.fig = plt.figure(figsize=(13,12))
        self.ax1 = self.fig.add_axes([0.1,0.65,.4,.25])
        self.ax1.xaxis.set_label_coords(.5, -.15)
        self.ax1.yaxis.set_label_coords(-.08, .5)

        self.ax2 = self.fig.add_axes([0.1,0.3,.4,.25])
        self.ax2.set_xlabel('z-position in cm')
        self.ax2.xaxis.set_label_coords(.5, -.15)
        self.ax2.set_ylabel('signal in V')
        self.ax2.yaxis.set_label_coords(-.08, .5)
        plt.setp(self.ax2.get_xticklabels(), visible=False) #looks nicer if the axis starts blank
        plt.setp(self.ax2.get_yticklabels(), visible=False)#looks nicer if the axis starts blank

        self.radio_x_axis = self.fig.add_axes([0.8, 0.65, .2, .2], axisbg=(0.7, 0.7, 0.9))
        self.radio_x_buttons = widgets.RadioButtons(self.radio_x_axis, self.df.columns, active = list(self.df.columns).index('temperature'))
        self.radio_x_buttons.on_clicked(self.on_clicked_radiox)
        self.radio_y_axis = self.fig.add_axes([0.8, 0.45, .2, .2], axisbg=(0.9, 0.7, 0.9))
        self.radio_y_buttons = widgets.RadioButtons(self.radio_y_axis,  self.df.columns, active = list(self.df.columns).index('magnetic moment'))
        self.radio_y_buttons.on_clicked(self.on_clicked_radioy)

        self.bounds_to_change = 'temperature' 
        self.radio_bounds_axis = self.fig.add_axes([0.8, 0.25, .2, .2], axisbg=(1, 0.6, 0.0))
        self.radio_bounds_buttons = widgets.RadioButtons(self.radio_bounds_axis, self.df.columns, active = list(self.df.columns).index(self.bounds_to_change))
        self.radio_bounds_buttons.on_clicked(self.on_clicked_radio_bounds)

        self.bounds = dict((name, [min(self.df[name]), max(self.df[name])]) for name in self.df.columns)
        self.overall_bounds = self.bounds.copy()
        self.current_line_bounds = self.overall_bounds['temperature']
        
        self.bounds_axis = self.fig.add_axes([0.6, 0.3, .1, .6], axisbg=(1, 0.6, 0.0))
        self.bounds_axis.set_xticks([])
        
        self.bounds_axis.set_yticks([])
        self.bounds_axis.set_xlim((-.1,.1))
        self.bounds_axis.set_ylim((-.1,1.1))
        self.change_bounds=0
#        self.bounds_axis.add_patch(patches.Rectangle((-0.1, -0.1),0.2,0.1,))
#        self.bounds_axis.add_patch(patches.Rectangle((-0.1, 1.0),0.2,0.1,))
        self.bound_line = [self.bounds_axis.axhline(y=0, linewidth=5, color='k'),
                           self.bounds_axis.axhline(y=1, linewidth=5, color='k')]

        self.bound_text = [self.bounds_axis.text(.1, 0, '{0:g}{1}'.format(self.current_line_bounds[0], self.df.units[self.bounds_to_change]), ha = 'left', va='center'),
                           self.bounds_axis.text(-.1, 1, '{0:g}{1}'.format(self.current_line_bounds[1], self.df.units[self.bounds_to_change]), ha = 'right', va='center')]
        
        for side in self.ax2.spines: 
            self.ax2.spines[side].set_color('red')
            self.ax2.spines[side].set_linewidth(4)

        self.fit_but_ax = self.fig.add_axes([0., .95, .1, .05])
        self.fit_button = widgets.Button(self.fit_but_ax, '(f)it', color='white', hovercolor='white')
        self.fit_button.on_clicked(self.fit_button_action)

        self.del_but_ax = self.fig.add_axes([0.1, .95, .1, .05])
        self.del_button = widgets.Button(self.del_but_ax, 'de(l)ete', color='white', hovercolor='white')
        self.del_button.on_clicked(self.del_button_action)

        self.reset_but_ax = self.fig.add_axes([0.2, .95, .1, .05])
        self.reset_button = widgets.Button(self.reset_but_ax, 'r(e)eset', color='white', hovercolor='white')
        self.reset_button.on_clicked(self.reset_button_action)

        self.add_but_ax = self.fig.add_axes([0.3, .95, .1, .05])
        self.add_button = widgets.Button(self.add_but_ax, 'enable add\n(shift)', color='white', hovercolor='white')
        self.add_button.on_clicked(self.add_button_action)

        self.save_but_ax = self.fig.add_axes([0.4, .95, .1, .05])
        self.save_button = widgets.Button(self.save_but_ax, '(s)ave', color='white', hovercolor='white')
        self.save_button.on_clicked(self.save_button_action)

        self.info_ax = self.fig.add_axes([.8, .85, .2, .1], axisbg = 'g', alpha=.5)
        self.info_ax.set_xticks([])
        self.info_ax.set_yticks([])
        self.info_ax.patch.set_alpha(0.5)
        self.info_text = self.info_ax.text(.1,.5,'', verticalalignment='center', horizontalalignment='left')
#        self.infofield = self.fig.text(.1, .8, 'boxed italics text in data coords', style='italic',
#        bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
        self.fig.canvas.draw()
            
        self.slider_axes = []
        self.slider = []
        self.slider_bounds = []
        for i in range(len(self.p)):
            self.slider_axes.append(self.fig.add_axes([.05, .025 + i*.05, .9, .025]))
            self.slider.append(widgets.Slider(self.slider_axes[-1], pnames[i], 0, 1, valinit=0.5, 
                                              valfmt='%1.2f', closedmin=True, closedmax=True, 
                                              slidermin=None, slidermax=None, dragging=True))

            self.slider_bounds.append([0,1])


        """------set the flags------"""
        self.add_enabled = False

        """------------------------------"""
        self.chosen_slider = None #Flag that tells us which slider has been chosen

        
        self.chosen_inds = []
        self.disregarded = {}
        self.update_data_plot()
        self.update_chosen_plot()

        self.raw_plts = []
        self.raw_plts_not = []
        self.fit_plts = []

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.data_pick = self.fig.canvas.mpl_connect('pick_event', self.onpick_data)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.keypress = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.keyrelease = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)


        plt.show()

    def add_button_action(self, event):
        self.add_enabling()
        
    def add_enabling(self):
        """enables to add data points and changes the color of the corresponding button"""
        self.add_enabled = not self.add_enabled
        if self.add_enabled:
            #change the button color
            self.add_button.color = 'gray'
            #change the button color when hovering
            self.add_button.hovercolor='gray'
            #change the curent color of the button (color and hover are only
            #taken into accountonly change when mouse moves)
            self.add_button.ax.set_axis_bgcolor('gray')
        else:
            #change the button color
            self.add_button.color = 'white'
            #change the button color when hovering
            self.add_button.hovercolor='white'
            #change the curent color of the button (color and hover are only
            #taken into accountonly change when mouse moves)
            self.add_button.ax.set_axis_bgcolor('white')
        self.fig.canvas.draw()
        
    """---------functions for the radiobuttons (choose x and y of plot"""
    def on_clicked_radiox(self, label):
        self.abscissa = label
        self.update_data_plot()
        self.update_chosen_plot()
        self.update_info_plot()
        self.ax1.relim()      # make sure all the data fits
        self.ax1.autoscale()
        self.fig.canvas.draw()

    def on_clicked_radioy(self, label):
        self.ordinate = label
        self.update_data_plot()
        self.update_chosen_plot()
        self.update_info_plot()
        self.ax1.relim()      # make sure all the data fits
        self.ax1.autoscale()
        self.fig.canvas.draw()
#
    def update_data_plot(self):
        self.ax1xlabel = self.ax1.set_xlabel(self.abscissa + ' in ' + self.df.units[self.abscissa])
        self.ax1xlabel.set_color('blue')
        self.ax1ylabel = self.ax1.set_ylabel(self.ordinate + ' in ' + self.df.units[self.ordinate])
        self.ax1ylabel.set_color('m')
        try:
            condition = np.all(np.array([(self.df[name] >= self.bounds[name][0]) & (self.df[name] <= self.bounds[name][1]) for name in self.df.columns]), axis =0)
            #condition = np.all(np.array([(ar[name] > ar.c.bounds[name][0]) & (ar[name] < ar.c.bounds[name][1]) for name in ar.columns]), axis =0)
            self.data_plt.set_data(self.df[self.abscissa].where(condition), self.df[self.ordinate].where(condition))
        except Exception as error:
            self.data_plt,   = self.ax1.plot(self.df[self.abscissa], self.df[self.ordinate], 'bo', markersize = 4, picker = 8)

    def update_chosen_plot(self):
        x = self.df[self.abscissa][np.array(self.chosen_inds, dtype  = int)]
        y = self.df[self.ordinate][np.array(self.chosen_inds, dtype  = int)]
        try:
            self.chosen_plt.set_data(x, y)
        except:
            self.chosen_plt, = self.ax1.plot([], [], 'ro', markersize = 10, markeredgewidth = 3, fillstyle = 'none')

    def z_fine(self, z):
        #returns a finer version of z with holes filled
        z_fine = np.arange(z.min(), z.max(), .01)
        if z.min() < z[0]: #if we did not start the scan in the middle
            z_fine = np.r_[np.arange(z[0], z.min(), -.01), z_fine]
        if z.max() > z[-1]: #if we did not start the scan in the middle
            z_fine = np.r_[z_fine, np.arange(z.max(), z[-1], -.01)]
        return z_fine

    def update_info_plot(self):
        try:
            x, y = self.df[self.abscissa][self.info_index], self.df[self.ordinate][self.info_index]
            try:
                self.info_plt.set_data(x, y)
            except:
                self.info_plt, = self.ax1.plot(x, y, 'gs', markersize = 10, markeredgewidth = 4, fillstyle = 'none', alpha =.5)
            info = self.df.loc[self.info_index]
            info_string = 't = {0:g} s\nB = {1:g} T\nT = {2:g} K\nm = {3:g} Am^2'.format(info[0], info[1], info[2], info [3])
            self.info_text.set_text(info_string)
        except Exception as error:
            pass
        
    def on_key_press(self, event):
        if event.key == 'shift':
            if not self.add_enabled:
                self.add_enabling()
        elif event.key == 'l':
            self.delete_data()
        elif event.key == 'f':
            self.fit_data()
        elif event.key == 'e':
            self.reset_data()
        elif event.key == 's':
            self.save_data()
        sys.stdout.flush()           

    def on_key_release(self, event):
        if event.key == 'shift':
            if self.add_enabled:
                self.add_enabling()

    def onpick_data(self, event):
        self.flag = not self.flag
        index = event.ind.min()
        if event.mouseevent.inaxes == self.ax1:
            if self.add_enabled: #if the shift key is pressed
                #reset the sliders
                for slider in self.slider:
                    slider.reset()
            
                self.update_raw(index)
                self.update_chosen_plot()
            else:
                self.info_index = index
                self.update_info_plot()
        elif event.mouseevent.inaxes == self.ax2:
            """---remove or add points to the fitting data depending on whether or not shift is pressed---"""
            #if shift is not pressed, we want to remove points
            if not self.add_enabled:
                if event.artist in self.raw_plts:
                    x, y = event.artist.get_data()
                    artist_ind = next(ind for ind, artist in enumerate(self.raw_plts) if artist == event.artist)
                    destination = self.raw_plts_not[artist_ind]
                    x_neu, y_neu = destination.get_data()
                    x_neu, y_neu= np.append(x_neu, x[index]), np.append(y_neu, y[index])
                    destination.set_data((x_neu, y_neu))
                    indices = np.r_[:index, index + 1:x.shape[0]]
                    event.artist.set_data((x[indices], y[indices]))
            elif self.add_enabled:
                if event.artist in self.raw_plts_not:
                    x, y = event.artist.get_data()
                    artist_ind = next(ind for ind, artist in enumerate(self.raw_plts_not) if artist == event.artist)
                    destination = self.raw_plts[artist_ind]
                    x_neu, y_neu = destination.get_data()
                    x_neu, y_neu= np.append(x_neu, x[index]), np.append(y_neu, y[index])
                    destination.set_data((x_neu, y_neu))
                    indices = np.r_[:index, index + 1:x.shape[0]]
                    event.artist.set_data((x[indices], y[indices]))
        self.fig.canvas.draw()

    def update_fit_est(self):
        x, y = self.raw_plts[-1].get_data()
#        data = self.df.raw_data[self.df.raw_data[:,8] == index]
        signal = voltage(self.p, x) + drift(self.p, x)
        try:
            self.fit_est.set_xdata(x)
            self.fit_est.set_ydata(signal)
        except:
            self.fit_est, = self.ax2.plot(x, signal, 'g-', alpha=.3, linewidth = 5)

    def update_raw(self, index):
        self.erase_fit_plts()
        if index in self.chosen_inds:
            #find the position of "index" in the list of displayed data
            delind = next(delind for delind, value in enumerate(self.chosen_inds) if value == index)
            #remove it from the list
            self.chosen_inds.pop(delind)
            #remove the corresponding plot
            self.raw_plts[delind].remove()
            #remove the entry in the list of plots
            self.raw_plts.pop(delind)
            #if we have no more chosen raw data to show
            if len(self.chosen_inds) == 0:
                self.fit_est.set_data([[],[]])
                return

            else:
                #set the index as the last on the list
                index = self.chosen_inds[-1]
                z, U = self.raw_plts[-1].get_data()

        else:       
            plt.setp(self.ax2.get_xticklabels(), visible=True)
            plt.setp(self.ax2.get_yticklabels(), visible=True)
            #find the corresponding raw-data
            data = self.df.raw_data[self.df.raw_data[:,8] == index]
            mask = self.outside_boundaries(data[:,5])
            data = data[mask]
            #plot it 
            plot, = self.ax2.plot(data[:,5], data[:,6], 'bo', markersize = 4, picker = 5, alpha = .2)
            #append the plot to the list of raw-data plots
            self.raw_plts_not.append(plot)
            #find the corresponding raw-data
            data = self.df.raw_data[self.df.raw_data[:,8] == index]
            z, U = data[:, 5], data[:, 6]
            #plot it 
            plot, = self.ax2.plot(z, U, 'bo', markersize = 4, picker = 5)
            #append the plot to the list of raw-data plots
            self.raw_plts.append(plot)
            #append the index to the list of indices to keep track of what is shown
            self.chosen_inds.append(index)
        
        self.ax2.relim()      # make sure all the data fits
        self.ax2.autoscale()
        self.p = init_params(np.c_[z, U])
        self.update_fit_est()
        self.set_slider_bounds(z, U)
        
    def erase_fit_plts(self):
        for plot in self.fit_plts:
            try:
                plot.remove()
            except Exception:
                pass
        self.fit_plts = []

    def within_boundaries(self, z):
        return np.ones(z.shape[0], dtype = bool)

    def outside_boundaries(self, z):
        return np.zeros(z.shape[0], dtype = bool)

    def button_flash(self, button, color, time):
        """lets the button flash in the chosen color for a given time"""
        original_color = button.color
        button.color = color
        button.hovercolor = color
        button.ax.set_axis_bgcolor(color)
        plt.pause(time)
        button.color = original_color
        button.hovercolor = original_color
        button.ax.set_axis_bgcolor(original_color)
        self.fig.canvas.draw()  


    """---------------------------buttons for reset, fit and delete---------------------------"""
    def fit_button_action(self, event):
        self.fit_data()

    def fit_data(self):
        #let the button flash quickly to acknowledge its use
        self.button_flash(self.fit_button, 'gray', .05)
        self.erase_fit_plts()
        for i, index in enumerate(self.chosen_inds):
            #get the sensitivity factor
            sf = self.df.raw_data[self.df.raw_data[:,8] == index][0, 7]
            z = self.raw_plts[i].get_xdata() #get the data directly from the plot
            U = self.raw_plts[i].get_ydata() #get the data directly from the plot
            p = fit(self.p, z, U)
            z_neu = self.z_fine(z)
            p[3] *= 1. * z.shape[0] / z_neu.shape[0]
            signal = voltage(p, z_neu) + drift(p, z_neu)
            plot, = self.ax2.plot(z_neu, signal, 'r-')
            self.fit_plts.append(plot)
            m = _UtoM(p[1]) / sf * 1e-3 #dividing by the sensitivity factor and bring the moment to SI-units
            self.df['magnetic moment'][index] = m
        self.update_data_plot()
        self.update_chosen_plot()
#        self.update_info_plot(self, index):
        self.update_chosen_plot()
        self.update_info_plot()
        self.fig.canvas.draw()

    def reset_button_action(self, event):
        self.reset_data()

    def reset_data(self):
        #let the button flash quickly to acknowledge its use
        self.button_flash(self.reset_button, 'gray', .05)
        for index in self.chosen_inds[::-1]:
            delind = next(delind for delind, value in enumerate(self.chosen_inds) if value == index)
            #remove it from the list
            self.chosen_inds.pop(delind)
            #remove the corresponding plot
            self.raw_plts[delind].remove()
            #remove the entry in the list of plots
            self.raw_plts.pop(delind)
        self.update_chosen_plot()
        self.erase_fit_plts()
        try:
            self.fit_est.set_data([[],[]])
        except:
            pass
        self.update_data_plot()
        self.fig.canvas.draw()  

    def save_button_action(self, event):
        self.save_data()
        
    def save_data(self):
        """saves the data and (if available) the raw data to file"""
        #let the button flash quickly to acknowledge its use
        self.button_flash(self.save_button, 'gray', .05)
        path = self.df.path
        if not '_analyzed' in path:
            #if there is an ending
            if len(path.split('.')[-1]) < 5:
                ending = '.' + path.split('.')[-1]
            #remove the ending
            basepath = path.strip(ending)
            path = basepath + '.analyzed' + '.data'
        with open(path, 'w') as datei:
            datei.write('#DATA' + '\n')
            header = ','.join(self.df.columns)
            datei.write('#' + header + '\n')
            #get the used units and write them to the file
            units = []
            for key in self.df.columns:
                units.append(self.df.units[key])
            units = ','.join(units)
            datei.write('#' + units + '\n')
            for index in range(self.df.shape[0]):
                line = ','.join(map(str, self.df.iloc[index])) + '\n'
                datei.write(line)
            try:
                header = ','.join(self.df.raw_header)
                datei.write('#RAWDATA' + '\n')
                datei.write('#' + header + '\n')
                for line in self.df.raw_data:
                    datei.write(','.join(map(str, line)) + '\n')
            except:
                pass
            
    def del_button_action(self, event):
        self.delete_data()
        
    def delete_data(self):
        """deletes the chosen data from the dataframe (data AND raw_data)"""
        #let the button flash quickly to acknowledge its use
        self.button_flash(self.del_button, 'gray', .05)
        indices_to_delete = sorted(self.chosen_inds)
        for index in self.chosen_inds[::-1]:
            delind = next(delind for delind, value in enumerate(self.chosen_inds) if value == index)
            #remove it from the list
            self.chosen_inds.pop(delind)
            #remove the corresponding plot
            self.raw_plts[delind].remove()
            #remove the entry in the list of plots
            self.raw_plts.pop(delind)
            #mark the points in the raw data that can be deleted
            self.df.raw_data[self.df.raw_data[:,8] == index][:,8] = -1 
            #reduce the index of all entries with higher indices than the one we want to remove
            self.df.raw_data[self.df.raw_data[:,8] > index][:,8] -= 1
        self.df.raw_data = self.df.raw_data[self.df.raw_data[:,8] > -1]
        self.update_chosen_plot()
        self.erase_fit_plts()
        self.info_index = None
        self.update_info_plot()

        try:
            self.fit_est.set_data([[],[]])
        except:
            pass
        self.df.drop(indices_to_delete, inplace=True) #^ the indices from the dataframe
        self.df.reset_index(inplace = True, drop = True)
        self.update_data_plot()
        self.fig.canvas.draw()  

    """---------------------------slider functions---------------------------"""
    def on_press(self, event):
        """Checks, in which slider axis we have clicked and stores this information"""
        if event.inaxes==self.bounds_axis:
            y_pos = event.ydata
            if abs(y_pos-self.bound_line[0].get_ydata()[0]) < .01:
                self.change_bounds=1
            elif abs(y_pos-self.bound_line[1].get_ydata()[0]) < .01:
                self.change_bounds=2
        else:
            for i in range(4):
                if event.inaxes == self.slider_axes[i]:
                    self.chosen_slider = self.slider[i]
                    lower = self.slider_bounds[i][0]
                    upper = self.slider_bounds[i][1]
                    value = self.slider[i].val -.5
                    self.p[i] = lower + (upper - lower) * value
        self.fig.canvas.draw()

    def on_motion(self, event):
        """If the mouse is moved, we check the value of the chosen slider
        this is necessary because slider.on_changed does not allow to identify
        the slider which has changed it value"""
        if self.change_bounds > 0:
            y_neu = event.ydata
            if 0 < y_neu < 1:
                self.bound_line[self.change_bounds - 1].set_ydata([y_neu, y_neu])
                self.bound_text[self.change_bounds - 1].set_y(y_neu)
                lower = min(self.bound_line[0].get_ydata()[0], self.bound_line[1].get_ydata()[0])
                upper = max(self.bound_line[0].get_ydata()[0], self.bound_line[1].get_ydata()[0])
                delta = self.current_line_bounds[1] - self.current_line_bounds[0]
                lower = self.current_line_bounds[0] + delta * lower
                upper = self.current_line_bounds[0] + delta * upper
                self.bounds[self.bounds_to_change] = [lower, upper]
                new_text = '{0:g}{1}'.format(self.current_line_bounds[0] + delta * y_neu, self.df.units[self.bounds_to_change])
                self.bound_text[self.change_bounds - 1].set_text(new_text)
                self.update_data_plot()
                
                self.fig.canvas.draw()
        if self.chosen_slider is None: return
        if len(self.chosen_inds) == 0: return
        for i in range(4):
            if self.chosen_slider ==self.slider[i]:
                lower = self.slider_bounds[i][0]
                upper = self.slider_bounds[i][1]
                value = self.slider[i].val
                self.p[i] = lower + (upper - lower) * value
        self.update_fit_est()
        self.fig.canvas.draw()
        
    def on_release(self, event):
        self.chosen_slider = None
        self.change_bounds = 0
        
    def on_clicked_radio_bounds(self, label):
        self.current_line_bounds = self.overall_bounds[label]
        self.bounds_to_change = label
        lower, upper = self.bounds[self.bounds_to_change]
        new_text = '{0:g}{1}'.format(lower, 
                                     self.df.units[self.bounds_to_change])
        self.bound_text[0].set_text(new_text)
        new_text = '{0:g}{1}'.format(upper, 
                                     self.df.units[self.bounds_to_change])
        self.bound_text[1].set_text(new_text)

        delta_overall = self.current_line_bounds[1] - self.current_line_bounds[0]
        if abs(delta_overall) > 0:
            lower = (lower - self.current_line_bounds[0]) / delta_overall
            upper = (upper - self.current_line_bounds[0]) / delta_overall
            self.bound_line[0].set_ydata([lower, lower])
            self.bound_line[1].set_ydata([upper, upper])
            self.bound_text[0].set_y(lower)
            self.bound_text[1].set_y(upper)
        self.fig.canvas.draw()


    def set_slider_bounds(self, z, U):
        """calculates the boundaries of the sliders according to the data as the slider boundaries itself can not be changed
        """
        #the centershould be in the range of the z-data, we give 25 percent additional space to each side
        zrange = abs(z.max() - z.min())
        self.slider_bounds[0] = self.p[0] + [ -zrange, zrange]

        #the amplitude should be given by the difference between minimum and maximum (we forget about the
        #necessary division by 1.8 and even give additional 50percent more space) The amplitude can be negative
        delta = abs(U.max() - U.min())  * 1.5 
        self.slider_bounds[1] = self.p[1] + [ -delta, delta ]

        #the offset should between the smallest and largest datapoint
        delta = max(abs(U - U.mean()))
        self.slider_bounds[2] = self.p[2] + 2*[ -delta, delta ]
        #the slope parameter should be between the smallest and largest data point, 
        #divided by the length of the dataset with a fudge factor of 1.5
        delta = abs(U.max() - U.min())
        self.slider_bounds[3] = self.p[3] + [ -delta/z.shape[0], delta/z.shape[0] ]