import streamlit as st
import numpy as np
import math, random

def main():
    st.set_page_config(page_title="Helical Indexing", layout="wide")

    title = "Helical indexing using the cylindrical projection of a 3D map"
    st.title(title)

    col1, col4, col2, col3 = st.beta_columns((1.0, 1.65, 0.3, 1.))

    with col1:
        with st.beta_expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as a 2D crystal that has been rolled up into a cylindrical tube while preserving the original lattice. The indexing process is thus to computationally reverse this process: the 3D helical structure is first unrolled into a 2D image using cylindrical projection, and then the 2D lattice parameters are automatically identified from which the helical parameters (twist, rise, and cyclic symmetry) are derived. The auto-correlation function of the cylindrical projection is used to provide a lattice with sharper peaks. Two distinct lattice identification methods, one for generical 2D lattice and one specifically for helical lattice, are used to find a consistent solution.  \n  \nTips: play with the rmin/rmax, #peaks, axial step size parameters if consistent helical parameters cannot be obtained with the default parameters.  \n  \nTips: maximize the browser window or zoom-out the browser view (using ctrl- or ⌘- key combinations) if the displayed images overlap each other.")
        
        data = None
        # make radio display horizontal
        st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        input_mode = st.radio(label="How to obtain the input map:", options=["upload a mrc file", "url", "emd-xxxx"], index=1)
        if input_mode == "upload a mrc file":
            fileobj = st.file_uploader("Upload a mrc file", type=['mrc', 'map', 'map.gz'])
            if fileobj is not None:
                data, apix = get_3d_map_from_uploaded_file(fileobj)
                nz, ny, nx = data.shape
                if nz<32:
                    st.warning(f"The uploaded file {fileobj.name} ({nx}x{ny}x{nz}) is not a 3D map")
                    data = None
        else:
            if input_mode == "url":
                label = "Input a url of a 3D map:"
                value = "ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-10499/map/emd_10499.map.gz"
                url = st.text_input(label=label, value=value)
                data, apix = get_3d_map_from_url(url.strip())
                nz, ny, nx = data.shape
                if nz<32:
                    st.warning(f"{url} points to a file ({nx}x{ny}x{nz}) that is not a 3D map")
                    data = None
            elif input_mode == "emd-xxxx":
                label = "Input an EMDB ID (emd-xxxx):"
                value = "emd-10499"
                emdid = st.text_input(label=label, value=value)
                emdb_ids = get_emdb_ids()
                if emdb_ids:
                    emd_id = emdid.lower().split("emd-")[-1]
                    if emd_id in emdb_ids:
                        data, apix = get_emdb_map(emd_id)
                    else:
                        emd_id_bad = emd_id
                        emd_id = random.choice(emdb_ids)
                        st.warning(f"EMD-{emd_id_bad} is not a helical structure. Please input a valid id (for example, a randomly selected valid id {emd_id})")
                else:
                    st.warning("failed to obtained a list of helical structures in EMDB")
        if data is None:
            return

        nz, ny, nx = data.shape
        st.write(f'Map size: {nx}x{ny}x{nz} &emsp;Sampling: {apix:.4f} Å/voxel')

        section_axis = st.radio(label="Display a section along this axis:", options="X Y Z".split(), index=0)
        mapping = {"X":(nx, 2), "Y":(ny, 1), "Z":(nz, 0)}
        n, axis = mapping[section_axis]
        section_index = st.slider(label="Choose a section to display:", min_value=1, max_value=n, value=n//2+1, step=1)
        container_image = st.beta_container()

        with st.beta_expander(label="Transform the map", expanded=False):
            rotx = st.slider(label="Rotate map around X-axis (°):", min_value=-90., max_value=90., value=0.0, step=1.0)
            roty = st.slider(label="Rotate map around Y-axis (°):", min_value=-90., max_value=90., value=0.0, step=1.0)
            shiftx = st.slider(label="Shift map along X-axis (Å):", min_value=-nx//2*apix, max_value=nx//2*apix, value=0.0, step=1.0)
            shifty = st.slider(label="Shift map along Y-axis (Å):", min_value=-ny//2*apix, max_value=ny//2*apix, value=0.0, step=1.0)

        image = np.squeeze(np.take(data, indices=[section_index-1], axis=axis))
        h, w = image.shape
        if rotx or roty or shiftx or shifty:
            data = transform_map(data, shift_x=shiftx/apix, shift_y=shifty/apix, angle_x=rotx, angle_y=roty)
            image2 = np.squeeze(np.take(data, indices=[section_index-1], axis=axis))
            with container_image:
                st.image([image, image2], clamp=True, width=w, caption=["Original", "Transformed"])
        else:
            with container_image:
                st.image(image, clamp=True, width=w)

        radprofile = compute_radial_profile(data)
        rad=np.arange(len(radprofile)) * apix
        from bokeh.plotting import figure
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        p = figure(title="density radial profile", x_axis_label="r (Å)", y_axis_label="pixel value", frame_height=ny, tools=tools)
        p.line(rad, radprofile, line_width=2, color='red')
        st.bokeh_chart(p, use_container_width=True)

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    with col2:
        da = st.number_input('Angular step size (°)', value=1.0, min_value=0.1, max_value=10., step=0.1, format="%.1f")
        dz = st.number_input('Axial step size (Å)', value=1.0, min_value=0.1, max_value=10., step=0.1, format="%.1f")
        rmin = st.number_input('rmin (Å)', value=0.0, min_value=0.0, max_value=nx//2*apix, step=1.0, format="%.1f")
        rmax = st.number_input('rmax (Å)', value=nx//2*apix, min_value=rmin+1.0, max_value=nx//2*apix, step=1.0, format="%.1f")

        data = auto_masking(data)
        data = minimal_grids(data)
        cylproj = cylindrical_projection(data, da=da, dz=dz/apix, dr=1, rmin=rmin/apix, rmax=rmax/apix, interpolation_order=1)

        cylproj_square = make_square_shape(cylproj)
        acf = auto_correlation(cylproj_square, high_pass_fraction=1./cylproj_square.shape[0])
        
        peaks = find_peaks(acf, da=da, dz=dz, peak_diameter=0.025, min_mass=0.5)
        npeaks_all = len(peaks)
        npeaks = st.number_input('# peaks to use', value=npeaks_all, min_value=3, max_value=npeaks_all, step=2)

        st.subheader("Display:")
        show_cylproj = st.checkbox(label="Cylindrical projection", value=True)
        show_acf = st.checkbox(label="Auto-correlation function", value=True)
        if show_acf:
            show_peaks = st.checkbox(label="Peaks", value=True)
        
    with col3:
        if show_cylproj:
            st.text("") # workaround for a streamlit layout bug
            h, w = cylproj.shape
            tooltips = [("angle", "$x°"), ('axial', '$yÅ'), ('cylproj', '@image')]
            fig = generate_bokeh_figure(cylproj, da, dz, title=f"Cylindrical Projection ({w}x{h})", title_location="below", 
                    plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False)
            fig.title.align = "center"
            fig.title.text_font_size = "18px"
            fig.title.text_font_style = "normal"
            st.bokeh_chart(fig, use_container_width=True)

        if show_acf:
            st.text("") # workaround for a streamlit layout bug
            h, w = acf.shape
            tooltips = [("twist", "$x°"), ('rise', '$yÅ'), ('acf', '@image')]
            fig = generate_bokeh_figure(acf, da, dz, title=f"Auto-correlation function ({w}x{h})", title_location="below", 
                    plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False)
            fig.title.align = "center"
            fig.title.text_font_size = "18px"
            fig.title.text_font_style = "normal"

            if show_peaks:
                x = peaks[:npeaks, 0]
                y = peaks[:npeaks, 1]
                xs = np.sort(x)
                ys = np.sort(y)
                size = 30/(da+dz)
                fig.circle(x, y, size=size, line_width=2, line_color='yellow', fill_alpha=0)

            st.bokeh_chart(fig, use_container_width=True)

    with col4:
        h, w = acf.shape
        h2 = 900   # final plot height
        w2 = int(round(w * h2/h))//2*2
        x_axis_label="twist (°)"
        y_axis_label="reise (Å)"
        tooltips = [("twist", "$x°"), ('rise', '$yÅ'), ('acf', '@image')]
        fig = generate_bokeh_figure(image=acf, dx=da, dy=dz, title="", title_location="above", 
                plot_width=w2, plot_height=h2, x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                tooltips=tooltips, show_axis=True, show_toolbar=True)

        # horizontal line along the equator
        from bokeh.models import LinearColorMapper, Arrow, VeeHead, Line
        fig.line([-w//2*dz, (w//2-1)*dz], [0, 0], line_width=2, line_color="yellow", line_dash="dashed")
        
        trc1, trc2 = fitHelicalLattice(peaks[:npeaks], acf, da=da, dz=dz)
        trc_mean = consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0)
        success = True if trc_mean else False

        if success:
            twist_tmp, rise_tmp, cn = trc_mean
            twist, rise = refine_twist_rise(acf_image=(acf, da, dz), twist=twist_tmp, rise=rise_tmp, cn=cn)

            fig.title.text = f"twist={twist:.2f}°  rise={rise:.2f}Å  csym=c{cn}"
            fig.title.align = "center"
            fig.title.text_font_size = "24px"
            fig.title.text_font_style = "normal"

            fig.add_layout(Arrow(x_start=0, y_start=0, x_end=twist, y_end=rise, line_color="yellow", line_width=4, 
                end=VeeHead(line_color="yellow", fill_color="yellow", line_width=2))
            )
        else:
            msg = "Failed to obtain consistent helical parameters. The two sollutions are:  \n"
            msg+= f"Twist per subunit: {trc1[0]:.2f}&emsp;{trc2[0]:.2f} °  \n"
            msg+= f"Rise &nbsp; per subunit: {trc1[1]:.2f}&emsp;&emsp;&emsp;{trc2[1]:.2f} Å  \n"
            msg+= f"Csym &emsp; &emsp; &emsp; &emsp; : c{trc1[2]}&emsp;&emsp;&emsp;&emsp;c{trc2[2]}"
            st.warning(msg)

        st.bokeh_chart(fig, use_container_width=True)

    return

def generate_bokeh_figure(image, dx, dy, title="", title_location="below", plot_width=None, plot_height=None, 
    x_axis_label='x', y_axis_label='y', tooltips=None, show_axis=True, show_toolbar=True):
    from bokeh.plotting import figure
    h, w = image.shape
    w2 = plot_width if plot_width else w
    h2 = plot_height if plot_height else h
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    fig = figure(title_location=title_location, 
        frame_width=w2, frame_height=h2, 
        x_axis_label=x_axis_label, y_axis_label=y_axis_label,
        x_range=(-w//2*dx, (w//2-1)*dx), y_range=(-h//2*dy, (h//2-1)*dy), 
        tools=tools)
    fig.grid.visible = False
    if title: fig.title.text=title
    if not show_axis: fig.axis.visible = False
    if not show_toolbar: fig.toolbar_location = None

    source_data = dict(image=[image], x=[-w//2*dx], y=[-h//2*dy], dw=[w*dx], dh=[h*dy])
    from bokeh.models import LinearColorMapper
    color_mapper = LinearColorMapper(palette='Greys256')    # Greys256, Viridis256
    image = fig.image(source=source_data, image='image', color_mapper=color_mapper,
                x='x', y='y', dw='dw', dh='dh'
            )

    # add hover tool only for the image
    from bokeh.models.tools import HoverTool
    if not tooltips:
        tooltips = [("x", "$x°"), ('y', '$yÅ'), ('val', '@image')]
    image_hover = HoverTool(renderers=[image], tooltips=tooltips)
    fig.add_tools(image_hover)
    return fig

@st.cache(persist=True, show_spinner=False)
def fitHelicalLattice(peaks, acf, da=1.0, dz=1.0):
    if len(peaks) < 3:
        st.warning(f"WARNING: only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (None, None, peaks)

    nmax = len(peaks) if len(peaks)%2 else len(peaks)-1
    for n in range(nmax, 3-1, -2):
        trc1 = getHelicalLattice(peaks[:n])
        trc2 = getGenericLattice(peaks[:n])
        if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0):
            break
    
    twist1, rise1, cn1 = trc1
    twist1, rise1 = refine_twist_rise(acf_image=(acf, da, dz), twist=twist1, rise=rise1, cn=cn1)
    twist2, rise2, cn2 = trc2
    twist2, rise2 = refine_twist_rise(acf_image=(acf, da, dz), twist=twist2, rise=rise2, cn=cn2)
    
    return (twist1, rise1, cn1), (twist2, rise2, cn2)

@st.cache(persist=True, show_spinner=False)
def consistent_twist_rise_cn_sets(twist_rise_cn_set_1, twist_rise_cn_set_2, epsilon=1.0):
    def consistent_twist_rise_cn_pair(twist_rise_cn_1, twist_rise_cn_2, epsilon=1.0):
        def good_twist_rise_cn(twist, rise, cn, epsilon=1):
            if abs(twist)>epsilon:
                if abs(rise)>epsilon: return True
                elif abs(rise*360./twist/cn)>epsilon: return True # pitch>epsilon
                else: return False
            else:
                if abs(rise)>epsilon: return True
                else: return False
        if twist_rise_cn_1 is None or twist_rise_cn_2 is None:
            return None
        twist1, rise1, cn1 = twist_rise_cn_1
        twist2, rise2, cn2 = twist_rise_cn_2
        if not good_twist_rise_cn(twist1, rise1, cn1, epsilon=1): return None
        if not good_twist_rise_cn(twist2, rise2, cn2, epsilon=1): return None
        if cn1==cn2 and abs(rise2-rise1)<epsilon and abs(twist2-twist1)<epsilon:
            cn = cn1
            rise_tmp = (rise1+rise2)/2
            twist_tmp = (twist1+twist2)/2
            return twist_tmp, rise_tmp, cn
        else:
            return None
    for twist_rise_cn_1 in twist_rise_cn_set_1:
        for twist_rise_cn_2 in twist_rise_cn_set_2:
            trc = consistent_twist_rise_cn_pair(twist_rise_cn_1, twist_rise_cn_2, epsilon=epsilon)
            if trc: return trc
    return None

@st.cache(persist=True, show_spinner=False)
def refine_twist_rise(acf_image, twist, rise, cn):
    from scipy.ndimage import map_coordinates
    from scipy.optimize import minimize
    if rise<=0: return twist, rise

    acf_image, da, dz = acf_image
    
    ny, nx = acf_image.shape
    try:
        npeak = max(3, min(100, int(ny/2/abs(rise)/2)))
    except:
        npeak = 3
    i = np.repeat(range(1, npeak), cn)
    w = np.power(i, 1./2)
    x_sym = np.tile(range(cn), npeak-1) * 360./cn    
    def score(x):
        twist, rise = x
        px = np.fmod(nx//2 + i * twist/da + x_sym + npeak*nx, nx)
        py = ny//2 + i * rise/dz
        v = map_coordinates(acf_image, (py, px))
        score = -np.sum(v*w)
        return score    
    res = minimize(score, (twist, rise), method='nelder-mead', options={'xatol': 1e-4, 'adaptive': True})
    twist_opt, rise_opt = res.x
    return twist_opt, rise_opt

@st.cache(persist=True, show_spinner=False)
def getHelicalLattice(peaks):
    if len(peaks) < 3:
        st.warning(f"only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (0, 0, 1)

    x = peaks[:, 0]
    y = peaks[:, 1]

    ys = np.sort(y)
    vys = ys[1:] - ys[:-1]
    vy = np.median(vys[np.abs(vys) > 1e-1])
    j = np.around(y / vy)
    nonzero = j != 0
    if np.count_nonzero(nonzero)>0:
        rise = np.median(y[nonzero] / j[nonzero])
        if np.isnan(rise):
            st.warning(f"failed to detect rise parameter. all {len(peaks)} peaks are in the same row?")
            return (0, 0, 1)
    else:
        st.warning(f"failed to detect rise parameter. all {len(peaks)} peaks are on the equator?")
        return (0, 0, 1)

    cn = 1
    js = np.rint(y / rise)
    spacing = []
    for j in sorted(list(set(js))):
        x_j = x[js == j]
        if len(x_j) > 1:
            x_j.sort()
            spacing += list(x_j[1:] - x_j[:-1])
    if len(spacing):
        best_spacing = max(0.01, np.median(spacing)) # avoid corner case crash
        cn_f = 360. / best_spacing
        expected_spacing = 360./round(cn_f)
        if abs(best_spacing - expected_spacing)/expected_spacing < 0.05:
            cn = int(round(cn_f))

    js = np.rint(y / rise)
    above_equator = js > 0
    if np.count_nonzero(above_equator)>0:
        min_j = js[above_equator].min()  # result might not be right if min_j>1
        vx = sorted(x[js == min_j] / min_j, key=lambda x: abs(x))[0]
        x2 = np.reshape(x, (len(x), 1))
        xdiffs = x2 - x2.T
        y2 = np.reshape(y, (len(y), 1))
        ydiffs = y2 - y2.T
        selected = (np.rint(ydiffs / rise) == min_j) & (np.rint(xdiffs / vx) == min_j)
        best_vx = np.mean(xdiffs[selected])
        if best_vx > 180: best_vx -= 360
        best_vx /= min_j
        twist = best_vx
        if cn>1 and abs(twist)>180./cn:
            if twist<0: twist+=360./cn
            elif twist>0: twist-=360./cn
        if np.isnan(twist):
            st.warning(f"failed to detect twist parameter using {len(peaks)} peaks")
            return (0, 0, 1)
    else:
        st.warning(f"failed to detect twist parameter using {len(peaks)} peaks")
        return (0, 0, 1)

    return (twist, rise, cn)

@st.cache(persist=True, show_spinner=False)
def getGenericLattice(peaks):
    if len(peaks) < 3:
        st.warning(f"only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (0, 0, 1)

    from scipy.spatial import cKDTree as KDTree

    mindist = 10 # minimal inter-subunit distance
    minang = 15 # minimal angle between a and b vectors
    epsilon = 0.5

    def angle(v1, v2=None):  # angle between two vectors, ignoring vector polarity
        p = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        p = np.clip(abs(p), 0, 1)
        ret = np.rad2deg(np.arccos(p))  # 0<=angle<90
        return ret

    def distance(v1, v2):
        d = math.hypot(v1[0] - v2[0], v1[1] - v2[1])
        return d

    def onEquator(v, epsilon=0.5):
        # test if b vector is on the equator
        if abs(v[1]) > epsilon: return 0
        return 1

    def pickTriplet(kdtree, index=-1):
        '''
        pick a point as the origin and find two points closest to the origin
        '''
        m, n = kdtree.data.shape  # number of data points, dimension of each data point
        if index < 0:
            index = random.randint(0, m - 1)
        origin = kdtree.data[index]
        distances, indices = kdtree.query(origin, k=m)
        first = None
        for i in range(1, m):
            v = kdtree.data[indices[i]]
            #if onEquator(v - origin, epsilon=epsilon):
            #    continue
            first = v
            break
        second = None
        for j in range(i + 1, m):
            v = kdtree.data[indices[j]]
            #if onEquator(v - origin, epsilon=epsilon):
            #    continue
            ang = angle(first - origin, v - origin)
            dist = distance(first - origin, v - origin)
            if dist > mindist and ang > minang:
                second = v
                break
        return (origin, first, second)

    def peaks2NaNbVaVbOrigin(peaks, va, vb, origin):
        # first find indices of each peak using current unit cell vectors
        A = np.vstack((va, vb)).transpose()
        b = (peaks - origin).transpose()
        x = np.linalg.solve(A, b)
        NaNb = np.around(x)
        # then refine unit cell vectors using these indices
        good = np.abs(x-NaNb).max(axis=0) < 0.1 # ignore bad peaks
        one = np.ones((1, (NaNb[:, good].shape)[1]))
        A = np.vstack((NaNb[:, good], one)).transpose()
        (p, residues, rank, s) = np.linalg.lstsq(A, peaks[good, :], rcond=-1)
        va = p[0]
        vb = p[1]
        origin = p[2]
        err = np.sqrt(sum(residues)) / len(peaks)
        return {"NaNb": NaNb, "a": va, "b": vb, "origin": origin, "err": err}

    kdt = KDTree(peaks)

    bestLattice = None
    minError = 1e30
    for i in range(len(peaks)):
        origin, first, second = pickTriplet(kdt, index=i)
        if first is None or second is None: continue
        va = first - origin
        vb = second - origin

        lattice = peaks2NaNbVaVbOrigin(peaks, va, vb, origin)
        lattice = peaks2NaNbVaVbOrigin(peaks, lattice["a"], lattice["b"], lattice["origin"])
        err = lattice["err"]
        if err < minError:
            dist = distance(lattice['a'], lattice['b'])
            ang = angle(lattice['a'], lattice['b'])
            if dist > mindist and ang > minang:
                minError = err
                bestLattice = lattice

    if bestLattice is None:
        # assume all peaks are along the same line of an arbitrary direction
        # fit a line through the peaks
        from scipy.odr import Data, ODR, unilinear
        x = peaks[:, 0]
        y = peaks[:, 1]
        odr_data = Data(x, y)
        odr_obj = ODR(odr_data, unilinear)
        output = odr_obj.run()
        x2 = x + output.delta   # positions on the fitted line
        y2 = y + output.eps
        v0 = np.array([x2[-1]-x2[0], y2[-1]-y2[0]])
        v0 = v0/np.linalg.norm(v0, ord=2)   # unit vector along the fitted line
        ref_i = 0
        t = (x2-x2[ref_i])*v0[0] + (y2-y2[ref_i])*v0[1] # coordinate along the fitted line
        t.sort()
        spacings = t[1:]-t[:-1]
        a = np.median(spacings[np.abs(spacings)>1e-1])
        a = v0 * a
        if a[1]<0: a *= -1
        bestLattice = {"a": a, "b": a}

    a, b = bestLattice["a"], bestLattice["b"]

    minLength = min(np.linalg.norm(a), np.linalg.norm(b)) * 0.9
    vs_on_equator = []
    vs_off_equator = []
    maxI = 10
    for i in range(-maxI, maxI + 1):
        for j in range(-maxI, maxI + 1):
            if i or j:
                v = i * a + j * b
                if -180 <= v[0] <= 180 and np.linalg.norm(v) > minLength:
                    if v[1]<0: v *= -1
                    if onEquator(v, epsilon=epsilon):
                        vs_on_equator.append(v)
                    else:
                        vs_off_equator.append(v)
    twist, rise, cn = 0, 0, 1
    if vs_on_equator:
        vs_on_equator.sort(key=lambda v: abs(v[0]))
        best_spacing = abs(vs_on_equator[0][0])
        cn_f = 360. / best_spacing
        expected_spacing = 360./round(cn_f)
        if abs(best_spacing - expected_spacing)/expected_spacing < 0.05:
            cn = int(round(cn_f))
    if vs_off_equator:
        vs_off_equator.sort(key=lambda v: (abs(round(v[1]/epsilon)), abs(v[0])))
        twist, rise = vs_off_equator[0]
        if cn>1 and abs(twist)>180./cn:
            if twist<0: twist+=360./cn
            elif twist>0: twist-=360./cn
    return twist, rise, cn

@st.cache(persist=True, show_spinner=False)
def find_peaks(acf, da, dz, peak_diameter=0.025, min_mass=0.5):
    from trackpy import locate
    # diameter: fraction of the maximal dimension of the image (acf)
    diameter = int(max(acf.shape)*peak_diameter)//2*2+1
    f = locate(acf, diameter=diameter, minmass=1)
    f = f.sort_values(["mass"], ascending=False)
    masses = f["mass"].values
    thresh=masses[1]*min_mass
    f = f[f["mass"] >= thresh]
    peaks = np.zeros((len(f), 2), dtype=float)
    peaks[:, 0] = f['x'].values - acf.shape[1]//2    # pixel
    peaks[:, 1] = f['y'].values - acf.shape[0]//2    # pixel
    peaks[:, 0] *= da  # the values are now in degree
    peaks[:, 1] *= dz  # the values are now in Angstrom
    return peaks

@st.cache(persist=True, show_spinner=False)
def auto_correlation(data, high_pass_fraction=0):
    from scipy.signal import correlate2d
    fft = np.fft.rfft2(data)
    product = fft*np.conj(fft)
    if 0<high_pass_fraction<=1:
        nz, na = product.shape
        Z, A = np.meshgrid(np.arange(-nz//2, nz//2, dtype=np.float), np.arange(-na//2, na//2, dtype=np.float), indexing='ij')
        Z /= nz//2
        A /= na//2
        f2 = np.log(2)/(high_pass_fraction**2)
        filter = 1.0 - np.exp(- f2 * Z**2) # Z-direction only
        product *= np.fft.fftshift(filter)
    corr = np.fft.fftshift(np.fft.irfft2(product))
    corr -= np.median(corr, axis=1, keepdims=True)
    corr = normalize(corr)
    return corr

@st.cache(persist=True, show_spinner=False)
def make_square_shape(cylproj):
    nz, na = cylproj.shape
    if nz<na:
        zeros_top = np.zeros((na//2-nz//2, na))
        zeros_bottom = np.zeros((na-nz-zeros_top.shape[0], na))
        ret = cylproj-cylproj[[0,-1], :].mean()  # subtract the mean values of top/bottom rows
        ret = np.vstack((zeros_top, ret, zeros_bottom))
    elif nz>na:
        row0 = nz//2-na//2
        ret = cylproj[row0:row0+na]
    else:
        ret = cylproj
    return ret

@st.cache(persist=True, show_spinner=False)
def cylindrical_projection(map3d, da=1, dz=1, dr=1, rmin=0, rmax=-1, interpolation_order=1):
    # da: degree
    # dr/dz/rmin/rmax: pixel
    assert(map3d.shape[0]>1)
    nz, ny, nx = map3d.shape
    if rmax<=rmin:
        rmax = min(nx//2, ny//2)
    assert(rmin<rmax)
    
    r = np.arange(rmin, rmax, dr, dtype=np.float32)
    theta = (np.arange(0, 360, da, dtype=np.float32) - 90) * np.pi/180.
    #z = np.arange(0, nz, dz)    # use entire length
    n_theta = len(theta)
    z = np.arange(max(0, nz//2-n_theta//2*dz), min(nz, nz//2+n_theta//2*dz), dz, dtype=np.float32)    # use only the central segment 

    z_grid, theta_grid, r_grid = np.meshgrid(z, theta, r, indexing='ij', copy=False)
    y_grid = ny//2 + r_grid * np.sin(theta_grid)
    x_grid = nx//2 + r_grid * np.cos(theta_grid)

    coords = np.vstack((z_grid.flatten(), y_grid.flatten(), x_grid.flatten()))

    from scipy.ndimage.interpolation import map_coordinates
    cylindrical_map = map_coordinates(map3d, coords, order=interpolation_order, mode='nearest').reshape(z_grid.shape)
    cylindrical_proj = (cylindrical_map*r_grid).sum(axis=2)/r_grid.sum(axis=2)

    cylindrical_proj = normalize(cylindrical_proj)

    return cylindrical_proj

@st.cache(persist=True, show_spinner=False)
def minimal_grids(map3d):
    from scipy.ndimage import find_objects
    labels = (abs(map3d) >1e-6) * 1
    objs = find_objects(labels)
    nz, ny, nx = map3d.shape
    zmin, zmax = objs[0][0].start, objs[0][0].stop
    ymin, ymax = objs[0][1].start, objs[0][1].stop
    xmin, xmax = objs[0][2].start, objs[0][2].stop
    dy = max(abs(ymin-ny//2), abs(ymax-ny//2))+1
    dx = max(abs(xmin-nx//2), abs(xmax-nx//2))+1
    ret = map3d[zmin:zmax, max(0, ny//2-dy):ny//2+dy, max(0, nx//2-dx):nx//2+dx]
    return ret

@st.cache(persist=True, show_spinner=False)
def auto_masking(map3d):
    from skimage.segmentation import watershed
    data = (map3d/map3d.max())
    data[data<0] = 0
    markers = np.zeros(data.shape, dtype = np.uint)
    markers[data < 0.02] = 1    # background
    markers[data > 0.2 ] = 2    # structure
    labels = watershed(data.astype(np.float64), markers=markers, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False)
    if np.count_nonzero(labels == 2)>map3d.size * 0.001:
        labels[labels != 2] = 0
        labels[labels == 2] = 1
        masked = data * labels
    else:
        masked = data
    return masked

@st.cache(persist=True, show_spinner=False)
def compute_radial_profile(data):
    proj = data.mean(axis=0)
    ny, nx = proj.shape
    rmax = min(nx//2, ny//2)
    
    r = np.arange(0, rmax, 1, dtype=np.float32)
    theta = np.arange(0, 360, 1, dtype=np.float32) * np.pi/180.
    n_theta = len(theta)

    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij', copy=False)
    y_grid = ny//2 + r_grid * np.sin(theta_grid)
    x_grid = nx//2 + r_grid * np.cos(theta_grid)

    coords = np.vstack((y_grid.flatten(), x_grid.flatten()))

    from scipy.ndimage.interpolation import map_coordinates
    polar = map_coordinates(proj, coords, order=1).reshape(r_grid.shape)

    rad_profile = polar.sum(axis=0)
    return rad_profile

@st.cache(persist=True, show_spinner=False)
def transform_map(data, shift_x=0, shift_y=0, angle_x=0, angle_y=0):
    from scipy.spatial.transform import Rotation as R
    from scipy.ndimage import affine_transform
    # note the convention change
    # xyz in scipy is zyx in cryoEM maps
    rot = R.from_euler('zy', [angle_x, -angle_y], degrees=True)
    m = rot.as_matrix()
    nx, ny, nz = data.shape
    bcenter = np.array((nx//2, ny//2, nz//2), dtype=np.float)
    offset = bcenter.T - np.dot(m, bcenter.T) + np.array([0.0, shift_y, -shift_x])
    ret = affine_transform(data, matrix=m, offset=offset, mode='nearest')
    return ret

@st.cache(persist=True, show_spinner=False)
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2

@st.cache(persist=True, show_spinner=False)
def get_3d_map_from_uploaded_file(fileobj):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        return get_3d_map_from_file(temp.name)

@st.cache(persist=True, show_spinner=False, ttl=24*60*60.) # refresh every day
def get_emdb_ids():
    try:
        import pandas as pd
        emdb_ids = pd.read_csv("https://wwwdev.ebi.ac.uk/emdb/api/search/*%20AND%20structure_determination_method:%22helical%22?wt=csv&download=true&fl=emdb_id")
        emdb_ids = list(emdb_ids.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
    except:
        emdb_ids = []
    return emdb_ids

@st.cache(persist=True, show_spinner=False)
def get_emdb_map(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    url = f"ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdid_number}/map/emd_{emdid_number}.map.gz"
    return get_3d_map_from_url(url)

@st.cache(persist=True, show_spinner=False)
def get_3d_map_from_url(url):
    ds = np.DataSource(None)
    fp=ds.open(url)
    return get_3d_map_from_file(fp.name)

@st.cache(persist=True, show_spinner=False)
def get_3d_map_from_file(filename):
    import mrcfile
    data = None
    with mrcfile.open(filename) as mrc:
        apix = mrc.voxel_size.x.item()
        is3d = mrc.is_volume() or mrc.is_volume_stack()
        data = normalize(mrc.data)
    return data, apix

@st.cache(persist=True, show_spinner=False)
def setup_anonymous_usage_tracking():
    try:
        import os, stat
        index_file = os.path.dirname(st.__file__) + "/static/index.html"
        os.chmod(index_file, stat.S_IRUSR|stat.S_IWUSR|stat.S_IROTH)
        with open(index_file, "r+") as fp:
            txt = fp.read()
            if txt.find("gtag.js")==-1:
                txt2 = txt.replace("<head>", '''<head><!-- Global site tag (gtag.js) - Google Analytics --><script async src="https://www.googletagmanager.com/gtag/js?id=G-8Z99BDVHTC"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-8Z99BDVHTC');</script>''')
                fp.seek(0)
                fp.write(txt2)
                fp.truncate()
    except:
        pass

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    main()
