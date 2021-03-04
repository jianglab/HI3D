""" 
MIT License

Copyright (c) 2020-2021 Wen Jiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def import_with_auto_install(packages, scope=locals()):
    if isinstance(packages, str): packages=[packages]
    for package in packages:
        if package.find(":")!=-1:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = __import__(package_import_name)
        except ImportError:
            import subprocess
            subprocess.call(f'pip install {package_pip_name}', shell=True)
            scope[package_import_name] =  __import__(package_import_name)
required_packages = "streamlit numpy scipy pandas bokeh skimage:scikit_image mrcfile trackpy".split()
import_with_auto_install(required_packages)

import streamlit as st
import numpy as np
import math, random

def main():
    title = "HI3D: Helical indexing using the cylindrical projection of a 3D map"
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)

    query_params = st.experimental_get_query_params()

    col1, col2, col3, col4 = st.beta_columns((1.0, 3.2, 0.6, 1.15))

    with col1:
        with st.beta_expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as a 2D crystal that has been rolled up into a cylindrical tube while preserving the original lattice. The indexing process is thus to computationally reverse this process: the 3D helical structure is first unrolled into a 2D image using cylindrical projection, and then the 2D lattice parameters are automatically identified from which the helical parameters (twist, rise, and cyclic symmetry) are derived. The auto-correlation function of the cylindrical projection is used to provide a lattice with sharper peaks. Two distinct lattice identification methods, one for generical 2D lattice and one specifically for helical lattice, are used to find a consistent solution.  \n  \nTips: play with the rmin/rmax, #peaks, axial step size parameters if consistent helical parameters cannot be obtained with the default parameters. Use a larger axial step size (for example 2Å) for a structure with large rise.\n  \nTips: maximize the browser window or zoom-out the browser view (using ctrl- or ⌘- key combinations) if the displayed images overlap each other.")
        
        data = None
        # make radio display horizontal
        st.markdown('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        input_modes = {0:"upload", 1:"url", 2:"emd-xxxx"}
        value = int(query_params["input_mode"][0]) if "input_mode" in query_params else 2
        input_mode = st.radio(label="How to obtain the input map:", options=list(input_modes.keys()), format_func=lambda i:input_modes[i], index=value)
        is_emd = False
        if input_mode == 0: # "upload a mrc file":
            fileobj = st.file_uploader("Upload a mrc file", type=['mrc', 'map', 'map.gz'])
            if fileobj is not None:
                is_emd = fileobj.name.find("emd_")!=-1
                data, apix = get_3d_map_from_uploaded_file(fileobj)
                nz, ny, nx = data.shape
                if nz<32:
                    st.warning(f"The uploaded file {fileobj.name} ({nx}x{ny}x{nz}) is not a 3D map")
                    data = None
        else:
            if input_mode == 1: # "url":
                label = "Input a url of a 3D map:"
                value = query_params["url"][0] if "url" in query_params else "ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-10499/map/emd_10499.map.gz"
                url = st.text_input(label=label, value=value)
                is_emd = url.find("emd_")!=-1
                data, apix = get_3d_map_from_url(url.strip())
                nz, ny, nx = data.shape
                if nz<32:
                    st.warning(f"{url} points to a file ({nx}x{ny}x{nz}) that is not a 3D map")
                    data = None
            elif input_mode == 2:   # "emd-xxxx":
                is_emd = True
                label = "Input an EMDB ID (emd-xxxx):"
                value = query_params["emdid"][0] if "emdid" in query_params else "emd-10499"
                emdid = st.text_input(label=label, value=value)
                emdb_ids = get_emdb_ids()
                if emdb_ids:
                    emd_id = emdid.lower().split("emd-")[-1]
                    if emd_id in emdb_ids:
                        data, apix = get_emdb_map(emd_id)
                    else:
                        emd_id_bad = emd_id
                        emd_id = random.choice(emdb_ids)
                        st.warning(f"EMD-{emd_id_bad} is not a helical structure. Please input a valid id (for example, a randomly selected valid id 'emd-{emd_id}')")
                else:
                    st.warning("failed to obtained a list of helical structures in EMDB")
                st.markdown(f'[EMD-{emd_id}](https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-{emd_id})')
        if data is None:
            return

        nz, ny, nx = data.shape
        st.markdown(f'{nx}x{ny}x{nz} voxels | {round(apix,4):g} Å/voxel')

        if np.std(data) == 0:
            st.warning(f"The map is blank: min={data.min()} max={data.max()} mean={data.mean()} sigma={np.std(data)}. Please provide a valid 3D map")
            st.stop()

        data = normalize(data)

        section_axis = st.radio(label="Display a section along this axis:", options="X Y Z".split(), index=0)
        mapping = {"X":(nx, 2), "Y":(ny, 1), "Z":(nz, 0)}
        n, axis = mapping[section_axis]
        section_index = st.slider(label="Choose a section to display:", min_value=1, max_value=n, value=n//2+1, step=1)
        container_image = st.beta_container()

        with st.beta_expander(label="Transform the map", expanded=False):
            if is_emd:
                rotx_auto, roty_auto, shiftx_auto, shifty_auto = 0., 0., 0., 0.
            else:
                rotx_auto, shifty_auto = auto_vertical_center(np.sum(data, axis=2))
                roty_auto, shiftx_auto = auto_vertical_center(np.sum(data, axis=1))
            rotx = st.number_input(label="Rotate map around X-axis (°):", min_value=-90., max_value=90., value=round(rotx_auto,2), step=1.0, format="%g")
            roty = st.number_input(label="Rotate map around Y-axis (°):", min_value=-90., max_value=90., value=round(roty_auto,2), step=1.0, format="%g")
            shiftx = st.number_input(label="Shift map along X-axis (Å):", min_value=-nx//2*apix, max_value=nx//2*apix, value=round(shiftx_auto*apix,2), step=1.0, format="%g")
            shifty = st.number_input(label="Shift map along Y-axis (Å):", min_value=-ny//2*apix, max_value=ny//2*apix, value=round(shifty_auto*apix,2), step=1.0, format="%g")

        image = np.squeeze(np.take(data, indices=[section_index-1], axis=axis))
        h, w = image.shape
        if rotx or roty or shiftx or shifty:
            data = transform_map(data, shift_x=shiftx/apix, shift_y=-shifty/apix, angle_x=-rotx, angle_y=-roty)
            image2 = np.squeeze(np.take(data, indices=[section_index-1], axis=axis))
            with container_image:
                tooltips = [("x", "$x"), ('y', '$y'), ('val', '@image')]
                fig1 = generate_bokeh_figure(image, apix, apix, title=f"Original", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)
                fig2 = generate_bokeh_figure(image2, apix, apix, title=f"Transformed", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)

                from bokeh.plotting import figure
                x = (np.arange(0, w)-w//2) * apix
                ymax = np.max(image2, axis=0)
                ymean = np.mean(image2, axis=0)
                fig4 = figure(x_axis_label=None, y_axis_label=None, x_range=fig2.x_range, aspect_ratio=3)
                fig4.line(x, ymax, line_width=2, color='red', legend_label="max")
                fig4.line(-x, ymax, line_width=2, color='red', line_dash="dashed", legend_label="max flipped")
                fig4.line(x, ymean, line_width=2, color='blue', legend_label="mean")
                fig4.line(-x, ymean, line_width=2, color='blue', line_dash="dashed", legend_label="mean flipped")
                fig4.xaxis.visible = False
                fig4.yaxis.visible = False
                fig4.legend.visible=False
                fig4.toolbar_location = None
                ymax = np.max(image, axis=0)
                ymean = np.mean(image, axis=0)
                fig3 = figure(x_axis_label=None, y_axis_label=None, x_range=fig1.x_range, aspect_ratio=3)
                fig3.line(x, ymax, line_width=2, color='red', legend_label="max")
                fig3.line(-x, ymax, line_width=2, color='red', line_dash="dashed", legend_label="max flipped")
                fig3.line(x, ymean, line_width=2, color='blue', legend_label="mean")
                fig3.line(-x, ymean, line_width=2, color='blue', line_dash="dashed", legend_label="mean flipped")
                fig3.xaxis.visible = False
                fig3.yaxis.visible = False
                fig3.legend.visible=False
                fig3.toolbar_location = None
                
                # create a linked crosshair tool among the figures
                from bokeh.models import CrosshairTool
                crosshair = CrosshairTool(dimensions="both")
                crosshair.line_color = 'red'
                fig1.add_tools(crosshair)
                fig2.add_tools(crosshair)
                crosshair = CrosshairTool(dimensions="height")
                crosshair.line_color = 'red'
                fig1.add_tools(crosshair)
                fig2.add_tools(crosshair)
                fig3.add_tools(crosshair)
                fig4.add_tools(crosshair)

                from bokeh.layouts import column
                fig_image = column([fig3, fig1, fig2, fig4], sizing_mode='scale_width')
                st.bokeh_chart(fig_image, use_container_width=True)
        else:
            with container_image:
                tooltips = [("x", "$x"), ('y', '$y'), ('val', '@image')]
                fig_image = generate_bokeh_figure(image, 1, 1, title=f"Original", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)
                st.bokeh_chart(fig_image, use_container_width=True)

        rad_plot = st.empty()

        with st.beta_expander(label="Select radial range", expanded=False):
            radprofile = compute_radial_profile(data)
            rad = np.arange(len(radprofile)) * apix
            rmin_auto, rmax_auto = estimate_radial_range(radprofile, thresh_ratio=0.1)
            rmin = st.number_input('Minimal radius (Å)', value=round(rmin_auto*apix,1), min_value=0.0, max_value=round(nx//2*apix,1), step=1.0, format="%g")
            rmax = st.number_input('Maximal radius (Å)', value=round(rmax_auto*apix,1), min_value=0.0, max_value=round(nx//2*apix,1), step=1.0, format="%g")
            if rmax<=rmin:
                st.warning(f"rmax(={rmax}) should be larger than rmin(={rmin})")
                return

        from bokeh.plotting import figure
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        tooltips = [("r", "@x{0.0}Å"), ("val", "@y{0.0}"),]
        fig_radprofile = figure(title="density radial profile", x_axis_label="r (Å)", y_axis_label="pixel value", tools=tools, tooltips=tooltips, aspect_ratio=2)
        fig_radprofile.line(rad, radprofile, line_width=2, color='red')
        
        from bokeh.models import Span
        rmin_span = Span(location=rmin, dimension='height', line_color='green', line_dash='dashed', line_width=3)
        rmax_span = Span(location=rmax, dimension='height', line_color='green', line_dash='dashed', line_width=3)
        fig_radprofile.add_layout(rmin_span)
        fig_radprofile.add_layout(rmax_span)
        fig_radprofile.yaxis.visible = False
        with rad_plot:
            st.bokeh_chart(fig_radprofile, use_container_width=True)

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    with col3:
        da = st.number_input('Angular step size (°)', value=1.0, min_value=0.1, max_value=10., step=0.1, format="%g")
        dz = st.number_input('Axial step size (Å)', value=1.0, min_value=0.1, max_value=10., step=0.1, format="%g")
        
        data = auto_masking(data)
        data = minimal_grids(data)
        cylproj = cylindrical_projection(data, da=da, dz=dz/apix, dr=1, rmin=rmin/apix, rmax=rmax/apix, interpolation_order=1)

        cylproj_work = cylproj
        draw_cylproj_box = False

        st.subheader("Display:")
        show_cylproj = st.checkbox(label="Cylindrical projection", value=True)
        if show_cylproj:
            nz, na = cylproj.shape
            ang_min = st.number_input('Minimal angle (°)', value=-180., min_value=-180.0, max_value=180., step=1.0, format="%g")
            ang_max = st.number_input('Maximal angle (°)', value=180., min_value=-180.0, max_value=180., step=1.0, format="%g")
            z_min = st.number_input('Minimal z (Å)', value=round(-nz//2*dz,1), min_value=-nz//2*dz, max_value=nz//2*dz, step=1.0, format="%g")
            z_max = st.number_input('Maximal z (Å)', value=round(nz//2*dz,1), min_value=-nz//2*dz, max_value=nz//2*dz, step=1.0, format="%g")
            if z_max<=z_min:
                st.warning(f"'Maximal z'(={z_max}) should be larger than 'Minimal z'(={z_min})")
                return

            if not (ang_min==-180 and ang_max==180 and z_min==-nz//2*dz and z_max==nz//2*dz):
                draw_cylproj_box = True
                cylproj_work = cylproj * 1.0
                if ang_min<ang_max:
                    if ang_min>-180.:
                        a0 = round(ang_min/da) + na//2
                        cylproj_work[:, 0:a0] = 0
                    if ang_max<180.:
                        a1 = round(ang_max/da)+ na//2
                        cylproj_work[:, a1:] = 0
                else: # wrap around
                    if ang_min<180:
                        a0 = round(ang_min/da) + na//2
                        cylproj_work[:, a0:] = 0
                    if ang_max>-180:
                        a1 = round(ang_max/da)+ na//2
                        cylproj_work[:, 0:a1] = 0
                if z_min>-nz//2*dz:
                    z0 = round(z_min/dz)+ nz//2
                    cylproj_work[0:z0, :] = 0
                if z_max<nz//2*dz:
                    z1 = round(z_max/dz)+ nz//2
                    cylproj_work[z1:, :] = 0

        cylproj_square = make_square_shape(cylproj_work)
        acf = auto_correlation(cylproj_square, high_pass_fraction=1./cylproj_square.shape[0])

        peaks = find_peaks(acf, da=da, dz=dz, peak_diameter=0.025, minmass=1.0)
        if peaks is None:
            st.warning("No peak was found from the auto-correlation image")
            return
        elif len(peaks)<3:
            st.warning(f"Only {len(peaks)} peaks were found. At least 3 peaks are required")
            return
        npeaks = len(peaks)
        
        show_acf = st.checkbox(label="Auto-correlation function", value=True)
        if show_acf:
            show_peaks = st.checkbox(label="Peaks", value=True)
            if show_peaks:
                npeaks_all = len(peaks)
                npeaks = st.number_input('# peaks to use', value=npeaks_all, min_value=3, max_value=npeaks_all, step=2)
        
    with col4:
        if show_cylproj:
            st.text("") # workaround for a streamlit layout bug
            h, w = cylproj.shape
            tooltips = [("angle", "$x°"), ('z', '$yÅ'), ('cylproj', '@image')]
            fig_cylproj = generate_bokeh_figure(cylproj, da, dz, title=f"Cylindrical Projection ({w}x{h})", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)

            if draw_cylproj_box:
                if ang_min<ang_max:
                    fig_cylproj.quad(left=ang_min, right=ang_max, bottom=z_min, top=z_max, line_color=None, fill_color='yellow', fill_alpha=0.5)
                else:
                    fig_cylproj.quad(left=ang_min, right=180, bottom=z_min, top=z_max, line_color=None, fill_color='yellow', fill_alpha=0.5)
                    fig_cylproj.quad(left=-180, right=ang_max, bottom=z_min, top=z_max, line_color=None, fill_color='yellow', fill_alpha=0.5)

            st.text("") # workaround for a layout bug in streamlit 
            st.bokeh_chart(fig_cylproj, use_container_width=True)

        if show_acf:
            st.text("") # workaround for a streamlit layout bug
            h, w = acf.shape
            tooltips = [("twist", "$x°"), ('rise', '$yÅ'), ('acf', '@image')]
            fig_acf = generate_bokeh_figure(acf, da, dz, title=f"Auto-Correlation Function ({w}x{h})", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)
            if show_peaks:
                x = peaks[:npeaks, 0]
                y = peaks[:npeaks, 1]
                xs = np.sort(x)
                ys = np.sort(y)
                size = 30/(da+dz)
                fig_acf.circle(x, y, size=size, line_width=2, line_color='yellow', fill_alpha=0)

            st.bokeh_chart(fig_acf, use_container_width=True)
            
    with col2:
        h, w = acf.shape
        h2 = 900   # final plot height
        w2 = int(round(w * h2/h))//2*2
        x_axis_label="twist (°)"
        y_axis_label="reise (Å)"
        tooltips = [("twist", "$x°"), ('rise', '$yÅ'), ('acf', '@image')]
        fig_indexing = generate_bokeh_figure(image=acf, dx=da, dy=dz, title="", title_location="above", plot_width=None, plot_height=None, x_axis_label=x_axis_label, y_axis_label=y_axis_label, tooltips=tooltips, show_axis=True, show_toolbar=True, crosshair_color="white", aspect_ratio=w/h)

        # horizontal line along the equator
        from bokeh.models import LinearColorMapper, Arrow, VeeHead, Line
        fig_indexing.line([-w//2*dz, (w//2-1)*dz], [0, 0], line_width=2, line_color="yellow", line_dash="dashed")
        
        trc1, trc2 = fitHelicalLattice(peaks[:npeaks], acf, da=da, dz=dz)
        trc_mean = consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0)
        success = True if trc_mean else False

        if success:
            twist_tmp, rise_tmp, cn = trc_mean
            twist, rise = refine_twist_rise(acf_image=(acf, da, dz), twist=twist_tmp, rise=rise_tmp, cn=cn)

            fig_indexing.title.text = f"twist={round(twist,2):g}°  rise={round(rise,2):g}Å  csym=c{cn}"
            fig_indexing.title.align = "center"
            fig_indexing.title.text_font_size = "24px"
            fig_indexing.title.text_font_style = "normal"

            fig_indexing.add_layout(Arrow(x_start=0, y_start=0, x_end=twist, y_end=rise, line_color="yellow", line_width=4, 
                end=VeeHead(line_color="yellow", fill_color="yellow", line_width=2))
            )
        else:
            msg = f"Failed to obtain consistent helical parameters using {npeaks} peaks. The two sollutions are:  \n"
            msg+= f"Twist per subunit: {round(trc1[0],2):g}&emsp;{round(trc2[0],2):g} °  \n"
            msg+= f"Rise &nbsp; per subunit: {round(trc1[1],2):g}&emsp;&emsp;&emsp;{round(trc2[1]):g} Å  \n"
            msg+= f"Csym &emsp; &emsp; &emsp; &emsp; : c{trc1[2]}&emsp;&emsp;&emsp;&emsp;c{trc2[2]}"
            st.warning(msg)

        st.text("") # workaround for a layout bug in streamlit 
        st.bokeh_chart(fig_indexing, use_container_width=True)

    if input_mode == 2:
        st.experimental_set_query_params(input_mode=2, emdid=emdid)
    elif input_mode == 1:
        st.experimental_set_query_params(input_mode=1, url=url)
    else:
        st.experimental_set_query_params()

def generate_bokeh_figure(image, dx, dy, title="", title_location="below", plot_width=None, plot_height=None, x_axis_label='x', y_axis_label='y', tooltips=None, show_axis=True, show_toolbar=True, crosshair_color="white", aspect_ratio=None):
    from bokeh.plotting import figure
    h, w = image.shape
    if aspect_ratio is None and (plot_width and plot_height):
        aspect_ratio = plot_width/plot_height
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    fig = figure(title_location=title_location, 
        frame_width=plot_width, frame_height=plot_height, 
        x_axis_label=x_axis_label, y_axis_label=y_axis_label,
        x_range=(-w//2*dx, (w//2-1)*dx), y_range=(-h//2*dy, (h//2-1)*dy), 
        tools=tools, aspect_ratio=aspect_ratio)
    fig.grid.visible = False
    if title:
        fig.title.text=title
        fig.title.align = "center"
        fig.title.text_font_size = "18px"
        fig.title.text_font_style = "normal"
    if not show_axis: fig.axis.visible = False
    if not show_toolbar: fig.toolbar_location = None

    source_data = dict(image=[image], x=[-w//2*dx], y=[-h//2*dy], dw=[w*dx], dh=[h*dy])
    from bokeh.models import LinearColorMapper
    color_mapper = LinearColorMapper(palette='Greys256')    # Greys256, Viridis256
    image = fig.image(source=source_data, image='image', color_mapper=color_mapper,
                x='x', y='y', dw='dw', dh='dh'
            )

    # add hover tool only for the image
    from bokeh.models.tools import HoverTool, CrosshairTool
    if not tooltips:
        tooltips = [("x", "$x°"), ('y', '$yÅ'), ('val', '@image')]
    image_hover = HoverTool(renderers=[image], tooltips=tooltips)
    fig.add_tools(image_hover)
    crosshair = [t for t in fig.tools if isinstance(t, CrosshairTool)]
    if crosshair: 
        for ch in crosshair: ch.line_color = crosshair_color
    return fig

# do not cache - so that the random process is effective upon rerun
def fitHelicalLattice(peaks, acf, da=1.0, dz=1.0):
    if len(peaks) < 3:
        #st.warning(f"WARNING: only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (None, None, peaks)

    consistent_solution_found = False
    nmax = len(peaks) if len(peaks)%2 else len(peaks)-1
    for n in range(nmax, 3-1, -2):
        trc1 = getHelicalLattice(peaks[:n])
        trc2 = getGenericLattice(peaks[:n])
        if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0):
            consistent_solution_found = True
            break
    
    if not consistent_solution_found: 
        for _ in range(100):
            from random import randint, sample
            n = randint(5, len(peaks)//2)
            random_choices = sorted(sample(range(n*2), k=n))
            if 0 not in random_choices: random_choices = [0] + random_choices
            peaks_random = peaks[random_choices]
            trc1 = getHelicalLattice(peaks_random)
            trc2 = getGenericLattice(peaks_random)
            if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0):
                consistent_solution_found = True
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

@st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
def getHelicalLattice(peaks):
    if len(peaks) < 3:
        #st.warning(f"only {len(peaks)} peaks were found. At least 3 peaks are required")
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
            #st.warning(f"failed to detect rise parameter. all {len(peaks)} peaks are in the same row?")
            return (0, 0, 1)
    else:
        #st.warning(f"failed to detect rise parameter. all {len(peaks)} peaks are on the equator?")
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
            #st.warning(f"failed to detect twist parameter using {len(peaks)} peaks")
            return (0, 0, 1)
    else:
        #st.warning(f"failed to detect twist parameter using {len(peaks)} peaks")
        return (0, 0, 1)

    return (twist, rise, cn)

@st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
def getGenericLattice(peaks):
    if len(peaks) < 3:
        #st.warning(f"only {len(peaks)} peaks were found. At least 3 peaks are required")
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
def find_peaks(acf, da, dz, peak_diameter=0.025, minmass=1.0, max_peaks=71):
    from trackpy import locate
    # diameter: fraction of the maximal dimension of the image (acf)
    diameter = int(max(acf.shape)*peak_diameter)//2*2+1
    while True:
        f = locate(acf, diameter=diameter, minmass=minmass)
        if len(f)>3: break
        minmass *= 0.9
        if minmass<0.1:
            return None
    f = f.sort_values(["mass"], ascending=False)[:max_peaks]
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
def estimate_radial_range(radprofile, thresh_ratio=0.1):
    background = np.mean(radprofile[-3:])
    thresh = (radprofile.max() - background) * thresh_ratio + background
    indices = np.nonzero(radprofile>thresh)
    rmin_auto = np.min(indices)
    rmax_auto = np.max(indices)
    return float(rmin_auto), float(rmax_auto)

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
    rot = R.from_euler('zy', [-angle_x, angle_y], degrees=True)
    m = rot.as_matrix()
    nx, ny, nz = data.shape
    bcenter = np.array((nx//2, ny//2, nz//2), dtype=np.float)
    offset = bcenter.T - np.dot(m, bcenter.T) + np.array([0.0, shift_y, -shift_x])
    ret = affine_transform(data, matrix=m, offset=offset, mode='nearest')
    return ret

@st.cache(persist=True, show_spinner=False)
def auto_vertical_center(image):
    background = np.mean(image[[0,1,2,-3,-2,-1],[0,1,2,-3,-2,-1]])
    thresh = (image.max()-background) * 0.2 + background
    image_work = 1.0 * image
    image_work[image<thresh] = 0

    # rough estimate of rotation
    def score_rotation(angle):
        tmp = rotate_shift_image(data=image_work, angle=angle)
        y_proj = tmp.sum(axis=0)
        percentiles = (100, 95, 90, 85, 80) # more robust than max alone
        y_values = np.percentile(y_proj, percentiles)
        err = -np.sum(y_values)
        return err
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(score_rotation, bounds=(-90, 90), method='bounded', options={'disp':0})
    angle = res.x

    # further refine rotation
    def score_rotation_shift(x):
        angle, dy, dx = x
        tmp1 = rotate_shift_image(data=image_work, angle=angle, pre_shift=(dy, dx))
        tmp2 = rotate_shift_image(data=image_work, angle=angle+180, pre_shift=(dy, dx))
        tmps = [tmp1, tmp2, tmp1[::-1,:], tmp2[::-1,:], tmp1[:,::-1], tmp2[:,::-1]]
        tmp_mean = np.zeros_like(image_work)
        for tmp in tmps: tmp_mean += tmp
        tmp_mean /= len(tmps)
        err = 0
        for tmp in tmps:
            err += np.sum(np.abs(tmp - tmp_mean))
        err /= len(tmps) * image_work.size
        return err
    from scipy.optimize import fmin
    res = fmin(score_rotation_shift, x0=(angle, 0, 0), xtol=1e-2, disp=0)
    angle = res[0]  # dy, dx are not robust enough
    if angle>90: angle-=180
    elif angle<-90: angle+=180

    # refine dx 
    image_work = rotate_shift_image(data=image_work, angle=angle)
    y = np.sum(image_work, axis=0)
    n = len(y)
    from scipy.ndimage.measurements import center_of_mass
    cx = int(round(center_of_mass(y)[0]))
    max_shift = abs((cx-n//2)*2)+3

    import scipy.interpolate as interpolate
    x = np.arange(3*n)
    f = interpolate.interp1d(x, np.tile(y, 3), kind='cubic')    # avoid out-of-bound errors
    def score_shift(dx):
        x_tmp = x[n:2*n]-dx
        tmp = f(x_tmp)
        err = np.sum(np.abs(tmp-tmp[::-1]))
        return err
    res = minimize_scalar(score_shift, bounds=(-max_shift, max_shift), method='bounded', options={'disp':0})
    dx = res.x + (0.0 if n%2 else 0.5)
    return angle, dx

@st.cache(persist=True, show_spinner=False)
def rotate_shift_image(data, angle=0, pre_shift=(0, 0), post_shift=(0, 0), rotation_center=None, order=1):
    # pre_shift/rotation_center/post_shift: [y, x]
    if angle==0 and pre_shift==[0,0] and post_shift==[0,0]: return data*1.0
    ny, nx = data.shape
    if rotation_center is None:
        rotation_center = np.array((ny//2, nx//2), dtype=np.float32)
    ang = np.deg2rad(angle)
    m = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    pre_dy, pre_dx = pre_shift    
    post_dy, post_dx = post_shift

    offset = -np.dot(m, np.array([post_dy, post_dx], dtype=np.float32).T) # post_rotation shift
    offset += np.array(rotation_center, dtype=np.float32).T - np.dot(m, np.array(rotation_center, dtype=np.float32).T)  # rotation around the specified center
    offset += -np.array([pre_dy, pre_dx], dtype=np.float32).T     # pre-rotation shift

    from scipy.ndimage import affine_transform
    ret = affine_transform(data, matrix=m, offset=offset, order=order, mode='constant')
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
        data = mrc.data * 1.0
    return data, apix

@st.cache(persist=True, show_spinner=False)
def setup_anonymous_usage_tracking():
    try:
        import pathlib, stat
        index_file = pathlib.Path(st.__file__).parent / "static/index.html"
        index_file.chmod(stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)
        txt = index_file.read_text()
        if txt.find("gtag/js?")==-1:
            txt = txt.replace("<head>", '''<head><script async src="https://www.googletagmanager.com/gtag/js?id=G-8Z99BDVHTC"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-8Z99BDVHTC');</script>''')
            index_file.write_text(txt)
    except:
        pass

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    main()
