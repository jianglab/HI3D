""" 
MIT License

Copyright (c) 2020-2024 Wen Jiang

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
required_packages = "streamlit numpy scipy bokeh trackpy kneebow".split()
import_with_auto_install(required_packages)

import streamlit as st
import numpy as np
from scipy.ndimage import map_coordinates
import math, random
import gc
gc.enable()

#from memory_profiler import profile
#@profile(precision=4)
def main():
    title = "HI3D: Helical indexing using the cylindrical projection of a 3D map"
    st.set_page_config(page_title=title, layout="wide")

    hosted, host = is_hosted(return_host=True)
    if hosted and host in ['heroku']:
        st.error(f"This app hosted on Heroku will be unavailable starting November 28, 2022 [when Heroku discontinues free hosting service](https://blog.heroku.com/next-chapter). Please switch to [the same app hosted elsewhere](https://helical-indexing-hi3d.streamlit.app)")

    st.title(title)

    st.elements.utils._shown_default_value_warning = True

    if "input_mode" not in st.session_state:  # only run once at the start of the session
        parse_query_parameters()
    
    if is_hosted():
        max_map_size  = mem_quota()/2    # MB
        max_map_dim   = int(pow(max_map_size*pow(2, 20)/4, 1./3.)//10*10)    # pixels in any dimension
        stop_map_size = mem_quota()*0.75 # MB
    else:
        max_map_size = -1   # no limit
        max_map_dim  = -1
    if max_map_size>0:
        warning_map_size = f"Due to the resource limit ({mem_quota():.1f} MB memory cap) of the hosting service, the maximal map size should be {max_map_size:.1f} MB ({max_map_dim}x{max_map_dim}x{max_map_dim} voxels) or less to avoid crashing the server process"

    msg_hint = f"There are a few things you can try:  \n"
    msg_hint+= f"· Ensure the map is vertical along Z-axis and centered in XY plane  \n"
    msg_hint+= f"· Change \"Axial step size\" to a larger value (1 → 2 or 3 Å) for large rise structures or a smaller value (1 → 0.2 Å) for small rise structures (for example, amyloids)  \n"
    msg_hint+= f"· Change \"Peak width\" and \"Peak height\" to approximate the size/shape of the peaks in the autocorrelation image  \n"
    msg_hint+= f"· Manually inspect the autocorrelation image at the center of the screen, use mouse hover tips to reveal the corresponding twist/rise values at the pixel under the mouse pointer  \n"

    col1, col2, col3, col4 = st.columns((1.0, 3.2, 0.6, 1.15))

    msg_empty = col2.empty()

    with col1:
        with st.expander(label="README", expanded=False):
            st.write("This Web app considers a biological helical structure as a 2D crystal that has been rolled up into a cylindrical tube while preserving the original lattice. The indexing process is thus to computationally reverse this process: the 3D helical structure is first unrolled into a 2D image using cylindrical projection, and then the 2D lattice parameters are automatically identified from which the helical parameters (twist, rise, and cyclic symmetry) are derived. The auto-correlation function (ACF) of the cylindrical projection is used to produce a lattice with sharper peaks. Two distinct lattice identification methods, one for generical 2D lattices and one specifically for helical lattices, are used to find a consistent solution.  \n  \nTips: play with the rmin/rmax, #peaks, axial step size parameters if consistent helical parameters cannot be obtained with the default parameters. Use a larger axial step size (for example 2Å) for a structure with large rise. Use a smaller axial step size (for example 0.2Å) for a structure (such Tau fibril) with small rise.")
        
        data = None
        da_auto = 1.0
        dz_auto = 1.0
        input_modes = {0:"upload", 1:"url", 2:"emd-xxxxx"}
        help = "Only maps in MRC (.mrc) or CCP4 (.map) format are supported. Compressed maps (.gz) will be automatically decompressed"
        if max_map_size>0: help += f". {warning_map_size}"
        input_mode = st.radio(label="How to obtain the input map:", options=list(input_modes.keys()), format_func=lambda i:input_modes[i], index=2, horizontal=True, help=help, key="input_mode")
        is_emd = False
        emdb_ids_all, emdb_ids_helical, methods = get_emdb_ids()
        if input_mode == 0: # "upload a MRC file":
            label = "Upload a map in MRC or CCP4 format"
            help = None
            if max_map_size>0: help = warning_map_size
            fileobj = st.file_uploader(label, type=['mrc', 'map', 'map.gz'], help=help, key="file_upload")
            if fileobj is not None:
                emd_id = extract_emd_id(fileobj.name)
                is_emd = emd_id is not None and emd_id in emdb_ids_helical
                data, map_crs, apix = get_3d_map_from_uploaded_file(fileobj)
                nz, ny, nx = data.shape
                if nz<32:
                    st.warning(f"The uploaded file {fileobj.name} ({nx}x{ny}x{nz}) is not a 3D map")
                    data = None
        elif input_mode == 1: # "url":
            url_default = get_emdb_map_url("emd-10499")
            help = "An online url (http:// or ftp://) or a local file path (/path/to/your/structure.mrc)"
            if max_map_size>0: help += f". {warning_map_size}"
            url = st.text_input(label="Input the url of a 3D map:", value=url_default, help=help, key="url")
            emd_id = extract_emd_id(url)
            is_emd = emd_id is not None and emd_id in emdb_ids_helical
            data, map_crs, apix = get_3d_map_from_url(url.strip())
            nz, ny, nx = data.shape
            if nz<32:
                st.warning(f"{url} points to a file ({nx}x{ny}x{nz}) that is not a 3D map")
                data = None
        elif input_mode == 2:
            if not emdb_ids_all:
                st.warning("failed to obtained a list of helical structures in EMDB")
                return
            url = "https://www.ebi.ac.uk/emdb/search/*%20AND%20structure_determination_method:%22helical%22?rows=10&sort=release_date%20desc"
            st.markdown(f'[All {len(emdb_ids_helical)} helical structures in EMDB]({url})')
            emd_id_default = "emd-10499"
            help = "Randomly select another helical structure in EMDB"
            if max_map_size>0: help += f". {warning_map_size}"
            button_clicked = st.button(label="Change EMDB ID", help=help)
            if button_clicked:
                import random
                st.session_state.emd_id = 'emd-' + random.choice(emdb_ids_helical)
            help = None
            if max_map_size>0: help = warning_map_size
            label = "Input an EMDB ID (emd-xxxxx):"
            emd_id = st.text_input(label=label, value=emd_id_default, key='emd_id', help=help)
            emd_id = emd_id.lower().split("emd-")[-1]
            if emd_id not in emdb_ids_all:
                import random
                msg = f"EMD-{emd_id} is not a valid EMDB entry. Please input a valid id (for example, a randomly selected valid id 'emd-{random.choice(emdb_ids_helical)}')"
                st.warning(msg)
                return
            elif emd_id not in emdb_ids_helical:
                msg= f"EMD-{emd_id} is in EMDB but annotated as a '{methods[emdb_ids_all.index(emd_id)]}' structure, not a helical structure" 
                st.warning(msg)
            params = get_emdb_parameters(emd_id)
            if params is None:
                msg = f"EMD-{emd_id}: could not retrieve information"
                st.warning(msg)
                return
            resolution = params['resolution']
            msg = f'[EMD-{emd_id}](https://www.ebi.ac.uk/emdb/entry/EMD-{emd_id}) | resolution={resolution}Å'
            if max_map_size>0 and params and "nz" in params and "ny" in params and "nx" in params:
                nz = params["nz"]
                ny = params["ny"]
                nx = params["nx"]
                map_size = nz*ny*nx*4 / pow(2, 20)
                if map_size>stop_map_size:
                    msg_map_too_large = f"As the map size ({map_size:.1f} MB, {nx}x{ny}x{nz} voxels) is too large for the resource limit ({mem_quota():.1f} MB memory cap) of the hosting service, HI3D will stop analyzing it to avoid crashing the server. Please bin/crop your map so that it is {max_map_size} MB ({max_map_dim}x{max_map_dim}x{max_map_dim} voxels) or less, and then try again. Please check the [HI3D web site](https://jiang.bio.purdue.edu/hi3d) to learn how to run HI3D on your local computer with larger memory to support large maps"
                    msg_empty.warning(msg_map_too_large)
                    st.stop()
            if params and is_amyloid(params, cutoff=6):
                dz_auto = 0.2
            if params and "twist" in params and "rise" in params:
                msg += f"  \ntwist={params['twist']}° | rise={params['rise']}Å"
                if "csym" in params: msg += f" | c{params['csym']}"
            else:
                msg +=  "  \n*helical params not available*"
            st.markdown(msg)
            data, map_crs, apix = get_emdb_map(emd_id)
            if data is None:
                st.warning(f"Failed to download [EMD-{emd_id}](https://www.ebi.ac.uk/emdb/entry/EMD-{emd_id})")
                return
            is_emd = emd_id in emdb_ids_helical

        if data is None:
            return
        
        if map_crs != [1, 2, 3]:
            map_crs_to_xyz = {1:'x', 2:'y', 3:'z'}
            xyz = ','.join([map_crs_to_xyz[int(i)] for i in map_crs])
            label = f":red[Change map axes order from {xyz} to:]"
            st.text_input(label=label, value="x,y,z", max_chars=5, help=f"Cryo-EM field assumes that the map axes are in the order of x,y,z (e.g. MRC/CCP4 header fields mapc=1, mapr=2, maps=3). However, you map header has a different order {xyz} (mapc={map_crs[0]}, mapr={map_crs[1]}, maps={map_crs[2]})", key="target_map_axes_order")
            try:
                target_map_axes_order = st.session_state.target_map_axes_order.lower().split(",")
                assert len(target_map_axes_order) == 3
                xyz_to_map_crs = {'x':1, 'y':2, 'z':3}
                target_map_crs = [xyz_to_map_crs[a] for a in target_map_axes_order]
            except:
                st.warning(f"Incorrect value {st.session_state.target_map_axes_order}. I will use the default value x,y,z")
                target_map_crs = [1, 2, 3]
            data = change_mrc_map_crs_order(data=data, current_order=map_crs, target_order=target_map_crs)

        nz, ny, nx = data.shape
        st.markdown(f'{nx}x{ny}x{nz} voxels | {round(apix,4):g} Å/voxel')

        if max_map_size>0:
            map_size = nz*ny*nx*4 / pow(2, 20)
            if map_size>stop_map_size:
                msg_map_too_large = f"As the map size ({map_size:.1f} MB, {nx}x{ny}x{nz} voxels) is too large for the resource limit ({mem_quota():.1f} MB memory cap) of the hosting service, HI3D will stop analyzing it to avoid crashing the server. Please bin/crop your map so that it is {max_map_size} MB ({max_map_dim}x{max_map_dim}x{max_map_dim} voxels) or less, and then try again. Please check the [HI3D web site](https://jiang.bio.purdue.edu/hi3d) to learn how to run HI3D on your local computer with larger memory to support large maps"
                msg_empty.warning(msg_map_too_large)
                st.stop()
            elif map_size>max_map_size:
                reduce_map_size = st.checkbox(f"Reduce map size to < {max_map_size} MB", value=True, key="reduce_map_size")
                if reduce_map_size:
                    data_small, bin = minimal_grids(data, max_map_dim)
                    del data
                    data = data_small * 1.0
                    del data_small
                    apix *= bin
                    nz, ny, nx = data.shape
                    st.markdown(f'{nx}x{ny}x{nz} voxels | {round(apix,4):g} Å/voxel')
                else:
                    msg = f"{warning_map_size}. If this map ({map_size:.1f}>{max_map_size } MB) indeed crashes the server process, please reduce the map size by binning the map or cropping the empty padding space around the structure, and then try again. If the crashing persists, please check the [HI3D web site](https://jiang.bio.purdue.edu/hi3d) to learn how to run HI3D on your local computer with larger memory to support large maps"
                    msg_empty.warning(msg)
        
        vmin, vmax = data.min(), data.max()
        if vmin == vmax:
            st.warning(f"The map is blank: min={vmin} max={vmax}. Please provide a meaningful 3D map")
            st.stop()

        axis_mapping = {3:'X/Y', 2:'X', 1:'Y', 0:'Z'}
        section_axis = st.radio(label="Display a section along this axis:", options=list(axis_mapping.keys()), format_func=lambda i:axis_mapping[i], index=0, horizontal=True, key="section_axis")
        mapping = {0:nz, 1:ny, 2:nx, 3:min(nx,ny)}
        n = mapping[section_axis]
        section_index = st.slider(label="Choose a section to display:", min_value=-n//2, max_value=n-n//2-1, value=0, step=1)
        section_index += n//2
        container_image = st.container()
        
        expanded = False if is_emd else True
        with st.expander(label="Transform the map", expanded=expanded):
            do_threshold = st.checkbox("Threshold the map", value=False, key="do_threshold")
            if do_threshold:
                data_min, data_max = float(data.min()), float(data.max())
                background = np.mean(data[[0,1,2,-3,-2,-1],[0,1,2,-3,-2,-1]])
                thresh_auto = (data_max-background) * 0.2 + background
                thresh = st.number_input(label="Minimal voxel value:", min_value=data_min, max_value=data_max, value=float(round(thresh_auto,6)), step=float((data_max-data_min)/1000.), format="%g", key="thresh")
            else:
                thresh = None
            if thresh is not None:
                data = data * 1.0
                data[data<thresh] = 0

            do_transform = st.checkbox("Center & verticalize", value= not (is_emd or is_hosted()), key="do_transform")
            if do_transform:
                with st.form("do_transform_form"):
                    rotx_auto, shifty_auto = auto_vertical_center(np.sum(data, axis=2))
                    roty_auto, shiftx_auto = auto_vertical_center(np.sum(data, axis=1))
                    if "rotx" in st.session_state: rotx_auto = st.session_state.rotx
                    if "roty" in st.session_state: roty_auto = st.session_state.roty
                    if "shiftx" in st.session_state: shiftx_auto = st.session_state.shiftx/apix # unit: pixel
                    if "shifty" in st.session_state: shifty_auto = st.session_state.shifty/apix # unit: pixel
                    rotx = st.number_input(label="Rotate map around X-axis (°):", min_value=-90., max_value=90., value=round(rotx_auto,2), step=1.0, format="%g", key="rotx")
                    roty = st.number_input(label="Rotate map around Y-axis (°):", min_value=-90., max_value=90., value=round(roty_auto,2), step=1.0, format="%g", key="roty")
                    shiftx = st.number_input(label="Shift map along X-axis (Å):", min_value=-nx//2*apix, max_value=nx//2*apix, value=round(min(max(-nx//2*apix, shiftx_auto*apix), nx//2*apix), 2), step=1.0, format="%g", key="shiftx")     # unit: Å
                    shifty = st.number_input(label="Shift map along Y-axis (Å):", min_value=-ny//2*apix, max_value=ny//2*apix, value=round(min(max(-ny//2*apix, shifty_auto*apix), ny//2*apix), 2), step=1.0, format="%g", key="shifty")     # unit: Å
                    shiftz = st.number_input(label="Shift map along Z-axis (Å):", min_value=-nz//2*apix, max_value=nz//2*apix, value=0.0, step=1.0, format="%g", key="shiftz")
                    st.form_submit_button("Submit")
            else:
                rotx, roty, shiftx, shifty, shiftz = 0., 0., 0., 0., 0.

        if section_axis == 3:
            image = np.zeros((nz, max(ny, nx)), dtype=data.dtype)
            image_x = np.squeeze(np.take(data, indices=[section_index], axis=2))
            image_y = np.squeeze(np.take(data, indices=[section_index], axis=1))
            image[:, :ny//2] = image_x[:, :ny//2]
            image[:, -nx//2:] = image_y[:, nx//2:]
            image[:, ny//2-1] = np.max(image)
        else:
            image = np.squeeze(np.take(data, indices=[section_index], axis=section_axis))

        h, w = image.shape
        if thresh is not None or rotx or roty or shiftx or shifty or shiftz:
            data = transform_map(data, shift_x=shiftx/apix, shift_y=-shifty/apix, shift_z=-shiftz/apix, angle_x=-rotx, angle_y=-roty)
            if section_axis == 3:
                image2 = np.squeeze(np.take(data, indices=[section_index], axis=2))
                image2_y = np.squeeze(np.take(data, indices=[section_index], axis=1))
                image2[:, ny//2:]= image2_y[:, nx//2:]
                image2[:, ny//2-1] = 0
            else:
                image2 = np.squeeze(np.take(data, indices=[section_index], axis=section_axis))
            with container_image:
                tooltips = [("x", "$x"), ('y', '$y'), ('val', '@image')]
                fig1 = generate_bokeh_figure(image, apix, apix, title=f"Original", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_angle_tooltip=True, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)
                fig2 = generate_bokeh_figure(image2, apix, apix, title=f"Transformed", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_angle_tooltip=True, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)

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
                del fig_image, image, image2
        else:
            with container_image:
                tooltips = [("x", "$x"), ('y', '$y'), ('val', '@image')]
                fig_image = generate_bokeh_figure(image, 1, 1, title=f"Original", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=False, crosshair_color="white", aspect_ratio=w/h)
                st.bokeh_chart(fig_image, use_container_width=True)
                del fig_image, image

        rad_plot = st.empty()

        with st.expander(label="Select radial range", expanded=False):
            radprofile = compute_radial_profile(data)
            rad = np.arange(len(radprofile)) * apix
            rmin_auto, rmax_auto = estimate_radial_range(radprofile, thresh_ratio=0.1)
            rmin = st.number_input('Minimal radius (Å)', value=round(rmin_auto*apix,1), min_value=0.0, max_value=round(nx//2*apix,1), step=1.0, format="%g", key="rmin")
            rmax = st.number_input('Maximal radius (Å)', value=round(rmax_auto*apix,1), min_value=0.0, max_value=round(nx//2*apix,1), step=1.0, format="%g", key="rmax")
            if rmax<=rmin:
                st.warning(f"rmax(={rmax}) should be larger than rmin(={rmin})")
                return

        with st.expander(label="Download data", expanded=False):
            import pandas as pd
            st.download_button(
                label="Radial profile",
                data=pd.DataFrame(np.hstack((rad.reshape(len(rad), 1), radprofile.reshape(len(rad), 1))), columns=["radius (Å)", "density"]).round(6).to_csv().encode('utf-8'),
                file_name='radial_profile.csv',
                mime='text/csv',
            )

        from bokeh.plotting import figure
        tools = 'box_zoom,crosshair,hover,pan,reset,save,wheel_zoom'
        tooltips = [("r", "@x{0.0}Å"), ("val", "@y{0.0}"),]
        fig_radprofile = figure(title="density radial profile", x_axis_label="r (Å)", y_axis_label="pixel value", tools=tools, tooltips=tooltips, aspect_ratio=2)
        fig_radprofile.line(rad, radprofile, line_width=2, color='red')
        del rad, radprofile
        
        from bokeh.models import Span
        rmin_span = Span(location=rmin, dimension='height', line_color='green', line_dash='dashed', line_width=3)
        rmax_span = Span(location=rmax, dimension='height', line_color='green', line_dash='dashed', line_width=3)
        fig_radprofile.add_layout(rmin_span)
        fig_radprofile.add_layout(rmax_span)
        fig_radprofile.yaxis.visible = False
        with rad_plot:
            st.bokeh_chart(fig_radprofile, use_container_width=True)
            del fig_radprofile
        
        with st.expander(label="Server info", expanded=False):
            server_info_empty = st.empty()
            #server_info = f"Host: {get_hostname()}  \n"
            #server_info+= f"Account: {get_username()}"
            server_info = f"Uptime: {up_time():.1f} s  \n"
            server_info+= f"Mem (total): {mem_info()[0]:.1f} MB  \n"
            server_info+= f"Mem (quota): {mem_quota():.1f} MB  \n"
            server_info+= "Mem (used): {mem_used:.1f} MB"
            server_info_empty.markdown(server_info.format(mem_used=mem_used()))

        share_url = st.checkbox('Show sharable URL', value=False, help="Include relevant parameters in the browser URL to allow you to share the URL and reproduce the plots", key="share_url")
        if share_url:
            show_qr = st.checkbox('Show QR code of the URL', value=False, help="Display the QR code of the sharable URL", key="show_qr")
        else:
            show_qr = False

        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    with col3:
        da = st.number_input('Angular step size (°)', value=da_auto, min_value=0.1, max_value=10., step=0.1, format="%g", help="Set the azimuthal angle step size for the computation of the cylindric projection", key="da")
        dz = st.number_input('Axial step size (Å)', value=dz_auto, min_value=0.1, max_value=10., step=0.1, format="%g", help="Set the axial step size for the computation of the cylindric projection. Use a smaller step size (such as 0.2 Å) for a helical structure with small rise (such as a protein fibril with rise ~2.3-2.4 Å or ~4.7-4.8 Å)", key="dz")
        peak_width = st.number_input('Peak width (°)', value=max(1.0, da*9.0), min_value=0.1, max_value=60.0, step=1.0, format="%g", help="Set the expected peak width (°) in the auto-correlation image", key="peak_width")
        peak_height = st.number_input('Peak height (Å)', value=max(1.0, dz*9.0), min_value=0.1, max_value=30.0, step=1.0, format="%g", help="Set the expected peak height (Å) in the auto-correlation image", key="peak_height")
        npeaks_empty = st.empty()
        acf_2rounds = st.checkbox(label="ACF 2x", value=False, help="Compute ACF of ACF", key="acf_2rounds")
        show_scf = st.checkbox(label="SCF", value=False, help="Use the self-correlation function (SCF) variant of ACF (e.g. |F|->sqrt(|F|)", key="show_scf")
        
        #data = auto_masking(data)
        #data = minimal_grids(data)
        cylproj = cylindrical_projection(data, da=da, dz=dz/apix, dr=1, rmin=rmin/apix, rmax=rmax/apix, interpolation_order=1)
        del data
        server_info_empty.markdown(server_info.format(mem_used=mem_used()))

        cylproj_work = cylproj
        draw_cylproj_box = False

        st.subheader("Display:")
        show_cylproj = st.checkbox(label="Cylindrical projection", value=True, help="Display the cylindric projection", key="show_cylproj")
        if show_cylproj:
            nz, na = cylproj.shape
            ang_min = st.number_input('Minimal angle (°)', value=-180., min_value=-180.0, max_value=180., step=1.0, format="%g", help="Set the minimal azimuthal angle of the cylindric projection to be included to compute the auto-correlation function", key="ang_min")
            ang_max = st.number_input('Maximal angle (°)', value=180., min_value=-180.0, max_value=180., step=1.0, format="%g", help="Set the maximal azimuthal angle of the cylindric projection to be included to compute the auto-correlation function. If this angle is smaller than *Minimal angle*, the angular range will be *Minimal angle* to 360 and -360 to *Maximal angle*", key="ang_max")
            z_min = st.number_input('Minimal z (Å)', value=round(-nz//2*dz,1), min_value=round(-nz//2*dz,1), max_value=round(nz//2*dz,1), step=1.0, format="%g", help="Set the minimal axial section of the cylindric projection to be included to compute the auto-correlation function", key="z_min")
            z_max = st.number_input('Maximal z (Å)', value=round(nz//2*dz,1), min_value=round(-nz//2*dz,1), max_value=round(nz//2*dz,1), step=1.0, format="%g", help="Set the maximal axial section of the cylindric projection to be included to compute the auto-correlation function", key="z_max")
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
        del cylproj_work
        show_acf = st.checkbox(label="ACF", value=True, help="Display the auto-correlation function (ACF)", key="show_acf")
        if show_acf:
            show_peaks_empty = st.empty()

        acf = auto_correlation(cylproj_square, sqrt=show_scf, high_pass_fraction=1./cylproj_square.shape[0])
        if acf_2rounds:
            acf = auto_correlation(acf, sqrt=show_scf, high_pass_fraction=1./cylproj_square.shape[0])
        del cylproj_square

        peaks, masses = find_peaks(acf, da=da, dz=dz, peak_width=peak_width, peak_height=peak_height, minmass=1.0)
        if peaks is not None:
            npeaks_all = len(peaks)
            from kneebow.rotor import Rotor 
            rotor = Rotor()
            rotor.fit_rotate( np.vstack((np.arange(len(masses)-3), masses.iloc[3:])).T )
            npeaks_guess = min(npeaks_all, rotor.get_elbow_index()+3)
            npeaks = int(npeaks_empty.number_input('# peaks to use', value=npeaks_guess, min_value=3, max_value=npeaks_all, step=2, help=f"The {npeaks_all} peaks detected in the auto-correlation function are sorted by peak quality. This input allows you to use only the best peaks instead of all {npeaks_all} peaks to determine the lattice parameters (i.e. helical twist, rise, and csym)", key="npeaks"))

        show_arrow_empty = st.empty()
        show_lattice_empty = st.empty()
        server_info_empty.markdown(server_info.format(mem_used=mem_used()))
        
    with col4:
        if show_cylproj:
            h, w = cylproj.shape
            tooltips = [("angle", "$x°"), ('z', '$yÅ'), ('cylproj', '@image')]
            fig_cylproj = generate_bokeh_figure(cylproj, da, dz, title=f"Cylindrical Projection ({w}x{h})", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=True, crosshair_color="white", aspect_ratio=w/h)

            if draw_cylproj_box:
                if ang_min<ang_max:
                    fig_cylproj.quad(left=ang_min, right=ang_max, bottom=z_min, top=z_max, line_color=None, fill_color='yellow', fill_alpha=0.5)
                else:
                    fig_cylproj.quad(left=ang_min, right=180, bottom=z_min, top=z_max, line_color=None, fill_color='yellow', fill_alpha=0.5)
                    fig_cylproj.quad(left=-180, right=ang_max, bottom=z_min, top=z_max, line_color=None, fill_color='yellow', fill_alpha=0.5)

            st.text("") # workaround for a layout bug in streamlit 
            st.bokeh_chart(fig_cylproj, use_container_width=True)
            del fig_cylproj
            del cylproj

        if show_acf:
            st.text("") # workaround for a streamlit layout bug
            h, w = acf.shape
            tooltips = [("twist", "$x°"), ('rise', '$yÅ'), ('acf', '@image')]
            fig_acf = generate_bokeh_figure(acf, da, dz, title=f"Auto-Correlation ({w}x{h})", title_location="below", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=tooltips, show_axis=False, show_toolbar=True, crosshair_color="white", aspect_ratio=w/h)

            if peaks is not None:
                show_peaks = show_peaks_empty.checkbox(label="Peaks", value=True, help=f"Mark the {len(peaks)} peaks detected in the auto-correlation function with yellow circles", key="show_peaks")
                if show_peaks:
                    x = peaks[:npeaks, 0]
                    y = peaks[:npeaks, 1]
                    fig_acf.ellipse(x, y, width=peak_width, height=peak_height, line_width=1, line_color='yellow', fill_alpha=0)

            st.bokeh_chart(fig_acf, use_container_width=True)
            del fig_acf

        if peaks is None:
            msg_empty.warning("No peak was found from the auto-correlation image  \n" + msg_hint)
            return
        elif len(peaks)<3:
            msg_empty.warning(f"Only {len(peaks)} peaks were found. At least 3 peaks are required  \n" + msg_hint)
            return

        twist_empty = st.empty()
        rise_empty = st.empty()
        csym_empty = st.empty()
        button_refine_twist_rise = st.button("Refine twist/rise")
        server_info_empty.markdown(server_info.format(mem_used=mem_used()))

    with col2:
        h, w = acf.shape
        h2 = 900   # final plot height
        w2 = int(round(w * h2/h))//2*2
        x_axis_label="twist (°)"
        y_axis_label="rise (Å)"
        tooltips = [("twist", "$x°"), ('rise', '$yÅ'), ('acf', '@image')]
        fig_indexing = generate_bokeh_figure(image=acf, dx=da, dy=dz, title="", title_location="above", plot_width=None, plot_height=None, x_axis_label=x_axis_label, y_axis_label=y_axis_label, tooltips=tooltips, show_axis=True, show_toolbar=True, crosshair_color="white", aspect_ratio=w/h)

        # horizontal line along the equator
        from bokeh.models import Arrow, VeeHead
        fig_indexing.line([-w//2*da, (w//2-1)*da], [0, 0], line_width=2, line_color="yellow", line_dash="dashed")
        
        trc1, trc2 = fitHelicalLattice(peaks[:npeaks], acf, da=da, dz=dz)
        trc_mean = consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0)
        success = True if trc_mean else False

        if success:
            twist_tmp, rise_tmp, cn = trc_mean[0]
            twist_auto, rise_auto = refine_twist_rise(acf_image=(acf, da, dz), twist=twist_tmp, rise=rise_tmp, cn=cn)
            csym_auto = cn
        else:
            twist_auto, rise_auto, csym_auto = trc1
            msg = f"The two automated methods with default parameters have failed to obtain consistent helical parameters using {npeaks} peaks. The two solutions are:  \n"
            msg+= f"Twist per subunit:&emsp;&emsp;{round(trc1[0],2):>6.2f}&emsp;{round(trc2[0],2):>6.2f} °  \n"
            msg+= f"Rise &nbsp; per subunit:&emsp;&emsp;{round(trc1[1],2):>6.2f}&emsp;{round(trc2[1]):>6.2f} Å  \n"
            msg+= f"Csym &emsp;&emsp;&emsp;&emsp;&emsp;:&emsp;&emsp;c{int(trc1[2]):5}&emsp;&emsp;c{int(trc2[2]):5}  \n  \n"
            msg+= msg_hint
            msg_empty.warning(msg)

        twist_manual = twist_empty.number_input(label="Twist (°):", min_value=-180., max_value=180., value=float(round(twist_auto,2)), step=0.01, format="%g", help="Manually set the helical twist instead of automatically detecting it from the lattice in the auto-correlation function")
        rise_manual = rise_empty.number_input(label="Rise (Å):", min_value=0., max_value=h*dz, value=float(round(rise_auto,2)), step=0.01, format="%g", help="Manually set the helical rise instead of automatically detecting it from the lattice in the auto-correlation function")
        csym = int(csym_empty.number_input(label="Csym:", min_value=1, max_value=64, value=int(csym_auto), step=1, format="%d", help="Manually set the cyclic symmetry instead of automatically detecting it from the lattice in the auto-correlation function", key="csym"))
        
        if button_refine_twist_rise:
            twist, rise = refine_twist_rise(acf_image=(acf, da, dz), twist=twist_manual, rise=rise_manual, cn=csym)
        else:
            twist, rise = twist_manual, rise_manual
        st.session_state.twist = twist
        st.session_state.rise = rise

        fig_indexing.title.text = f"twist={round(twist,2):g}° (pitch={round(360/abs(twist)*rise, 2):g}Å) rise={round(rise,2):g}Å  csym=c{int(csym):d}"
        fig_indexing.title.align = "center"
        fig_indexing.title.text_font_size = "24px"
        fig_indexing.title.text_font_style = "normal"
        fig_indexing.xaxis.axis_label_text_font_size = "16pt"
        fig_indexing.yaxis.axis_label_text_font_size = "16pt"
        fig_indexing.xaxis.major_label_text_font_size = "12pt"
        fig_indexing.yaxis.major_label_text_font_size = "12pt"
        fig_indexing.hover[0].attachment = "vertical"

        show_arrow = show_arrow_empty.checkbox(label="Arrow", value=True, help="Show an arrow in the central panel from the center to the first lattice point corresponding to the helical twist/rise", key="show_arrow")
        if show_arrow:
            fig_indexing.add_layout(Arrow(x_start=0, y_start=0, x_end=twist, y_end=rise, line_color="yellow", line_width=4, end=VeeHead(line_color="yellow", fill_color="yellow", line_width=2)))

        show_lattice = show_lattice_empty.checkbox(label="Lattice", value=True, help="Show the helical twist/rise lattice in the central panel", key="show_lattice")
        if show_lattice:
            from bokeh.models import ColumnDataSource, Scatter
            from bokeh.palettes import Category10
            colors_avail = Category10[max(3, min(10, csym))]
            colors = [colors_avail[si % len(colors_avail)] for si in range(csym)]
            n = int(h/2*dz/rise)+1
            n = np.arange(-n, n+1)
            x = np.fmod(twist * n + np.max(n)*360, 360)
            x[x>180] -= 360
            y = rise * n
            for si in range(csym):
                xsym = np.fmod(x + 360/csym * si, 360)
                xsym[xsym>180] -= 360
                source = ColumnDataSource(dict(x=xsym, y=y))
                scatter = Scatter(x='x', y='y', marker='circle', size=10, line_width=3, line_color=colors[si], fill_color=None)
                fig_indexing.add_glyph(source, scatter)

        from bokeh.models import CustomJS
        from bokeh.events import MouseEnter
        title_js = CustomJS(args=dict(title=title), code="""
            document.title=title
        """)
        fig_indexing.js_on_event(MouseEnter, title_js)

        st.text("") # workaround for a layout bug in streamlit 
        st.bokeh_chart(fig_indexing, use_container_width=True)
        del fig_indexing
        del acf

    if share_url:
        set_query_parameters()
        if show_qr:
            with col2:
                qr_image = qr_code()
                st.image(qr_image)
    else:
        st.query_params.clear()

    with col2:
        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu/HI3D). Report problems to [HI3D@GitHub](https://github.com/jianglab/hi3d/issues)*")
        st.markdown("Please cite: *Sun, C., Gonzalez, B., & Jiang, W. (2022). Helical Indexing in Real Space. Scientific Reports, 12(1), 1–11. https://doi.org/10.1038/s41598-022-11382-7*")

    server_info_empty.markdown(server_info.format(mem_used=mem_used()))

def generate_bokeh_figure(image, dx, dy, title="", title_location="below", plot_width=None, plot_height=None, x_axis_label='x', y_axis_label='y', tooltips=None, show_angle_tooltip=False, show_axis=True, show_toolbar=True, crosshair_color="white", aspect_ratio=None):
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

    # add hover tool only for the image
    from bokeh.models.tools import HoverTool, CrosshairTool
    if tooltips is not None and show_angle_tooltip:
        tooltips = tooltips + [("angle", '°')]
    from bokeh.models import LinearColorMapper
    color_mapper = LinearColorMapper(palette='Greys256')    # Greys256, Viridis256
    fig_image = fig.image(source=source_data, image='image', color_mapper=color_mapper, x='x', y='y', dw='dw', dh='dh')

    image_hover = HoverTool(renderers=[fig_image], tooltips=tooltips)
    fig.add_tools(image_hover)

    # avoid the need for embedding angle image -> smaller fig object and less data to transfer
    if tooltips is not None and show_angle_tooltip:
        mousemove_callback_code = """
        var x = cb_obj.x
        var y = cb_obj.y
        var angle = Math.atan2(y, x) * 180 / Math.PI - 90
        if (angle < -180) angle += 360
        angle = Math.round(angle*10)/10
        hover.tooltips[hover.tooltips.length - 1][1] = angle.toString() + "°"
        """
        from bokeh.models import CustomJS
        from bokeh.events import MouseMove
        mousemove_callback = CustomJS(args={"hover":fig.hover[0]}, code=mousemove_callback_code)
        fig.js_on_event(MouseMove, mousemove_callback)

    crosshair = [t for t in fig.tools if isinstance(t, CrosshairTool)]
    if crosshair: 
        for ch in crosshair: ch.line_color = crosshair_color
    return fig

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def fitHelicalLattice(peaks, acf, da=1.0, dz=1.0):
    if len(peaks) < 3:
        #st.warning(f"WARNING: only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (None, None, peaks)

    trc1s = []
    trc2s = []
    consistent_solution_found = False
    nmax = len(peaks) if len(peaks)%2 else len(peaks)-1
    for n in range(nmax, min(7, nmax)-1, -2):
        trc1 = getHelicalLattice(peaks[:n])
        trc2 = getGenericLattice(peaks[:n])
        trc1s.append(trc1)
        trc2s.append(trc2)
        if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0):
            consistent_solution_found = True
            break
    
    if not consistent_solution_found: 
        for _ in range(100):
            from random import randint, sample
            if len(peaks)//2 > 5:   # stronger peaks
                n = randint(5, len(peaks)//2)
                random_choices = sorted(sample(range(2*n), k=n))
            else:
                n = randint(3, len(peaks))
                random_choices = sorted(sample(range(len(peaks)), k=n))
            if 0 not in random_choices: random_choices = [0] + random_choices
            peaks_random = peaks[random_choices]
            trc1 = getHelicalLattice(peaks_random)
            trc2 = getGenericLattice(peaks_random)
            trc1s.append(trc1)
            trc2s.append(trc2)
            if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1):
                consistent_solution_found = True
                break
    
    if not consistent_solution_found: 
        trc_mean = consistent_twist_rise_cn_sets(trc1s, trc2s, epsilon=1)
        if trc_mean:
            _, trc1, trc2 = trc_mean
        else:
            trc1s = np.array(trc1s)
            trc2s = np.array(trc2s)
            trc1 = list(geometric_median(X=trc1s[:,:2])) + [int(np.median(trc1s[:,2]))]
            trc2 = list(geometric_median(X=trc2s[:,:2])) + [int(np.median(trc2s[:,2]))]

    twist1, rise1, cn1 = trc1
    twist1, rise1 = refine_twist_rise(acf_image=(acf, da, dz), twist=twist1, rise=rise1, cn=cn1)
    twist2, rise2, cn2 = trc2
    twist2, rise2 = refine_twist_rise(acf_image=(acf, da, dz), twist=twist2, rise=rise2, cn=cn2)
    
    return (twist1, rise1, cn1), (twist2, rise2, cn2)

def consistent_twist_rise_cn_sets(twist_rise_cn_set_1, twist_rise_cn_set_2, epsilon=1.0):
    def angle_difference(angle1, angle2):
        err = abs((angle1 - angle2) % 360)
        if err > 180: err -= 360
        err = abs(err)
        return err

    def angle_mean(angle1, angle2):
        angles = np.deg2rad([angle1, angle2])
        ret = np.rad2deg(np.arctan2( np.sin(angles).sum(), np.cos(angles).sum()))
        return ret

    def consistent_twist_rise_cn_pair(twist_rise_cn_1, twist_rise_cn_2, epsilon=1.0):
        def good_twist_rise_cn(twist, rise, cn, epsilon=0.1):
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
        if not good_twist_rise_cn(twist1, rise1, cn1, epsilon=0.1): return None
        if not good_twist_rise_cn(twist2, rise2, cn2, epsilon=0.1): return None
        if cn1==cn2 and abs(rise2-rise1)<epsilon and angle_difference(twist1, twist2)<epsilon:
            cn = cn1
            rise_tmp = (rise1+rise2)/2
            twist_tmp = angle_mean(twist1, twist2)
            return twist_tmp, rise_tmp, int(cn)
        else:
            return None
    for twist_rise_cn_1 in twist_rise_cn_set_1:
        for twist_rise_cn_2 in twist_rise_cn_set_2:
            trc = consistent_twist_rise_cn_pair(twist_rise_cn_1, twist_rise_cn_2, epsilon=epsilon)
            if trc: return (trc, twist_rise_cn_1, twist_rise_cn_2)
    return None

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def refine_twist_rise(acf_image, twist, rise, cn):
    from scipy.optimize import minimize
    if rise<=0: return twist, rise
    cn = int(cn)

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
    twist_opt = set_to_periodic_range(twist_opt, min=-180, max=180)
    return twist_opt, rise_opt

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
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

    return (twist, rise, int(cn))

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
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

    minLength = max(1.0, min(np.linalg.norm(a), np.linalg.norm(b)) * 0.9)
    vs_on_equator = []
    vs_off_equator = []
    maxI = 10
    for i in range(-maxI, maxI + 1):
        for j in range(-maxI, maxI + 1):
            if i or j:
                v = i * a + j * b
                v[0] = set_to_periodic_range(v[0], min=-180, max=180)
                if np.linalg.norm(v) > minLength:
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
    return twist, rise, int(cn)

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def find_peaks(acf, da, dz, peak_width=9.0, peak_height=9.0, minmass=1.0, max_peaks=71):
    from trackpy import locate, refine_com
    # diameter: fraction of the maximal dimension of the image (acf)
    diameter_height = int(peak_height/dz+0.5)//2*2+1
    diameter_width = int(peak_width/da+0.5)//2*2+1
    pad_width = diameter_width * 3
    acf2 = np.hstack((acf[:, -pad_width:], acf, acf[:, :pad_width]))   # to handle peaks at left/right edges
    
    # try a few different shapes around the starting height/width
    params = []
    for hf, wf in ((1, 1), (1, 2), (0.5, 0.5), (0.5, 1)):
        params += [(int(diameter_height*hf+0.5)//2*2+1, int(diameter_width*wf+0.5)//2*2+1)]
    while True:
        results = []
        for h, w in params:
            if h<1 or w<1: continue
            try:
                f = locate(acf2, diameter=(h, w), minmass=minmass, separation=(h*2, w*2))
                if len(f):
                    results.append((f["mass"].sum()*np.power(len(f), -0.5), len(f), f, h, w))
                    try:
                        f_refined = refine_com(raw_image=acf2, image=acf2, radius=(h//2, w//2), coords=f)    # radius must be even integers?
                        results.append((f_refined["mass"].sum()*np.power(len(f_refined), -0.5), len(f_refined), f_refined, h, w))
                    except:
                        pass
            except:
                pass
            if len(results) and results[-1][1] > 31: break
        results.sort(key=lambda x: x[0], reverse=True)
        if len(results) and results[0][1]>3: break
        minmass *= 0.9
        if minmass<0.1: return None, None
    f = results[0][2]

    f.loc[:, 'x'] -= pad_width
    f = f.loc[ (f['x'] >= 0) & (f['x'] < acf.shape[1]) ]
    f = f.sort_values(["mass"], ascending=False)[:max_peaks]
    peaks = np.zeros((len(f), 2), dtype=float)
    peaks[:, 0] = f['x'].values - acf.shape[1]//2    # pixel
    peaks[:, 1] = f['y'].values - acf.shape[0]//2    # pixel
    peaks[:, 0] *= da  # the values are now in degree
    peaks[:, 1] *= dz  # the values are now in Angstrom
    return peaks, f["mass"]

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def auto_correlation(data, sqrt=False, high_pass_fraction=0):
    from scipy.signal import correlate2d
    fft = np.fft.rfft2(data)
    product = fft*np.conj(fft)
    if sqrt: product = np.sqrt(product)
    if 0<high_pass_fraction<=1:
        nz, na = product.shape
        Z, A = np.meshgrid(np.arange(-nz//2, nz//2, dtype=float), np.arange(-na//2, na//2, dtype=float), indexing='ij')
        Z /= nz//2
        A /= na//2
        f2 = np.log(2)/(high_pass_fraction**2)
        filter = 1.0 - np.exp(- f2 * Z**2) # Z-direction only
        product *= np.fft.fftshift(filter)
    corr = np.fft.fftshift(np.fft.irfft2(product))
    corr -= np.median(corr, axis=1, keepdims=True)
    corr = normalize(corr)
    if sqrt:
        corr = np.power(np.log1p(corr), 1/3)   # make weaker peaks brighter
        corr = normalize(corr)
    return corr

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
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

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def cylindrical_projection(map3d, da=1, dz=1, dr=1, rmin=0, rmax=-1, interpolation_order=1):
    # da: degree
    # dr/dz/rmin/rmax: pixel
    assert(map3d.shape[0]>1)
    nz, ny, nx = map3d.shape
    if rmax<=rmin:
        rmax = min(nx//2, ny//2)
    assert(rmin<rmax)
    
    theta = (np.arange(0, 360, da, dtype=np.float32) - 90) * np.pi/180.
    #z = np.arange(0, nz, dz)    # use entire length
    n_theta = len(theta)
    z = np.arange(max(0, nz//2-n_theta//2*dz), min(nz, nz//2+n_theta//2*dz), dz, dtype=np.float32)    # use only the central segment 

    cylindrical_proj = np.zeros((len(z), len(theta)), dtype=np.float32)
    for r in np.arange(rmin, rmax, dr, dtype=np.float32):
        z_grid, theta_grid = np.meshgrid(z, theta, indexing='ij', copy=False)
        y_grid = ny//2 + r * np.sin(theta_grid)
        x_grid = nx//2 + r * np.cos(theta_grid)
        coords = np.vstack((z_grid.flatten(), y_grid.flatten(), x_grid.flatten()))
        cylindrical_image = map_coordinates(map3d, coords, order=interpolation_order, mode='nearest').reshape(z_grid.shape)
        cylindrical_proj += cylindrical_image * r
    cylindrical_proj = normalize(cylindrical_proj)

    return cylindrical_proj

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def minimal_grids(map3d, max_map_dim=300):
    nz, ny, nx = map3d.shape
    n_min_xy = min([ny, nx])
    n_min_z = min(nz, n_min_xy)
    bin = max(1, n_min_xy//max_map_dim+1)
    ret = map3d[nz//2-n_min_xy//2:nz//2+n_min_xy//2:bin, ny//2-n_min_xy//2:ny//2+n_min_xy//2:bin, nx//2-n_min_z//2:nx//2+n_min_z//2:bin]
    return ret, bin

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def auto_masking(map3d):
    required_packages = "skimage:scikit_image".split()
    import_with_auto_install(required_packages)
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

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def estimate_radial_range(radprofile, thresh_ratio=0.1):
    background = np.mean(radprofile[-3:])
    thresh = (radprofile.max() - background) * thresh_ratio + background
    indices = np.nonzero(radprofile>thresh)
    rmin_auto = np.min(indices)
    rmax_auto = np.max(indices)
    return float(rmin_auto), float(rmax_auto)

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
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

    polar = map_coordinates(proj, coords, order=1).reshape(r_grid.shape)

    rad_profile = polar.mean(axis=0)
    return rad_profile

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def transform_map(data, shift_x=0, shift_y=0, shift_z=0, angle_x=0, angle_y=0):
    if not (shift_x or shift_y or angle_x or angle_y):
        return data
    from scipy.spatial.transform import Rotation as R
    from scipy.ndimage import affine_transform
    # note the convention change
    # xyz in scipy is zyx in cryoEM maps
    rot = R.from_euler('zy', [-angle_x, angle_y], degrees=True)
    m = rot.as_matrix()
    nx, ny, nz = data.shape
    bcenter = np.array((nx//2, ny//2, nz//2), dtype=m.dtype)
    offset = bcenter.T - np.dot(m, bcenter.T) + np.array([shift_z, shift_y, -shift_x])
    ret = affine_transform(data, matrix=m, offset=offset, mode='nearest')
    return ret

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def auto_vertical_center(image, max_angle=15):
    image_work = 1.0 * image

    # rough estimate of rotation
    def score_rotation(angle):
        tmp = rotate_shift_image(data=image_work, angle=angle)
        y_proj = tmp.sum(axis=0)
        percentiles = (100, 95, 90, 85, 80) # more robust than max alone
        y_values = np.percentile(y_proj, percentiles)
        err = -np.sum(y_values)
        return err
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(score_rotation, bounds=(-max_angle, max_angle), method='bounded', options={'disp':0})
    res_90 = minimize_scalar(score_rotation, bounds=(90-max_angle, 90+max_angle), method='bounded', options={'disp':0})
    if res.fun < res_90.fun:
        angle = res.x
    else:
        angle = res_90.x

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
    y -= y.min()
    n = len(y)
    from scipy.ndimage import center_of_mass
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

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
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

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2

def get_3d_map_from_uploaded_file(fileobj):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        return get_3d_map_from_file(temp.name)

def extract_emd_id(text):
    import re
    pattern = r'.*emd_([0-9]+)\.map.*'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        emd_id = match.group(1)
    else:
        emd_id = None
    return emd_id

st.cache_resource(show_spinner=False)
def get_emdb_ids():
    try:
        import_with_auto_install(["pandas"])
        import pandas as pd
        entries_all = pd.read_csv('https://www.ebi.ac.uk/emdb/api/search/current_status:"REL"?rows=1000000&wt=csv&download=true&fl=emdb_id,structure_determination_method,resolution')
        methods = list(entries_all["structure_determination_method"])
        entries_helical = entries_all[entries_all["structure_determination_method"]=="helical"]
        emdb_ids_all     = list(entries_all.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
        emdb_ids_helical = list(entries_helical.iloc[:,0].str.split('-', expand=True).iloc[:, 1].values)
    except:
        emdb_ids_all = []
        emdb_ids_helical = []
        methods = {}
    return emdb_ids_all, emdb_ids_helical, methods

@st.cache_data(persist='disk', max_entries=1, show_spinner=False)
def get_emdb_parameters(emd_id):
    try:
        emd_id2 = ''.join([s for s in str(emd_id) if s.isdigit()])
        url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id2}/header/emd-{emd_id2}.xml"
        from urllib.request import urlopen
        with urlopen(url) as response:
            xml_data = response.read()
        import_with_auto_install(["xmltodict"])
        import xmltodict
        data = xmltodict.parse(xml_data)
    except:
        return None
    ret = {}
    try:
      ret['sample'] = data['emd']['sample']['name']
      ret["method"] = data['emd']['structure_determination_list']['structure_determination']['method']
      dimensions = data['emd']['map']['dimensions']
      ret["nz"] = int(dimensions["sec"])
      ret["ny"] = int(dimensions["row"])
      ret["nx"] = int(dimensions["col"])
      res_dict = dict_recursive_search(data, 'resolution')
      if res_dict:
        ret["resolution"] = float(res_dict['#text'])
      if ret["method"] == 'helical':
          #ret["resolution"] = float(data['emd']['structure_determination_list']['structure_determination']['helical_processing']['final_reconstruction']['resolution']['#text'])
          helical_parameters = data['emd']['structure_determination_list']['structure_determination']['helical_processing']['final_reconstruction']['applied_symmetry']['helical_parameters']
          #assert(helical_parameters['delta_phi']['@units'] == 'deg')
          #assert(helical_parameters['delta_z']['@units'] == 'Å')
          ret["twist"] = float(helical_parameters['delta_phi']['#text'])
          ret["rise"] = float(helical_parameters['delta_z']['#text'])
          ret["csym"] = int(helical_parameters['axial_symmetry'][1:])
    finally:
      return ret

def is_amyloid(params, cutoff=6):
    if "twist" in params and "rise" in params:
        twist = params["twist"]
        rise = params["rise"]
        r = np.hypot(twist, rise)
        if r<cutoff: return True
        twist2 = abs(twist)-180
        r = np.hypot(twist2, rise)
        if r<cutoff: return True
    if "sample" in params:
        sample = params["sample"].lower()
        for target in "tau synuclein amyloid tdp-43".split():
            if sample.find(target)!=-1: return True
    return False

def get_emdb_map_url(emdid):
    emdid_number = emdid.lower().split("emd-")[-1]
    #server = "https://files.wwpdb.org/pub"    # Rutgers University, USA
    server = "https://ftp.ebi.ac.uk/pub/databases" # European Bioinformatics Institute, England
    #server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
    url = f"{server}/emdb/structures/EMD-{emdid_number}/map/emd_{emdid_number}.map.gz"
    return url

@st.cache_resource(show_spinner=False)
def get_emdb_map(emdid):
    url = get_emdb_map_url(emdid)
    data = get_3d_map_from_url(url)
    return data

@st.cache_resource(show_spinner=False)
def get_3d_map_from_url(url):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    with download_file_from_url(url_final) as fileobj:
        if fileobj is None:
            st.error(f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
            st.stop()
        data = get_3d_map_from_file(fileobj.name)
    import os
    file_to_remove = fileobj.name.removesuffix(".gz")
    if os.path.exists(file_to_remove):
        os.unlink(file_to_remove)
    return data

def get_3d_map_from_file(filename):
    if filename.endswith(".gz"):
        filename_final = filename[:-3]
        import gzip, shutil
        with gzip.open(filename, 'r') as f_in, open(filename_final, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        filename_final = filename
    import mrcfile
    with mrcfile.open(filename_final,mode="r") as mrc:
        data = mrc.data
        map_crs = [int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps)]
        apix = mrc.voxel_size.x.item()
        is3d = mrc.is_volume() or mrc.is_volume_stack()
    return data, map_crs, apix

def change_mrc_map_crs_order(data, current_order, target_order=[1, 2, 3]):
    if current_order == target_order: return data
    map_crs_to_np_axes = {1:2, 2:1, 3:0}
    current_np_axes_order = [map_crs_to_np_axes[int(i)] for i in current_order]
    target_np_axes_order = [map_crs_to_np_axes[int(i)] for i in target_order]
    import numpy as np
    ret = np.moveaxis(data, current_np_axes_order, target_np_axes_order)
    return ret

def download_file_from_url(url):
    import tempfile
    import requests
    try:
        filesize = get_file_size(url)
        local_filename = url.split('/')[-1]
        suffix = '.' + local_filename
        fileobj = tempfile.NamedTemporaryFile(suffix=suffix)
        msg = f'Downloading {url}'
        if filesize is not None:
            msg += f" ({filesize/2**20:.1f} MB)"
        with st.spinner(msg):
            with requests.get(url) as r:
                r.raise_for_status()  # Check for request success
                fileobj.write(r.content)
        return fileobj
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def get_file_size(url):
    import requests
    response = requests.head(url)
    if 'Content-Length' in response.headers:
        file_size = int(response.headers['Content-Length'])
        return file_size
    else:
        return None

def get_direct_url(url):
    import re
    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1")!=-1: return url
        elif url.find("dl=0")!=-1: return url.replace("dl=0", "dl=1")
        else: return url+"?dl=1"
    elif url.find("sharepoint.com")!=-1 and url.find("guestaccess.aspx")!=-1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64
        data_bytes64 = base64.b64encode(bytes(url, 'utf-8'))
        data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    else:
        return url

int_types = ['csym', 'do_threshold', 'do_transform', 'input_mode', 'npeaks', 'random_embid', 'section_axis', 'share_url', 'show_acf', 'show_scf', 'show_arrow', 'show_cylproj', 'show_peaks', 'show_qr']
float_types = ['ang_max', 'ang_min', 'da', 'dz', 'peak_width', 'peak_height', 'rise', 'rmax', 'rmin', 'shiftx', 'shifty', 'shiftz', 'rotx', 'roty', 'thresh', 'twist', 'z_max', 'z_min']
default_values = {'csym':1, 'do_threshold':0, 'do_transform':0, 'input_mode':2, 'npeaks':71, 'random_embid':1, 'section_axis':2, 'share_url':0, 'show_acf':1, 'show_scf':0, 'show_arrow':1, 'show_cylproj':1, 'show_peaks':1, 'show_qr':0, 'ang_max':180., 'ang_min':-180., 'da':1.0, 'dz':1.0, 'peak_width':9.0, 'peak_height':9.0, 'rmin':0, 'shiftx':0, 'shifty':0, 'shiftz':0, 'rotx':0, 'roty':0, 'target_map_axes_order':'x,y,z', 'thresh':0, 'z_max':-180., 'z_min':180.}
def set_query_parameters():
    d = {}
    attrs = sorted(st.session_state.keys())
    for attr in attrs:
        v = st.session_state[attr]
        if attr in default_values and v==default_values[attr]: continue
        if attr in int_types or isinstance(v, bool):
            d[attr] = int(v)
        elif attr in float_types:
            d[attr] = f'{float(v):g}'
        else:
            d[attr] = v
    st.query_params.update(d)

def parse_query_parameters():
    query_params = st.query_params
    for attr in query_params:
        if attr in int_types:
            st.session_state[attr] = int(query_params[attr])
        elif attr in float_types:
            st.session_state[attr] = float(query_params[attr])
        else:
            st.session_state[attr] = query_params[attr]

def dict_recursive_search(d, key, default=None):
    stack = [iter(d.items())]
    while stack:
        for k, v in stack[-1]:
            if k == key:          
                return v
            elif isinstance(v, dict):
                stack.append(iter(v.items()))
                break
        else:
            stack.pop()
    return default

def qr_code(url=None, size = 8):
    import_with_auto_install(["qrcode"])
    import qrcode
    if url is None: # ad hoc way before streamlit can return the url
        _, host = is_hosted(return_host=True)
        if len(host)<1: return None
        if host == "streamlit":
            url = "https://helical-indexing-hi3d.streamlit.app/"
        elif host == "heroku":
            url = "https://helical-indexing-hi3d.herokuapp.com/"
        else:
            url = f"http://{host}:8501/"
        import urllib
        params = st.query_params
        d = {k:params[k][0] for k in params}
        url += "?" + urllib.parse.urlencode(d)
    if not url: return None
    img = qrcode.make(url)  # qrcode.image.pil.PilImage
    data = np.array(img.convert("RGBA"))
    return data

@st.cache_resource(show_spinner=False)
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

def mem_info():
    import_with_auto_install(["psutil"])
    from psutil import virtual_memory
    mem = virtual_memory()
    mb = pow(2, 20)
    return (mem.total/mb, mem.available/mb, mem.used/mb, mem.percent)

def mem_quota():
    fqdn = get_hostname()
    if fqdn.find("heroku")!=-1:
        return 512  # MB
    username = get_username()
    if username.find("appuser")!=-1:    # streamlit share
        return 1024  # MB
    available_mem = mem_info()[1]
    return available_mem

def mem_used():
    import_with_auto_install(["psutil"])
    from psutil import Process
    from os import getpid
    mem = Process(getpid()).memory_info().rss / 1024**2   # MB
    return mem

def up_time():
    import_with_auto_install(["uptime"])
    from uptime import uptime
    t = uptime()
    if t is None: return 0
    else: return t

def get_username():
    from getpass import getuser
    return getuser()

def get_hostname():
    import socket
    fqdn = socket.getfqdn()
    return fqdn

def is_hosted(return_host=False):
    hosted = False
    host = ""
    fqdn = get_hostname()
    if fqdn.find("heroku")!=-1:
        hosted = True
        host = "heroku"
    username = get_username()
    if username.find("appuser")!=-1:
        hosted = True
        host = "streamlit"
    if not host:
        host = "localhost"
    if return_host:
        return hosted, host
    else:
        return hosted

# https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
def geometric_median(X, eps=1e-5):
    import numpy as np
    from scipy.spatial.distance import cdist, euclidean
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def set_to_periodic_range(v, min=-180, max=180):
    from math import fmod
    tmp = fmod(v-min, max-min)
    if tmp>=0: tmp+=min
    else: tmp+=max
    return tmp

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    main()
    gc.collect(2)
