import json
import re
import shutil
import pathlib

from osgeo import gdal
import pyproj

from proj_sql_dict_classes import ProjDB, GenericRecord

local_raster_path = pathlib.Path(r'C:\GIT_Repos\SVN_Collaboration\VyperScratch\data\NOAA_NOS_Vertical_Datum_Repo\grids\cog')
local_tin_path = pathlib.Path(r'C:\GIT_Repos\SVN_Collaboration\VyperScratch\data\NOAA_NOS_Vertical_Datum_Repo\tins\json')

use_radians = False

rad_to_deg_step = '+step +proj=unitconvert +xy_in=rad +xy_out=deg '
rad_to_deg = "" if use_radians else rad_to_deg_step
deg_to_rad_step = '+step +proj=unitconvert +xy_in=deg +xy_out=rad '
deg_to_rad = "" if use_radians else deg_to_rad_step
axisswap_step = "+step +proj=axisswap +order=2,1 "

# FIXME: read the last id from the json - or have all the ids in the json to start with
current_id = 0

NOAA_AUTH_CODE = 'NOAATEST'


def unique_id():
    global current_id
    current_id += 1
    return current_id


# FIXME:  Some of the 3d pivots don't transform to NAD83(2011) or have no-op transformations when vdatum is not a no-op.
#   An example was IGS14 (9018) would not transform so you have to use ITRF2014 (7912) instead.
#   Another would be NAD83(HARN) to NAD83(2011) which is a no-op.
#   @TODO need to determine if we can force a transform that exists, like 9173 which is NAD83(NSRS2007) (4893) to NAVD88 (5703) but is a no-op by proj
#   This shows a transform exists:
#   projinfo -k operation EPSG:9173
#   These get no-ops
#   projinfo -t EPSG:5703 -s EPSG:4893
#   projinfo -t EPSG:6318+EPSG:5703 -s EPSG:4893

datum_abbreviations = {
    "MLLW": "Mean Lower Low Water",
    "MLW": "Mean Low Water",
    "MHW": "Mean High Water",
    "MSL": "Mean Sea Level",
    "MHHW": "Mean Higher High Water",
    "MTL": "Mean Tide Level",
    "LWD": "Low Water Datum",
    "LMSL": "Local Mean Sea Level",
    "CRD": "Colombia River Datum",
    "HRD": "Hudson River Datum",
    "geoid": "geoid",
    "LWRP": "Low Water Reference Plane",
    "CD": "Chart Datum",
}


def insert_json_into_proj(json_path, proj_db_path, auth_name=NOAA_AUTH_CODE, commit=True):
    # Here is the plan:
    # 1) Find and add datums
    # 2) Find and add region extents
    # 3) Find and add region coordinate reference systems
    # 4) Add Usage for each region coordinate reference system
    # 5) Create grid or other transformations for each region
    # 6) Add Usage for each transformation

    noaa_registry = json.load(open(json_path))['NOAA_NOS_Vertical_Datum_Manifest']
    noaa_code_names = noaa_registry['code_register'][NOAA_AUTH_CODE]['vcrs']
    noaa_codes = {code: code_name for code_name, code in noaa_registry['code_register'][NOAA_AUTH_CODE]['vcrs'].items()}

    # @TODO - change the ID creation if we want to create ID ranges for various types (datums from 0-100, grids xform from 10k-100k, etc)
    # If we make any IDs set them after the last values in the register
    global current_id
    if current_id <= max(noaa_codes.keys()):
        current_id = max(noaa_codes.keys()) + 1
    pdb = ProjDB(proj_db_path)

    VerticalDatum, VerticalCRS, Extent, Usage, Scope = pdb.get_vertical_classes()
    GridTransform, OtherTransform = pdb.get_class(pdb.GRID_TRANSFORMATION), pdb.get_class(pdb.OTHER_TRANSFORMATION)
    CoordSys = pdb.get_class(pdb.COORD_SYSTEM)
    CoordOpMethod = pdb.get_class(pdb.COORD_OP_METHOD)
    CoordOpParam = pdb.get_class(pdb.COORD_OP_PARAM)
    GeodeticCRS = pdb.get_class(pdb.GEODETIC_CRS)
    CompoundCRS = pdb.get_class(pdb.COMPOUND_CRS)
    ConcatOp = pdb.get_class(pdb.CONCAT_OP)
    ConcatStep = pdb.get_class(pdb.CONCAT_STEP)

    # get some of the known, existing EPSG codes
    # EPSG 5103 = NAVD88 datum
    # EPSG 6318 = 2D  NAD83(2011) CRS
    # EPSG 6319 = 3D  NAD83(2011) CRS
    # EPSG 6349 = compound NAD83 + NAVD88 height
    # EPSG 32618 WGS84 UTM Zone 18
    # EPSG 26918 NAD83 UTM Zone 18
    # EPSG 9596 = transform from NAD83(2011) to NAD83+NAVD88 height  (6319 -> 6349)
    # EPSG 1511 = area CONUS and AK
    # EPSG 1024 = scope - unknown
    # @TODO find the right scopes (this doesn't stop proj from working though)
    change_height_scope = Scope(auth_name='EPSG', code=1059)
    coastal_hydrography = Scope(auth_name='EPSG', code=1103)
    geoid_model = Scope(auth_name='EPSG', code=1270)  # EPSG 1270 = scope - reversible geoid model transformation
    cs_up = CoordSys(auth_name='EPSG', code=6499)  # EPSG 6499 = Vertical CS. Axis: height (H). Orientation: up. UoM: m.
    cs_down = CoordSys(auth_name='EPSG', code=6498)  # EPSG 6498 = Vertical CS. Axis: depth (D). Orientation: down. UoM: m.
    geo3d_to_gravity_height_method = CoordOpMethod(auth_name='EPSG', code=9615, name='Geographic3D to GravityRelatedHeight (gtx)')
    geoid_model_param = CoordOpParam(auth_name='EPSG', code=8666, name='Geoid (height correction) model file')
    nad83_2007_2d_crs = GeodeticCRS(auth_name='EPSG', code=4759)  # EPSG 4759 = 2D  NAD83(NSRS2007) CRS
    nad83_2007_3d_crs = GeodeticCRS(auth_name='EPSG', code=4893)  # EPSG 4893 = 3D  NAD83(NSRS2007) CRS
    nad83_2007_navd88_crs = GeodeticCRS(auth_name='EPSG', code=5500)  # EPSG 5500 = 3D  NAD83(2007) CRS + NAVD88 (geoid09) height
    nad83_2011_2d_crs = GeodeticCRS(auth_name='EPSG', code=6318)  # EPSG 6318 = 2D  NAD83(2011) CRS
    nad83_2011_3d_crs = GeodeticCRS(auth_name='EPSG', code=6319)  # EPSG 6319 = 3D  NAD83(2011) CRS
    nad83_2011_navd88_crs = GeodeticCRS(auth_name='EPSG', code=6349)  # EPSG 6349 = 3D  NAD83(2011) CRS + NAVD88 (geoid12b or geoid18) height

    igs14_3d_crs = GeodeticCRS(auth_name='EPSG', code=9018)  # EPSG 9018 = 3D IGS14
    igs14_2d_crs = GeodeticCRS(auth_name='EPSG', code=9019)  # EPSG 9018 = 2D IGS14
    navd88_height_crs = VerticalCRS(auth_name='EPSG', code=5703)  # EPSG 5703 for NAVD88 height
    # This is for a 3d to vertical, like EPSG:6319 to EPSG:5703, see grid_transformation EPSG:6326 for an example
    gravity_gtx = GenericRecord(auth_name='EPSG', code=9665, name='Geographic3D to GravityRelatedHeight (gtx)')
    # to use this you need to supply the interpolation_crs  see grid_transformation "EPSG:9595" for an example
    gravity2d_to_3d_gtx = GenericRecord(auth_name='EPSG', code=1088, name='Geog3D to Geog2D+GravityRelatedHeight (gtx)')
    # Use this to specify a geoid gtx file for the transform
    geoid_height_model = GenericRecord(auth_name='EPSG', code=8666, name='Geoid (height correction) model file')
    vertical_offset = GenericRecord(auth_name='EPSG', code=1084, name='Vertical Offset by Grid Interpolation (gtx)')
    # Use this to specify a vertical gtx file for the transform
    vertical_offset_model = GenericRecord(auth_name='EPSG', code=8732, name='Vertical offset file')

    model_regions = {}
    datums = {}
    extents = {}
    crses = {}  # "geoid09": nad83_2007_navd88_crs, "geoid18": nad83_2011_navd88_crs, "geoid12b": nad83_2011_navd88_crs, "navd88": navd88_height_crs
    compound_crses = {}
    usages = {}
    other_transforms = {}
    grid_transforms = {}
    xform_usages = {}
    concat_operations = {}
    concat_steps = {}

    def make_crs(name, crs_id=None, extent=(-180, -90, 180, 90)):  # make a quick crs for testing or xgeoids that aren't registered
        extents[name] = Extent(auth_name=auth_name, code=unique_id(), name=name + " extent", description=name + " extent description",
                               west_lon=extent[0],
                               south_lat=extent[1], east_lon=extent[2], north_lat=extent[3], deprecated=False)
        datums[name] = VerticalDatum(auth_name=auth_name, code=unique_id(), name=name + " datum",
                                     description=name + " datum description", deprecated=False)
        if crs_id is None:
            crs_id = unique_id()
        crses[name] = VerticalCRS(auth_name=auth_name, code=crs_id, name=name + " crs",
                                  description=name + " crs description", coordinate_system=cs_up,
                                  datum=datums[name], deprecated=False)
        usages[name] = Usage(auth_name=auth_name, code=unique_id(), object=crses[name], extent=extents[name], scope=change_height_scope)

    def add_to_steps(xform_name, steps):
        """ Add steps to the concat_steps dictionary for a given transform name

        Parameters
        ----------
        xform_name: str
            name of the transform, the key in concat_operations and concat_steps
        steps: list
            iterable of [str, int, SQLRecord]
            each step can be an integer (assumed to be EPSG) a string "AUTH:CODE" or a SQLRecord with auth_name and code attributes.

        Returns
        -------
        None
        """
        step_start = 1
        while (xform_name, step_start) in concat_steps:
            step_start += 1
        for n, step in enumerate(steps):
            if isinstance(step, int):
                concat_steps[(xform_name, n + step_start)] = ConcatStep(operation=concat_operations[xform_name], step_number=n + step_start,
                                                                        step_auth_name='EPSG', step_code=step)
            elif isinstance(step, str):
                auth, code = step.split(":")
                concat_steps[(xform_name, n + step_start)] = ConcatStep(operation=concat_operations[xform_name], step_number=n + step_start,
                                                                        step_auth_name=auth, step_code=code)
            else:
                concat_steps[(xform_name, n + step_start)] = ConcatStep(operation=concat_operations[xform_name], step_number=n + step_start,
                                                                        step=step)
    def get_step_list(transform_name):
        """ Get the list of steps which can be passes to add_to_steps for a given transform name"""
        steps = []
        nstep = 1
        while (transform_name, nstep) in concat_steps:
            step_obj = concat_steps[(transform_name, nstep)]
            try:
                steps.append(step_obj['step'])
            except KeyError:
                steps.append(f"{step_obj['step_auth_name']}:{step_obj['step_code']}")
            nstep += 1
        return steps


    def add_to_extents(vcrs):
        # fixme service_uri or source/filename?
        if vcrs.data_path.lower().endswith('.json'):
            local_path = local_tin_path
        else:
            local_path = local_raster_path
        full_path_to_grid = str(local_path.joinpath(vcrs.data_path)).replace("\\", "/")
        full_path_to_extent = str(local_path.joinpath(vcrs.extent_path)).replace("\\", "/")

        if vcrs.region_name not in extents:
            extent = None
            for geom_path in (full_path_to_extent, full_path_to_grid):
                # get the bounding box from the geometry
                ds = gdal.OpenEx(geom_path, gdal.OF_VECTOR)
                if ds is None:
                    ds = gdal.OpenEx(geom_path, gdal.OF_RASTER)
                    if ds is None:
                        if geom_path.lower().endswith('.json'):
                            with open(geom_path, 'r') as f:
                                extent = json.load(f)['extent']['parameters']['bbox']
                    else:  #
                        x1, dxx, dyx, y1, dyx, dyy = ds.GetGeoTransform()
                        nx, ny = ds.RasterXSize, ds.RasterYSize
                        xs = (x1, x1 + (nx + 1) * dxx + (ny + 1) * dyx)
                        xs = [x if x <= 180 else x - 360 for x in xs]  # fix coords given as, for example, 264 degrees east
                        ys = (y1, y1 + (nx + 1) * dyx + (ny + 1) * dyy)
                        extent = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    layer = ds.GetLayer()
                    extent = layer.GetExtent()
                    del layer
                del ds
                if extent is not None:
                    break

            if extent is None:
                print(f"For {vcrs.region_name} could not open either:\n  {full_path_to_grid}  or \n  {full_path_to_extent}")
                extent = [-180, -90, 180, 90]
            horz = pyproj.crs.CRS(vcrs.horz_crs)
            if horz.is_projected:
                transform = pyproj.Transformer.from_crs(horz, "EPSG:6318")
                miny, minx = transform.transform(extent[0], extent[1])
                maxy, maxx = transform.transform(extent[2], extent[3])
                extent = [minx, miny, maxx, maxy]
                print(f"{vcrs.region_name} has a projected coordinate system, make sure extent is correct: {extent}")

            extents[vcrs.region_name] = Extent(auth_name=auth_name, code=unique_id(), name=vcrs.region_name, description=vcrs.region_long_description, west_lon=extent[0],
                                          south_lat=extent[1], east_lon=extent[2], north_lat=extent[3], deprecated=False)
        return extents[vcrs.region_name]


    def add_to_datum_and_crs(vcrs, region_extent):
        # 3) Add region coordinate reference systems
        # @TODO use the make_crs function.  Also check if the datum is a geoid and has epsg registry entry.
        if vcrs.datum_name not in datums:
            datums[vcrs.datum_name] = VerticalDatum(auth_name=auth_name, code=unique_id(), name=vcrs.datum_name,
                                               description=vcrs.region_long_description, deprecated=False)

        if vcrs.crs_name not in crses and vcrs.crs_name not in compound_crses:
            # code_name = noaa_code_names["_".join([NOAA_AUTH_CODE, "CODE", grid['source']['model'], vdatum_version, region_name, "DEPTH"])]
            # code_name = grid['crs']['input']['compound_height']['code'].split(NOAA_AUTH_CODE + ":")[1]
            code = noaa_code_names[vcrs.crs_name]
            vert_crs = VerticalCRS(auth_name=auth_name, code=code, name=vcrs.db_crs_name,
                                   description=vcrs.region_long_description, coordinate_system=vcrs.orientation,
                                   datum=datums[vcrs.datum_name], deprecated=False)
            crses[vcrs.crs_name] = vert_crs
            # 4) Add Usage for each region coordinate reference system
            region_crs_usage = Usage(auth_name=auth_name, code=unique_id(), object=vert_crs, extent=region_extent, scope=change_height_scope)
            usages[vcrs.crs_name] = region_crs_usage


    def add_to_transforms(transform_name, vcrs1, vcrs2, transform_extent):
        path_to_grid = vcrs1.data_path
        try:
            # @TODO figure out uncertainty (quadrature?)
            uncertainty = (vcrs1.uncertainty**2 + vcrs2.uncertainty**2 + vcrs1.xform_uncertainty**2)**.5  # + float(grid['uncertainty']['regional_source_sigma'])
        except (KeyError, TypeError, Exception):
            uncertainty = 0.10  # 10 cm default
            print("Uncertainty ValueError for ", transform_name)
        xdescription = vcrs1.region_description + " " + vcrs1.long_descr + " to " + vcrs2.long_descr
        if path_to_grid.lower().endswith('.json'):  # tins for other_transforms
            if transform_name not in other_transforms:
                xform = OtherTransform(auth_name=auth_name, code=unique_id(), name=transform_name, description=xdescription,
                                       method_auth_name='PROJ', method_code='PROJString', method_name='+proj=pipeline ' +
                                                                                                      f'{rad_to_deg}' +
                                                                                                      f'+step +proj=tinshift +file={path_to_grid} +inv ' +
                                                                                                      f'{deg_to_rad}',
                                       source_crs=crses[vcrs1.crs_name], target_crs=crses[vcrs2.crs_name], accuracy=uncertainty,
                                       deprecated=False)
                other_transforms[transform_name] = xform
                xform_usages[transform_name] = Usage(auth_name=auth_name, code=unique_id(), object=xform, extent=transform_extent,
                                                     scope=coastal_hydrography)
        else:  # rasters, like tif or gtx, for grid_transforms
            if transform_name not in grid_transforms:
                # this works for a transform to 3D like EPSG:6319 but not if we are just trying to do a vertical offset
                # xform = GridTransform(auth_name=auth_name, code=unique_id(), name=transform_name, description=xdescription,
                #          method = gravity_gtx,
                #          source_crs=crses[crs1], target_crs=crses[crs2], accuracy=uncertainty,
                #          grid_param = geoid_height_model, grid_name=path_to_grid,
                #          deprecated=False)
                # This is for a vertical offset only (not a 3D transform)
                xform = GridTransform(auth_name=auth_name, code=unique_id(), name=transform_name, description=xdescription,
                                      method=vertical_offset,  # epsg:1084
                                      source_crs=crses[vcrs1.crs_name], target_crs=crses[vcrs2.crs_name], accuracy=uncertainty,
                                      grid_param=vertical_offset_model, grid_name=path_to_grid,  # epsg:8732
                                      deprecated=False)

                grid_transforms[transform_name] = xform
                xform_usages[transform_name] = Usage(auth_name=auth_name, code=unique_id(), object=xform, extent=transform_extent,
                                                     scope=coastal_hydrography)


    def add_geoid_and_pivot_operations(vcrs, transform_name, pivot_xform_name, nad83_xform_name, pivot):
        # FIXME @TODO This may not be true -- This works IF the operations are in the same direction.  See notes in test_noaa_transforms.py
        # Make the to concatenated operations with steps to go from Tidal datum (usually LMSL) to NAD83(2011)
        # and tidal datum to pivot_3d (which might be NAD83(2011)).
        # LMSL -> geoid realization -> 3d pivot -> NAD83(2011) (if needed)
        # If the 3d pivot is not NAD83(2011) we can just reuse the steps but leave off the last one
        # Then when we process the other tidal datums during the second loop (MLLW, MHW, etc) we can just reuse the steps to go from LMSL to NAD83(2011)

        # concat_operations[xform_name] = ConcatOp(auth_name=auth_name, code=unique_id(), name=xform_name, description=f"{crs1} to {pivot['name']}",
        #                                           source_crs=crses[crs1], target_crs=pivot_crs, accuracy=uncertainty,
        #                                           deprecated=False)

        # this will be the basic raster or tin transformation we made above
        lmsl_geoid_step = other_transforms[transform_name] if transform_name in other_transforms else grid_transforms[transform_name]

        ortho_3dpivot_step = noaa_registry['orthometric_system'][vcrs.ortho_key]['crs']['geoid_model']['code']
        add_to_steps(pivot_xform_name, [lmsl_geoid_step, ortho_3dpivot_step])
        xform_usages[pivot_xform_name] = Usage(auth_name=auth_name, code=unique_id(), object=concat_operations[pivot_xform_name],
                                         extent=extents[vcrs.region_name],
                                         scope=coastal_hydrography)
        # Now make the transform to a common 3d CRS so everything can connect to everything else
        if pivot['code'] != 'EPSG:6319':
            concat_operations[nad83_xform_name] = ConcatOp(auth_name=auth_name, code=unique_id(), name=nad83_xform_name, description=f"{vcrs.db_crs_name} to NAD83(2011)",
                                                     source_crs=crses[vcrs.crs_name], target_crs=nad83_2011_3d_crs, accuracy=vcrs.uncertainty,
                                                     deprecated=False)
            add_to_steps(nad83_xform_name, [lmsl_geoid_step, ortho_3dpivot_step])
            # Unfortunately PROJ will not allow a concatenated operation to have a concatenated operation as a step --
            # sqlite3.IntegrityError: insert on concatenated_operation_step violates constraint: step should not be a concatenated_operation
            if pivot['code'] == 'EPSG:8542':  # NAD83(FBN)
                # add_to_steps(nad83_xform_name, ["PROJ:NAD83_FBN_TO_NAD83_2011_CONUS"])
                steps = (8862, 8559)
                add_to_steps(nad83_xform_name, steps)
            elif pivot['code'] == "EPSG:4957":  # NAD83(HARN)
                # add_steps(nad83_xform_name, ["PROJ:NAD83_HARN_TO_NAD83_2011_CONUS"])
                steps = (8861, 8862, 8559)
                add_to_steps(nad83_xform_name, steps)
            elif pivot['code'] == "EPSG:6782":  # NAD83(CORS96)
                print("Need to add NAD83(CORS96) to NAD83(2011) concatenated operations")
            elif pivot['code'] == "EPSG:4893":  # NAD83(NSRS2007)
                steps = (8559,)
                add_to_steps(nad83_xform_name, steps)
            elif pivot['code'] == "EPSG:7912":  # ITRF2014
                steps = (9602,)
                add_to_steps(nad83_xform_name, steps)
            else:
                print(f"{pivot['code']} not found, Need to add {pivot} to NAD83(2011) concatenated operations?")
            xform_usages[nad83_xform_name] = Usage(auth_name=auth_name, code=unique_id(), object=concat_operations[nad83_xform_name],
                                             extent=extents[vcrs.region_name],
                                             scope=coastal_hydrography)

    def add_tidal_operations(vcrs1, vcrs2, transform_name, pivot_xform_name, nad83_xform_name, pivot):
        # everything is now registered with NOAA codes, so make concatenated operations to get to 3d pivots and NAD83(2011)
        # make concatenated operations for tidal datum to pivot_3d and NAD83(2011) 3d
        # just means to add the steps already done with the geoid to the xform for this tidal datum

        geoid_transform_name = "_".join([vcrs1.region_name, vcrs2.datum_name, vcrs2.height_or_depth, pivot['name']])
        datum1_datum2_step = other_transforms[transform_name] if transform_name in other_transforms else grid_transforms[transform_name]
        steps = [datum1_datum2_step] + get_step_list(geoid_transform_name)
        if len(steps) == 1:
            print(f"WARNING: steps for {geoid_transform_name} not found, does it have a {vcrs2.abbr}-geoid entry in the registry?")
            get_step_list(geoid_transform_name)
        add_to_steps(pivot_xform_name, steps)
        # add_to_steps(pivot_xform_name, [lmsl_geoid_step, ortho_3dpivot_step])
        xform_usages[pivot_xform_name] = Usage(auth_name=auth_name, code=unique_id(), object=concat_operations[pivot_xform_name],
                                         extent=extents[vcrs1.region_name],
                                         scope=coastal_hydrography)
        if pivot['code'] != 'EPSG:6319':
            concat_operations[nad83_xform_name] = ConcatOp(auth_name=auth_name, code=unique_id(), name=nad83_xform_name, description=f"{vcrs1.db_crs_name} to NAD83(2011)",
                                                     source_crs=crses[vcrs1.crs_name], target_crs=nad83_2011_3d_crs, accuracy=vcrs1.uncertainty,
                                                     deprecated=False)
            geoid_nad83_xform_name = "_".join([vcrs1.region_name, vcrs2.datum_name, vcrs2.height_or_depth, "NAD83(2011)"])
            steps = [datum1_datum2_step] + get_step_list(geoid_nad83_xform_name)
            if len(steps) == 1:
                print(f"WARNING: steps for {geoid_nad83_xform_name} not found, does it have a {vcrs2.abbr}-geoid entry in the registry?")
                get_step_list(geoid_nad83_xform_name)
            add_to_steps(nad83_xform_name, steps)
            xform_usages[nad83_xform_name] = Usage(auth_name=auth_name, code=unique_id(), object=concat_operations[nad83_xform_name],
                                             extent=extents[vcrs1.region_name],
                                             scope=coastal_hydrography)

    # NOTE the geoid_model in the orthometric_system if the transform code number not the crs code number
    # 1) Find and add datums
    def find_geoid_to_3d_pivots():
        # find the geoid to 3d pivot transforms
        # if the compound_height is not a single EPSG code then register a compound crs.
        # @TODO If the second part is a placeholder {NOAA_AUTH...} then make a new vertical crs OR just use 5703?
        # Register a transform from the compound crs to the 3d pivot using the service_uri if needed.
        for ortho_name, ortho in noaa_registry['orthometric_system'].items():
            # FIXME -- this was checking the compound_height.  We should check the geoid_model instead and register a compound 2d+geoid. (Geoid 96, 99, XGeoids)
            #   Then see if the compound_height is a single EPSG code or a 2D epsg with a vertical crs NOAA code. (Geoid96 and XGeoids will need this)
            #   Do we need to use height or depth?  Only height is in the json geoid stanza.
            #   So should we look at ortho['crs']['compound_height']['code'] or ortho['crs']['geoid_model']['code'] or both?
            # FIXME -- I think I need an extra step?  We are just making the vertical CRS so do we need horizontal+geoid explicitly?
            #   to make the transform from MLLW to LMSL to GEOID to horizontal+geoid to 3D

            # @TODO
            #   Ok, geoid_model is the transform to use to go from compound_height to pivot3d
            #   The compound height used to supply prebuilt compounds (like GEOID09 had EPSG:5500 instead of EPSG:4759 + EPSG:5703)
            #   So, do we want the old pre-combined or the separate or just use the vertical of the separate?
            #   looking at the geoid_model transforms, they go from 3D to gravity height - meaning just use the vertical of the compound_height (for now?)
            #   GEOID09 lists geoid_model with transform EPSG:9173 has source as EPSG:4893 and target as EPSG:5703

            # if the crs has a curly brace then it needs to have a NOAA Code made.  Create the Datum, CRS
            if NOAA_AUTH_CODE in ortho['crs']['compound_height']['code']:  # ortho['crs']['geoid_model']['code']
                horiz_auth, horiz_code = (ortho['crs']['compound_height']['code'].replace(" ", "").split("+")[0]).split(":")
                vert_auth, vert_code_name = (ortho['crs']['compound_height']['code'].replace(" ", "").split("+")[1]).split(":")
                vert_code = noaa_code_names[vert_code_name]
                # get the gravitational vertical datum
                if ortho_name == "NAVD88_GEOID96_CONUS":
                    # FIXME - register a new CORS96 geoid
                    crses[ortho_name] = navd88_height_crs  # use NAVD88 (EPSG:5703) for NAD83(CORS96)
                    print("FIXME - register a new CORS96 geoid")
                if ortho_name not in crses and ortho_name not in compound_crses:
                    make_crs(ortho_name, crs_id=vert_code)
                # register a compound crs for the ortho_name using the gravity vertical datum and the horizontal crs
                horiz = GeodeticCRS(auth_name=horiz_auth, code=horiz_code)
                vert = crses[ortho_name]
                compound_crses[ortho_name] = CompoundCRS(auth_name=auth_name, code=unique_id(), name=ortho_name + " gravity compound crs",
                                                         horiz_crs=horiz, vertical_crs=vert, deprecated=False)
                # use the compound CRS as the compound height for the ortho crs
                ortho['crs']['compound_height']['code'] = f"{auth_name}:{compound_crses[ortho_name]['code']}"
            elif "+" in ortho['crs']['compound_height']['code']:  # two EPSG codes
                # Just use the vertical (often EPSG:5703) as the vertical crs
                horiz_auth, horiz_code = (ortho['crs']['compound_height']['code'].replace(" ", "").split("+")[0]).split(":")
                horiz = GeodeticCRS(auth_name=horiz_auth, code=horiz_code)
                vert_auth, vert_code = (ortho['crs']['compound_height']['code'].replace(" ", "").split("+")[1]).split(":")
                vert = GeodeticCRS(auth_name=vert_auth, code=vert_code)
                crses[ortho_name] = vert
                ## use the compound CRS as the compound height for the ortho crs, this gets registered later
                # compound_crses[ortho_name] = CompoundCRS(auth_name=auth_name, code=unique_id(), name=ortho_name + " gravity compound crs",
                #                                          horiz_crs=horiz, vertical_crs=vert, deprecated=False)
                # ortho['crs']['compound_height']['code'] = f"{auth_name}:{compound_crses[ortho_name]['code']}"
            else:
                raise ValueError("Compound Height is a single EPSG code or a compound EPSG code")
                auth, code = ortho['crs']['compound_height']['code'].split(":")
                crses[ortho_name] = GeodeticCRS(auth_name=auth, code=code)

            # register the geoid to 3d pivot transform if needed
            if NOAA_AUTH_CODE in ortho['crs']['geoid_model']['code']:
                transform_name = ortho_name + "_to_" + ortho['crs']['pivot_3d']['name']
                # @TODO figure out uncertainty (quadrature?)
                try:
                    uncertainty = float(grid['uncertainty']['regional_transform_sigma']) + float(grid['uncertainty']['regional_source_sigma'])
                except:
                    uncertainty = 0.0
                    print("Uncertainty ValueError for ", transform_name)
                xdescription = " ".join([ortho_name, "to", ortho['crs']['pivot_3d']['name']])
                # src_auth, src_code = ortho['crs']['compound_height']['code'].split(":")
                # source_crs = GeodeticCRS(auth_name=src_auth, code=src_code)
                source_crs = crses[ortho_name]
                tgt_auth, tgt_code = ortho['crs']['pivot_3d']['code'].split(":")
                target_crs = GeodeticCRS(auth_name=tgt_auth, code=tgt_code)
                path_to_grid = ortho['service_uri'].split("/")[-1]
                # FIXME get the real extent
                print(f"Using whole world for extent of {ortho_name}")
                extent = [-180, -90, 180, 90]
                extents[ortho_name] = Extent(auth_name=auth_name, code=unique_id(), name=ortho_name, description=xdescription, west_lon=extent[0],
                                             south_lat=extent[1], east_lon=extent[2], north_lat=extent[3], deprecated=False)
                region_extent = extents[ortho_name]
                if ortho['service_uri'].lower().endswith('.json'):  # tins for other_transforms
                    if transform_name not in other_transforms:
                        xform = OtherTransform(auth_name=auth_name, code=unique_id(), name=transform_name, description=xdescription,
                                               method_auth_name='PROJ', method_code='PROJString', method_name='+proj=pipeline ' +
                                                                                                              f'{rad_to_deg}' +
                                                                                                              f'+step +proj=tinshift +file={path_to_grid} +inv ' +
                                                                                                              f'{deg_to_rad}',
                                               source_crs=source_crs,
                                               target_crs=target_crs,
                                               accuracy=uncertainty, deprecated=False)
                        other_transforms[transform_name] = xform
                        xform_usages[transform_name] = Usage(auth_name=auth_name, code=unique_id(), object=xform, extent=region_extent,
                                                             scope=coastal_hydrography)
                else:  # rasters, like tif or gtx, for grid_transforms
                    if transform_name not in grid_transforms:
                        # this works for a transform to 3D like EPSG:6319 but not if we are just trying to do a vertical offset
                        xform = GridTransform(auth_name=auth_name, code=unique_id(), name=transform_name, description=xdescription,
                                 method = gravity_gtx,
                                 source_crs=source_crs, target_crs=target_crs, accuracy=uncertainty,
                                 grid_param = geoid_height_model, grid_name=path_to_grid,
                                 deprecated=False)
                        # This is for a vertical offset only (not a 3D transform)
                        # xform = GridTransform(auth_name=auth_name, code=unique_id(), name=transform_name, description=xdescription,
                        #                       method=vertical_offset,  # epsg:1084
                        #                       source_crs=crses[crs1], target_crs=crses[crs2], accuracy=uncertainty,
                        #                       grid_param=vertical_offset_model, grid_name=path_to_grid,  # epsg:8732
                        #                       deprecated=False)

                        grid_transforms[transform_name] = xform
                        xform_usages[transform_name] = Usage(auth_name=auth_name, code=unique_id(), object=xform, extent=region_extent,
                                                             scope=coastal_hydrography)
                # Put the new code we made into the orthometric_system object
                ortho['crs']['geoid_model']['code'] = f"{xform['auth_name']}:{xform['code']}"


    def find_model_regions():
        # Iterate the main TIN model regions (not the gtx grids which are subdivisions of the main regions)
        # Make a dictionary of the subregions and what main region they are in as well as the orthometric system they use
        # basically a reverse lookup of what is in the register
        for model_name, model in noaa_registry['model'].items():
            for release_name, release in model['releases'].items():
                if 'regions' in release:
                    for region_name, region_desc in release['regions'].items():
                        if region_name in model_regions:
                            pass  # raise ValueError(f"Region {region_name} already exists in model_regions")
                        try:
                            model_regions[region_name] = {"description": region_desc, "model": model_name,
                                                          "release": release_name, "ortho_sys_key": release['ortho_sys'],
                                                          'ortho_sys_name': noaa_registry['orthometric_system'][release['ortho_sys']]['crs']['geoid_model']['name'],
                                                          'epoch': model['epoch']}
                        except KeyError:
                            print(f"KeyError for ortho sys '{release['ortho_sys']}' in {region_name} of {model_name} {release_name}")
                            continue
                else:
                    region_name = region_desc = release_name
                    if region_name in model_regions:
                        pass  # raise ValueError(f"Region {region_name} already exists in model_regions")
                    try:
                        model_regions[region_name] = {"description": region_desc, "model": model_name, "release": release_name, "ortho_sys_key": release['ortho_sys'],
                                                  'ortho_sys_name': noaa_registry['orthometric_system'][release['ortho_sys']]['crs']['geoid_model']['name']}
                    except KeyError:
                        print(f"KeyError for ortho sys '{release['ortho_sys']}' in {region_name} of {model_name} {release_name}")
                        continue

    class VertCRS:
        """ This class encapsulates an unregistered tidal datum from the NOAA VDatum registry,
        specifically uses the grid structure of the json file.
        """
        compound_json_key = {'height': 'compound_height', 'depth': 'compound_depth'}
        orientations = {'height': cs_up,
                        'depth': cs_down,
                        }

        def __init__(self, region_name, abbr, desc, grid, ortho_key, epoch, height_or_depth, in_out):
            self.region_name = region_name
            self.grid = grid
            self.abbr = abbr
            self.desc = desc
            self.ortho_key = ortho_key
            self.height_or_depth = height_or_depth
            self.in_out = in_out
            self.epoch = epoch
            if self.desc is not None:
                try:
                    self.ntde = re.search(r'NOS VDatum-modeled .* height, realized per the .* (National Tidal Datum Epoch).*', desc).groups()[0]
                except AttributeError:
                    try:
                        self.ntde = re.search(r'NOS VDatum-modeled .* orthometric height, realized per the .* (National Tidal Datum Epoch).*', desc).groups()[0]
                    except AttributeError:
                        re.search("NOS VDatum-modeled (.*) orthometric height", desc).groups()
                        self.ntde = ""
            else:
                self.ntde = ""

            if 'geoid' not in self.abbr.lower():
                self.datum_name = self.abbr + '_' + self.epoch
                # crs1 = region_name + "_" + datum1
                try:
                    self.crs_name = self.grid['crs'][self.in_out][self.compound_json_key[self.height_or_depth]]['code'].split(NOAA_AUTH_CODE + ":")[1]
                except IndexError:
                    self.crs_name = self.grid['crs'][self.in_out][self.compound_json_key[self.height_or_depth]]['code'].replace(" ", "").split("+")[1]
                    crses[self.crs_name] = VerticalCRS(auth_name='EPSG', code=int(self.crs_name.split(":")[1]))

                try:
                    self.long_descr = " ".join([datum_abbreviations[self.abbr], self.epoch, self.ntde])
                except KeyError:
                    self.long_descr = " ".join([self.abbr, self.epoch, self.ntde])
                    print(f"Abbreviation KeyError for {self.abbr} in {self.region_name}")
            else:
                self.datum_name, self.crs_name, self.long_descr = self.ortho_key, self.ortho_key, self.ortho_key
            self.horz_crs = self.grid['crs'][self.in_out][self.compound_json_key[self.height_or_depth]]['code'].replace(" ", "").split("+")[0]
            if 'EPSG' not in self.horz_crs:
                print("----------------HORZ CRS----------------")
                print(self.region_name, self.abbr, self.horz_crs)

        def __repr__(self):
            return f"{self.region_name} {self.abbr} {self.height_or_depth}"

        @property
        def db_crs_name(self):
            return self.crs_name.replace("NOAATEST_CODE_", "")

        @property
        def orientation(self):
            return self.orientations[self.height_or_depth]

        @property
        def region_description(self):
            return self.grid['region']

        @property
        def region_long_description(self):
            return self.region_description + " " + self.long_descr

        @property
        def data_path(self):
            path_to_grid = self.grid['data_uri']
            return pathlib.Path(path_to_grid).name

        @property
        def extent_path(self):
            path_to_data = self.grid['extent_uri']
            return pathlib.Path(path_to_data).name

        @property
        def uncertainty(self):
            return self.grid['uncertainty'][self.abbr.lower()+'_sigma']

        @property
        def xform_uncertainty(self):
            return self.grid['uncertainty']['transform_sigma']

    def process_datums():
        process_gtx_grids()
        process_model_tins()

    def process_gtx_grids():
        # Find the hydrographic datums
        for vdatum_version, vdatum in noaa_registry.items():
            # skip the model_manifest and orthometric_system which we already processed
            if vdatum_version in ('model', 'orthometric_system', "code_register"):
                continue

            # process everything with a geoid first then the tidal datums and do the concatenated transforms on the second pass.
            for is_geoid in (True, False):
                for d_e in vdatum['datum_elevation']:
                    # gridded uncertainty models are not supported yet
                    if bool("geoid" in d_e.lower()) == is_geoid and "uncertainty" not in d_e.lower() and "bathymetry" not in d_e.lower():
                        abbr1, abbr2 = d_e.split('-')
                        desc = vdatum['datum_elevation'][d_e]['description']
                        # regular expression to find the datum name after "NOS VDatum-modeled"
                        # "description" : "NOS VDatum-modeled Mean Higher High Water - Local Mean Sea Level height, realized per the ? National Tidal Datum Epoch."
                        # 2) Find and add region extents
                        for region_name, grid in vdatum['datum_elevation'][d_e]['grid'].items():
                            # try:
                                if not grid['source']['superseded']:
                                    try:
                                        ortho_key = model_regions[region_name]['ortho_sys_key']
                                        epoch = model_regions[region_name]['epoch']
                                        ortho_sys = model_regions[region_name]['ortho_sys_name']
                                    except KeyError:
                                        print(f"KeyError for ortho sys in {region_name} of {grid['source']['model']} {vdatum_version}")
                                        continue
                                    for height_or_depth in VertCRS.orientations.keys():
                                        try:
                                            vcrs_input = VertCRS(region_name, abbr1, desc, grid, ortho_key, epoch, height_or_depth, 'input')
                                            vcrs_output = VertCRS(region_name, abbr2, desc, grid, ortho_key, epoch, height_or_depth, 'output')
                                        except AttributeError:
                                            print(f"Couldn't parse datum description: {vdatum_version} {d_e} {desc}")
                                            continue
                                        process_crses(vcrs_input, vcrs_output, is_geoid)

    def process_model_tins():
        # Find the hydrographic datums
        for model_name, model in noaa_registry['model'].items():
            # process everything with a geoid first then the tidal datums and do the concatenated transforms on the second pass.
            epoch = model['epoch']
            orthos = []
            for release_name, release in model['releases'].items():
                if release['ortho_sys']:  # skip any blank entries
                    orthos.append(release['ortho_sys'])
                    if not orthos[0] == orthos[-1]:
                        raise ValueError(f"Multiple orthometric systems in a single model: {model_name} {release_name}")
            for is_geoid in (True, False):
                for d_e, grid_or_instances in model['datum_elevation'].items():
                    if 'bathymetry' in d_e.lower() or "uncertainty" in d_e.lower():
                        continue
                    if 'instance' in grid_or_instances:
                        grids = grid_or_instances['instance'].values()
                    else:
                        grids = [grid_or_instances]
                    for grid in grids:
                        if 'data_uri' not in grid:
                            print(f"No data_uri for {model_name} {d_e}")
                            continue
                        # FIXME - hacks to make the tins fit with rasters, should just subclass VertCRS for tins.
                        # gridded uncertainty models are not supported yet
                        if bool("geoid" in d_e.lower()) == is_geoid and "uncertainty" not in d_e.lower():
                            abbr1, abbr2 = d_e.split('-')
                            grid['region'] = model['domain']
                            grid['uncertainty'] = {}
                            grid['uncertainty']['transform_sigma'] = 0.10
                            grid['uncertainty'][abbr1.lower()+'_sigma'] = 0.10
                            grid['uncertainty'][abbr2.lower()+'_sigma'] = 0.10
                            desc = None

                            try:
                                ortho_key = orthos[0]
                                ortho_sys = noaa_registry['orthometric_system'][ortho_key]['crs']['geoid_model']['name']
                            except KeyError:
                                print(f"KeyError for orthometric_system of {ortho_key}")
                                continue
                            for height_or_depth in VertCRS.orientations.keys():
                                try:
                                    vcrs_input = VertCRS(model_name, abbr1, desc, grid, ortho_key, epoch, height_or_depth, 'input')
                                    vcrs_output = VertCRS(model_name, abbr2, desc, grid, ortho_key, epoch, height_or_depth, 'output')
                                except AttributeError:
                                    print(f"Couldn't parse datum description: {model_name} {d_e} {desc}")
                                    continue
                                try:
                                    process_crses(vcrs_input, vcrs_output, is_geoid)
                                except FileNotFoundError as e:
                                    print(f"Couldn't find file for {model_name} {d_e} {e.filename}")

    def process_crses(vcrs_input, vcrs_output, is_geoid):
        """ Add the two crses to the proj.db and make transforms between the two plus
        """
        # Add the rectangular extent for the GTX/TIF or TIN to the db
        # -- this is applicable to both input and output vcrs objects
        region_extent = add_to_extents(vcrs_input)
        # 3, 4) Add the input CRS to the db
        if 'EPSG' not in vcrs_input.crs_name:
            add_to_datum_and_crs(vcrs_input, region_extent)
        # Add the output CRS to the db
        # the output crs could be a standard EPSG geoid/3d so don't add it.
        if 'EPSG' not in vcrs_output.crs_name:
            add_to_datum_and_crs(vcrs_output, region_extent)

        # 5) Create grid_transformation or other_transformation for each region
        transform_name = vcrs_input.region_name + "_" + vcrs_input.datum_name + "_" + vcrs_input.height_or_depth + "_" + vcrs_output.datum_name + "_" + vcrs_output.height_or_depth
        add_to_transforms(transform_name, vcrs_input, vcrs_output, region_extent)

        # 6) make a concatenated operation to go from  tidal datum to 3d pivot and then to NAD83(2011)
        # e.g. mllw -> lmsl -> navd88(in reference frame)->reference frame 3D -> nad83 2011 3D
        # FIXME - do geoid first with a concatenated operations for 3d pivot and NAD83(2011)
        #  then do the other tidal datums and concat with the geoid ones to get, for example, MLLW to NAD83(2011)
        pivot = noaa_registry['orthometric_system'][vcrs_input.ortho_key]['crs']['pivot_3d']
        pivot_auth, pivot_code = pivot['code'].split(":")
        pivot_crs = GeodeticCRS(auth_name=pivot_auth, code=pivot_code)
        pivot_xform_name = "_".join([vcrs_input.region_name, vcrs_input.datum_name, vcrs_input.height_or_depth, pivot['name']])
        nad83_xform_name = "_".join([vcrs_input.region_name, vcrs_input.datum_name, vcrs_input.height_or_depth, "NAD83(2011)"])
        # FIXME - need to confirm uncertainty for the geoid to 3d pivot and sum the steps?
        concat_operations[pivot_xform_name] = ConcatOp(auth_name=auth_name, code=unique_id(), name=pivot_xform_name, description=f"{vcrs_input.db_crs_name} to {pivot['name']}",
                                                 source_crs=crses[vcrs_input.crs_name], target_crs=pivot_crs, accuracy=vcrs_input.uncertainty, deprecated=False)
        if is_geoid:
            # Make the concatenated operations with steps to go from Tidal datum (usually LMSL) to pivot_3d
            # and tidal datum to NAD83(2011) (note: pivot 3d might be NAD83(2011)).
            # e.g.  LMSL -> geoid realization -> 3d pivot -> NAD83(2011) (if needed)
            # If the 3d pivot is not NAD83(2011) we can just reuse the steps but leave off the last one
            # Then when we process the other tidal datums during the second loop (MLLW, MHW, etc) we can just reuse the steps to go from LMSL to NAD83(2011)
            add_geoid_and_pivot_operations(vcrs_input, transform_name, pivot_xform_name, nad83_xform_name, pivot)
        else:
            # everything is now registered with NOAA codes, so make concatenated operations to get to 3d pivots and NAD83(2011)
            # make concatenated operations for tidal datum to pivot_3d and NAD83(2011) 3d
            # just means to add the steps already done with the geoid to the xform for this tidal datum
            add_tidal_operations(vcrs_input, vcrs_output, transform_name, pivot_xform_name, nad83_xform_name, pivot)

                            # except Exception as e:
                            #     print(f"Error processing {vdatum_version, d_e, region_name}: {e}")
    find_geoid_to_3d_pivots()
    find_model_regions()
    process_datums()

    # remove any EPSG entries that we added to our dictionary - they will cause uniqueness errors if you try to add them to the database
    for k in list(crses.keys()):
        if crses[k]['auth_name'] == 'EPSG':
            crses.pop(k)

    for table in [datums, extents, crses, compound_crses, usages, other_transforms, grid_transforms, concat_operations, concat_steps, xform_usages]:
        pdb.add_all(table.values())
        if commit:
            pdb.commit()
    # FIXME this should only run if the authority is not already in the database, otherwise update or skip
    if commit:
        pdb.cursor.execute(
        f"""INSERT INTO "authority_to_authority_preference" (source_auth_name,target_auth_name, allowed_authorities) VALUES ('{auth_name}', 'EPSG', '{auth_name},PROJ,EPSG')""")
        pdb.cursor.execute(
        f"""INSERT INTO "authority_to_authority_preference" (source_auth_name,target_auth_name, allowed_authorities) VALUES ('EPSG', '{auth_name}', '{auth_name},PROJ,EPSG')""")
        pdb.commit()

    return pdb


def create_insert_statements_from_json(json_path, proj_db_path, output_path=None):
    pdb = insert_json_into_proj(json_path, proj_db_path, commit=True)
    stmts = pdb.get_insert_statements()
    if output_path is not None:
        with open(output_path, 'w') as f:
            f.writelines("\n".join(stmts))
            # FIXME this should come from the pdb but isn't right now
            f.write(f"""\nINSERT INTO "authority_to_authority_preference" (source_auth_name,target_auth_name, allowed_authorities) VALUES ('{"NOAA"}', 'EPSG', '{"NOAA"},PROJ,EPSG')""")
    return pdb


if __name__ == "__main__":
    shutil.copyfile("proj.db", "datum_files/proj.db")
    pdb = create_insert_statements_from_json("NOAA_Register.json", "datum_files/proj.db", 'noaa_inserts.sql')


# projinfo -t EPSG:6318+EPSG:5703 -s EPSG:6319 --spatial-test intersects