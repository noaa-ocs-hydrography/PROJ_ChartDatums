# sqlalchemy does a nice job of reading a database and creating classes for each table.
# it also creates a nice relationship between the tables.
# however, it has problems with the INT_OR_TEXT column type.
# Adding a non-int primary key (using the TEXT part) fails with a sort error.
# it also doesn't easily print the insert statements we may want.
# we have to play games to get SQL, capturing the echo or using "compile".
# compile also has issues with the INT_OR_TEXT column type.
#
# Instead we will try a light wrapper using dictionaries (and also avoid a dependency on sqlalchemy)

import sqlite3
import functools


use_radians = False

rad_to_deg_step = '+step +proj=unitconvert +xy_in=rad +xy_out=deg '
rad_to_deg = "" if use_radians else rad_to_deg_step
deg_to_rad_step = '+step +proj=unitconvert +xy_in=deg +xy_out=rad '
deg_to_rad = "" if use_radians else deg_to_rad_step
axisswap_step = "+step +proj=axisswap +order=2,1 "


# proj has their foreign keys as TABLENAME_auth_name and TABLENAME_code - so we can take advantage of that
class ProjRecord(dict):
    """ A dictionary that can be used to create a record in a proj database table.
    """
    def __init__(self, tablename: str, columns: dict, *args, **kwargs):
        """ Create a record for a proj database table.  The keys must be in the column names for the table.

        When creating a record for a table that has a foreign key to another table, the foreign key
        can be a ProjRecord.  The ProjRecord will be used to fill in the auth_name and code columns
        from the foreign key.  Also the tablename and name column will be used if they are wanted by the record
        making the insert string.

        If using this to create an object just to fill the auth_name/code for a related object (foriegn key),
        the tablename doesn't really have to exist in the database, but it is used to create the insert statement.
        Similarly the columns don't have to exist in the database, but they are used to create the insert statement.

        Parameters
        ----------
        tablename
            The name of the table in the proj database.
        columns
            The columns in the table and their types.
        args
            args will be passed to the dict constructor
        kwargs
            kwargs will be passed to the dict constructor.  They would normally be the column names and values.
        """
        super().__init__(*args, **kwargs)
        self.__tablename__ = tablename
        self.columns = columns
        extra_keys = set(self.keys()).difference(set(self.columns.keys()))
        if extra_keys:
            for k in list(extra_keys):
                if k + "_auth_name" in self.columns:
                    extra_keys.remove(k)
            if extra_keys:
                raise ValueError(f"Keys {extra_keys} not in table {tablename}")

    @staticmethod
    def simple_format(val, dtype) -> str:
        """ Format a value for use in an insert statement.  Boolean values are converted to 1 or 0.
        Strings are quoted.  All other values are converted to strings.

        Parameters
        ----------
        val
            The value to format
        dtype
            The type of the column in the database.  This is used to determine how to format the value.

        Returns
        -------
        str
            The formatted value.
        """
        if dtype == 'BOOLEAN':
            return '1' if val else '0'
        # elif dtype == 'INT_OR_TEXT':
        elif isinstance(val, str):
            return f"'{val}'"
        else:
            return str(val)

    @property
    def insert_statement(self) -> str:
        """ Makes an insert statement for the record.
        Formats any ProjRecord values as foreign keys to fill xxx_auth_name, xxx_code and optionally xxx_table_name and xxx_name.

        Returns
        -------
        str
        """
        converted_values = {}
        for key, val in self.items():
            if isinstance(val, ProjRecord):
                # treat ProjRecord as a foriegn key
                if key+"_table_name" in self.columns:
                    converted_values[key+"_table_name"] = self.simple_format(val.__tablename__, 'TEXT')
                if key+"_name" in self.columns:
                    converted_values[key+"_name"] = self.simple_format(val['name'], 'TEXT')
                converted_values[key + "_auth_name"] = self.simple_format(val['auth_name'], 'TEXT')
                converted_values[key + "_code"] = self.simple_format(val['code'], 'INT_OR_TEXT')
            else:
                converted_values[key] = self.simple_format(val, self.columns[key])
        return f"""INSERT INTO "{self.__tablename__}" ({", ".join(converted_values.keys())}) VALUES ({", ".join(converted_values.values())})"""

    def insert(self, cursor):
        """ Insert the record into a database using the cursor."""
        try:
            cursor.execute(self.insert_statement)
        except sqlite3.IntegrityError as e:
            print(self.insert_statement)
            raise e


# The following classes are used to create records for the proj database tables to refer to.
# The coord_operation_method and coord_operation_parameter are not actually stored in the proj database,
# but are used in the records proj tracks
CoordOperationParam = functools.partial(ProjRecord, "coord_operation_parameter", {"auth_name": "TEXT", "code": "INT_OR_TEXT", "name": "TEXT"})
GenericRecord = functools.partial(ProjRecord, "generic", {"auth_name": "TEXT", "code": "INT_OR_TEXT", "name": "TEXT"})
CoordOperationMethod = functools.partial(ProjRecord, "coord_operation_method", {"auth_name": "TEXT", "code": "INT_OR_TEXT", "name": "TEXT"})


class ProjDB:
    SCOPE = "scope"
    VERTICAL_DATUM = "vertical_datum"
    VERTICAL_CRS = "vertical_crs"
    EXTENT = "extent"
    USAGE = "usage"
    GRID_TRANSFORMATION = "grid_transformation"
    OTHER_TRANSFORMATION = "other_transformation"
    COORD_SYSTEM = "coordinate_system"
    COMPOUND_CRS = "compound_crs"
    COORD_OP_METHOD = "coord_operation_method"
    COORD_OP_PARAM = "coord_operation_parameter"
    GEODETIC_CRS = "geodetic_crs"
    CONCAT_OP = 'concatenated_operation'
    CONCAT_STEP = 'concatenated_operation_step'

    def __init__(self, database_path):
        """ Create a ProjDB object to insert records into a proj database.
        The datatbase is read and the tables and columns are stored in the object.
        Classes are created for dynamically for each table/record type so any version of proj should be supported.

        Parameters
        ----------
        database_path
        """
        self.conn = None
        try:
            self.conn = sqlite3.connect(database_path)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print('Failed to establish SQLite database connection.')
            raise e
        if self.conn is not None:
            self.cursor = self.conn.cursor()
        self.adding = []
        self.inserted = []
        # These first two don't exist in the ProjDB - so these are just for convenience
        self.classes = {self.COORD_OP_PARAM: CoordOperationParam, self.COORD_OP_METHOD: CoordOperationMethod}
        self.cursor.execute("""SELECT name FROM sqlite_master WHERE type='table';""")
        data = self.cursor.fetchall()
        for rec in data:
            table = rec[0]
            # sqlite3 allows loose types, find them here.
            typs = self.cursor.execute(f"""SELECT name, type FROM pragma_table_info('{table}')""").fetchall()
            columns = {t[0]: t[1] for t in typs}
            # data_desc = self.cursor.execute(f'''SELECT * FROM {table}''')
            # columns = [column[0] for column in data_desc.description]
            self.classes[table] = functools.partial(ProjRecord, table, columns)

    def get_class(self, class_name):
        return self.classes[class_name]

    def get_vertical_classes(self):
        """Return the classes needed to create records to allow adding Vertical datums and transforms.
        These are the vertical datum, vertical crs, extent, and usage classes"""
        VerticalDatum = self.get_class(self.VERTICAL_DATUM)
        VerticalCRS = self.get_class(self.VERTICAL_CRS)
        Extent = self.get_class(self.EXTENT)
        Usage = self.get_class(self.USAGE)
        Scope = self.get_class(self.SCOPE)
        return VerticalDatum, VerticalCRS, Extent, Usage, Scope

    def add(self, record):
        """ Add a record to the database.  Similar to sqlalchemy session.add()"""
        self.adding.append(record)
        self.inserted.append(record)

    def add_all(self, records):
        """ Add a list of records to the database.  Similar to sqlalchemy session.add_all()"""
        for record in records:
            self.add(record)

    def commit(self):
        """ Commit the changes to the database.  Similar to sqlalchemy session.commit()"""
        for record in self.adding:
            record.insert(self.cursor)
        self.adding = []
        self.conn.commit()

    def get_insert_statements(self, classes=None):
        """ Get the insert statements for the records that have been added to the database.

        Parameters
        ----------
        classes
            filter to a specific list of classes.
        Returns
        -------
        list
            strings of the insert statements
        """
        statements = []
        for record in self.inserted:
            if classes is None or record.__class__ in classes:
                statements.append(record.insert_statement)
        return statements


if __name__ == "__main__":
    pdb = ProjDB("proj - copy.db")
    VerticalDatum, VerticalCRS, Extent, Usage, Scope = pdb.get_vertical_classes()
    GridTransform, OtherTransform = pdb.get_class("grid_transformation"), pdb.get_class("other_transformation")
    CoordSys = pdb.get_class("coordinate_system")

    # @TODO find the right scopes (this doesn't stop proj from working though)
    change_height_scope = Scope(auth_name='EPSG', code=1059)  # pdb.session.query(Scope).where(Scope.code == 1059 and Scope.auth_name=='EPSG').first()
    print("for a Scope object the columns are:", change_height_scope.columns)
    coastal_hydrography = Scope(auth_name='EPSG', code=1103)  # pdb.session.query(Scope).where(Scope.code == 1103 and Scope.auth_name=='EPSG').first()
    geoid_model = Scope(auth_name='EPSG', code=1270)  # pdb.session.query(Scope).where(Scope.code == 1270 and Scope.auth_name=='EPSG').first()
    cs_up = CoordSys(auth_name='EPSG', code=6499)
    current_id = 0
    def unique_id():
        global current_id
        current_id += 1
        return current_id
    mllw_datum = VerticalDatum(auth_name="NOAA", code='txt', name="Mean Lower Low Water", deprecated=False)
    print("available columns for vertical datum:", mllw_datum.columns)
    print("the insert statement for the mllw datum:", mllw_datum.insert_str())
    lmsl_datum = VerticalDatum(auth_name="NOAA", code=unique_id(), name="Mean Sea Level", deprecated=False)
    navd88_datum = VerticalDatum(auth_name="NOAA", code=unique_id(), name="North American Vertical Datum 1988", deprecated=False)
    all_datums = [mllw_datum, lmsl_datum, navd88_datum]
    pdb.add_all(all_datums)

    region_name = "NOAA_Region_1"
    region_extent = Extent(auth_name="NOAA", code=unique_id(), name=region_name +" extent", description=region_name+"An extent description from noaa", south_lat=-90, north_lat=90, west_lon=-180, east_lon=180, deprecated=False)
    pdb.add(region_extent)

    region_mllw_crs = VerticalCRS(auth_name="NOAA", code=unique_id(), name=region_name+"_MLLW", description=region_name+"A description from noaa", coordinate_system=cs_up, datum=mllw_datum, deprecated=False)
    region_lmsl_crs = VerticalCRS(auth_name="NOAA", code=unique_id(), name=region_name+"_LMSL", description=region_name+"A description from noaa", coordinate_system_auth_name='EPSG', coordinate_system_code=6499, datum=lmsl_datum, deprecated=False)
    region_navd88_crs = VerticalCRS(auth_name="NOAA", code=unique_id(), name=region_name+"_NAVD88", description=region_name+"A description from noaa", coordinate_system=cs_up, datum=navd88_datum, deprecated=False)
    region_crses = [region_mllw_crs, region_lmsl_crs, region_navd88_crs]
    pdb.add_all(region_crses)

    region_mllw_crs_usage = Usage(auth_name="NOAA", code=unique_id(), object=region_mllw_crs, extent=region_extent, scope=change_height_scope)
    print(region_mllw_crs_usage.insert_str())
    region_lmsl_crs_usage = Usage(auth_name="NOAA", code=unique_id(), object=region_lmsl_crs, extent=region_extent, scope=change_height_scope)
    region_navd88_crs_usage = Usage(auth_name="NOAA", code=unique_id(), object=region_navd88_crs, extent=region_extent, scope=change_height_scope)
    # @TODO Do we need to specify a usage per region?  Can we just make a generic one for the world?
    mllw_datum_usage = Usage(auth_name="NOAA", code=unique_id(), object=mllw_datum, extent=region_extent, scope=coastal_hydrography)
    lmsl_datum_usage = Usage(auth_name="NOAA", code=unique_id(), object=lmsl_datum, extent=region_extent, scope=coastal_hydrography)
    navd88_datum_usage = Usage(auth_name="NOAA", code=unique_id(), object=navd88_datum, extent=region_extent, scope=coastal_hydrography)

    pdb.add_all([region_mllw_crs_usage, region_lmsl_crs_usage, region_navd88_crs_usage])
    pdb.add_all([mllw_datum_usage, lmsl_datum_usage, navd88_datum_usage])

    # Example of using a grid transform to convert from NAVD88 to NAD83(2011) by vdatum's geoid12b grid
    GEOID_PATH = "geoid12b.gtx"
    navd88_nad83_2011_xform = GridTransform(auth_name="NOAA", code=unique_id(), name='NAVD88_to_NAD83_2011_VDATUM', description='NAVD88 to NAD83(2011) by vdatum grid',
                                   method_auth_name='EPSG', method_code=9615, method_name='Geographic3D to GravityRelatedHeight (gtx)',
                                   source_crs_auth_name='EPSG', source_crs_code=6319,  # EPSG 6319 = 3D  NAD83(2011) CRS
                                   target_crs=region_navd88_crs,
                                   grid_param_auth_name='EPSG', grid_param_code=8666, grid_param_name='Geoid (height correction) model file', grid_name=GEOID_PATH,
                                   deprecated=False)
    pdb.add(navd88_nad83_2011_xform)

    grid_xform_usage = Usage(auth_name="NOAA", code=unique_id(), object=navd88_nad83_2011_xform, extent=region_extent, scope=geoid_model)
    pdb.add(grid_xform_usage)

    CHES_MLLW_LMSL_PATH = "somepath_to_tin.json"
    uncertainty = 0.1
    CHES_LMSL_NAVD88_PATH = "somepath_to_tin2.json"
    uncertainty2 = 0.2
    mllw_lmsl_xform = OtherTransform(auth_name="NOAA", code=unique_id(), name="test TIN transform", description='test TIN description',
                                 method_auth_name='PROJ', method_code='PROJString', method_name='+proj=pipeline ' +
                                                                                                f'{rad_to_deg}' +
                                                                                                f'+step +proj=tinshift +file={CHES_MLLW_LMSL_PATH} +inv ' +
                                                                                                f'{deg_to_rad}',
                                 source_crs=region_mllw_crs,
                                 target_crs=region_lmsl_crs,
                                 accuracy=uncertainty,
                                 deprecated=False)
    # LMSL to NAVD88 for chesapeake testing only
    lmsl_custom_navd88_xform = OtherTransform(auth_name="NOAA", code=unique_id(), name="test TIN transform", description='test TIN description',
                                 method_auth_name='PROJ', method_code='PROJString', method_name='+proj=pipeline ' +
                                                                                                f'{rad_to_deg}' +
                                                                                                f'+step +proj=tinshift +file={CHES_LMSL_NAVD88_PATH} +inv ' +
                                                                                                f'{deg_to_rad}',
                                 source_crs=region_lmsl_crs,
                                 target_crs=region_navd88_crs,
                                 accuracy=uncertainty2,
                                 deprecated=False)
    # LMSL to NAVD88 height using EPSG 5703
    lmsl_navd88_xform = OtherTransform(auth_name="NOAA", code=unique_id(), name="test TIN transform", description='test TIN description',
                                 method_auth_name='PROJ', method_code='PROJString', method_name='+proj=pipeline ' +
                                                                                                f'{rad_to_deg}' +
                                                                                                f'+step +proj=tinshift +file={CHES_LMSL_NAVD88_PATH} +inv ' +
                                                                                                f'{deg_to_rad}',
                                 source_crs=region_lmsl_crs,
                                 target_crs_auth_name='EPSG', target_crs_code=5703,
                                 accuracy=uncertainty2,
                                 deprecated=False)
    pdb.add_all([mllw_lmsl_xform, lmsl_custom_navd88_xform, lmsl_navd88_xform])
    # pdb.commit()

    mllw_lmsl_xform_usage = Usage(auth_name="NOAA", code=unique_id(), object=mllw_lmsl_xform, extent=region_extent, scope=coastal_hydrography)
    lmsl_custom_navd88_xform_usage = Usage(auth_name="NOAA", code=unique_id(), object=lmsl_custom_navd88_xform, extent=region_extent, scope=coastal_hydrography)
    lmsl_navd88_xform_usage = Usage(auth_name="NOAA", code=unique_id(), object=lmsl_navd88_xform, extent=region_extent, scope=coastal_hydrography)
    pdb.add_all([mllw_lmsl_xform_usage, lmsl_custom_navd88_xform_usage, lmsl_navd88_xform_usage])
    pdb.commit()


    for s in pdb.get_insert_statements():
        print(s)
