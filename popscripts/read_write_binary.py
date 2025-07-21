#!/usr/bin/env python3
import numpy as np
import xarray as xr

def get_variables(filetype):
    if filetype == "ws":
        fields_file_binary = ("TAUX", "TAUY")
        fields_descr_file_binary = ("zonal surface wind stress", "meridional surface wind stress")
        fields_unit_file_binary = ("dyne/cm^2", "dyne/cm^2")
    elif filetype == "shf":
        fields_file_binary = ("SST", "TEMP10", "Q10", "FDS", "FDL", "WS10", "ARHFLX")
        fields_descr_file_binary = ("Restoring SST (used only to determine location of sea ice)", "10m atmospheric temperature", "10m atmospheric humidity", "Downward shortwave", "Downward longwave", "10m atmospheric wind speed", "Applied restoring heat flux")
        fields_unit_file_binary = ("C", "K", "kg/kg", "W/m^2", "W/m^2", "m/s", "W/m^2")
    elif filetype == "sfwf":
        fields_file_binary = ("SSS", "PREC", "RUNOFF", "ARSFLX")
        fields_descr_file_binary = ("Restoring SSS", "Precipitation", "Runoff", "Applied restoring S flux")
        fields_unit_file_binary = ("psu", "m/y", "kg/m2/s", "kg/m2/s")
    elif filetype == "shf_restoring":
        fields_file_binary = ["SST"]
        fields_descr_file_binary = ["Restoring SST"]
        fields_unit_file_binary = ["C"]
    elif filetype == "sfwf_restoring":
        fields_file_binary = ["SSS"]
        fields_descr_file_binary = ["Restoring SSS"]
        fields_unit_file_binary = ["psu"]
    else:
        raise ValueError(f"File type {filetype} is not supported, must be one of 'ws', 'shf', 'sfwf', or *_restoring")
    return fields_file_binary, fields_descr_file_binary, fields_unit_file_binary


####################### READING ########################

def read_pop_binary(file, nrec_in, num_rows=384, num_cols=320, rec_length=8):
    total_elements = num_rows * num_cols
    dtype = f'>f{rec_length}'
    
    with open(file, 'rb') as f:
        # Move to the desired record (Fortran uses 1-based indexing)
        f.seek((nrec_in - 1) * total_elements * rec_length)
        
        # Read the data into a NumPy array with big-endian format
        field = np.fromfile(f, dtype=dtype, count=total_elements)
    
    return field.reshape((num_rows, num_cols))


def read_pop_fields(file, filetype, num_rows=384, num_cols=320, rec_length=8, nrframes=12, return_type="numpy", xr_coord_file=None):
    # Load coordinates if xarray output is desired
    if return_type == "xarray":
        if xr_coord_file is not None:
            xr_coords = xr.open_dataset(xr_coord_file)["TAREA"]
        else:
            raise ValueError("Must provide 'xr_coord_file' argument with return type 'xarray'")
    elif return_type != "numpy":
        raise ValueError(f"Return type {return_type} is not supported, must be one of 'numpy', 'xarray'")

    # filetype is one of ws, shf or sfwf
    fields, fields_descr, fields_unit = get_variables(filetype)
    nfields = len(fields)
    vars = []
    # Read field for each variable
    for i in range(nfields):
        records = []
        for record in range(1,nrframes+1):
            records.append(read_pop_binary(file, record+nrframes*i, num_rows=num_rows, num_cols=num_cols, rec_length=rec_length))
        field_np = np.stack(records)
        if return_type == "numpy":
            vars.append(field_np)
        else:
            field_xr = xr.DataArray(field_np, dims=["record", "nlat", "nlon"], coords=xr_coords.coords)
            field_xr.name = fields[i]
            field_xr.attrs = {"long_name": fields_descr[i], "units": fields_unit[i]}
            vars.append(field_xr.to_dataset())

    # Merge, attributes
    if return_type == "numpy":
        return np.stack(vars)
    elif return_type == "xarray":
        return xr.merge(vars)


def parse_pop_header(file_path):
    entries = {}
    current_entry = {}
    current_key = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('&'):
                if current_entry:
                    entries[current_key] = current_entry
                current_key = line[1:]  # Remove the '&' character
                current_entry = {}
            elif line == '/':
                if current_entry:
                    entries[current_key] = current_entry
                    current_entry = {}
                    current_key = None
            else:
                linesep = line.split(':')
                if len(linesep)<3:
                    continue
                key = linesep[0]; dtype = linesep[1]; value = linesep[2]
                current_entry[key.strip()] = value.strip()
                if dtype.strip() == "int":
                    current_entry[key.strip()] = int(value.strip())

    return entries

####################### WRITING ########################

def write_pop_binary(field, file, nrec_out, num_rows=384, num_cols=320, rec_length=8):
    total_elements = num_rows * num_cols
    dtype = f'>f{rec_length}'
    field = field.ravel().astype(dtype) # convert to big-endian double-precision float

    with open(file, 'wb') as f:
        # Move to the desired record (Fortran uses 1-based indexing)
        f.seek((nrec_out - 1) * total_elements * rec_length)

        # Write the data to file
        field.tofile(f)

    print(f"Data written to {file} at record {nrec_out}.")


def write_pop_fields(input_field, file, filetype, num_rows=384, num_cols=320, nrframes=12):
    # filetype is one of ws, shf or sfwf
    fields, _, _ = get_variables(filetype)
    nfields = len(fields)

    if type(input_field) == xr.Dataset: # If xarray, convert to numpy array
        # TODO: missing variables
        input_np = np.zeros((nfields, nrframes, num_rows, num_cols))
        for i in range(nfields):
            input_np[i,:,:,:] = input_field[fields[i]].transpose("record", "nlat", "nlon").values
    else: # Numpy
        # Check dimensions of input file
        if input_field.shape == (nfields, nrframes, num_rows, num_cols):
            input_np = input_field
        else:
            raise ValueError(f"Input is of shape {input_field.shape} but expected {(nfields, nrframes, num_rows, num_cols)}")
    
    # Check if any field contains NaNs (this will probably lead to unintended model behavior)
    for i in range(nfields):
        if np.isnan(input_np[i,:,:,:]).sum() > 0:
            print(f"Warning: {fields[i]} contains NaN values. This will probably lead to unintended behavior in POP!")

    # Write field for each variable
    with open(file, 'wb') as f:
        for i in range(nfields):
            for record in range(1,nrframes+1):
                write_field = input_np[i,record-1,:,:].copy().astype('>f8')
                f.write(write_field.tobytes())
            print(f"{fields[i]} written to {file} at record {nrframes*i+1}-{nrframes*(i+1)}.")