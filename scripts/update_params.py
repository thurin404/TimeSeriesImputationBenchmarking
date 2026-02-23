#!/usr/bin/env python3
"""Update parameters in YAML configuration files for models."""
import os
import sys
import yaml

def update_params(file_path: str, field: str|None = None, parameter: str = None, value: any = None):
    """Update parameters in a YAML file."""
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)

    # Update parameters
    if field:
        try: 
            params[field]
        except KeyError:
            print(f"Field {field} not found in {file_path}.")
            return
        try: 
            params[field][parameter]
        except KeyError:
            print(f"Parameter {parameter} not found in field {field} of {file_path}.")
            return
        params[field][parameter] = value
    else:
        if parameter not in params:
            print(f"Parameter {parameter} not found in {file_path}.")
            return
        params[parameter] = value

    # Write back to the file
    with open(file_path, 'w') as file:
        yaml.dump(params, file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/update_params.py <prameter> <value> or \n"
        "<field> <parameter> <value>")
        sys.exit(1)
    parameter = sys.argv[-2]
    value = sys.argv[-1]
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            # else keep as string
    if len(sys.argv) == 4:
        field = sys.argv[1]
    else:
        field = None

    for file_name in os.listdir('params'):
        if file_name.endswith('.yaml'):
            file_path = os.path.join('params', file_name)
            update_params(file_path, field, parameter, value)
            print(f"Updated {parameter} in {file_name} to {value}")
    sys.exit(0)

        