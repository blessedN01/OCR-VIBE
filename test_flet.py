import flet as ft
import inspect

fp = ft.FilePicker()
print("FilePicker MRO:", [c.__name__ for c in ft.FilePicker.__mro__])
print()

# Check how control type is serialized
for attr in sorted(dir(fp)):
    if 'type' in attr.lower() and not attr.startswith('__'):
        try:
            val = getattr(fp, attr, None)
            print(f"{attr}: {val}")
        except:
            pass

print()
# Check the base_control module
from flet.controls.base_control import control
print("control function:", control)

# Check what the Service class registers
from flet.controls.services.service import Service
print()
print("Service MRO:", [c.__name__ for c in Service.__mro__])
