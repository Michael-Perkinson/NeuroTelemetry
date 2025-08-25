# test_main.py
import importlib

# Flags to control which tests to run
RUN_PRESSURE = True
RUN_PHOTOMETRY = True

if RUN_PRESSURE:
    print("\n=== Running test_pressure_pipeline ===")
    importlib.import_module("tests.test_pressure_pipeline")

if RUN_PHOTOMETRY:
    print("\n=== Running test_photometry_pipeline ===")
    importlib.import_module("tests.test_photometry_pipeline")

print("\n✅ Finished running selected tests")
