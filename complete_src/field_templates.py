#!/usr/bin/env python3
"""
HRRR Field Templates System
Defines base templates for common HRRR parameter types with inheritance
"""

import numpy as np
from typing import Dict, Any, Optional
import copy

class FieldTemplates:
    """Base templates for common HRRR field types"""
    
    # Common access patterns
    ACCESS_PATTERNS = {
        'surface_instant': {'typeOfLevel': 'surface', 'stepType': 'instant'},
        'surface_accum': {'typeOfLevel': 'surface', 'stepType': 'accum'},
        'atmosphere': {'typeOfLevel': 'atmosphere'},
        'atmosphere_single_layer': {'typeOfLevel': 'entireAtmosphere'},
        'height_agl': {'typeOfLevel': 'heightAboveGround'},
        'pressure_layer': {'typeOfLevel': 'pressureFromGroundLayer', 'stepType': 'instant'},
        'pressure_layer_max': {'typeOfLevel': 'pressureFromGroundLayer', 'stepType': 'max'},
        'height_layer': {'typeOfLevel': 'heightAboveGroundLayer', 'stepType': 'instant'},
        'param_id': {}  # Will be filled with paramId
    }
    
    # Base field templates
    TEMPLATES = {
        # === INSTABILITY TEMPLATES ===
        'cape_base': {
            'units': 'J/kg',
            'cmap': 'CAPE',
            'levels': [0, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
            'extend': 'max',
            'var': 'cape',
            'category': 'instability',
            'description': 'Convective Available Potential Energy indicates updraft strength potential'
        },
        
        'cin_base': {
            'units': 'J/kg', 
            'cmap': 'CIN',
            'levels': [-500, -300, -200, -150, -100, -75, -50, -25, -10, 0],
            'extend': 'min',
            'var': 'cin',
            'category': 'instability',
            'description': 'Convective Inhibition represents energy barrier preventing convection'
        },
        
        'surface_cape': {
            'template': 'cape_base',
            'access_pattern': 'surface_instant',
            'title_prefix': 'Surface-Based',
            'description': 'HRRR model-calculated SBCAPE using proper thermodynamic parcel lifting'
        },
        
        'mixed_layer_cape': {
            'template': 'cape_base', 
            'access_pattern': 'pressure_layer',
            'level': 18000,
            'title_prefix': 'Mixed-Layer',
            'title_suffix': '(180-0 mb)',
            'description': 'HRRR model-calculated MLCAPE for lowest 100mb averaged parcel'
        },
        
        'most_unstable_cape': {
            'template': 'cape_base',
            'access_pattern': 'pressure_layer', 
            'level': 25500,
            'title_prefix': 'Most Unstable',
            'title_suffix': '(255-0 mb)',
            'description': 'HRRR model-calculated MUCAPE for most unstable parcel in lowest 300mb'
        },
        
        'low_level_cape': {
            'template': 'cape_base',
            'access_pattern': 'pressure_layer',
            'level': 30000,
            'levels': [25, 50, 100, 150, 200, 300, 400, 500, 750],
            'title_prefix': '0-3 km AGL',
            'title_suffix': '(300-0 mb)'
        },
        
        'low_level_cin': {
            'template': 'cin_base',
            'access_pattern': 'pressure_layer',
            'level': 30000,
            'title_prefix': '0-3 km AGL', 
            'title_suffix': '(300-0 mb)'
        },
        
        'surface_cin': {
            'template': 'cin_base',
            'access_pattern': 'surface_instant', 
            'title_prefix': 'Surface-Based',
            'description': 'HRRR model-calculated SBCIN using proper integration to Level of Free Convection'
        },
        
        'mixed_layer_cin': {
            'template': 'cin_base',
            'access_pattern': 'pressure_layer',
            'level': 18000,
            'title_prefix': 'Mixed-Layer',
            'title_suffix': '(180-0 mb)',
            'description': 'HRRR model-calculated MLCIN for lowest 100mb mixed parcel'
        },
        
        'most_unstable_cin': {
            'template': 'cin_base',
            'access_pattern': 'pressure_layer',
            'level': 25500,
            'title_prefix': 'Most Unstable',
            'title_suffix': '(255-0 mb)',
            'description': 'HRRR model-calculated MUCIN (often zero since MU parcel chosen for instability)'
        },
        
        'lifted_index': {
            'access_pattern': 'pressure_layer',
            'level': 18000,
            'var': 'lftx4',
            'title': 'Surface-Based Lifted Index',
            'units': '°C',
            'cmap': 'LiftedIndex',
            'levels': [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8],
            'extend': 'both',
            'category': 'instability'
        },
        
        # === REFLECTIVITY TEMPLATES ===
        'reflectivity_base': {
            'units': 'dBZ',
            'cmap': 'NWSReflectivity', 
            'levels': list(range(5, 75, 5)),
            'extend': 'max',
            'var': 'refd',
            'category': 'reflectivity'
        },
        
        'composite_reflectivity': {
            'access_pattern': 'atmosphere',
            'var': 'refc',
            'title': 'Composite Reflectivity',
            'units': 'dBZ',
            'cmap': 'NWSReflectivity',
            'levels': list(range(5, 75, 5)),
            'extend': 'max',
            'category': 'reflectivity'
        },
        
        'height_reflectivity': {
            'template': 'reflectivity_base',
            'access_pattern': 'height_agl'
        },
        
        # === SURFACE/NEAR-SURFACE TEMPLATES ===
        'surface_temperature': {
            'access_pattern': 'param_id',
            'units': '°C',
            'cmap': 'RdYlBu_r',
            'levels': list(range(-30, 45, 5)),
            'extend': 'both',
            'transform': 'celsius',
            'category': 'surface'
        },
        
        'surface_moisture': {
            'access_pattern': 'param_id',
            'units': '%',
            'cmap': 'BrBG',
            'extend': 'max',
            'category': 'surface'
        },
        
        'surface_wind': {
            'access_pattern': 'param_id',
            'units': 'm/s',
            'cmap': 'RdBu_r',
            'extend': 'both',
            'category': 'surface'
        },
        
        'surface_pressure_field': {
            'access_pattern': 'surface_instant',
            'var': 'sp',
            'title': 'Surface Pressure',
            'units': 'mb',
            'cmap': 'viridis',
            'levels': list(range(960, 1040, 4)),
            'extend': 'both',
            'transform': 'mb',
            'category': 'surface'
        },
        
        # === PRECIPITATION TEMPLATES ===
        'precipitation_rate': {
            'access_pattern': 'surface_instant',
            'stepType': 'instant',
            'shortName': 'prate',
            'var': 'prate',
            'title': 'Precipitation Rate',
            'units': 'mm/hr',
            'cmap': 'viridis',
            'levels': [0.01, 0.1, 1, 5, 10, 25, 50, 100, 200, 400],
            'extend': 'max',
            'transform': 'prate_units',
            'category': 'precipitation'
        },
        
        'precipitation_accumulation': {
            'access_pattern': 'surface_accum',
            'var': 'tp',
            'title': 'Total Precipitation',
            'units': 'mm',
            'cmap': 'viridis',
            'levels': [1, 2.5, 5, 10, 25, 50, 75, 100, 150],
            'extend': 'max',
            'category': 'precipitation'
        },
        
        # === SEVERE WEATHER TEMPLATES ===
        'hail_field': {
            'access_pattern': 'atmosphere',
            'var': 'hail',
            'title': 'Maximum Hail Size',
            'units': 'mm',
            'cmap': 'Hail',
            'levels': [5, 10, 15, 20, 25, 30, 40, 50, 60, 75],
            'extend': 'max',
            'transform': 'hail_size',
            'category': 'severe'
        },
        
        'storm_helicity': {
            'access_pattern': 'height_layer',
            'var': 'hlcy',
            'units': 'm²/s²',
            'cmap': 'plasma',
            'extend': 'max',
            'process': 'select_layer',
            'category': 'severe'
        },
        
        'updraft_helicity': {
            'access_pattern': 'height_layer',
            'var': 'MXUPHL',
            'level': [3000, 0],
            'title': 'Hourly Maximum Updraft Helicity (0-3 km)',
            'units': 'm²/s²',
            'cmap': 'plasma',
            'levels': [5, 10, 15, 25, 40, 60, 80, 100, 150, 200],
            'extend': 'max',
            'stepType': 'max',
            'requires_multi_dataset': True,
            'level_selection': {'bottom': 0, 'top': 3000},
            'wgrib2_pattern': 'MXUPHL:3000-0 m above ground',
            'category': 'updraft_helicity',
            'plot_style': 'filled'
        },
        
        'updraft_helicity_25km': {
            'access_pattern': 'height_layer',
            'var': 'MXUPHL',
            'level': [5000, 2000],
            'title': 'Updraft Helicity (2-5 km AGL)',
            'units': 'm²/s²',
            'cmap': 'plasma',
            'levels': [5, 10, 15, 25, 40, 60, 80, 100, 150, 200],
            'extend': 'max',
            'stepType': 'max',
            'requires_multi_dataset': True,
            'level_selection': {'bottom': 2000, 'top': 5000},
            'wgrib2_pattern': 'MXUPHL:5000-2000 m above ground',
            'category': 'updraft_helicity',
            'plot_style': 'filled'
        },
        
        'updraft_helicity_01km': {
            'access_pattern': 'height_layer',
            'var': 'MXUPHL',
            'level': [2000, 0],
            'title': 'Updraft Helicity (0-2 km AGL)',
            'units': 'm²/s²',
            'cmap': 'plasma',
            'levels': [5, 10, 15, 25, 40, 60, 80, 100, 150, 200],
            'extend': 'max',
            'stepType': 'max',
            'requires_multi_dataset': True,
            'level_selection': {'bottom': 0, 'top': 2000},
            'wgrib2_pattern': 'MXUPHL:2000-0 m above ground',
            'category': 'updraft_helicity',
            'plot_style': 'filled'
        },
        
        'wind_shear_base': {
            'access_pattern': 'height_layer',
            'units': 'm/s',
            'cmap': 'viridis',
            'extend': 'max',
            'category': 'severe'
        },
        
        'wind_shear_u_06km': {
            'access_pattern': 'height_layer',
            'var': 'vucsh',
            'level': [0, 6000],
            'title': '0-6 km U-Component Wind Shear',
            'units': 'm/s',
            'cmap': 'RdBu_r',
            'levels': [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25],
            'extend': 'both',
            'requires_multi_dataset': True,
            'level_selection': {'bottom': 0, 'top': 6000},
            'wgrib2_pattern': 'VUCSH:0-6000 m above ground',
            'category': 'severe'
        },
        
        'wind_shear_v_06km': {
            'access_pattern': 'height_layer',
            'var': 'vvcsh',
            'level': [0, 6000],
            'title': '0-6 km V-Component Wind Shear',
            'units': 'm/s',
            'cmap': 'RdBu_r',
            'levels': [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25],
            'extend': 'both',
            'requires_multi_dataset': True,
            'level_selection': {'bottom': 0, 'top': 6000},
            'wgrib2_pattern': 'VVCSH:0-6000 m above ground',
            'category': 'severe'
        },
        
        'wind_shear_u_01km': {
            'access_pattern': 'height_layer',
            'var': 'vucsh',
            'level': [0, 1000],
            'title': '0-1 km U-Component Wind Shear',
            'units': 'm/s',
            'cmap': 'RdBu_r',
            'levels': [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15],
            'extend': 'both',
            'requires_multi_dataset': True,
            'level_selection': {'bottom': 0, 'top': 1000},
            'wgrib2_pattern': 'VUCSH:0-1000 m above ground',
            'category': 'severe'
        },
        
        'wind_shear_v_01km': {
            'access_pattern': 'height_layer',
            'var': 'vvcsh',
            'level': [0, 1000],
            'title': '0-1 km V-Component Wind Shear',
            'units': 'm/s',
            'cmap': 'RdBu_r',
            'levels': [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15],
            'extend': 'both',
            'requires_multi_dataset': True,
            'level_selection': {'bottom': 0, 'top': 1000},
            'wgrib2_pattern': 'VVCSH:0-1000 m above ground',
            'category': 'severe'
        },
        
        'lightning_strike_density': {
            'access_pattern': 'height_agl',
            'level': 1,
            'var': 'ltngsd',
            'title': 'Lightning Strike Density',
            'units': 'strikes/km²/min',
            'cmap': 'plasma',
            'levels': [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30],
            'extend': 'max',
            'category': 'severe'
        },
        
        'lightning_flash_rate': {
            'access_pattern': 'atmosphere',
            'var': 'ltng',
            'title': 'Lightning Flash Rate',
            'units': 'flashes/km²/min',
            'cmap': 'plasma',
            'levels': [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30],
            'extend': 'max',
            'category': 'severe'
        },
        
        'echo_top': {
            'access_pattern': 'surface_instant',
            'var': 'RETOP',
            'title': 'Echo Top Height',
            'units': 'm',
            'cmap': 'viridis',
            'levels': [3000, 6000, 9000, 12000, 15000, 18000, 21000],
            'extend': 'max',
            'requires_multi_dataset': True,
            'grib_shortname_match': 'RETOP',
            'wgrib2_pattern': 'RETOP:cloud top',
            'category': 'severe'
        },
        
        'vil': {
            'access_pattern': 'atmosphere',
            'var': 'VIL',
            'title': 'Vertically Integrated Liquid',
            'units': 'kg/m²',
            'cmap': 'viridis',
            'levels': [5, 10, 20, 30, 40, 50, 60, 70, 80],
            'extend': 'max',
            'requires_multi_dataset': True,
            'grib_shortname_match': 'VIL',
            'wgrib2_pattern': 'VIL:entire atmosphere',
            'category': 'severe'
        },
        
        # === ATMOSPHERIC TEMPLATES ===
        'atmospheric_field': {
            'access_pattern': 'atmosphere',
            'cmap': 'viridis',
            'extend': 'max',
            'category': 'atmospheric'
        },
        
        'boundary_layer': {
            'access_pattern': 'surface_instant',
            'var': 'blh',
            'title': 'Planetary Boundary Layer Height',
            'units': 'm',
            'cmap': 'viridis',
            'levels': [250, 500, 750, 1000, 1500, 2000, 2500, 3000],
            'extend': 'max',
            'category': 'atmospheric'
        },
        
        # === SMOKE TEMPLATES ===
        'smoke_surface': {
            'access_pattern': 'height_agl',
            'level': 8,
            'var': 'mdens',
            'title': 'Near-Surface Smoke (8m AGL)',
            'units': 'μg/m³',
            'cmap': 'NOAASmoke',
            'levels': [1, 2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 60, 100, 200],
            'extend': 'max',
            'transform': 'smoke_concentration',
            'category': 'smoke'
        },
        
        'smoke_column': {
            'access_pattern': 'param_id',
            'param_id': 400001,
            'var': 'colmd',
            'title': 'Column-Integrated Smoke Mass',
            'units': 'mg/m²',
            'cmap': 'Reds',
            'levels': [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
            'extend': 'max',
            'transform': 'smoke_column',
            'category': 'smoke'
        },
        
        'smoke_aloft': {
            'access_pattern': 'height_agl',
            'var': 'MASSDEN',
            'title': 'Smoke Concentration at Altitude',
            'units': 'μg/m³',
            'cmap': 'OrRd',
            'levels': [1, 5, 10, 25, 50, 100, 150, 200, 300],
            'extend': 'max',
            'transform': 'smoke_concentration',
            'category': 'smoke'
        },
        
        'smoke_visibility': {
            'access_pattern': 'height_agl',
            'level': 8,
            'var': 'vis',
            'title': 'Visibility (Smoke Impact)',
            'units': 'm',
            'cmap': 'viridis_r',
            'levels': [800, 1600, 3200, 4800, 8000, 16000, 24000],
            'extend': 'min',
            'category': 'smoke'
        },
    }

    @classmethod
    def resolve_template(cls, field_config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template inheritance and apply overrides"""
        result = {}
        
        # If this config references a template, resolve it first
        if 'template' in field_config:
            template_name = field_config['template']
            if template_name in cls.TEMPLATES:
                # Recursively resolve template (in case template references another template)
                base_template = cls.resolve_template(cls.TEMPLATES[template_name])
                result.update(base_template)
            else:
                raise ValueError(f"Template '{template_name}' not found")
        
        # Apply current config, overriding template values
        for key, value in field_config.items():
            if key != 'template':  # Don't include template reference in final config
                result[key] = value
        
        # Resolve access pattern
        if 'access_pattern' in result:
            pattern_name = result['access_pattern']
            if pattern_name in cls.ACCESS_PATTERNS:
                access_base = copy.deepcopy(cls.ACCESS_PATTERNS[pattern_name])
                
                # Add level if specified
                if 'level' in result:
                    access_base['level'] = result['level']
                    
                # Add paramId if this is param_id pattern
                if pattern_name == 'param_id' and 'param_id' in result:
                    access_base['paramId'] = result['param_id']
                
                # Add stepType if specified
                if 'stepType' in result:
                    access_base['stepType'] = result['stepType']
                
                # Add shortName if specified
                if 'shortName' in result:
                    access_base['shortName'] = result['shortName']
                
                result['access'] = access_base
                # Clean up pattern references
                del result['access_pattern']
                if 'level' in result and 'level' in access_base:
                    del result['level']
                if 'param_id' in result:
                    del result['param_id']
                if 'stepType' in result and 'stepType' in access_base:
                    del result['stepType']
                if 'shortName' in result and 'shortName' in access_base:
                    del result['shortName']
            else:
                raise ValueError(f"Access pattern '{pattern_name}' not found")
        
        # Build title if using prefixes/suffixes
        if 'title_prefix' in result:
            base_title = result.get('title', result.get('var', 'Unknown'))
            if 'CAPE' in base_title or result.get('var') == 'cape':
                base_title = 'CAPE'
            elif 'CIN' in base_title or result.get('var') == 'cin':
                base_title = 'CIN'
            
            title_parts = []
            if 'title_prefix' in result:
                title_parts.append(result['title_prefix'])
            title_parts.append(base_title)
            if 'title_suffix' in result:
                title_parts.append(result['title_suffix'])
            
            result['title'] = ' '.join(title_parts)
            
            # Clean up title building components
            for key in ['title_prefix', 'title_suffix']:
                if key in result:
                    del result[key]
        
        return result

    @classmethod
    def get_available_templates(cls) -> Dict[str, str]:
        """Get list of available templates with descriptions"""
        descriptions = {
            'cape_base': 'Base CAPE configuration',
            'cin_base': 'Base CIN configuration', 
            'surface_cape': 'Surface-based CAPE',
            'mixed_layer_cape': 'Mixed-layer CAPE (180-0 mb)',
            'most_unstable_cape': 'Most unstable CAPE (90-0 mb)',
            'low_level_cape': 'Low-level CAPE (0-3 km)',
            'surface_cin': 'Surface-based CIN',
            'mixed_layer_cin': 'Mixed-layer CIN (180-0 mb)',
            'low_level_cin': 'Low-level CIN (0-3 km)',
            'lifted_index': 'Surface-based lifted index',
            'reflectivity_base': 'Base reflectivity configuration',
            'composite_reflectivity': 'Composite reflectivity',
            'height_reflectivity': 'Height-based reflectivity',
            'surface_temperature': 'Surface temperature fields',
            'surface_moisture': 'Surface moisture fields',
            'surface_wind': 'Surface wind fields',
            'surface_pressure_field': 'Surface pressure',
            'precipitation_rate': 'Precipitation rate',
            'precipitation_accumulation': 'Precipitation accumulation',
            'hail_field': 'Hail size',
            'storm_helicity': 'Storm relative helicity',
            'updraft_helicity': 'Hourly maximum updraft helicity (0-3 km)',
            'updraft_helicity_25km': 'Updraft helicity 2-5 km AGL',
            'updraft_helicity_01km': 'Updraft helicity 0-2 km AGL',
            'wind_shear_base': 'Base wind shear configuration',
            'wind_shear_u_06km': '0-6 km U-component wind shear',
            'wind_shear_v_06km': '0-6 km V-component wind shear',
            'wind_shear_u_01km': '0-1 km U-component wind shear',
            'wind_shear_v_01km': '0-1 km V-component wind shear',
            'lightning_strike_density': 'Lightning strike density',
            'lightning_flash_rate': 'Lightning flash rate',
            'echo_top': 'Echo top height',
            'vil': 'Vertically integrated liquid',
            'atmospheric_field': 'General atmospheric field',
            'boundary_layer': 'Boundary layer height',
            'smoke_surface': 'Surface smoke concentration (8m AGL)',
            'smoke_column': 'Column-integrated smoke mass',
            'smoke_aloft': 'Smoke concentration at altitude',
            'dust_concentration': 'Fine dust concentration',
            'vertically_integrated_smoke': 'Vertically integrated smoke mass',
            'column_smoke': 'General column smoke products'
        }
        return descriptions

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate a field configuration"""
        required_keys = ['title', 'units', 'cmap', 'levels', 'extend']
        
        try:
            # Resolve template to get full config
            resolved = cls.resolve_template(config)
            
            # Check required keys
            for key in required_keys:
                if key not in resolved:
                    print(f"Missing required key: {key}")
                    return False
            
            # Check if this is a derived parameter
            if resolved.get('derived'):
                # Derived parameters need inputs and function instead of access/var
                if 'inputs' not in resolved:
                    print("Missing inputs specification for derived parameter")
                    return False
                if 'function' not in resolved:
                    print("Missing function specification for derived parameter")
                    return False
            else:
                # Regular fields need access method and var
                if 'access' not in resolved:
                    print("Missing access configuration")
                    return False
                    
                # Validate var
                if 'var' not in resolved:
                    print("Missing var specification")
                    return False
                
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False