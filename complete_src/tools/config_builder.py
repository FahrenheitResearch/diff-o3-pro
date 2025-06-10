#!/usr/bin/env python3
"""
HRRR Configuration Builder
Loads and builds field configurations from templates and parameter files
"""

import json
from pathlib import Path
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from typing import Dict, Any, List, Optional
import copy

from field_templates import FieldTemplates


class ConfigBuilder:
    """Builds field configurations from templates and parameter definitions"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration builder
        
        Args:
            config_dir: Directory containing parameter configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent / 'parameters'
        self.templates = FieldTemplates()
        self._field_cache = {}
        
    def load_parameter_file(self, file_path: Path) -> Dict[str, Any]:
        """Load parameter configuration from JSON or YAML file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() == '.json':
                    return json.load(f)
                elif file_path.suffix.lower() in ['.yml', '.yaml']:
                    if YAML_AVAILABLE:
                        return yaml.safe_load(f)
                    else:
                        print(f"YAML not available, skipping {file_path}")
                        return {}
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def load_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Load all parameter configurations from config directory"""
        all_params = {}
        
        if not self.config_dir.exists():
            print(f"Config directory not found: {self.config_dir}")
            return {}
        
        # Load all configuration files
        for config_file in self.config_dir.glob('*.json'):
            file_params = self.load_parameter_file(config_file)
            if file_params:
                print(f"Loaded {len(file_params)} parameters from {config_file.name}")
                all_params.update(file_params)
        
        for config_file in self.config_dir.glob('*.yml'):
            file_params = self.load_parameter_file(config_file)
            if file_params:
                print(f"Loaded {len(file_params)} parameters from {config_file.name}")
                all_params.update(file_params)
                
        return all_params
    
    def build_field_config(self, field_name: str, field_def: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete field configuration from definition"""
        try:
            # Use cached config if available
            cache_key = f"{field_name}:{hash(str(sorted(field_def.items())))}"
            if cache_key in self._field_cache:
                return copy.deepcopy(self._field_cache[cache_key])
            
            # Resolve template and build config
            resolved_config = self.templates.resolve_template(field_def)
            
            # Add field name if not present
            if 'name' not in resolved_config:
                resolved_config['name'] = field_name
            
            # Validate configuration
            if not self.templates.validate_config(resolved_config):
                raise ValueError(f"Invalid configuration for field: {field_name}")
            
            # Cache and return
            self._field_cache[cache_key] = copy.deepcopy(resolved_config)
            return resolved_config
            
        except Exception as e:
            print(f"Error building config for {field_name}: {e}")
            return None
    
    def build_all_configs(self, parameter_defs: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Build all field configurations"""
        if parameter_defs is None:
            parameter_defs = self.load_all_parameters()
        
        configs = {}
        failed_fields = []
        
        for field_name, field_def in parameter_defs.items():
            config = self.build_field_config(field_name, field_def)
            if config:
                configs[field_name] = config
            else:
                failed_fields.append(field_name)
        
        if failed_fields:
            print(f"Failed to build configs for: {failed_fields}")
        
        print(f"Successfully built {len(configs)} field configurations")
        return configs
    
    def get_fields_by_category(self, configs: Dict[str, Dict[str, Any]], category: str) -> Dict[str, Dict[str, Any]]:
        """Get all fields in a specific category"""
        return {
            name: config for name, config in configs.items()
            if config.get('category') == category
        }
    
    def get_available_categories(self, configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get list of available categories"""
        categories = set()
        for config in configs.values():
            if 'category' in config:
                categories.add(config['category'])
        return sorted(list(categories))
    
    def create_field_from_template(self, field_name: str, template_name: str, 
                                 overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new field configuration from template with overrides"""
        field_def = {'template': template_name}
        if overrides:
            field_def.update(overrides)
        
        return self.build_field_config(field_name, field_def)
    
    def export_config(self, configs: Dict[str, Dict[str, Any]], output_file: Path):
        """Export configurations to file for inspection/backup"""
        try:
            with open(output_file, 'w') as f:
                if output_file.suffix.lower() == '.json':
                    json.dump(configs, f, indent=2, default=str)
                elif YAML_AVAILABLE:
                    yaml.dump(configs, f, default_flow_style=False)
                else:
                    # Fall back to JSON if YAML not available
                    json.dump(configs, f, indent=2, default=str)
            print(f"Exported configurations to {output_file}")
        except Exception as e:
            print(f"Error exporting configurations: {e}")
    
    def validate_all_configs(self, configs: Dict[str, Dict[str, Any]]) -> bool:
        """Validate all configurations"""
        valid = True
        for field_name, config in configs.items():
            if not self.templates.validate_config(config):
                print(f"Invalid configuration for field: {field_name}")
                valid = False
        return valid
    
    def get_config_summary(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics of configurations"""
        summary = {
            'total_fields': len(configs),
            'categories': {},
            'colormaps': {},
            'access_patterns': {},
            'units': {}
        }
        
        for config in configs.values():
            # Count by category
            category = config.get('category', 'unknown')
            summary['categories'][category] = summary['categories'].get(category, 0) + 1
            
            # Count by colormap
            cmap = config.get('cmap', 'unknown')
            summary['colormaps'][cmap] = summary['colormaps'].get(cmap, 0) + 1
            
            # Count by access pattern
            access = config.get('access', {})
            access_type = access.get('typeOfLevel', access.get('paramId', 'unknown'))
            summary['access_patterns'][access_type] = summary['access_patterns'].get(access_type, 0) + 1
            
            # Count by units
            units = config.get('units', 'unknown')
            summary['units'][units] = summary['units'].get(units, 0) + 1
        
        return summary


def create_example_parameter_config():
    """Create example parameter configuration for demonstration"""
    config_dir = Path(__file__).parent / 'parameters'
    config_dir.mkdir(exist_ok=True)
    
    # Example: Easy way to add new CAPE field
    new_field_example = {
        "my_custom_cape": {
            "template": "surface_cape",
            "overrides": {
                "title": "My Custom CAPE",
                "levels": [200, 500, 1000, 2000, 3000, 4000, 5000]
            }
        }
    }
    
    example_file = config_dir / 'example_custom.json'
    with open(example_file, 'w') as f:
        json.dump(new_field_example, f, indent=2)
    
    print(f"Created example configuration: {example_file}")
    

if __name__ == '__main__':
    # Demo usage
    builder = ConfigBuilder()
    
    # Create example config
    create_example_parameter_config()
    
    # Show available templates
    print("Available templates:")
    for template, desc in builder.templates.get_available_templates().items():
        print(f"  {template}: {desc}")
    
    # Create a field from template
    test_field = builder.create_field_from_template(
        'test_reflectivity',
        'height_reflectivity', 
        {'level': 2000, 'title': '2 km Reflectivity'}
    )
    
    if test_field:
        print(f"\nCreated test field:")
        for key, value in test_field.items():
            print(f"  {key}: {value}")