import os
import yaml
import argparse
import warnings
import pprint

def get_configs_from_file(script_file_name, config_file_path):
    args = { }
    
    if config_file_path is None:
        return args
    
    if not os.path.exists(config_file_path):
        warnings.warn(f"local config file {config_file_path} does not exists")
        return args

    with open(config_file_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    args.update(configs.get('common', { })) # first get the common arguments
    args.update(configs.get(script_file_name, { })) # then update with the script-specific ones

    return args

def _remove_defaults(args: dict, raw_args: list):
    modified_args = { }
    for k, v in args.items():
        if ('--' + k) in raw_args:
            modified_args[k] = v
    return modified_args

class ProxyArgumentParser(object):
    def __init__(self, args) -> None:
        self.args = args
    def parse_args(self):
        return self.args

def to_yaml_interface(script_file_path, globa_config_folder="./configs"):
    # wrapper for accepting decorator arguments

    def _to_yaml_interface_decorator(create_argparse_fn):
        # The decorator that enhances OpenAI's command line interface

        def create_yaml_enhanced_argparser():
            script_file_name = os.path.basename(script_file_path)
            orig_parser: argparse.ArgumentParser = create_argparse_fn() # original argparser
            openai_default = vars(orig_parser.parse_args([])) # default args set by scripts
            
            parser = argparse.ArgumentParser()
            # required positional argument specifying the dataset (add ..
            # .. more choices as we explore more datasets). config .yml files
            # must be named `<dataset>.yml`.
            # parser.add_argument('dataset', type=str,
                                    # choices=['cifar10', 'tinyimagenet', 'celeba', 'lsun'],
                                    # help='select a dataset from available choices')
            parser.add_argument('--config', type=str, required=False,
                                    help='specify local yaml config file to override arguments')
            parser.add_argument('--export', type=str, required=False,
                                    help='specify yaml file to export final arguments')
            knowns, unknowns = parser.parse_known_args() # only the above arguments are parsed
            
            # The global configs (These files should be tracked by git)
            # These should contain the best configuration parameters proved from experimentation
            # global_file_args = get_configs_from_file(script_file_name,
                            # os.path.join(globa_config_folder, f'{knowns.dataset}.yml'))

            # The local config file (These shouldn't be tracked by git). For local dev purpose.
            local_file_args = get_configs_from_file(script_file_name, knowns.config)
            
            # The command line arguments; helpful for quickly changing an argument
            cmd_args = vars(orig_parser.parse_args(unknowns)) # let the original parser parse the unknown ones
            cmd_args = _remove_defaults(cmd_args, unknowns) # only the ones passed through command line

            # Following determines the precedence of arguments from different sources.
            # Precedence: OpenAI defaults -> Global .yml -> Local .yml -> cmd line
            args = { }
            args.update(openai_default)
            # args.update(global_file_args)
            args.update(local_file_args)
            args.update(cmd_args)

            # optional exporting functionalities
            if knowns.export is not None:
                with open(knowns.export, 'w') as f:
                    yaml.dump(
                        {
                            # following the same format as global/local .yaml config files
                            script_file_name: args
                        }, f, default_flow_style=False)
                    print(f'Configuration file {knowns.export} written')
                    exit(0)
            
            # just show the whole configuration on console for visual inspection
            pprint.pprint(args)
            
            # Just to keep it compatible with majority of OpenAI's code
            return ProxyArgumentParser(argparse.Namespace(**args))
        
        return create_yaml_enhanced_argparser # new argparser creator

    return _to_yaml_interface_decorator