import configparser
debug = True

def read_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Debug
    if debug:
        print(parser['config']['input'])
        print(parser['config']['output'])

    return parser
