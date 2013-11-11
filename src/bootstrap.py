import sys
import os

params = {}

def bootstrap():
    global params

    Handlers = {
        'images_path': str,
        'videos_path': str,
    }

    config_file_name = 'config'
    try:
        reader = open(config_file_name, 'r')
    except IOError:
        print >> sys.stderr, \
            'Configuration file named "%s" is required' % config_file_name
        sys.exit(1)

    for line in reader:
        line = line.split('#')[0].strip()
        if not line:
            continue

        name, value = line.split()
        if name not in Handlers:
            print >> sys.stderr, 'Bad parameter name "%s"' % name
            sys.exit(1)
        if name in params:
            print >> sys.stderr, 'Duplicate parameter name "%s"' % name
            sys.exit(1)

        conversion_func = Handlers[name]
        params[name] = conversion_func(value)

    if not os.path.isdir(params['images_path']):
        os.makedirs(params['images_path'])

    if not os.path.isdir(params['videos_path']):
        os.makedirs(params['videos_path'])
